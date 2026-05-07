"""Experiment runner: orchestrates runs from config manifests.

Handles model loading, task generation, forward passes,
intervention application, metrics collection, and artifact saving.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

from analysis.metrics.core import MetricsBundle
from configs.schema import RunConfig, load_run_config
from models.adapters.factory import create_adapter


def _coerce_progress_value(value: Any) -> Any:
    """Convert progress payload values to JSON-safe primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_progress_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_progress_value(v) for k, v in value.items()}
    return str(value)


def _special_positions(token_ids: List[int], tokenizer: Any) -> Set[int]:
    """Return set of positions occupied by special tokens."""
    special_ids = set(tokenizer.all_special_ids)
    return {i for i, tid in enumerate(token_ids) if tid in special_ids}


def _is_causal(adapter: Any) -> bool:
    """Check if the adapter is a decoder-only causal LM."""
    from models.adapters.base import ModelFamily
    return adapter.family == ModelFamily.DECODER_ONLY


def _add_full_cache_encoder_lens_metrics(
    runner: "ExperimentRunner",
    cache_dir: Path,
) -> bool:
    """Populate encoder lens-decoding metrics from a full probe cache."""
    from analysis.probes.lens_cache import compute_encoder_lens_metrics_from_full_cache

    def _progress(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            runner._write_progress(
                "lens_decoding_from_full_cache",
                done=done,
                total=total,
                cache_dir=cache_dir,
            )

    runner._write_progress("lens_decoding_from_full_cache", done=0, total=1, cache_dir=cache_dir)
    refreshed = compute_encoder_lens_metrics_from_full_cache(
        cache_dir,
        runner.adapter,
        progress_callback=_progress,
    )
    runner.metrics.add("avg_retention_by_layer_ns", refreshed["avg_retention_by_layer_ns"])
    runner.metrics.add("early_layers_avg_retention_ns", refreshed["early_layers_avg_retention_ns"])
    runner.metrics.add("late_layers_avg_retention_ns", refreshed["late_layers_avg_retention_ns"])
    runner.metrics.add("all_layers_avg_retention_ns", refreshed["all_layers_avg_retention_ns"])
    runner.metrics.add("num_layers", refreshed["num_layers"])
    runner.metrics.add("lens_modes_run", ["normed"])
    runner.metrics.add("lens_decode_source", "full_cache_output_head")
    runner.metrics.add("lens_full_cache_total_examples", refreshed["lens_full_cache_total_examples"])
    runner.metrics.add("lens_full_cache_total_tokens", refreshed["lens_full_cache_total_tokens"])
    runner.metrics.add("lens_full_cache_storage_dtype", refreshed["lens_full_cache_storage_dtype"])
    for example_id in refreshed["example_ids"]:
        runner.metrics.add_per_example(example_id, {"num_layers": refreshed["num_layers"]})
    runner._write_progress("lens_decoding_from_full_cache", done=1, total=1, cache_dir=cache_dir)
    return True


def _get_attention_sublayer(layer_module: Any) -> Optional[Any]:
    """Try to find the self-attention sub-module within a transformer layer.

    Covers: GPT-2 (attn), Pythia (attention), LLaMA/Qwen/SmolLM (self_attn).
    """
    for attr in ["self_attn", "attention", "attn"]:
        if hasattr(layer_module, attr):
            return getattr(layer_module, attr)
    return None


def _get_ffn_sublayer(layer_module: Any) -> Optional[Any]:
    """Try to find the FFN sub-module within a transformer layer.

    Covers: GPT-2 (mlp), Pythia (mlp), LLaMA/Qwen/SmolLM (mlp).
    """
    for attr in ["mlp", "feed_forward", "ffn"]:
        if hasattr(layer_module, attr):
            return getattr(layer_module, attr)
    return None


class ExperimentRunner:
    """Top-level runner that executes a single RunConfig."""

    _split_cache: Dict[tuple, Any] = {}

    def __init__(self, config: RunConfig, base_dir: Path | None = None):
        self.config = config
        self.base_dir = base_dir or Path(".")
        self.adapter = None
        self.metrics = MetricsBundle(
            run_id=config.run_id,
            phase=config.phase,
            model_name=config.model.name,
            task_family=config.task_family,
            dataset_split=config.dataset.split,
            seed=config.seed,
        )

    def setup(self) -> None:
        """Load model and tokenizer."""
        family_override = None
        if self.config.model.family != "auto":
            from models.adapters.base import ModelFamily
            family_override = ModelFamily[self.config.model.family.upper()]

        self.adapter = create_adapter(
            self.config.model.name,
            device=self.config.model.device,
            precision=self.config.model.precision,
            output_attentions=self.config.model.output_attentions,
            family=family_override,
        )
        self.adapter.load()
        self._write_progress("setup_complete", done=1, total=1, device=self.config.model.device)

    def run(self) -> MetricsBundle:
        """Execute the experiment and return metrics."""
        start = time.time()
        self.metrics.add("start_time", datetime.now().isoformat())
        self._write_progress("running", done=0, total=1, task_family=self.config.task_family)

        self._run_task()

        elapsed = time.time() - start
        self.metrics.add("elapsed_seconds", round(elapsed, 2))
        self.metrics.add("end_time", datetime.now().isoformat())
        self._write_progress("completed", done=1, total=1, elapsed_seconds=round(elapsed, 2))

        self._save_artifacts()
        return self.metrics

    def _progress_path(self) -> Path:
        return self.base_dir / self.config.output_dir / self.config.run_id / "progress.json"

    def _write_progress(self, stage: str, done: int, total: int, **extra: Any) -> None:
        """Write a lightweight JSON progress marker for long-running tasks."""
        path = self._progress_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.config.run_id,
            "task_family": self.config.task_family,
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "updated_at": datetime.now().isoformat(),
        }
        payload.update({k: _coerce_progress_value(v) for k, v in extra.items()})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _run_task(self) -> None:
        """Dispatch to the appropriate task family."""
        family = self.config.task_family
        dispatch = {
            "clean_passthrough": self._run_clean_passthrough,
            "swap_repair": self._run_swap_repair,
            "lens_decoding": self._run_lens_decoding,
            "probe_training": self._run_probe_training,
            "single_corrupt_repair": self._run_single_corrupt_repair,
            "special_token_intervention": self._run_special_token_intervention,
            "swap_independence": self._run_swap_independence,
            "distant_swap_repair": self._run_distant_swap_repair,
            "trajectory_geometry": self._run_trajectory_geometry,
            "probe_holdout": self._run_probe_holdout,
            "weak_example_replay": self._run_weak_example_replay,
            "cyclic_shuffle_comparison": self._run_cyclic_shuffle,
            "attention_centrality": self._run_attention_centrality,
            "repeated_layer_robustness": self._run_repeated_layer_robustness,
            "decoder_control_intervention": self._run_decoder_control_intervention,
            "decoder_repeated_layer": self._run_decoder_repeated_layer,
            "decoder_attention_centrality": self._run_decoder_attention_centrality,
            "decoder_early_exit": self._run_decoder_early_exit,
            "decoder_prefix_corruption": self._run_decoder_prefix_corruption,
            "decoder_attractor": self._run_decoder_attractor,
            "decoder_tuned_lens": self._run_decoder_tuned_lens,
        }
        handler = dispatch.get(family)
        if handler:
            handler()
        else:
            self.metrics.add("error", f"Unknown task family: {family}")

    def _get_split(self):
        """Load the dataset split for this run."""
        from datasets.data_manager import get_corpus_split, make_causal_prefix_trimmed_split

        cache_key = (
            self.config.dataset.name,
            self.config.dataset.split,
            self.config.dataset.max_examples,
            self.config.dataset.variant,
            self.config.dataset.trim_min_tokens,
            self.config.dataset.trim_max_tokens,
            self.config.dataset.min_visible_tokens,
            self.config.seed,
            self.config.model.name,
        )
        cached = self._split_cache.get(cache_key)
        if cached is not None:
            return cached

        split = get_corpus_split(
            split=self.config.dataset.split,
            max_examples=self.config.dataset.max_examples,
        )
        if self.config.dataset.variant == "clm_prefix_trimmed":
            split = make_causal_prefix_trimmed_split(
                split=split,
                tokenizer=self.adapter.tokenizer,
                seed=self.config.seed,
                trim_min_tokens=max(1, self.config.dataset.trim_min_tokens or 2),
                trim_max_tokens=max(1, self.config.dataset.trim_max_tokens or 5),
                min_visible_tokens=max(1, self.config.dataset.min_visible_tokens or 4),
            )
            trim_meta = getattr(split, "trim_metadata", {})
            for key, value in trim_meta.items():
                self.metrics.add(f"dataset_{key}", value)

        self._split_cache[cache_key] = split
        return split

    def _run_clean_passthrough(self) -> None:
        """Family B: unmasked denoising scan.

        Metrics exclude special tokens. For causal LMs, logits[i] predicts
        input_ids[i+1], so we shift the comparison accordingly.
        """

        split = self._get_split()
        changed_count = 0
        total_count = 0
        causal = _is_causal(self.adapter)
        num_examples = min(len(split.examples), self.config.batch_size) if self.config.batch_size else len(split.examples)
        # Per-position in-seq retention at last two output positions
        _last_inseq_sum, _last_inseq_n = 0, 0
        _penult_inseq_sum, _penult_inseq_n = 0, 0

        for ex_idx, ex in enumerate(split.examples[:num_examples], start=1):
            enc = self.adapter.encode_text(ex["text"])
            out = self.adapter.forward(**enc)

            orig_ids = enc["input_ids"].squeeze(0).tolist()
            pred_ids = out.logits.argmax(dim=-1).squeeze(0).tolist()
            sp = _special_positions(orig_ids, self.adapter.tokenizer)

            ex_changes = []
            ex_changed = 0
            ex_total = 0

            if causal:
                # logits[i] predicts orig_ids[i+1]
                for i in range(len(orig_ids) - 1):
                    target = orig_ids[i + 1]
                    if (i + 1) in sp:
                        continue
                    ex_total += 1
                    if pred_ids[i] != target:
                        ex_changed += 1
                        ex_changes.append({
                            "pos": i + 1,
                            "orig": self.adapter.tokenizer.convert_ids_to_tokens(target),
                            "pred": self.adapter.tokenizer.convert_ids_to_tokens(pred_ids[i]),
                        })
            else:
                for i, (o, p) in enumerate(zip(orig_ids, pred_ids)):
                    if i in sp:
                        continue
                    ex_total += 1
                    if o != p:
                        ex_changed += 1
                        ex_changes.append({
                            "pos": i,
                            "orig": self.adapter.tokenizer.convert_ids_to_tokens(o),
                            "pred": self.adapter.tokenizer.convert_ids_to_tokens(p),
                        })

            retention = 1.0 - (ex_changed / max(ex_total, 1))
            changed_count += ex_changed
            total_count += ex_total

            self.metrics.add_per_example(ex["example_id"], {
                "text": ex["text"],
                "retention": round(retention, 4),
                "num_changes": ex_changed,
                "num_non_special": ex_total,
                "changes": ex_changes,
            })

            # Track in-seq retention at last two output positions (causal only)
            if causal:
                n = len(orig_ids)
                # output pos -1: logits[n-2] predicts token[n-1]
                if n >= 2 and (n - 1) not in sp:
                    _last_inseq_sum += int(pred_ids[n - 2] == orig_ids[n - 1])
                    _last_inseq_n += 1
                # output pos -2: logits[n-3] predicts token[n-2]
                if n >= 3 and (n - 2) not in sp:
                    _penult_inseq_sum += int(pred_ids[n - 3] == orig_ids[n - 2])
                    _penult_inseq_n += 1

        self.metrics.add("avg_change_rate", round(changed_count / max(total_count, 1), 4))
        self.metrics.add("total_non_special_tokens", total_count)
        self.metrics.add("causal_shift", causal)
        if causal:
            self.metrics.add("avg_last_inseq_retention",
                             round(_last_inseq_sum / max(_last_inseq_n, 1), 4))
            self.metrics.add("avg_penult_inseq_retention",
                             round(_penult_inseq_sum / max(_penult_inseq_n, 1), 4))

    def _run_swap_repair(self) -> None:
        """Family C: local swap repair.

        Metrics exclude special tokens. Per-example output includes full
        position-by-position comparison for all non-special tokens.
        """
        from models.hooks.token_metadata import extract_token_metadata
        from tasks.generators.corruption import generate_swap_task

        split = self._get_split()
        fully_restored = 0
        swap_pos_restored = 0
        at_least_one_restored = 0
        total = 0
        retention_sum = 0.0
        num_examples = min(len(split.examples), self.config.batch_size) if self.config.batch_size else len(split.examples)

        for ex_idx, ex in enumerate(split.examples[:num_examples], start=1):
            enc = self.adapter.encode_text(ex["text"])
            orig_ids = enc["input_ids"].squeeze(0).tolist()
            tokens = [self.adapter.tokenizer.convert_ids_to_tokens(t) for t in orig_ids]
            metas = extract_token_metadata(orig_ids, self.adapter.tokenizer)
            sp = _special_positions(orig_ids, self.adapter.tokenizer)
            causal = _is_causal(self.adapter)

            task = generate_swap_task(
                ex["text"], tokens, orig_ids, metas,
                seed=self.config.seed,
                tokenizer_id=self.adapter.model_name,
                exclude_positions={0} if causal else None,
            )
            if task is None:
                continue

            corrupted_ids = task.metadata["corrupted_ids"]
            swap_a, swap_b = task.metadata["swap_positions"]
            corrupted_tensor = torch.tensor([corrupted_ids], device=self.adapter.device)
            mask = enc.get("attention_mask")

            out = self.adapter.forward(corrupted_tensor, attention_mask=mask)
            pred_ids = out.logits.argmax(dim=-1).squeeze(0).tolist()

            ns_match = 0
            ns_total = 0
            pos_details = []

            def _pred_for_pos(target_pos: int) -> int:
                """Get the model's prediction for what token should be at target_pos."""
                if causal:
                    return pred_ids[target_pos - 1] if target_pos > 0 else -1
                return pred_ids[target_pos]

            for i in range(len(orig_ids)):
                if i in sp:
                    continue
                if causal and i == 0:
                    continue
                ns_total += 1
                pred_for_i = _pred_for_pos(i)
                matched = pred_for_i == orig_ids[i]
                if matched:
                    ns_match += 1
                pos_details.append({
                    "pos": i,
                    "orig": self.adapter.tokenizer.convert_ids_to_tokens(orig_ids[i]),
                    "input": self.adapter.tokenizer.convert_ids_to_tokens(corrupted_ids[i]),
                    "pred": self.adapter.tokenizer.convert_ids_to_tokens(pred_for_i) if pred_for_i >= 0 else "N/A",
                    "match_orig": matched,
                    "is_swapped": i in (swap_a, swap_b),
                })

            ns_retention = ns_match / max(ns_total, 1)
            ns_full_match = ns_match == ns_total
            pos_a_fixed = _pred_for_pos(swap_a) == orig_ids[swap_a]
            pos_b_fixed = _pred_for_pos(swap_b) == orig_ids[swap_b]
            both_fixed = pos_a_fixed and pos_b_fixed
            either_fixed = pos_a_fixed or pos_b_fixed

            fully_restored += int(ns_full_match)
            swap_pos_restored += int(both_fixed)
            at_least_one_restored += int(either_fixed)
            retention_sum += ns_retention
            total += 1

            self.metrics.add_per_example(ex["example_id"], {
                "text": ex["text"],
                "ns_full_restored": ns_full_match,
                "both_swapped_restored": both_fixed,
                "pos_a_restored": pos_a_fixed,
                "pos_b_restored": pos_b_fixed,
                "swap_positions": [swap_a, swap_b],
                "ns_retention": round(ns_retention, 4),
                "positions": pos_details,
            })

        self.metrics.add("ns_exact_restoration_rate", round(fully_restored / max(total, 1), 4))
        self.metrics.add("both_swap_positions_restored_rate", round(swap_pos_restored / max(total, 1), 4))
        self.metrics.add("at_least_one_swap_restored_rate", round(at_least_one_restored / max(total, 1), 4))
        self.metrics.add("avg_ns_retention", round(retention_sum / max(total, 1), 4))
        self.metrics.add("total_examples", total)

    def _run_lens_decoding(self) -> None:
        """Family A: intermediate hidden-state decoding.

        Retention metrics exclude special tokens and Layer 0.
        For causal LMs, position i predicts token i+1, so we shift the
        comparison target accordingly.

        For decoder models, runs multiple lens modes (normed, optimal, etc.)
        and reports both standard and weighted retention.
        """
        from collections import defaultdict
        from models.lenses.logit_lens import LogitLens, LENS_MODES
        from models.hooks.token_metadata import extract_token_metadata
        from analysis.metrics.core import weighted_in_sequence_retention
        from configs.schema import resolve_lens_mode

        split = self._get_split()
        causal = _is_causal(self.adapter)
        if (not causal) and self.config.dataset.split == "benchmark":
            try:
                from analysis.probes.feature_cache import cache_status

                cache = cache_status(self.base_dir, self.config)
                if cache.exists:
                    if _add_full_cache_encoder_lens_metrics(self, cache.cache_dir):
                        return
            except Exception as e:
                self.metrics.add("lens_full_cache_error", str(e))

        num_examples = min(len(split.examples), self.config.batch_size) if self.config.batch_size else len(split.examples)
        if (not causal) and self.config.dataset.split == "benchmark" and num_examples > 256:
            num_examples = 256
            self.metrics.add("effective_lens_examples", num_examples)
            self.metrics.add("lens_example_cap_reason", "benchmark_encoder_memory_guard")
        self._write_progress("lens_decoding", done=0, total=num_examples)

        # Determine which lens modes to run
        if causal:
            optimal_std = resolve_lens_mode(self.config.model.name, "standard")
            optimal_wt = resolve_lens_mode(self.config.model.name, "weighted")
            modes_to_run = list(dict.fromkeys(["normed", optimal_std, optimal_wt]))
            self.metrics.add("optimal_lens_standard", optimal_std)
            self.metrics.add("optimal_lens_weighted", optimal_wt)
        else:
            modes_to_run = ["normed"]  # encoders: mode doesn't matter

        lenses = {mode: LogitLens(self.adapter, top_k=5, mode=mode)
                  for mode in modes_to_run}

        # Per-mode aggregators
        agg_retention: Dict[str, dict] = {m: defaultdict(list) for m in modes_to_run}
        agg_pos_matches: Dict[str, Dict[int, List[List[bool]]]] | None = None
        if causal:
            agg_pos_matches = {m: defaultdict(list) for m in modes_to_run}
        # Position-specific retention: last_visible and last_visible-1
        agg_lastvis: Dict[str, dict] = {m: defaultdict(list) for m in modes_to_run}
        agg_lastvis_m1: Dict[str, dict] = {m: defaultdict(list) for m in modes_to_run}
        num_layers = 0

        for ex_idx, ex in enumerate(split.examples[:num_examples], start=1):
            enc = self.adapter.encode_text(ex["text"])
            out = self.adapter.forward(**enc)
            orig_ids = enc["input_ids"].squeeze(0).tolist()
            sp = _special_positions(orig_ids, self.adapter.tokenizer)
            seq_len = len(orig_ids)

            if causal:
                shifted_ids = orig_ids[1:] + [orig_ids[-1]]
                compare_ids = torch.tensor([shifted_ids])
                eval_positions = [i for i in range(seq_len - 1) if (i + 1) not in sp]
                last_vis_pos = seq_len - 1
                last_inseq_pos = last_vis_pos - 1 if last_vis_pos > 0 else None
                prev_inseq_pos = last_vis_pos - 2 if last_vis_pos > 1 else None
            else:
                compare_ids = enc["input_ids"]
                eval_positions = [i for i in range(seq_len) if i not in sp]
                last_vis_pos = None
                last_inseq_pos = None
                prev_inseq_pos = None

            for mode in modes_to_run:
                results = lenses[mode].decode_all_layers(
                    out.hidden_states,
                    original_ids=compare_ids,
                    tokenizer=self.adapter.tokenizer,
                )

                for lr in results:
                    matches_list = [
                        bool(lr.top1_matches_original[pos])
                        for pos in eval_positions
                        if pos < len(lr.top1_matches_original)
                    ]
                    ns_matches = sum(matches_list)
                    ns_total = len(eval_positions)
                    rate = ns_matches / max(ns_total, 1)
                    agg_retention[mode][lr.layer_idx].append(rate)
                    if agg_pos_matches is not None:
                        agg_pos_matches[mode][lr.layer_idx].append(matches_list)

                    if causal:
                        # last_visible position retention (gen position lens)
                        if last_vis_pos is not None and last_vis_pos < len(lr.top1_matches_original):
                            agg_lastvis[mode][lr.layer_idx].append(
                                float(bool(lr.top1_matches_original[last_vis_pos])))
                        # last_visible - 1 (last in-seq position)
                        if last_inseq_pos is not None and last_inseq_pos < len(lr.top1_matches_original):
                            agg_lastvis_m1[mode][lr.layer_idx].append(
                                float(bool(lr.top1_matches_original[last_inseq_pos])))

                num_layers = len(results)

            self.metrics.add_per_example(ex["example_id"], {
                "num_layers": num_layers,
            })
            if ex_idx == num_examples or ex_idx % 25 == 0:
                self._write_progress("lens_decoding", done=ex_idx, total=num_examples)

        # Compute summaries per mode
        for mode in modes_to_run:
            avg_by_layer = {
                k: round(sum(v) / len(v), 4)
                for k, v in sorted(agg_retention[mode].items())
            }
            prefix = f"mode_{mode}_" if len(modes_to_run) > 1 else ""

            self.metrics.add(f"{prefix}avg_retention_by_layer_ns", avg_by_layer)

            valid_layers = {k: v for k, v in avg_by_layer.items() if k >= 1}
            early_keys = [k for k in valid_layers if k <= 3]
            late_keys = [k for k in valid_layers if k >= num_layers - 4]
            early_avg = sum(valid_layers[k] for k in early_keys) / max(len(early_keys), 1)
            late_avg = sum(valid_layers[k] for k in late_keys) / max(len(late_keys), 1)
            all_avg = sum(valid_layers[k] for k in valid_layers) / max(len(valid_layers), 1)

            self.metrics.add(f"{prefix}early_layers_avg_retention_ns", round(early_avg, 4))
            self.metrics.add(f"{prefix}late_layers_avg_retention_ns", round(late_avg, 4))
            self.metrics.add(f"{prefix}all_layers_avg_retention_ns", round(all_avg, 4))

            # Weighted retention
            if causal and num_layers >= 2 and agg_pos_matches is not None:
                w_rets = []
                n_ex = len(agg_pos_matches[mode].get(0, []))
                for ex_idx in range(n_ex):
                    ex_matches: Dict[int, List[bool]] = {}
                    ex_seq = 0
                    for L in range(num_layers):
                        if ex_idx < len(agg_pos_matches[mode][L]):
                            ex_matches[L] = agg_pos_matches[mode][L][ex_idx]
                            ex_seq = max(ex_seq, len(ex_matches[L]))
                    if ex_seq > 0:
                        w = weighted_in_sequence_retention(
                            ex_matches, num_layers, ex_seq + 1
                        )
                        w_rets.append(w)
                if w_rets:
                    self.metrics.add(f"{prefix}weighted_retention",
                                     round(float(sum(w_rets) / len(w_rets)), 4))

        # Position-specific retention for causal models
        if causal:
            for mode in modes_to_run:
                prefix = f"mode_{mode}_" if len(modes_to_run) > 1 else ""

                # last_visible retention by layer (gen position)
                if agg_lastvis[mode]:
                    lv_by_layer = {
                        k: round(sum(v) / len(v), 4)
                        for k, v in sorted(agg_lastvis[mode].items())
                    }
                    self.metrics.add(f"{prefix}lastvis_retention_by_layer", lv_by_layer)
                    vl = {k: v for k, v in lv_by_layer.items() if k >= 1}
                    if vl:
                        late_k = [k for k in vl if k >= num_layers - 4]
                        self.metrics.add(f"{prefix}lastvis_late_avg",
                                         round(sum(vl[k] for k in late_k) / max(len(late_k), 1), 4))
                        self.metrics.add(f"{prefix}lastvis_all_avg",
                                         round(sum(vl.values()) / len(vl), 4))

                # last_visible - 1 retention by layer
                if agg_lastvis_m1[mode]:
                    lvm1_by_layer = {
                        k: round(sum(v) / len(v), 4)
                        for k, v in sorted(agg_lastvis_m1[mode].items())
                    }
                    self.metrics.add(f"{prefix}lastvis_m1_retention_by_layer", lvm1_by_layer)
                    vl = {k: v for k, v in lvm1_by_layer.items() if k >= 1}
                    if vl:
                        late_k = [k for k in vl if k >= num_layers - 4]
                        self.metrics.add(f"{prefix}lastvis_m1_late_avg",
                                         round(sum(vl[k] for k in late_k) / max(len(late_k), 1), 4))
                        self.metrics.add(f"{prefix}lastvis_m1_all_avg",
                                         round(sum(vl.values()) / len(vl), 4))

        # Legacy keys (backward compat): point to normed mode
        if "normed" in modes_to_run and len(modes_to_run) > 1:
            normed_by_layer = {
                k: round(sum(v) / len(v), 4)
                for k, v in sorted(agg_retention["normed"].items())
            }
            self.metrics.add("avg_retention_by_layer_ns", normed_by_layer)
            vl = {k: v for k, v in normed_by_layer.items() if k >= 1}
            ek = [k for k in vl if k <= 3]
            lk = [k for k in vl if k >= num_layers - 4]
            self.metrics.add("early_layers_avg_retention_ns",
                             round(sum(vl[k] for k in ek) / max(len(ek), 1), 4))
            self.metrics.add("late_layers_avg_retention_ns",
                             round(sum(vl[k] for k in lk) / max(len(lk), 1), 4))

        self.metrics.add("num_layers", num_layers)
        self.metrics.add("lens_modes_run", modes_to_run)

    def _run_probe_training(self) -> None:
        """Family G: layer prediction probes.

        Uses hidden_states from forward output (not hooks) to correctly
        handle weight-shared architectures like ALBERT.
        Probes use ALL tokens including special — depth code is everywhere.
        """
        import numpy as np
        from models.probes.linear_probe import LinearProbe

        split = self._get_split()

        all_X, all_y = [], []
        probe_limit = min(50, len(split.examples))
        for ex in split.examples[:probe_limit]:
            enc = self.adapter.encode_text(ex["text"])
            out = self.adapter.forward(**enc)

            for layer_idx, hs in enumerate(out.hidden_states):
                flat = hs.squeeze(0).detach().cpu().numpy()  # (seq_len, hidden)
                all_X.append(flat)
                all_y.append(np.full(flat.shape[0], layer_idx))

        if all_X:
            X = np.concatenate(all_X)
            y = np.concatenate(all_y)
            if len(set(y)) > 1:
                probe = LinearProbe()
                probe.fit(X, y)
                acc = probe.score(X, y)
                self.metrics.add("layer_probe_accuracy", acc)
                self.metrics.add("num_layers_probed", int(y.max()) + 1)
                self.metrics.add("num_samples", len(y))

    # ── intervention helper ──────────────────────────────────────

    def _forward_with_layer_fn(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        modify_fn,
    ):
        """Forward pass where *modify_fn(layer_idx, hidden)* is called after each layer.

        Uses a call counter so it works correctly with weight-shared models
        (ALBERT) where all virtual layers share a single module.
        """
        call_counter = [0]

        def hook_fn(module, inp, output):
            layer_idx = call_counter[0]
            call_counter[0] += 1
            if isinstance(output, tuple):
                h = output[0].clone()
            else:
                h = output.clone()
            h = modify_fn(layer_idx, h)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        modules = self.adapter.get_layer_modules()
        seen_ids: Set[int] = set()
        handles = []
        for m in modules:
            mid = id(m)
            if mid not in seen_ids:
                seen_ids.add(mid)
                handles.append(m.register_forward_hook(hook_fn))

        try:
            out = self.adapter.forward(input_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()
        return out

    def _find_special_positions(self, orig_ids: List[int]):
        """Return (cls_pos, sep_positions, all_special) for a token sequence."""
        tokenizer = self.adapter.tokenizer
        sp = _special_positions(orig_ids, tokenizer)
        cls_pos = None
        sep_positions = []

        cls_id = getattr(tokenizer, "cls_token_id", None)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        sep_id = getattr(tokenizer, "sep_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)

        for i, tid in enumerate(orig_ids):
            if cls_id is not None and tid == cls_id:
                cls_pos = i
            elif bos_id is not None and tid == bos_id and cls_pos is None:
                cls_pos = i
            if sep_id is not None and tid == sep_id:
                sep_positions.append(i)
            elif eos_id is not None and tid == eos_id:
                sep_positions.append(i)

        return cls_pos, sep_positions, sp

    # ── Family B extended: single-token corruption repair ────

    def _run_single_corrupt_repair(self) -> None:
        """Test if the model corrects a single corrupted (non-masked) token.

        For each sentence, replaces one random non-special token with another
        sentence-local token. Measures: was it changed? restored? Also runs
        a masked control (same position masked instead of corrupted).
        """
        import random as _random
        from models.hooks.token_metadata import extract_token_metadata

        split = self._get_split()
        num_examples = min(len(split.examples), self.config.batch_size) if self.config.batch_size else len(split.examples)
        rng = _random.Random(self.config.seed)

        stats = {
            "unmasked": {"changed": 0, "restored": 0, "total": 0, "collateral_sum": 0, "retention_sum": 0.0},
            "masked": {"changed": 0, "restored": 0, "total": 0},
        }

        for ex in split.examples[:num_examples]:
            enc = self.adapter.encode_text(ex["text"])
            orig_ids = enc["input_ids"].squeeze(0).tolist()
            att_mask = enc.get("attention_mask")
            sp = _special_positions(orig_ids, self.adapter.tokenizer)
            causal = _is_causal(self.adapter)
            ns_positions = [
                i for i in range(len(orig_ids))
                if i not in sp and not (causal and i == 0)
            ]

            if len(ns_positions) < 3:
                continue

            corrupt_pos = rng.choice(ns_positions)
            pool = [orig_ids[i] for i in ns_positions if i != corrupt_pos and orig_ids[i] != orig_ids[corrupt_pos]]
            if not pool:
                continue
            replacement_id = rng.choice(pool)

            baseline_out = self.adapter.forward(enc["input_ids"], attention_mask=att_mask)
            baseline_preds = baseline_out.logits.argmax(dim=-1).squeeze(0).tolist()

            # --- Unmasked corruption ---
            corrupted_ids = list(orig_ids)
            corrupted_ids[corrupt_pos] = replacement_id
            corrupted_tensor = torch.tensor([corrupted_ids], device=self.adapter.device)

            corrupt_out = self.adapter.forward(corrupted_tensor, attention_mask=att_mask)
            corrupt_preds = corrupt_out.logits.argmax(dim=-1).squeeze(0).tolist()

            def _pred(preds, pos):
                if causal:
                    return preds[pos - 1] if pos > 0 else -1
                return preds[pos]

            pred_at_corrupt = _pred(corrupt_preds, corrupt_pos)
            changed = pred_at_corrupt != replacement_id
            restored = pred_at_corrupt == orig_ids[corrupt_pos]

            collateral = 0
            ns_match = 0
            for i in ns_positions:
                if i == corrupt_pos:
                    continue
                if causal and i == 0:
                    continue
                p = _pred(corrupt_preds, i)
                bp = _pred(baseline_preds, i)
                if p == bp:
                    ns_match += 1
                else:
                    collateral += 1
            ns_other = len([i for i in ns_positions if i != corrupt_pos and not (causal and i == 0)])
            retention = ns_match / max(ns_other, 1)

            stats["unmasked"]["changed"] += int(changed)
            stats["unmasked"]["restored"] += int(restored)
            stats["unmasked"]["total"] += 1
            stats["unmasked"]["collateral_sum"] += collateral
            stats["unmasked"]["retention_sum"] += retention

            # --- Masked control ---
            mask_id = getattr(self.adapter.tokenizer, "mask_token_id", None)
            if mask_id is not None:
                masked_ids = list(orig_ids)
                masked_ids[corrupt_pos] = mask_id
                masked_tensor = torch.tensor([masked_ids], device=self.adapter.device)
                masked_out = self.adapter.forward(masked_tensor, attention_mask=att_mask)
                masked_preds = masked_out.logits.argmax(dim=-1).squeeze(0).tolist()
                m_pred = _pred(masked_preds, corrupt_pos)
                stats["masked"]["changed"] += int(m_pred != mask_id)
                stats["masked"]["restored"] += int(m_pred == orig_ids[corrupt_pos])
                stats["masked"]["total"] += 1

            self.metrics.add_per_example(ex["example_id"], {
                "text": ex["text"],
                "corrupt_pos": corrupt_pos,
                "original_token": self.adapter.tokenizer.convert_ids_to_tokens(orig_ids[corrupt_pos]),
                "replacement_token": self.adapter.tokenizer.convert_ids_to_tokens(replacement_id),
                "predicted_token": self.adapter.tokenizer.convert_ids_to_tokens(pred_at_corrupt) if pred_at_corrupt >= 0 else "N/A",
                "changed": changed,
                "restored": restored,
                "collateral": collateral,
                "ns_retention": round(retention, 4),
            })

        n_um = stats["unmasked"]["total"]
        n_m = stats["masked"]["total"]
        self.metrics.add("unmasked_changed_rate", round(stats["unmasked"]["changed"] / max(n_um, 1), 4))
        self.metrics.add("unmasked_restored_rate", round(stats["unmasked"]["restored"] / max(n_um, 1), 4))
        self.metrics.add("unmasked_avg_collateral", round(stats["unmasked"]["collateral_sum"] / max(n_um, 1), 2))
        self.metrics.add("unmasked_avg_ns_retention", round(stats["unmasked"]["retention_sum"] / max(n_um, 1), 4))
        self.metrics.add("masked_changed_rate", round(stats["masked"]["changed"] / max(n_m, 1), 4))
        self.metrics.add("masked_restored_rate", round(stats["masked"]["restored"] / max(n_m, 1), 4))
        self.metrics.add("total_examples", n_um)

    # ── Family F: special-token interventions ──────────────────

    def _run_special_token_intervention(self) -> None:
        """Family F: causally test CLS/SEP roles via masked-token recovery.

        Conditions:
          baseline              -- no intervention
          freeze_cls            -- lock CLS to embedding value at every layer
          freeze_sep            -- lock SEP to embedding value at every layer
          freeze_both           -- lock BOTH CLS and SEP to embedding values
          zero_cls              -- set CLS to zero at every layer
          zero_sep              -- set SEP to zero at every layer
          zero_both             -- set BOTH CLS and SEP to zero
          zero_ordinary_mean    -- average of zeroing 3 random ordinary positions (control)
          swap_cls_sep_input    -- swap CLS and SEP token IDs in raw input
          replace_cls_with_sep_init -- at every layer, overwrite CLS with SEP's embedding value
          replace_sep_with_cls_init -- at every layer, overwrite SEP with CLS's embedding value
          replace_cls_with_sep_current -- at every layer, overwrite CLS with SEP's CURRENT value (from same layer)
          replace_sep_with_cls_current -- at every layer, overwrite SEP with CLS's CURRENT value (from same layer)
          skip_cls_alternating  -- freeze CLS on even layers only
          skip_sep_alternating  -- freeze SEP on even layers only

        Evaluation:
          choose one random non-special, non-control position and replace it with [MASK]
          for each condition, score the masked position against the original token
          with exact top-1 recovery as the primary metric
        """
        import random as _random

        split = self._get_split()
        num_examples = min(len(split.examples), self.config.batch_size) if self.config.batch_size else len(split.examples)
        rng = _random.Random(self.config.seed)

        CONDITIONS = [
            "baseline",
            "freeze_cls", "freeze_sep", "freeze_both",
            "zero_cls", "zero_sep", "zero_both", "zero_ordinary_mean",
            "swap_cls_sep_input",
            "replace_cls_with_sep_init", "replace_sep_with_cls_init",
            "replace_cls_with_sep_current", "replace_sep_with_cls_current",
            "skip_cls_alternating", "skip_sep_alternating",
        ]
        agg = {
            c: {"exact_sum": 0.0, "agree_sum": 0.0, "changed_sum": 0.0, "count": 0}
            for c in CONDITIONS
        }

        for ex in split.examples[:num_examples]:
            enc = self.adapter.encode_text(ex["text"])
            input_ids = enc["input_ids"]
            att_mask = enc.get("attention_mask")
            orig_ids = input_ids.squeeze(0).tolist()
            sp = _special_positions(orig_ids, self.adapter.tokenizer)
            cls_pos, sep_positions, _ = self._find_special_positions(orig_ids)
            mask_id = getattr(self.adapter.tokenizer, "mask_token_id", None)

            if cls_pos is None or not sep_positions or mask_id is None:
                continue
            sep_pos = sep_positions[0]

            ns_positions = [i for i in range(len(orig_ids)) if i not in sp]
            mask_candidates = [i for i in ns_positions if i not in {cls_pos, sep_pos}]
            if not mask_candidates:
                continue
            mask_pos = rng.choice(mask_candidates)

            masked_ids = list(orig_ids)
            masked_ids[mask_pos] = mask_id
            masked_tensor = torch.tensor([masked_ids], device=self.adapter.device)

            baseline_out = self.adapter.forward(masked_tensor, attention_mask=att_mask)
            baseline_preds = baseline_out.logits.argmax(dim=-1).squeeze(0).tolist()
            clean_hs = baseline_out.hidden_states
            baseline_mask_pred = baseline_preds[mask_pos]
            ordinary_candidates = [
                p for p in ns_positions
                if p not in {cls_pos, sep_pos, mask_pos}
            ]

            ex_results = {}

            for cond in CONDITIONS:
                if cond == "baseline":
                    preds = baseline_preds

                elif cond == "swap_cls_sep_input":
                    swapped_ids = list(masked_ids)
                    swapped_ids[cls_pos], swapped_ids[sep_pos] = swapped_ids[sep_pos], swapped_ids[cls_pos]
                    swapped_tensor = torch.tensor([swapped_ids], device=self.adapter.device)
                    swap_out = self.adapter.forward(swapped_tensor, attention_mask=att_mask)
                    preds = swap_out.logits.argmax(dim=-1).squeeze(0).tolist()

                elif cond == "zero_ordinary_mean":
                    sample_n = min(3, len(ordinary_candidates))
                    if sample_n == 0:
                        pred_id = baseline_mask_pred
                        exact = 1.0 if pred_id == orig_ids[mask_pos] else 0.0
                        agree = 1.0
                        changed = 0.0
                        ex_results[cond] = {
                            "predicted_token_id": int(pred_id),
                            "exact": round(exact, 4),
                            "agrees_with_baseline": round(agree, 4),
                            "changed": round(changed, 4),
                        }
                        agg[cond]["exact_sum"] += exact
                        agg[cond]["agree_sum"] += agree
                        agg[cond]["changed_sum"] += changed
                        agg[cond]["count"] += 1
                        continue
                    chosen = rng.sample(ordinary_candidates, sample_n)
                    exact_sum = 0.0
                    agree_sum = 0.0
                    changed_sum = 0.0
                    pred_ids = []
                    for target_pos in chosen:
                        def _make_zero_fn(pos):
                            def modify_fn(layer_idx, h):
                                h[:, pos, :] = 0.0
                                return h
                            return modify_fn
                        z_out = self._forward_with_layer_fn(masked_tensor, att_mask, _make_zero_fn(target_pos))
                        z_preds = z_out.logits.argmax(dim=-1).squeeze(0).tolist()
                        pred_id = z_preds[mask_pos]
                        pred_ids.append(int(pred_id))
                        exact_sum += 1.0 if pred_id == orig_ids[mask_pos] else 0.0
                        agree_sum += 1.0 if pred_id == baseline_mask_pred else 0.0
                        changed_sum += 1.0 if pred_id != baseline_mask_pred else 0.0
                    exact = exact_sum / sample_n
                    agree = agree_sum / sample_n
                    changed = changed_sum / sample_n
                    ex_results[cond] = {
                        "predicted_token_ids": pred_ids,
                        "exact": round(exact, 4),
                        "agrees_with_baseline": round(agree, 4),
                        "changed": round(changed, 4),
                    }
                    agg[cond]["exact_sum"] += exact
                    agg[cond]["agree_sum"] += agree
                    agg[cond]["changed_sum"] += changed
                    agg[cond]["count"] += 1
                    continue

                else:
                    def _make_modify_fn(condition, hs_tuple, cp, sp_p):
                        def _embed(pos):
                            src = hs_tuple[0]
                            return src[:, pos, :].to(src.device) if src.dim() == 3 else src[pos, :].to(src.device)

                        def modify_fn(layer_idx, h):
                            if condition == "freeze_cls":
                                h[:, cp, :] = _embed(cp)
                            elif condition == "freeze_sep":
                                h[:, sp_p, :] = _embed(sp_p)
                            elif condition == "freeze_both":
                                h[:, cp, :] = _embed(cp)
                                h[:, sp_p, :] = _embed(sp_p)
                            elif condition == "zero_cls":
                                h[:, cp, :] = 0.0
                            elif condition == "zero_sep":
                                h[:, sp_p, :] = 0.0
                            elif condition == "zero_both":
                                h[:, cp, :] = 0.0
                                h[:, sp_p, :] = 0.0
                            elif condition == "replace_cls_with_sep_init":
                                h[:, cp, :] = _embed(sp_p)
                            elif condition == "replace_sep_with_cls_init":
                                h[:, sp_p, :] = _embed(cp)
                            elif condition == "replace_cls_with_sep_current":
                                h[:, cp, :] = h[:, sp_p, :].clone()
                            elif condition == "replace_sep_with_cls_current":
                                h[:, sp_p, :] = h[:, cp, :].clone()
                            elif condition == "skip_cls_alternating":
                                if layer_idx % 2 == 0:
                                    h[:, cp, :] = _embed(cp)
                            elif condition == "skip_sep_alternating":
                                if layer_idx % 2 == 0:
                                    h[:, sp_p, :] = _embed(sp_p)
                            return h
                        return modify_fn

                    fn = _make_modify_fn(cond, clean_hs, cls_pos, sep_pos)
                    intervened = self._forward_with_layer_fn(masked_tensor, att_mask, fn)
                    preds = intervened.logits.argmax(dim=-1).squeeze(0).tolist()

                pred_id = preds[mask_pos]
                exact = 1.0 if pred_id == orig_ids[mask_pos] else 0.0
                agree = 1.0 if pred_id == baseline_mask_pred else 0.0
                changed = 1.0 - agree
                ex_results[cond] = {
                    "predicted_token_id": int(pred_id),
                    "exact": round(exact, 4),
                    "agrees_with_baseline": round(agree, 4),
                    "changed": round(changed, 4),
                }
                agg[cond]["exact_sum"] += exact
                agg[cond]["agree_sum"] += agree
                agg[cond]["changed_sum"] += changed
                agg[cond]["count"] += 1

            self.metrics.add_per_example(ex["example_id"], {
                "text": ex["text"],
                "cls_pos": cls_pos,
                "sep_pos": sep_pos,
                "mask_pos": mask_pos,
                "target_token_id": int(orig_ids[mask_pos]),
                "baseline_mask_pred_id": int(baseline_mask_pred),
                "conditions": ex_results,
            })

        self.metrics.add("evaluation_target", "masked_token_recovery")
        for cond in CONDITIONS:
            n = agg[cond]["count"]
            exact = agg[cond]["exact_sum"] / max(n, 1)
            agree = agg[cond]["agree_sum"] / max(n, 1)
            changed = agg[cond]["changed_sum"] / max(n, 1)
            # Keep avg_retention_* for backward compatibility; it now means
            # masked-token exact top-1 recovery under the intervention.
            self.metrics.add(f"avg_retention_{cond}", round(exact, 4))
            self.metrics.add(f"mask_exact_{cond}", round(exact, 4))
            self.metrics.add(f"mask_agree_with_baseline_{cond}", round(agree, 4))
            self.metrics.add(f"mask_top1_changed_rate_{cond}", round(changed, 4))
        self.metrics.add("num_conditions", len(CONDITIONS))

    # ── Family D: swap-recovery independence ───────────────────

    def _run_swap_independence(self) -> None:
        """Family D: test whether swap repair is coordinated or independent."""
        from models.hooks.token_metadata import extract_token_metadata
        from tasks.generators.corruption import generate_swap_task, generate_swap_control_conditions

        split = self._get_split()
        num_examples = min(len(split.examples), self.config.batch_size) if self.config.batch_size else len(split.examples)

        CONDITIONS = ["standard_swap", "correct_duplicate", "correct_random", "swapped_random", "random_random"]
        agg = {c: {"any_restored": 0, "retention_sum": 0.0, "count": 0} for c in CONDITIONS}

        for ex in split.examples[:num_examples]:
            enc = self.adapter.encode_text(ex["text"])
            orig_ids = enc["input_ids"].squeeze(0).tolist()
            tokens = [self.adapter.tokenizer.convert_ids_to_tokens(t) for t in orig_ids]
            metas = extract_token_metadata(orig_ids, self.adapter.tokenizer)
            sp = _special_positions(orig_ids, self.adapter.tokenizer)
            mask = enc.get("attention_mask")
            causal = _is_causal(self.adapter)

            swap_task = generate_swap_task(
                ex["text"], tokens, orig_ids, metas,
                seed=self.config.seed, tokenizer_id=self.adapter.model_name,
                exclude_positions={0} if causal else None,
            )
            if swap_task is None:
                continue

            swap_a, swap_b = swap_task.metadata["swap_positions"]
            controls = generate_swap_control_conditions(
                ex["text"], tokens, orig_ids, metas,
                swap_a, swap_b,
                seed=self.config.seed, tokenizer_id=self.adapter.model_name,
            )
            if not controls:
                continue

            ex_results = {}
            all_conditions = {"standard_swap": swap_task}
            all_conditions.update(controls)

            for cond_name, task_ex in all_conditions.items():
                c_ids = task_ex.metadata["corrupted_ids"]
                c_tensor = torch.tensor([c_ids], device=self.adapter.device)
                out = self.adapter.forward(c_tensor, attention_mask=mask)
                pred_ids = out.logits.argmax(dim=-1).squeeze(0).tolist()

                pos_a_pred = pred_ids[swap_a - 1] if causal and swap_a > 0 else pred_ids[swap_a]
                pos_b_pred = pred_ids[swap_b - 1] if causal and swap_b > 0 else pred_ids[swap_b]

                a_restored = pos_a_pred == orig_ids[swap_a]
                b_restored = pos_b_pred == orig_ids[swap_b]
                either = a_restored or b_restored

                ns_match = 0
                ns_total = 0
                for i in range(len(orig_ids)):
                    if i in sp:
                        continue
                    if causal and i == 0:
                        continue
                    ns_total += 1
                    p = pred_ids[i - 1] if causal else pred_ids[i]
                    if p == orig_ids[i]:
                        ns_match += 1

                retention = ns_match / max(ns_total, 1)
                ex_results[cond_name] = {
                    "a_restored": a_restored,
                    "b_restored": b_restored,
                    "either_restored": either,
                    "ns_retention": round(retention, 4),
                }
                agg[cond_name]["any_restored"] += int(either)
                agg[cond_name]["retention_sum"] += retention
                agg[cond_name]["count"] += 1

            self.metrics.add_per_example(ex["example_id"], {
                "text": ex["text"],
                "swap_positions": [swap_a, swap_b],
                "conditions": ex_results,
            })

        for cond in CONDITIONS:
            n = agg[cond]["count"]
            self.metrics.add(f"{cond}_any_restored_rate", round(agg[cond]["any_restored"] / max(n, 1), 4))
            self.metrics.add(f"{cond}_avg_ns_retention", round(agg[cond]["retention_sum"] / max(n, 1), 4))
        self.metrics.add("total_examples", agg["standard_swap"]["count"])

    # ── Distant swap repair ──────────────────────────────────

    def _run_distant_swap_repair(self) -> None:
        """Test swap repair at multiple distances (non-local swaps).

        Runs swap repair at distances 1, 3, 5, and 8+ to see whether
        repair rate changes with distance between swapped positions.
        """
        from models.hooks.token_metadata import extract_token_metadata
        from tasks.generators.corruption import generate_swap_task

        split = self._get_split()
        num_examples = min(len(split.examples), self.config.batch_size) if self.config.batch_size else len(split.examples)
        causal = _is_causal(self.adapter)

        DISTANCES = [1, 3, 5, 8]
        agg = {d: {"any_restored": 0, "both_restored": 0, "retention_sum": 0.0, "count": 0} for d in DISTANCES}

        for ex in split.examples[:num_examples]:
            enc = self.adapter.encode_text(ex["text"])
            orig_ids = enc["input_ids"].squeeze(0).tolist()
            tokens = [self.adapter.tokenizer.convert_ids_to_tokens(t) for t in orig_ids]
            metas = extract_token_metadata(orig_ids, self.adapter.tokenizer)
            sp = _special_positions(orig_ids, self.adapter.tokenizer)
            mask = enc.get("attention_mask")

            for dist in DISTANCES:
                task = generate_swap_task(
                    ex["text"], tokens, orig_ids, metas,
                    seed=self.config.seed, tokenizer_id=self.adapter.model_name,
                    distance=dist,
                    exclude_positions={0} if causal else None,
                )
                if task is None:
                    continue

                c_ids = task.metadata["corrupted_ids"]
                swap_a, swap_b = task.metadata["swap_positions"]
                c_tensor = torch.tensor([c_ids], device=self.adapter.device)
                out = self.adapter.forward(c_tensor, attention_mask=mask)
                pred_ids = out.logits.argmax(dim=-1).squeeze(0).tolist()

                def _pred(pos):
                    if causal:
                        return pred_ids[pos - 1] if pos > 0 else -1
                    return pred_ids[pos]

                a_fixed = _pred(swap_a) == orig_ids[swap_a]
                b_fixed = _pred(swap_b) == orig_ids[swap_b]

                ns_match = 0
                ns_total = 0
                for i in range(len(orig_ids)):
                    if i in sp:
                        continue
                    if causal and i == 0:
                        continue
                    ns_total += 1
                    if _pred(i) == orig_ids[i]:
                        ns_match += 1

                retention = ns_match / max(ns_total, 1)
                agg[dist]["any_restored"] += int(a_fixed or b_fixed)
                agg[dist]["both_restored"] += int(a_fixed and b_fixed)
                agg[dist]["retention_sum"] += retention
                agg[dist]["count"] += 1

        for dist in DISTANCES:
            n = agg[dist]["count"]
            self.metrics.add(f"dist{dist}_any_restored_rate", round(agg[dist]["any_restored"] / max(n, 1), 4))
            self.metrics.add(f"dist{dist}_both_restored_rate", round(agg[dist]["both_restored"] / max(n, 1), 4))
            self.metrics.add(f"dist{dist}_avg_ns_retention", round(agg[dist]["retention_sum"] / max(n, 1), 4))
            self.metrics.add(f"dist{dist}_count", n)

    # ── Family G full: trajectory dumps + geometry ─────────────

    def _run_trajectory_geometry(self) -> None:
        """Family G: trajectory dumps, drift curves, CKA, and clustering."""
        import numpy as np
        from analysis.geometry.drift import (
            layerwise_drift_curve, cka_matrix, run_pca, nearest_token_change,
        )
        from analysis.clustering.cluster import run_dbscan, cluster_composition
        from models.hooks.token_metadata import extract_token_metadata

        split = self._get_split()
        limit = min(len(split.examples), self.config.batch_size or 100, 100)
        self._write_progress("trajectory_geometry_collect", done=0, total=limit)

        all_hs_by_layer: Dict[int, List] = {}
        all_families: List[str] = []
        drift_curves = []

        for idx, ex in enumerate(split.examples[:limit], start=1):
            enc = self.adapter.encode_text(ex["text"])
            out = self.adapter.forward(**enc)
            orig_ids = enc["input_ids"].squeeze(0).tolist()
            metas = extract_token_metadata(orig_ids, self.adapter.tokenizer)

            hs = out.hidden_states
            drift = layerwise_drift_curve(hs, metric="cosine")
            drift_curves.append(drift.mean(axis=1))

            for layer_idx, h in enumerate(hs):
                flat = h.squeeze(0).detach().cpu().numpy()
                if layer_idx not in all_hs_by_layer:
                    all_hs_by_layer[layer_idx] = []
                all_hs_by_layer[layer_idx].append(flat)

            for m in metas:
                all_families.append(m.family.name)
            if idx == limit or idx % 10 == 0:
                self._write_progress("trajectory_geometry_collect", done=idx, total=limit)

        num_layers = len(all_hs_by_layer)
        avg_drift = np.mean(drift_curves, axis=0).tolist() if drift_curves else []
        self.metrics.add("avg_cosine_drift_by_transition", [round(d, 4) for d in avg_drift])
        self.metrics.add("num_layers", num_layers)

        hs_concat = {}
        for layer_idx, arrays in all_hs_by_layer.items():
            hs_concat[layer_idx] = np.concatenate(arrays, axis=0)

        if num_layers >= 2:
            self._write_progress("trajectory_geometry_cka", done=0, total=1, num_layers=num_layers)
            hs_tensors = tuple(
                torch.tensor(hs_concat[i]) for i in range(num_layers)
            )
            cka = cka_matrix(hs_tensors)
            self.metrics.add("cka_matrix", [[round(float(c), 4) for c in row] for row in cka])
            self._write_progress("trajectory_geometry_cka", done=1, total=1, num_layers=num_layers)

        late_layer = num_layers - 1
        if late_layer in hs_concat and hs_concat[late_layer].shape[0] >= 10:
            self._write_progress("trajectory_geometry_cluster", done=0, total=1, late_layer=late_layer)
            data = hs_concat[late_layer]
            pca_2d, pca_obj = run_pca(data, n_components=min(2, data.shape[1]))
            self.metrics.add("pca_explained_variance_late", [round(float(v), 4) for v in pca_obj.explained_variance_ratio_])

            cr = run_dbscan(data, eps=0.3, min_samples=5, metric="cosine")
            self.metrics.add("late_layer_clusters", cr.n_clusters)
            self.metrics.add("late_layer_noise_points", cr.n_noise)
            if cr.silhouette is not None:
                self.metrics.add("late_layer_silhouette", round(cr.silhouette, 4))

            families_for_late = all_families[:data.shape[0]]
            if len(families_for_late) == data.shape[0]:
                comp = cluster_composition(cr.labels, families_for_late)
                self.metrics.add("late_layer_cluster_composition", {str(k): v for k, v in comp.items()})
            self._write_progress("trajectory_geometry_cluster", done=1, total=1, late_layer=late_layer)

    # ── Family G upgrade: holdout probes ───────────────────────

    def _run_probe_holdout(self) -> None:
        """Family G: publication-grade probes with holdout and token-family transfer.

        Breakdowns:
        - Layer scope: all / late / last
        - Token position: all / first / gen(-1) / last_visible(-2) / last_visible_m1(-3)
        """
        import numpy as np
        from models.probes.linear_probe import LinearProbe, QDAProbe, MLPProbeTrainer, prepare_token_family_data
        from models.hooks.token_metadata import extract_token_metadata, TokenFamily

        split = self._get_split()
        causal = _is_causal(self.adapter)
        examples = split.examples
        # Full benchmark-scale encoder probe matrices are unnecessarily large:
        # hidden states are expanded across tokens and layers, then fed into
        # CPU sklearn probes. Cap encoder benchmark probe collection to keep
        # RAM and CPU within a usable range while staying on the benchmark split.
        encoder_benchmark_guard = (not causal) and self.config.dataset.split == "benchmark"
        decoder_benchmark_guard = causal and self.config.dataset.split == "benchmark"
        benchmark_token_cap = None
        probe_collection_mode = "full_split"
        if encoder_benchmark_guard and len(examples) > 48:
            examples = examples[:48]
            self.metrics.add("effective_probe_examples", len(examples))
            self.metrics.add("probe_example_cap_reason", "benchmark_encoder_memory_guard")
            benchmark_token_cap = 8
            self.metrics.add("effective_probe_tokens_per_example", benchmark_token_cap)
            probe_collection_mode = "guarded_benchmark_encoder_subset"
        if decoder_benchmark_guard and len(examples) > 64:
            examples = examples[:64]
            self.metrics.add("effective_probe_examples", len(examples))
            self.metrics.add("probe_example_cap_reason", "benchmark_decoder_memory_guard")
            benchmark_token_cap = 12
            self.metrics.add("effective_probe_tokens_per_example", benchmark_token_cap)
            probe_collection_mode = "guarded_benchmark_decoder_subset"
        n = len(examples)
        train_n = max(1, int(n * 0.7))
        train_exs = examples[:train_n]
        test_exs = examples[train_n:]
        self.metrics.add("probe_collection_mode", probe_collection_mode)
        self.metrics.add("probe_collection_train_examples", len(train_exs))
        self.metrics.add("probe_collection_test_examples", len(test_exs))
        if len(test_exs) < 5:
            self.metrics.add("error", "Not enough examples for holdout split")
            return

        total_collect = len(train_exs) + len(test_exs)
        progress_state = {"done": 0, "total": total_collect}
        self._write_progress("probe_holdout_collect", done=0, total=total_collect)

        def _collect(exs, split_name: str):
            X_parts, y_layer, fam_parts, pos_parts = [], [], [], []
            for ex_idx, ex in enumerate(exs, start=1):
                enc = self.adapter.encode_text(ex["text"])
                out = self.adapter.forward(**enc)
                orig_ids = enc["input_ids"].squeeze(0).tolist()
                metas = extract_token_metadata(orig_ids, self.adapter.tokenizer)
                ex_families = [m.family.value for m in metas]
                seq_len = len(orig_ids)
                # Relative positions: 0=first, -1=gen(last), -2=last_visible, -3=prev
                rel_pos = list(range(seq_len))
                for i in range(seq_len):
                    rel_pos[i] = i - seq_len  # negative indexing

                for layer_idx, hs in enumerate(out.hidden_states):
                    flat = hs.squeeze(0).detach().cpu().numpy()
                    keep_idx = None
                    if benchmark_token_cap is not None and flat.shape[0] > benchmark_token_cap:
                        keep_idx = np.linspace(0, flat.shape[0] - 1, benchmark_token_cap, dtype=int)
                        flat = flat[keep_idx]
                    X_parts.append(flat)
                    y_layer.append(np.full(flat.shape[0], layer_idx))
                    if keep_idx is None:
                        fam_parts.extend(ex_families)
                        pos_parts.extend(rel_pos)
                    else:
                        fam_parts.extend(ex_families[i] for i in keep_idx)
                        pos_parts.extend(rel_pos[i] for i in keep_idx)
                progress_state["done"] += 1
                if progress_state["done"] == progress_state["total"] or progress_state["done"] % 25 == 0:
                    self._write_progress(
                        "probe_holdout_collect",
                        done=progress_state["done"],
                        total=progress_state["total"],
                        current_split=split_name,
                        split_progress=ex_idx,
                        split_total=len(exs),
                    )
            X = np.concatenate(X_parts)
            y = np.concatenate(y_layer)
            return X, y, fam_parts, np.array(pos_parts)

        X_train, y_train, fam_train, pos_train = _collect(train_exs, "train")
        X_test, y_test, fam_test, pos_test = _collect(test_exs, "test")

        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            self.metrics.add("error", "Degenerate label set")
            return

        num_layers_total = int(y_train.max()) + 1

        # Train probes on ALL data
        lp = LinearProbe()
        self._write_progress("probe_holdout_linear", done=0, total=1)
        lp.fit(X_train, y_train)
        self.metrics.add("holdout_layer_linear_acc", round(lp.score(X_test, y_test), 4))
        self.metrics.add("insample_layer_linear_acc", round(lp.score(X_train, y_train), 4))
        self._write_progress("probe_holdout_linear", done=1, total=1)

        if encoder_benchmark_guard or decoder_benchmark_guard:
            self.metrics.add("holdout_layer_qda_acc", -1)
            reason = "benchmark_decoder_memory_guard" if decoder_benchmark_guard else "benchmark_encoder_memory_guard"
            self._write_progress("probe_holdout_qda_skipped", done=1, total=1, reason=reason)
        else:
            try:
                self._write_progress("probe_holdout_qda", done=0, total=1)
                qp = QDAProbe()
                qp.fit(X_train, y_train)
                self.metrics.add("holdout_layer_qda_acc", round(qp.score(X_test, y_test), 4))
            except Exception:
                self.metrics.add("holdout_layer_qda_acc", -1)
            finally:
                self._write_progress("probe_holdout_qda", done=1, total=1)

        hidden_dim = X_train.shape[1]
        num_classes = num_layers_total
        mp = None
        try:
            probe_device = self.adapter.device if isinstance(self.adapter.device, str) else "cpu"
            self._write_progress("probe_holdout_mlp", done=0, total=30, device=probe_device)

            def _mlp_progress(epoch: int, total_epochs: int, epoch_loss: float) -> None:
                self._write_progress(
                    "probe_holdout_mlp",
                    done=epoch,
                    total=total_epochs,
                    device=probe_device,
                    epoch_loss=round(float(epoch_loss), 6),
                )

            mlp_cache_used = False
            if self.config.dataset.split == "benchmark":
                try:
                    from analysis.probes.feature_cache import (
                        LayerProbeCacheDataset,
                        cache_status,
                        load_cache_metadata,
                    )
                    cache = cache_status(self.base_dir, self.config)
                    if cache.exists:
                        cache_meta = load_cache_metadata(cache.cache_dir)
                        train_ds = LayerProbeCacheDataset(cache.cache_dir, split="train")
                        test_ds = LayerProbeCacheDataset(cache.cache_dir, split="test")
                        mlp_batch_size = 4096 if str(probe_device).startswith("cuda") else 1024
                        mp = MLPProbeTrainer(
                            int(cache_meta["hidden_size"]),
                            int(cache_meta["num_layers"]),
                            epochs=30,
                            batch_size=mlp_batch_size,
                            device=probe_device,
                            seed=self.config.seed,
                            progress_callback=_mlp_progress,
                        )
                        mp.fit_dataset(train_ds)
                        self.metrics.add("holdout_layer_mlp_acc", round(mp.score_dataset(test_ds), 4))
                        self.metrics.add("mlp_probe_source", "full_cache_torch")
                        self.metrics.add("mlp_full_cache_total_examples", int(cache_meta["total_examples"]))
                        self.metrics.add("mlp_full_cache_total_tokens", int(cache_meta["total_tokens"]))
                        self.metrics.add("mlp_full_cache_train_examples", int(train_ds.train_examples))
                        self.metrics.add("mlp_full_cache_test_examples", int(train_ds.test_examples))
                        self.metrics.add("mlp_full_cache_train_samples", len(train_ds))
                        self.metrics.add("mlp_full_cache_test_samples", len(test_ds))
                        mlp_cache_used = True
                except Exception as e:
                    self.metrics.add("mlp_full_cache_error", str(e))

            if not mlp_cache_used:
                mp = MLPProbeTrainer(
                    hidden_dim,
                    num_classes,
                    epochs=30,
                    device=probe_device,
                    seed=self.config.seed,
                    progress_callback=_mlp_progress,
                )
                mp.fit(X_train, y_train)
                self.metrics.add("holdout_layer_mlp_acc", round(mp.score(X_test, y_test), 4))
                self.metrics.add("mlp_probe_source", "guarded_numpy")
        except Exception:
            self.metrics.add("holdout_layer_mlp_acc", -1)
        finally:
            self._write_progress("probe_holdout_mlp", done=30, total=30, device=self.adapter.device)

        # Position-specific breakdowns (evaluate trained probes on subsets)
        if causal:
            pos_slices = {
                "first": pos_test == (0 - len(set(pos_test))),  # will handle below
                "gen": pos_test == -1,
                "lastvis": pos_test == -2,
                "lastvis_m1": pos_test == -3,
            }
            # Fix first: position 0 in absolute terms maps to -(seq_len) in relative
            # But seq_len varies per example. Use: relative position 0 maps to first token.
            # Since rel_pos[i] = i - seq_len, first token has rel_pos = -seq_len.
            # Simplification: first token = the most negative value per example.
            # Better approach: just check the absolute position modulo
            # Actually the rel_pos encoding I used is -(seq_len) + i, so first = -seq_len.
            # For breakdowns by position, let's use the modular positions:
            # pos == -1 means last token, pos == -2 means second-to-last, etc.
            for name, mask in [("gen", pos_test == -1),
                               ("lastvis", pos_test == -2),
                               ("lastvis_m1", pos_test == -3)]:
                if mask.sum() >= 10:
                    self.metrics.add(f"holdout_linear_at_{name}",
                                     round(lp.score(X_test[mask], y_test[mask]), 4))
                    if mp is not None:
                        self.metrics.add(f"holdout_mlp_at_{name}",
                                         round(mp.score(X_test[mask], y_test[mask]), 4))

            # Layer scope: late layers only (last 4)
            late_mask_test = y_test >= (num_layers_total - 4)
            late_mask_train = y_train >= (num_layers_total - 4)
            if late_mask_test.sum() >= 10 and len(set(y_test[late_mask_test])) >= 2:
                try:
                    lp_late = LinearProbe()
                    lp_late.fit(X_train[late_mask_train], y_train[late_mask_train])
                    self.metrics.add("holdout_linear_late_layers",
                                     round(lp_late.score(X_test[late_mask_test], y_test[late_mask_test]), 4))
                except Exception:
                    pass

            # Layer scope: last layer only
            last_mask_test = y_test == (num_layers_total - 1)
            if last_mask_test.sum() >= 10:
                self.metrics.add("holdout_linear_at_last_layer",
                                 round(lp.score(X_test[last_mask_test], y_test[last_mask_test]), 4))

        # Token-family transfer
        fam_arr_train = np.array(fam_train)
        fam_arr_test = np.array(fam_test)
        content_val = TokenFamily.CONTENT.value
        function_val = TokenFamily.FUNCTION.value
        content_train_mask = fam_arr_train == content_val
        function_train_mask = fam_arr_train == function_val
        content_test_mask = fam_arr_test == content_val
        function_test_mask = fam_arr_test == function_val

        if content_train_mask.sum() > 10 and function_test_mask.sum() > 10:
            probe_c2f = LinearProbe()
            probe_c2f.fit(X_train[content_train_mask], y_train[content_train_mask])
            self.metrics.add("content_to_function_transfer_acc",
                             round(probe_c2f.score(X_test[function_test_mask], y_test[function_test_mask]), 4))

        if function_train_mask.sum() > 10 and content_test_mask.sum() > 10:
            probe_f2c = LinearProbe()
            probe_f2c.fit(X_train[function_train_mask], y_train[function_train_mask])
            self.metrics.add("function_to_content_transfer_acc",
                             round(probe_f2c.score(X_test[content_test_mask], y_test[content_test_mask]), 4))

        self.metrics.add("train_examples", len(train_exs))
        self.metrics.add("test_examples", len(test_exs))
        self.metrics.add("train_samples", len(y_train))
        self.metrics.add("test_samples", len(y_test))
        self.metrics.add("probe_collection_train_samples", len(y_train))
        self.metrics.add("probe_collection_test_samples", len(y_test))
        self.metrics.add("num_layers_total", num_layers_total)

    # ── Family H: weak-example mining + replay ─────────────────

    def _run_weak_example_replay(self) -> None:
        """Family H: mine uncertain masked examples and steer via hidden-state replay.

        Tests whether injecting CONTEXT hidden states (never at the mask position
        itself) from a donor can steer the model's prediction at the masked position.

        Donor types:
          unmasked  -- same sentence without masking (full context)
          spam      -- target word repeated to fill the sequence length

        Patch scopes (where to inject donor states — never at mask position):
          cls           -- CLS position only
          sep           -- SEP position only
          one_ordinary  -- one random non-special, non-mask position (control)
          all_context   -- all non-special positions except the mask

        Layer windows (which layers to patch):
          all_layers    -- patch at every layer
          early         -- patch at layers 0-3 only
          late          -- patch at layers N-4 to N-1 only
        """
        from tasks.generators.weak_examples import mine_weak_examples

        if _is_causal(self.adapter):
            self.metrics.add("error", "Family H requires an MLM model (needs mask token)")
            return

        import random as _random
        rng = _random.Random(self.config.seed)

        split = self._get_split()
        sentences = [ex["text"] for ex in split.examples]

        weak = mine_weak_examples(
            self.adapter, sentences,
            top1_threshold=0.5, min_entropy=1.0,
            max_examples=200, seed=self.config.seed,
        )
        self.metrics.add("num_weak_examples_mined", len(weak))
        if len(weak) == 0:
            return

        tokenizer = self.adapter.tokenizer
        mask_id = tokenizer.mask_token_id
        num_layers = self.adapter.num_layers

        DONORS = ["unmasked", "spam"]
        SCOPES = ["cls", "sep", "cls_sep", "one_ordinary", "all_context"]
        WINDOWS = ["all_layers", "early", "late"]

        results = {}
        for donor_type in DONORS:
            for scope in SCOPES:
                for window in WINDOWS:
                    key = f"{donor_type}__{scope}__{window}"
                    results[key] = {"any": 0, "top1": 0, "rank_up": 0, "total": 0}

        for wex in weak[:50]:
            enc = self.adapter.encode_text(wex.text)
            orig_ids = enc["input_ids"].squeeze(0).tolist()
            att_mask = enc.get("attention_mask")
            sp = _special_positions(orig_ids, tokenizer)
            cls_pos, sep_positions, _ = self._find_special_positions(orig_ids)
            sep_pos = sep_positions[0] if sep_positions else None

            masked_ids = list(orig_ids)
            masked_ids[wex.mask_position] = mask_id
            masked_tensor = torch.tensor([masked_ids], device=self.adapter.device)

            baseline_out = self.adapter.forward(masked_tensor, attention_mask=att_mask)
            baseline_top1 = baseline_out.logits[0, wex.mask_position].argmax().item()
            baseline_topk = set(baseline_out.logits[0, wex.mask_position].topk(10).indices.tolist())

            ns_no_mask = [i for i in range(len(orig_ids)) if i not in sp and i != wex.mask_position]
            if not ns_no_mask:
                continue

            donor_hs_cache = {}

            for donor_type in DONORS:
                if donor_type == "unmasked":
                    donor_out = self.adapter.forward(enc["input_ids"], attention_mask=att_mask)
                    donor_hs_cache[donor_type] = donor_out.hidden_states
                elif donor_type == "spam":
                    target_id = wex.original_token_id
                    spam_ids = list(orig_ids)
                    for i in range(len(spam_ids)):
                        if spam_ids[i] not in set(tokenizer.all_special_ids):
                            spam_ids[i] = target_id
                    spam_tensor = torch.tensor([spam_ids], device=self.adapter.device)
                    spam_out = self.adapter.forward(spam_tensor, attention_mask=att_mask)
                    donor_hs_cache[donor_type] = spam_out.hidden_states

            # Pre-compute baseline rank of target
            baseline_logits_at_mask = baseline_out.logits[0, wex.mask_position]
            baseline_sorted = baseline_logits_at_mask.argsort(descending=True).tolist()
            baseline_target_rank = baseline_sorted.index(wex.original_token_id) if wex.original_token_id in baseline_sorted else len(baseline_sorted)

            for donor_type in DONORS:
                donor_hs = donor_hs_cache[donor_type]

                for scope in SCOPES:
                    if scope == "cls" and cls_pos is None:
                        continue
                    if scope == "sep" and sep_pos is None:
                        continue
                    if scope == "cls_sep" and (cls_pos is None or sep_pos is None):
                        continue

                    if scope == "cls":
                        patch_positions = {cls_pos}
                    elif scope == "sep":
                        patch_positions = {sep_pos}
                    elif scope == "cls_sep":
                        patch_positions = {cls_pos, sep_pos}
                    elif scope == "one_ordinary":
                        patch_positions = {rng.choice(ns_no_mask)}
                    else:
                        patch_positions = set(ns_no_mask)

                    for window in WINDOWS:
                        if window == "all_layers":
                            active_layers = set(range(num_layers))
                        elif window == "early":
                            active_layers = set(range(min(4, num_layers)))
                        else:
                            active_layers = set(range(max(0, num_layers - 4), num_layers))

                        def _make_fn(d_hs, positions, layers_set):
                            def modify_fn(layer_idx, h):
                                if layer_idx not in layers_set:
                                    return h
                                src_idx = min(layer_idx + 1, len(d_hs) - 1)
                                src = d_hs[src_idx]
                                if src.dim() == 3:
                                    src = src.squeeze(0)
                                for p in positions:
                                    if p < h.shape[1] and p < src.shape[0]:
                                        h[:, p, :] = src[p, :].to(h.device)
                                return h
                            return modify_fn

                        fn = _make_fn(donor_hs, patch_positions, active_layers)
                        patched_out = self._forward_with_layer_fn(masked_tensor, att_mask, fn)

                        new_logits = patched_out.logits[0, wex.mask_position]
                        new_top1 = new_logits.argmax().item()
                        new_sorted = new_logits.argsort(descending=True).tolist()
                        new_target_rank = new_sorted.index(wex.original_token_id) if wex.original_token_id in new_sorted else len(new_sorted)

                        any_changed = new_top1 != baseline_top1
                        target_is_top1 = new_top1 == wex.original_token_id
                        rank_improved = new_target_rank < baseline_target_rank

                        key = f"{donor_type}__{scope}__{window}"
                        results[key]["any"] += int(any_changed)
                        results[key]["top1"] += int(target_is_top1)
                        results[key]["rank_up"] += int(rank_improved)
                        results[key]["total"] += 1

        for key, r in results.items():
            n = r["total"]
            self.metrics.add(f"{key}__any_rate", round(r["any"] / max(n, 1), 4))
            self.metrics.add(f"{key}__top1_rate", round(r["top1"] / max(n, 1), 4))
            self.metrics.add(f"{key}__rank_up_rate", round(r["rank_up"] / max(n, 1), 4))
        self.metrics.add("num_conditions", len(results))
        self.metrics.add("num_weak_used", min(50, len(weak)))

    # ── Cyclic/shuffle comparison ──────────────────────────────

    def _run_cyclic_shuffle(self) -> None:
        """Compare per-corrupted-position repair across corruption types.

        For each corruption type, measures what fraction of the CORRUPTED
        positions specifically are restored to their original tokens.
        Also measures collateral damage to non-corrupted positions.
        """
        from models.hooks.token_metadata import extract_token_metadata
        from tasks.generators.corruption import (
            generate_swap_task, generate_cyclic_permutation_task, generate_shuffle_task,
        )

        split = self._get_split()
        num_examples = min(len(split.examples), self.config.batch_size) if self.config.batch_size else len(split.examples)
        causal = _is_causal(self.adapter)

        TYPES = ["swap", "cyclic", "shuffle"]
        agg = {t: {"corrupted_restored": 0, "corrupted_total": 0,
                    "clean_kept": 0, "clean_total": 0, "count": 0} for t in TYPES}

        for ex in split.examples[:num_examples]:
            enc = self.adapter.encode_text(ex["text"])
            orig_ids = enc["input_ids"].squeeze(0).tolist()
            tokens = [self.adapter.tokenizer.convert_ids_to_tokens(t) for t in orig_ids]
            metas = extract_token_metadata(orig_ids, self.adapter.tokenizer)
            sp = _special_positions(orig_ids, self.adapter.tokenizer)
            mask = enc.get("attention_mask")

            generators = {
                "swap": generate_swap_task,
                "cyclic": generate_cyclic_permutation_task,
                "shuffle": generate_shuffle_task,
            }

            for ctype, gen_fn in generators.items():
                task = gen_fn(
                    ex["text"], tokens, orig_ids, metas,
                    seed=self.config.seed, tokenizer_id=self.adapter.model_name,
                )
                if task is None:
                    continue

                c_ids = task.metadata["corrupted_ids"]
                c_tensor = torch.tensor([c_ids], device=self.adapter.device)
                out = self.adapter.forward(c_tensor, attention_mask=mask)
                pred_ids = out.logits.argmax(dim=-1).squeeze(0).tolist()

                corrupted_positions = set()
                if ctype == "swap":
                    corrupted_positions = set(task.metadata["swap_positions"])
                elif ctype == "cyclic":
                    corrupted_positions = set(task.metadata["cycle_positions"])
                elif ctype == "shuffle":
                    corrupted_positions = set(task.metadata["shuffled_positions"])

                for i in range(len(orig_ids)):
                    if i in sp:
                        continue
                    if causal and i == 0:
                        continue
                    p = pred_ids[i - 1] if causal else pred_ids[i]
                    restored = p == orig_ids[i]

                    if i in corrupted_positions:
                        agg[ctype]["corrupted_restored"] += int(restored)
                        agg[ctype]["corrupted_total"] += 1
                    else:
                        agg[ctype]["clean_kept"] += int(restored)
                        agg[ctype]["clean_total"] += 1

                agg[ctype]["count"] += 1

        for ctype in TYPES:
            ct = agg[ctype]["corrupted_total"]
            cl = agg[ctype]["clean_total"]
            self.metrics.add(f"{ctype}_corrupted_restore_rate",
                             round(agg[ctype]["corrupted_restored"] / max(ct, 1), 4))
            self.metrics.add(f"{ctype}_clean_retention_rate",
                             round(agg[ctype]["clean_kept"] / max(cl, 1), 4))
            self.metrics.add(f"{ctype}_num_corrupted_positions", ct)
            self.metrics.add(f"{ctype}_count", agg[ctype]["count"])

    # ── Family E: attention and special-token centrality ─────

    def _run_attention_centrality(self) -> None:
        """Family E: quantify whether CLS and SEP act as privileged aggregation points.

        For each example and layer, computes:
          - mean attention mass directed TO CLS / SEP / ordinary from all positions
          - mean attention mass FROM CLS / SEP to other positions
          - relative-position attention profile (distance-binned)
          - token-family-conditioned attention to special positions

        Supplementary to Family F (causal interventions). Attention mass alone
        is NOT causal evidence — every claim must be read against Family F controls.
        """
        import numpy as np
        from models.hooks.token_metadata import extract_token_metadata

        split = self._get_split()
        num_examples = (
            min(len(split.examples), self.config.batch_size)
            if self.config.batch_size
            else len(split.examples)
        )
        self._write_progress("decoder_attention_centrality", done=0, total=num_examples)

        num_layers = None
        # Per-layer aggregators
        layer_attn_to_cls = {}      # layer -> list of mean attn mass to cls
        layer_attn_to_sep = {}      # layer -> list of mean attn mass to sep
        layer_attn_to_ordinary = {} # layer -> list of mean attn mass to ordinary
        layer_attn_from_cls = {}    # layer -> list of mean attn mass from cls
        layer_attn_from_sep = {}    # layer -> list of mean attn mass from sep
        # Relative-position profile: layer -> {distance -> [attention_values]}
        layer_rel_pos = {}
        # Token-family to special: family -> layer -> [attn_to_special]
        family_to_special = {}

        for ex_idx, ex in enumerate(split.examples[:num_examples], start=1):
            enc = self.adapter.encode_text(ex["text"])
            input_ids = enc["input_ids"]
            att_mask = enc.get("attention_mask")
            orig_ids = input_ids.squeeze(0).tolist()
            cls_pos, sep_positions, sp = self._find_special_positions(orig_ids)

            if cls_pos is None or not sep_positions:
                continue
            sep_pos = sep_positions[0]

            out = self.adapter.forward(input_ids, attention_mask=att_mask)
            if out.attentions is None:
                continue

            metas = extract_token_metadata(orig_ids, self.adapter.tokenizer)
            seq_len = len(orig_ids)
            ns_positions = [i for i in range(seq_len) if i not in sp]

            if num_layers is None:
                num_layers = len(out.attentions)
                for L in range(num_layers):
                    layer_attn_to_cls[L] = []
                    layer_attn_to_sep[L] = []
                    layer_attn_to_ordinary[L] = []
                    layer_attn_from_cls[L] = []
                    layer_attn_from_sep[L] = []
                    layer_rel_pos[L] = {}

            for L in range(num_layers):
                # attn shape: (batch, num_heads, seq_len, seq_len)
                attn = out.attentions[L].squeeze(0)  # (num_heads, seq_len, seq_len)
                # Average across heads
                avg_attn = attn.mean(dim=0)  # (seq_len, seq_len) — [from, to]

                # Attention TO positions: column sums (from all positions to target)
                attn_col_cls = avg_attn[:, cls_pos].mean().item()
                attn_col_sep = avg_attn[:, sep_pos].mean().item()

                if ns_positions:
                    attn_col_ordinary = avg_attn[:, ns_positions].mean().item()
                else:
                    attn_col_ordinary = 0.0

                layer_attn_to_cls[L].append(attn_col_cls)
                layer_attn_to_sep[L].append(attn_col_sep)
                layer_attn_to_ordinary[L].append(attn_col_ordinary)

                # Attention FROM special positions: row sums
                attn_from_cls_val = avg_attn[cls_pos, :].mean().item()
                attn_from_sep_val = avg_attn[sep_pos, :].mean().item()
                layer_attn_from_cls[L].append(attn_from_cls_val)
                layer_attn_from_sep[L].append(attn_from_sep_val)

                # Relative-position attention profile
                for i in range(seq_len):
                    for j in range(seq_len):
                        dist = abs(i - j)
                        if dist not in layer_rel_pos[L]:
                            layer_rel_pos[L][dist] = []
                        layer_rel_pos[L][dist].append(avg_attn[i, j].item())

                # Token-family to special attention
                for i in ns_positions:
                    if i < len(metas):
                        fam = metas[i].family.name
                        if fam not in family_to_special:
                            family_to_special[fam] = {}
                        if L not in family_to_special[fam]:
                            family_to_special[fam][L] = []
                        attn_i_to_special = (
                            avg_attn[i, cls_pos].item() + avg_attn[i, sep_pos].item()
                        )
                        family_to_special[fam][L].append(attn_i_to_special)

            # Per-example summary
            self.metrics.add_per_example(ex["example_id"], {
                "text": ex["text"],
                "cls_pos": cls_pos,
                "sep_pos": sep_pos,
                "seq_len": seq_len,
            })

        if num_layers is None:
            self.metrics.add("error", "No attentions available — model needs output_attentions=True")
            return

        # Aggregate per-layer metrics
        attn_to_cls_by_layer = []
        attn_to_sep_by_layer = []
        attn_to_ordinary_by_layer = []
        attn_from_cls_by_layer = []
        attn_from_sep_by_layer = []

        for L in range(num_layers):
            attn_to_cls_by_layer.append(round(float(np.mean(layer_attn_to_cls[L])), 6))
            attn_to_sep_by_layer.append(round(float(np.mean(layer_attn_to_sep[L])), 6))
            attn_to_ordinary_by_layer.append(round(float(np.mean(layer_attn_to_ordinary[L])), 6))
            attn_from_cls_by_layer.append(round(float(np.mean(layer_attn_from_cls[L])), 6))
            attn_from_sep_by_layer.append(round(float(np.mean(layer_attn_from_sep[L])), 6))

        self.metrics.add("attn_to_cls_by_layer", attn_to_cls_by_layer)
        self.metrics.add("attn_to_sep_by_layer", attn_to_sep_by_layer)
        self.metrics.add("attn_to_ordinary_by_layer", attn_to_ordinary_by_layer)
        self.metrics.add("attn_from_cls_by_layer", attn_from_cls_by_layer)
        self.metrics.add("attn_from_sep_by_layer", attn_from_sep_by_layer)
        self.metrics.add("num_layers", num_layers)

        # Relative-position profile: average by distance per layer
        max_dist = 20  # Cap reporting to 20 positions
        rel_pos_profile = {}
        for L in range(num_layers):
            profile = {}
            for dist in sorted(layer_rel_pos[L].keys()):
                if dist > max_dist:
                    break
                profile[str(dist)] = round(float(np.mean(layer_rel_pos[L][dist])), 6)
            rel_pos_profile[str(L)] = profile
        self.metrics.add("relative_position_attention_profile", rel_pos_profile)

        # Token-family conditioned attention to special positions
        family_special_summary = {}
        for fam, layers_data in family_to_special.items():
            fam_by_layer = []
            for L in range(num_layers):
                if L in layers_data and layers_data[L]:
                    fam_by_layer.append(round(float(np.mean(layers_data[L])), 6))
                else:
                    fam_by_layer.append(0.0)
            family_special_summary[fam] = fam_by_layer
        self.metrics.add("token_family_attn_to_special_by_layer", family_special_summary)

        # Summary scalars
        self.metrics.add("avg_attn_to_cls", round(float(np.mean(attn_to_cls_by_layer)), 6))
        self.metrics.add("avg_attn_to_sep", round(float(np.mean(attn_to_sep_by_layer)), 6))
        self.metrics.add("avg_attn_to_ordinary", round(float(np.mean(attn_to_ordinary_by_layer)), 6))
        self.metrics.add("cls_vs_ordinary_ratio",
                         round(float(np.mean(attn_to_cls_by_layer)) /
                               max(float(np.mean(attn_to_ordinary_by_layer)), 1e-12), 4))
        self.metrics.add("sep_vs_ordinary_ratio",
                         round(float(np.mean(attn_to_sep_by_layer)) /
                               max(float(np.mean(attn_to_ordinary_by_layer)), 1e-12), 4))

    # ── Family I: repeated-layer robustness and geometric correction ──

    def _run_repeated_layer_robustness(self) -> None:
        """Family I: test whether some layers are safer to repeat and whether
        linear/low-rank correction stabilizes extra depth.

        Conditions for each (target_layer, repeat_count) pair:
          1. no_correction — repeat layer with no modification
          2. freeze_control — freeze CLS/SEP to their pre-repeat values
          3. low_rank_correction — apply low-rank additive correction on ordinary tokens
          4. both — freeze control + low-rank correction together

        Correction matrices are derived from fit_layer_transition() and truncated
        to candidate ranks via SVD.
        """
        import numpy as np
        from analysis.geometry.drift import fit_layer_transition, low_rank_approximation_error

        split = self._get_split()
        num_examples = (
            min(len(split.examples), self.config.batch_size)
            if self.config.batch_size
            else len(split.examples)
        )

        # Parameters
        REPEAT_COUNTS = [1, 2, 4]
        CORRECTION_RANKS = [1, 3, 5, 8]
        CONDITIONS = ["no_correction", "freeze_control", "low_rank_correction", "both"]

        modules = self.adapter.get_layer_modules()
        n_layers = len(modules)
        # Representative layer subset: early, mid, late
        if n_layers >= 12:
            TARGET_LAYERS = [3, 6, 9]
        elif n_layers >= 6:
            TARGET_LAYERS = [1, 3, 5]
        else:
            TARGET_LAYERS = list(range(n_layers))
        TARGET_LAYERS = [L for L in TARGET_LAYERS if L < n_layers]

        # Phase 1: collect hidden states and fit transition matrices
        # Use a small sample to fit corrections
        fit_limit = min(num_examples, 50)
        fit_hs_by_layer: Dict[int, List[torch.Tensor]] = {}

        for ex in split.examples[:fit_limit]:
            enc = self.adapter.encode_text(ex["text"])
            out = self.adapter.forward(**enc)
            for layer_idx, h in enumerate(out.hidden_states):
                if layer_idx not in fit_hs_by_layer:
                    fit_hs_by_layer[layer_idx] = []
                fit_hs_by_layer[layer_idx].append(h.squeeze(0).detach())

        # Fit transition matrices for target layers
        transition_matrices = {}
        transition_residuals = {}
        lr_errors = {}
        for target_L in TARGET_LAYERS:
            if target_L < n_layers and (target_L + 1) in fit_hs_by_layer:
                hs_from = torch.cat(fit_hs_by_layer[target_L], dim=0)
                hs_to = torch.cat(fit_hs_by_layer[target_L + 1], dim=0)
                W, resid = fit_layer_transition(hs_from, hs_to)
                transition_matrices[target_L] = W
                transition_residuals[target_L] = resid
                lr_errors[target_L] = low_rank_approximation_error(W, CORRECTION_RANKS)

        # Build low-rank correction matrices: (W_approx - I) for additive correction
        # h_corrected = h + h @ (W_low_rank - I)
        correction_by_layer_rank: Dict[int, Dict[int, torch.Tensor]] = {}
        for target_L in TARGET_LAYERS:
            if target_L in transition_matrices:
                W = transition_matrices[target_L]
                U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                correction_by_layer_rank[target_L] = {}
                for r in CORRECTION_RANKS:
                    W_approx = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
                    correction = W_approx - torch.eye(W.shape[0], device=W.device)
                    correction_by_layer_rank[target_L][r] = correction

        # Phase 2: run repeated-layer experiments
        # agg[target_L][repeat_count][condition] -> {"token_match_sum", "sent_match", "count", "total_tokens"}
        agg: Dict[int, Dict[int, Dict[str, Dict[str, float]]]] = {}
        for target_L in TARGET_LAYERS:
            agg[target_L] = {}
            for rc in REPEAT_COUNTS:
                agg[target_L][rc] = {}
                for cond in CONDITIONS:
                    agg[target_L][rc][cond] = {
                        "token_match_sum": 0, "sent_match": 0,
                        "count": 0, "total_tokens": 0,
                    }

        for ex in split.examples[:num_examples]:
            enc = self.adapter.encode_text(ex["text"])
            input_ids = enc["input_ids"]
            att_mask = enc.get("attention_mask")
            orig_ids = input_ids.squeeze(0).tolist()
            sp = _special_positions(orig_ids, self.adapter.tokenizer)
            cls_pos, sep_positions, _ = self._find_special_positions(orig_ids)
            sep_pos = sep_positions[0] if sep_positions else None
            ns_positions = [i for i in range(len(orig_ids)) if i not in sp]
            ns_count = len(ns_positions)
            if ns_count == 0:
                continue

            # Baseline
            baseline_out = self.adapter.forward(input_ids, attention_mask=att_mask)
            baseline_preds = baseline_out.logits.argmax(dim=-1).squeeze(0).tolist()

            for target_L in TARGET_LAYERS:
                for rc in REPEAT_COUNTS:
                    for cond in CONDITIONS:
                        # Build the modify_fn for this condition
                        # Pre-capture the hidden state before the target layer
                        # for control-position freezing
                        pre_repeat_hs = None
                        if cond in ("freeze_control", "both"):
                            if target_L < len(baseline_out.hidden_states):
                                pre_repeat_hs = baseline_out.hidden_states[target_L].clone()

                        corr_matrix = None
                        if cond in ("low_rank_correction", "both"):
                            # Use rank 5 as default
                            default_rank = 5
                            if target_L in correction_by_layer_rank:
                                if default_rank in correction_by_layer_rank[target_L]:
                                    corr_matrix = correction_by_layer_rank[target_L][default_rank]

                        # Pre-compute rotary position embeddings if needed
                        # (ModernBERT uses layer_type-specific rotary; decoder
                        # models like Pythia/Llama use generic position_embeddings)
                        _pos_emb_by_type = {}
                        _generic_pos_emb = None
                        _base_model = getattr(self.adapter.model, "model",
                                              getattr(self.adapter.model, "gpt_neox",
                                                      self.adapter.model))
                        if hasattr(_base_model, "rotary_emb"):
                            seq_len = input_ids.shape[1]
                            _dummy = torch.randn(1, seq_len, self.adapter.hidden_size,
                                                 device=input_ids.device)
                            _pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                            # Try ModernBERT-style (with layer_type)
                            for lt in ("full_attention", "sliding_attention"):
                                try:
                                    _pos_emb_by_type[lt] = _base_model.rotary_emb(
                                        _dummy, _pos_ids, layer_type=lt)
                                except Exception:
                                    pass
                            # Try generic rotary (Pythia, Llama, Qwen)
                            if not _pos_emb_by_type:
                                try:
                                    _generic_pos_emb = _base_model.rotary_emb(
                                        _dummy, _pos_ids)
                                except Exception:
                                    pass

                        def _make_repeat_fn(
                            t_layer, rcount, freeze_hs, c_matrix,
                            cp, sp_p, condition
                        ):
                            repeat_counter = [0]

                            def modify_fn(layer_idx, h):
                                if layer_idx != t_layer:
                                    return h
                                mod = modules[t_layer]
                                current = h
                                for _ in range(rcount):
                                    extra_kwargs = {}
                                    if _pos_emb_by_type:
                                        attn_type = getattr(mod, "attention_type",
                                                            "full_attention")
                                        pos_emb = _pos_emb_by_type.get(attn_type)
                                        if pos_emb is not None:
                                            extra_kwargs["position_embeddings"] = pos_emb
                                    elif _generic_pos_emb is not None:
                                        extra_kwargs["position_embeddings"] = _generic_pos_emb
                                    with torch.no_grad():
                                        try:
                                            layer_out = mod(current, **extra_kwargs)
                                        except TypeError:
                                            layer_out = mod(current)
                                    if isinstance(layer_out, tuple):
                                        current = layer_out[0]
                                    else:
                                        current = layer_out

                                    # Apply corrections after each extra pass
                                    if condition in ("freeze_control", "both"):
                                        if freeze_hs is not None:
                                            if cp is not None and cp < current.shape[1]:
                                                current[:, cp, :] = freeze_hs[:, cp, :]
                                            if sp_p is not None and sp_p < current.shape[1]:
                                                current[:, sp_p, :] = freeze_hs[:, sp_p, :]

                                    if condition in ("low_rank_correction", "both"):
                                        if c_matrix is not None:
                                            for p in range(current.shape[1]):
                                                if p not in (cp, sp_p):
                                                    current[:, p, :] = (
                                                        current[:, p, :]
                                                        + current[:, p, :] @ c_matrix.to(current.device)
                                                    )

                                return current

                            return modify_fn

                        fn = _make_repeat_fn(
                            target_L, rc, pre_repeat_hs, corr_matrix,
                            cls_pos, sep_pos, cond,
                        )
                        rep_out = self._forward_with_layer_fn(input_ids, att_mask, fn)
                        rep_preds = rep_out.logits.argmax(dim=-1).squeeze(0).tolist()

                        # Token-level match to baseline
                        token_match = sum(
                            1 for i in ns_positions
                            if rep_preds[i] == baseline_preds[i]
                        )
                        sent_match = 1 if token_match == ns_count else 0

                        agg[target_L][rc][cond]["token_match_sum"] += token_match
                        agg[target_L][rc][cond]["total_tokens"] += ns_count
                        agg[target_L][rc][cond]["sent_match"] += sent_match
                        agg[target_L][rc][cond]["count"] += 1

            self.metrics.add_per_example(ex["example_id"], {
                "text": ex["text"],
            })

        # Aggregate and save
        for target_L in TARGET_LAYERS:
            for rc in REPEAT_COUNTS:
                for cond in CONDITIONS:
                    d = agg[target_L][rc][cond]
                    n = d["count"]
                    token_pres = d["token_match_sum"] / max(d["total_tokens"], 1)
                    sent_pres = d["sent_match"] / max(n, 1)
                    key_prefix = f"layer{target_L}_repeat{rc}_{cond}"
                    self.metrics.add(f"{key_prefix}_token_preservation", round(token_pres, 4))
                    self.metrics.add(f"{key_prefix}_sentence_preservation", round(sent_pres, 4))

        # Save transition diagnostics
        for target_L in TARGET_LAYERS:
            if target_L in transition_residuals:
                self.metrics.add(f"layer{target_L}_transition_residual_norm",
                                 round(transition_residuals[target_L], 4))
            if target_L in lr_errors:
                for r, err in lr_errors[target_L].items():
                    self.metrics.add(f"layer{target_L}_lowrank_error_r{r}", round(err, 4))

        self.metrics.add("target_layers", TARGET_LAYERS)
        self.metrics.add("repeat_counts", REPEAT_COUNTS)
        self.metrics.add("correction_ranks", CORRECTION_RANKS)
        self.metrics.add("conditions", CONDITIONS)
        self.metrics.add("num_layers", n_layers)

    # ── decoder position helpers ──────────────────────────────

    def _find_decoder_positions(self, orig_ids: List[int]):
        """Return (bos_pos, last_visible_pos, delimiter_positions, ordinary_positions, all_special)
        for a decoder token sequence.

        BOS: the first token if it is a special BOS/CLS token, else None.
        last_visible: last non-pad token position (always set).
        delimiters: positions of period, comma, newline tokens.
        ordinary: all non-special, non-delimiter positions.
        """
        tokenizer = self.adapter.tokenizer
        sp = _special_positions(orig_ids, tokenizer)

        bos_pos = None
        bos_id = getattr(tokenizer, "bos_token_id", None)
        cls_id = getattr(tokenizer, "cls_token_id", None)
        if bos_id is not None and len(orig_ids) > 0 and orig_ids[0] == bos_id:
            bos_pos = 0
        elif cls_id is not None and len(orig_ids) > 0 and orig_ids[0] == cls_id:
            bos_pos = 0

        # Last visible = last non-pad token
        pad_id = tokenizer.pad_token_id
        last_visible = len(orig_ids) - 1
        if pad_id is not None:
            while last_visible > 0 and orig_ids[last_visible] == pad_id:
                last_visible -= 1

        # Delimiter detection: check decoded token text for punctuation characters.
        # We use character-level matching instead of tokenizer.encode() because
        # sentencepiece tokenizers (TinyLlama) encode bare punctuation differently
        # from how it appears in actual tokenized text.
        _DELIM_CHARS = {".", ",", ";", ":", "!", "?", "\n"}
        delimiter_positions = []
        for i, tid in enumerate(orig_ids):
            if i in sp:
                continue
            tok_text = tokenizer.convert_ids_to_tokens(tid)
            clean = tok_text.replace("\u0120", "").replace("\u2581", "").replace("##", "").strip()
            if clean in _DELIM_CHARS:
                delimiter_positions.append(i)

        # Ordinary = non-special, non-delimiter
        ordinary_positions = [
            i for i in range(len(orig_ids))
            if i not in sp and i not in delimiter_positions
        ]

        return bos_pos, last_visible, delimiter_positions, ordinary_positions, sp

    # ── Family J: decoder control-position interventions ──────

    def _run_decoder_control_intervention(self) -> None:
        """Family J: causally test BOS / last_visible / delimiter roles via per-layer
        interventions with ordinary-position control.

        8 conditions testing whether specific positions are causally important
        for decoder next-token prediction.

        in-seq metrics compare to original input (shifted): logits[i] vs orig_ids[i+1].
        gen metrics compare to baseline generation at last_visible.
        """
        import random as _random

        split = self._get_split()
        num_examples = (
            min(len(split.examples), self.config.batch_size)
            if self.config.batch_size
            else len(split.examples)
        )
        self._write_progress("decoder_control_intervention", done=0, total=num_examples)
        rng = _random.Random(self.config.seed)

        CONDITIONS = [
            "baseline",
            "freeze_bos", "freeze_delimiters", "freeze_first",
            "zero_bos", "zero_delimiters", "zero_first",
            "zero_ordinary_mean",
        ]
        agg = {c: {"in_all_sum": 0.0, "in_all_weighted_sum": 0.0,
                    "in_last_sum": 0.0, "gen_sum": 0.0,
                    "kl_sum": 0.0, "count": 0} for c in CONDITIONS}

        for ex_idx, ex in enumerate(split.examples[:num_examples], start=1):
            enc = self.adapter.encode_text(ex["text"])
            input_ids = enc["input_ids"]
            att_mask = enc.get("attention_mask")
            orig_ids = input_ids.squeeze(0).tolist()

            bos_pos, last_vis, delim_positions, ordinary_pos, sp = \
                self._find_decoder_positions(orig_ids)

            if last_vis < 1:
                continue

            baseline_out = self.adapter.forward(input_ids, attention_mask=att_mask)
            baseline_logits = baseline_out.logits.squeeze(0)
            baseline_gen = baseline_logits[last_vis].argmax().item()
            clean_hs = baseline_out.hidden_states

            # Ground truth for in-seq: orig_ids shifted by 1
            ground_truth = orig_ids[1:] + [orig_ids[-1]]

            eval_positions = [i for i in range(len(orig_ids) - 1) if (i + 1) not in sp]
            ns_count = len(eval_positions)
            if ns_count == 0:
                continue

            # last in-seq position = last_vis - 1 (predicts orig_ids[last_vis])
            last_inseq_pos = last_vis - 1 if last_vis > 0 else None

            seq_len = len(orig_ids)
            ex_results = {}

            for cond in CONDITIONS:
                if cond == "baseline":
                    preds = baseline_logits.argmax(dim=-1).tolist()
                    logits_cond = baseline_logits

                elif cond == "zero_ordinary_mean":
                    candidates = [p for p in ordinary_pos
                                  if p != bos_pos and p != last_vis
                                  and p not in delim_positions]
                    sample_n = min(3, len(candidates))
                    if sample_n == 0:
                        # Fallback: no ordinary candidates, skip averaging
                        preds = baseline_logits.argmax(dim=-1).tolist()
                        logits_cond = baseline_logits
                    else:
                        chosen = rng.sample(candidates, sample_n)
                        # Run each zeroed candidate, average metrics
                        in_all_s, in_last_s, gen_s, kl_s = 0.0, 0.0, 0.0, 0.0
                        in_w_s = 0.0
                        for target_pos in chosen:
                            def _make_zero_fn(pos):
                                def modify_fn(layer_idx, h):
                                    h[:, pos, :] = 0.0
                                    return h
                                return modify_fn
                            z_out = self._forward_with_layer_fn(input_ids, att_mask, _make_zero_fn(target_pos))
                            z_preds = z_out.logits.argmax(dim=-1).squeeze(0).tolist()
                            z_logits = z_out.logits.squeeze(0)
                            mc = sum(1 for i in eval_positions if z_preds[i] == ground_truth[i])
                            in_all_s += mc / ns_count
                            if last_inseq_pos is not None and last_inseq_pos in eval_positions:
                                in_last_s += 1.0 if z_preds[last_inseq_pos] == ground_truth[last_inseq_pos] else 0.0
                            z_gen = z_logits[last_vis].argmax().item()
                            gen_s += 1.0 if z_gen == baseline_gen else 0.0
                            from analysis.metrics.core import kl_divergence
                            kl_vals = kl_divergence(z_logits[eval_positions], baseline_logits[eval_positions])
                            kl_s += kl_vals.mean().item()
                            w_s, w_t = 0.0, 0.0
                            for i in eval_positions:
                                w = (i + 1) / seq_len
                                w_t += w
                                if z_preds[i] == ground_truth[i]:
                                    w_s += w
                            in_w_s += w_s / max(w_t, 1e-9)

                        ex_results[cond] = {
                            "in_all": round(in_all_s / sample_n, 4),
                            "in_all_weighted": round(in_w_s / sample_n, 4),
                            "in_last": round(in_last_s / sample_n, 4),
                            "gen": round(gen_s / sample_n, 4),
                            "kl": round(kl_s / sample_n, 4),
                        }
                        agg[cond]["in_all_sum"] += in_all_s / sample_n
                        agg[cond]["in_all_weighted_sum"] += in_w_s / sample_n
                        agg[cond]["in_last_sum"] += in_last_s / sample_n
                        agg[cond]["gen_sum"] += gen_s / sample_n
                        agg[cond]["kl_sum"] += kl_s / sample_n
                        agg[cond]["count"] += 1
                        continue

                else:
                    def _make_modify_fn(condition, hs_tuple, bp, lv, delims):
                        def _embed(pos):
                            src = hs_tuple[0]
                            return src[:, pos, :].clone() if src.dim() == 3 else src[pos, :].clone()

                        def modify_fn(layer_idx, h):
                            if condition == "freeze_bos":
                                if bp is not None and bp < h.shape[1]:
                                    h[:, bp, :] = _embed(bp)
                            elif condition == "freeze_delimiters":
                                for dp in delims:
                                    if dp < h.shape[1]:
                                        h[:, dp, :] = _embed(dp)
                            elif condition == "zero_bos":
                                if bp is not None and bp < h.shape[1]:
                                    h[:, bp, :] = 0.0
                            elif condition == "zero_delimiters":
                                for dp in delims:
                                    if dp < h.shape[1]:
                                        h[:, dp, :] = 0.0
                            elif condition == "freeze_first":
                                if 0 < h.shape[1]:
                                    h[:, 0, :] = _embed(0)
                            elif condition == "zero_first":
                                if 0 < h.shape[1]:
                                    h[:, 0, :] = 0.0
                            return h
                        return modify_fn

                    fn = _make_modify_fn(cond, clean_hs, bos_pos, last_vis, delim_positions)
                    intervened = self._forward_with_layer_fn(input_ids, att_mask, fn)
                    preds = intervened.logits.argmax(dim=-1).squeeze(0).tolist()
                    logits_cond = intervened.logits.squeeze(0)

                # in_all: fraction of eval positions matching ground truth
                in_all = sum(1 for i in eval_positions if preds[i] == ground_truth[i]) / ns_count

                # in_all_weighted: position-weighted
                w_sum, w_total = 0.0, 0.0
                for i in eval_positions:
                    w = (i + 1) / seq_len
                    w_total += w
                    if preds[i] == ground_truth[i]:
                        w_sum += w
                in_all_weighted = w_sum / max(w_total, 1e-9)

                # in_last: accuracy at last in-seq position (predicts last_visible token)
                in_last = 0.0
                if last_inseq_pos is not None and last_inseq_pos in eval_positions:
                    in_last = 1.0 if preds[last_inseq_pos] == ground_truth[last_inseq_pos] else 0.0

                # gen: does next-token prediction at last_vis match baseline?
                gen_pred = logits_cond[last_vis].argmax().item()
                gen_match = 1.0 if gen_pred == baseline_gen else 0.0

                from analysis.metrics.core import kl_divergence
                kl_val = 0.0
                if cond != "baseline":
                    kl_vals = kl_divergence(
                        logits_cond[eval_positions], baseline_logits[eval_positions])
                    kl_val = kl_vals.mean().item()

                ex_results[cond] = {
                    "in_all": round(in_all, 4),
                    "in_all_weighted": round(in_all_weighted, 4),
                    "in_last": round(in_last, 4),
                    "gen": round(gen_match, 4),
                    "kl": round(kl_val, 4),
                }
                agg[cond]["in_all_sum"] += in_all
                agg[cond]["in_all_weighted_sum"] += in_all_weighted
                agg[cond]["in_last_sum"] += in_last
                agg[cond]["gen_sum"] += gen_match
                agg[cond]["kl_sum"] += kl_val
                agg[cond]["count"] += 1

            self.metrics.add_per_example(ex["example_id"], {
                "text": ex["text"],
                "bos_pos": bos_pos,
                "has_bos": bos_pos is not None,
                "last_visible": last_vis,
                "num_delimiters": len(delim_positions),
                "conditions": ex_results,
            })
            if ex_idx == num_examples or ex_idx % 25 == 0:
                self._write_progress("decoder_control_intervention", done=ex_idx, total=num_examples)

        for cond in CONDITIONS:
            n = agg[cond]["count"]
            if n > 0:
                self.metrics.add(f"in_all_{cond}", round(agg[cond]["in_all_sum"] / n, 4))
                self.metrics.add(f"in_all_weighted_{cond}", round(agg[cond]["in_all_weighted_sum"] / n, 4))
                self.metrics.add(f"in_last_{cond}", round(agg[cond]["in_last_sum"] / n, 4))
                gen_val = round(agg[cond]["gen_sum"] / n, 4)
                self.metrics.add(f"gen_{cond}", gen_val)
                self.metrics.add(f"next_token_recovery_{cond}", gen_val)
                self.metrics.add(f"kl_{cond}", round(agg[cond]["kl_sum"] / n, 4))
        self.metrics.add("evaluation_target", "next_token_prediction_recovery")
        self.metrics.add("conditions", CONDITIONS)
        self.metrics.add("has_bos", bos_pos is not None)
        self.metrics.add("num_examples", agg["baseline"]["count"])

    # ── Family M: decoder repeated-layer robustness ───────────

    def _run_decoder_repeated_layer(self) -> None:
        """Family M Part 1: repeated-layer robustness for decoder models.

        Extends encoder Family I with decoder-specific conditions:
          - repeat_attention_only: repeat only the attention sublayer
          - repeat_ffn_only: repeat only the FFN sublayer
          - higher-rank corrections (16, 32, 64)
          - freeze_control uses BOS + last_visible instead of CLS + SEP

        Uses causal shift for prediction comparison.
        """
        import numpy as np
        from analysis.geometry.drift import fit_layer_transition, low_rank_approximation_error

        split = self._get_split()
        num_examples = (
            min(len(split.examples), self.config.batch_size)
            if self.config.batch_size
            else len(split.examples)
        )

        REPEAT_COUNTS = [1, 2, 4, 8]
        CORRECTION_RANKS = [16, 32, 64]
        CONDITIONS = [
            "no_correction", "freeze_control",
            "low_rank_correction", "both",
            "repeat_attention_only", "repeat_ffn_only",
        ]

        modules = self.adapter.get_layer_modules()
        n_layers = len(modules)

        # Target layers: early, quarter, mid, three-quarter, late
        if n_layers >= 12:
            TARGET_LAYERS = [2, n_layers // 4, n_layers // 2,
                             3 * n_layers // 4, n_layers - 3]
        elif n_layers >= 6:
            TARGET_LAYERS = [1, 2, 3, n_layers - 3, n_layers - 2]
        else:
            TARGET_LAYERS = list(range(min(3, n_layers)))
        TARGET_LAYERS = [L for L in TARGET_LAYERS if L < n_layers]

        # Phase 1: collect hidden states and fit transition matrices
        fit_limit = min(num_examples, 50)
        fit_hs_by_layer: Dict[int, List[torch.Tensor]] = {}

        for ex in split.examples[:fit_limit]:
            enc = self.adapter.encode_text(ex["text"])
            out = self.adapter.forward(**enc)
            for layer_idx, h in enumerate(out.hidden_states):
                if layer_idx not in fit_hs_by_layer:
                    fit_hs_by_layer[layer_idx] = []
                fit_hs_by_layer[layer_idx].append(h.squeeze(0).detach())

        transition_matrices = {}
        transition_residuals = {}
        lr_errors = {}
        for target_L in TARGET_LAYERS:
            if target_L < n_layers and (target_L + 1) in fit_hs_by_layer:
                hs_from = torch.cat(fit_hs_by_layer[target_L], dim=0)
                hs_to = torch.cat(fit_hs_by_layer[target_L + 1], dim=0)
                W, resid = fit_layer_transition(hs_from, hs_to)
                transition_matrices[target_L] = W
                transition_residuals[target_L] = resid
                lr_errors[target_L] = low_rank_approximation_error(W, CORRECTION_RANKS)

        correction_by_layer_rank: Dict[int, Dict[int, torch.Tensor]] = {}
        for target_L in TARGET_LAYERS:
            if target_L in transition_matrices:
                W = transition_matrices[target_L]
                U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                correction_by_layer_rank[target_L] = {}
                for r in CORRECTION_RANKS:
                    r_eff = min(r, len(S))
                    W_approx = U[:, :r_eff] @ torch.diag(S[:r_eff]) @ Vh[:r_eff, :]
                    correction = W_approx - torch.eye(W.shape[0], device=W.device)
                    correction_by_layer_rank[target_L][r] = correction

        # Phase 2: run repeated-layer experiments
        agg: Dict[int, Dict[int, Dict[str, Dict[str, float]]]] = {}
        for target_L in TARGET_LAYERS:
            agg[target_L] = {}
            for rc in REPEAT_COUNTS:
                agg[target_L][rc] = {}
                for cond in CONDITIONS:
                    agg[target_L][rc][cond] = {
                        "token_match_sum": 0, "sent_match": 0,
                        "count": 0, "total_tokens": 0,
                        "gen_match_sum": 0,
                        "weighted_match_sum": 0.0, "weighted_total": 0.0,
                    }

        # Pre-compute rotary position embeddings if model uses RoPE
        _pos_embeddings = None
        _base_model = getattr(self.adapter.model, "model",
                              getattr(self.adapter.model, "gpt_neox", self.adapter.model))
        if hasattr(_base_model, "rotary_emb"):
            # We'll compute per-example since seq_len may vary
            pass

        _conditions_with_fallback = set()

        for ex in split.examples[:num_examples]:
            enc = self.adapter.encode_text(ex["text"])
            input_ids = enc["input_ids"]
            att_mask = enc.get("attention_mask")
            orig_ids = input_ids.squeeze(0).tolist()
            seq_len = input_ids.shape[1]

            bos_pos, last_vis, delim_pos, ordinary_pos, sp = \
                self._find_decoder_positions(orig_ids)

            # Causal eval positions: i where logits[i] predicts orig_ids[i+1]
            eval_positions = [i for i in range(len(orig_ids) - 1) if (i + 1) not in sp]
            ns_count = len(eval_positions)
            if ns_count == 0:
                continue

            # Compute position embeddings for this sequence length (RoPE models)
            _pos_emb = None
            if hasattr(_base_model, "rotary_emb"):
                try:
                    _dummy = torch.randn(1, seq_len, self.adapter.hidden_size,
                                         device=input_ids.device)
                    _pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                    _pos_emb = _base_model.rotary_emb(_dummy, _pos_ids)
                except Exception:
                    # Some models have different rotary_emb signatures
                    try:
                        _pos_emb = _base_model.rotary_emb(
                            _dummy, position_ids=_pos_ids)
                    except Exception:
                        _pos_emb = None

            baseline_out = self.adapter.forward(input_ids, attention_mask=att_mask)
            baseline_gen = baseline_out.logits.squeeze(0)[last_vis].argmax().item()
            # Ground truth: orig_ids shifted by 1 (causal: logits[i] predicts token[i+1])
            ground_truth = orig_ids[1:] + [orig_ids[-1]]

            for target_L in TARGET_LAYERS:
                for rc in REPEAT_COUNTS:
                    for cond in CONDITIONS:
                        pre_repeat_hs = None
                        if cond in ("freeze_control", "both"):
                            if target_L < len(baseline_out.hidden_states):
                                pre_repeat_hs = baseline_out.hidden_states[target_L].clone()

                        corr_matrix = None
                        if cond in ("low_rank_correction", "both"):
                            default_rank = 32
                            if target_L in correction_by_layer_rank:
                                if default_rank in correction_by_layer_rank[target_L]:
                                    corr_matrix = correction_by_layer_rank[target_L][default_rank]

                        def _make_repeat_fn(
                            t_layer, rcount, freeze_hs, c_matrix,
                            bp, lv, condition, pos_emb, delim_positions=()
                        ):
                            # Track whether sublayer call fell back to full layer
                            _sublayer_fell_back = [False]

                            def modify_fn(layer_idx, h):
                                if layer_idx != t_layer:
                                    return h

                                mod = modules[t_layer]
                                current = h
                                extra_kw = {}
                                if pos_emb is not None:
                                    extra_kw["position_embeddings"] = pos_emb

                                def _run_full_layer(m, x, kw):
                                    """Run full layer with graceful kwarg fallback."""
                                    with torch.no_grad():
                                        try:
                                            out = m(x, **kw)
                                        except TypeError:
                                            out = m(x)
                                    return out[0] if isinstance(out, tuple) else out

                                for _ in range(rcount):
                                    if condition == "repeat_attention_only":
                                        attn_mod = _get_attention_sublayer(mod)
                                        if attn_mod is not None:
                                            with torch.no_grad():
                                                try:
                                                    attn_out = attn_mod(current, **extra_kw)
                                                except (TypeError, Exception):
                                                    _sublayer_fell_back[0] = True
                                                    current = _run_full_layer(mod, current, extra_kw)
                                                    continue
                                            current = attn_out[0] if isinstance(attn_out, tuple) else attn_out
                                        else:
                                            _sublayer_fell_back[0] = True
                                            current = _run_full_layer(mod, current, extra_kw)
                                    elif condition == "repeat_ffn_only":
                                        ffn_mod = _get_ffn_sublayer(mod)
                                        if ffn_mod is not None:
                                            with torch.no_grad():
                                                try:
                                                    ffn_out = ffn_mod(current)
                                                except (TypeError, Exception):
                                                    _sublayer_fell_back[0] = True
                                                    current = _run_full_layer(mod, current, extra_kw)
                                                    continue
                                            current = ffn_out[0] if isinstance(ffn_out, tuple) else ffn_out
                                        else:
                                            _sublayer_fell_back[0] = True
                                            current = _run_full_layer(mod, current, extra_kw)
                                    else:
                                        current = _run_full_layer(mod, current, extra_kw)

                                    # Apply corrections after each extra pass
                                    if condition in ("freeze_control", "both"):
                                        if freeze_hs is not None:
                                            if bp is not None and bp < current.shape[1]:
                                                current[:, bp, :] = freeze_hs[:, bp, :]
                                            if lv is not None and lv < current.shape[1]:
                                                current[:, lv, :] = freeze_hs[:, lv, :]

                                    if condition in ("low_rank_correction", "both"):
                                        if c_matrix is not None:
                                            # Exclude control + delimiter positions (spec: ordinary tokens only)
                                            skip = {bp, lv} - {None}
                                            skip |= set(delim_positions)
                                            for p in range(current.shape[1]):
                                                if p not in skip:
                                                    current[:, p, :] = (
                                                        current[:, p, :]
                                                        + current[:, p, :] @ c_matrix.to(current.device)
                                                    )

                                return current
                            return modify_fn, _sublayer_fell_back

                        fn, _fell_back = _make_repeat_fn(
                            target_L, rc, pre_repeat_hs, corr_matrix,
                            bos_pos, last_vis, cond, _pos_emb, delim_pos,
                        )
                        rep_out = self._forward_with_layer_fn(input_ids, att_mask, fn)
                        if _fell_back[0]:
                            _conditions_with_fallback.add(cond)
                        rep_preds = rep_out.logits.argmax(dim=-1).squeeze(0).tolist()

                        token_match = sum(
                            1 for i in eval_positions
                            if rep_preds[i] == ground_truth[i]
                        )
                        sent_match = 1 if token_match == ns_count else 0

                        rep_gen = rep_out.logits.squeeze(0)[last_vis].argmax().item()
                        gen_match = 1.0 if rep_gen == baseline_gen else 0.0

                        _seq_len = len(orig_ids)
                        _w_sum, _w_total = 0.0, 0.0
                        for i in eval_positions:
                            w = (i + 1) / _seq_len
                            _w_total += w
                            if rep_preds[i] == ground_truth[i]:
                                _w_sum += w

                        agg[target_L][rc][cond]["token_match_sum"] += token_match
                        agg[target_L][rc][cond]["total_tokens"] += ns_count
                        agg[target_L][rc][cond]["sent_match"] += sent_match
                        agg[target_L][rc][cond]["gen_match_sum"] += gen_match
                        agg[target_L][rc][cond]["weighted_match_sum"] += _w_sum
                        agg[target_L][rc][cond]["weighted_total"] += _w_total
                        agg[target_L][rc][cond]["count"] += 1

            self.metrics.add_per_example(ex["example_id"], {"text": ex["text"]})

        # Aggregate
        for target_L in TARGET_LAYERS:
            for rc in REPEAT_COUNTS:
                for cond in CONDITIONS:
                    d = agg[target_L][rc][cond]
                    n = d["count"]
                    token_pres = d["token_match_sum"] / max(d["total_tokens"], 1)
                    sent_pres = d["sent_match"] / max(n, 1)
                    gen_ret = d["gen_match_sum"] / max(n, 1)
                    w_ret = d["weighted_match_sum"] / max(d["weighted_total"], 1e-9)
                    key_prefix = f"layer{target_L}_repeat{rc}_{cond}"
                    self.metrics.add(f"{key_prefix}_token_preservation",
                                     round(token_pres, 4))
                    self.metrics.add(f"{key_prefix}_sentence_preservation",
                                     round(sent_pres, 4))
                    self.metrics.add(f"{key_prefix}_gen_retention",
                                     round(gen_ret, 4))
                    self.metrics.add(f"{key_prefix}_weighted_in_seq_retention",
                                     round(w_ret, 4))

        for target_L in TARGET_LAYERS:
            if target_L in transition_residuals:
                self.metrics.add(f"layer{target_L}_transition_residual_norm",
                                 round(transition_residuals[target_L], 4))
            if target_L in lr_errors:
                for r, err in lr_errors[target_L].items():
                    self.metrics.add(f"layer{target_L}_lowrank_error_r{r}", round(err, 4))

        self.metrics.add("target_layers", TARGET_LAYERS)
        self.metrics.add("repeat_counts", REPEAT_COUNTS)
        self.metrics.add("correction_ranks", CORRECTION_RANKS)
        self.metrics.add("conditions", CONDITIONS)
        self.metrics.add("num_layers", n_layers)
        # Record which sublayer conditions fell back to full layer
        self.metrics.add("sublayer_fallback_conditions",
                         sorted(_conditions_with_fallback))
        # Degradation slope per target layer (preservation vs repeat count)
        from analysis.metrics.core import degradation_slope
        for target_L in TARGET_LAYERS:
            for cond in CONDITIONS:
                pres_by_rc = {}
                for rc in REPEAT_COUNTS:
                    key = f"layer{target_L}_repeat{rc}_{cond}_token_preservation"
                    if key in self.metrics.metrics:
                        pres_by_rc[rc] = self.metrics.metrics[key]
                if len(pres_by_rc) >= 2:
                    slope = degradation_slope(pres_by_rc)
                    self.metrics.add(
                        f"layer{target_L}_{cond}_degradation_slope",
                        round(slope, 6))

    # ── Family E-decoder: attention centrality for decoders ───

    def _run_decoder_attention_centrality(self) -> None:
        """Family E-decoder: attention centrality for decoder models.

        Measures per-layer attention mass to BOS, last_visible, delimiters,
        and ordinary positions. Identifies the attention sink pattern.
        Uses decoder-appropriate position identification (no CLS/SEP).
        """
        import numpy as np
        from models.hooks.token_metadata import extract_token_metadata

        split = self._get_split()
        num_examples = (
            min(len(split.examples), self.config.batch_size)
            if self.config.batch_size
            else len(split.examples)
        )

        num_layers = None
        layer_attn_to_bos = {}
        layer_attn_to_pos0 = {}
        layer_attn_to_last = {}
        layer_attn_to_delimiters = {}
        layer_attn_to_ordinary = {}
        layer_attn_sink_pos = {}  # layer -> list of sink position indices

        for ex_idx, ex in enumerate(split.examples[:num_examples], start=1):
            enc = self.adapter.encode_text(ex["text"])
            input_ids = enc["input_ids"]
            att_mask = enc.get("attention_mask")
            orig_ids = input_ids.squeeze(0).tolist()

            bos_pos, last_vis, delim_positions, ordinary_pos, sp = \
                self._find_decoder_positions(orig_ids)

            out = self.adapter.forward(input_ids, attention_mask=att_mask)
            if out.attentions is None:
                continue

            seq_len = len(orig_ids)

            if num_layers is None:
                num_layers = len(out.attentions)
                for L in range(num_layers):
                    layer_attn_to_bos[L] = []
                    layer_attn_to_pos0[L] = []
                    layer_attn_to_last[L] = []
                    layer_attn_to_delimiters[L] = []
                    layer_attn_to_ordinary[L] = []
                    layer_attn_sink_pos[L] = []

            for L in range(num_layers):
                attn = out.attentions[L].squeeze(0)  # (heads, seq, seq)
                avg_attn = attn.mean(dim=0)  # (seq, seq) — [from, to]

                # Attention mass TO each position type (column averages)
                if bos_pos is not None:
                    attn_to_bos = avg_attn[:, bos_pos].mean().item()
                else:
                    attn_to_bos = 0.0
                layer_attn_to_bos[L].append(attn_to_bos)

                attn_to_pos0 = avg_attn[:, 0].mean().item()
                layer_attn_to_pos0[L].append(attn_to_pos0)

                attn_to_last = avg_attn[:, last_vis].mean().item()
                layer_attn_to_last[L].append(attn_to_last)

                if delim_positions:
                    attn_to_delim = avg_attn[:, delim_positions].mean().item()
                else:
                    attn_to_delim = 0.0
                layer_attn_to_delimiters[L].append(attn_to_delim)

                if ordinary_pos:
                    attn_to_ord = avg_attn[:, ordinary_pos].mean().item()
                else:
                    attn_to_ord = 0.0
                layer_attn_to_ordinary[L].append(attn_to_ord)

                # Attention sink: which position receives maximal column-sum attention?
                col_sums = avg_attn.sum(dim=0)  # (seq_len,)
                sink_pos = col_sums.argmax().item()
                layer_attn_sink_pos[L].append(sink_pos)

            self.metrics.add_per_example(ex["example_id"], {
                "text": ex["text"],
                "bos_pos": bos_pos,
                "last_visible": last_vis,
                "num_delimiters": len(delim_positions),
                "seq_len": seq_len,
            })
            if ex_idx == num_examples or ex_idx % 25 == 0:
                self._write_progress("decoder_attention_centrality", done=ex_idx, total=num_examples)

        if num_layers is None:
            self.metrics.add("error", "No attentions — model needs output_attentions=True")
            return

        attn_bos_by_layer = []
        attn_pos0_by_layer = []
        attn_last_by_layer = []
        attn_delim_by_layer = []
        attn_ordinary_by_layer = []
        sink_pos_by_layer = []
        pos0_sink_rate_by_layer = []

        for L in range(num_layers):
            attn_bos_by_layer.append(round(float(np.mean(layer_attn_to_bos[L])), 6))
            attn_pos0_by_layer.append(round(float(np.mean(layer_attn_to_pos0[L])), 6))
            attn_last_by_layer.append(round(float(np.mean(layer_attn_to_last[L])), 6))
            attn_delim_by_layer.append(round(float(np.mean(layer_attn_to_delimiters[L])), 6))
            attn_ordinary_by_layer.append(round(float(np.mean(layer_attn_to_ordinary[L])), 6))
            # Modal sink position
            from collections import Counter
            sink_counts = Counter(layer_attn_sink_pos[L])
            sink_pos_by_layer.append(sink_counts.most_common(1)[0][0])
            pos0_sink_rate_by_layer.append(round(sum(1 for s in layer_attn_sink_pos[L] if s == 0) / max(len(layer_attn_sink_pos[L]), 1), 6))

        self.metrics.add("attn_to_bos_by_layer", attn_bos_by_layer)
        self.metrics.add("attn_to_pos0_by_layer", attn_pos0_by_layer)
        self.metrics.add("attn_to_last_by_layer", attn_last_by_layer)
        self.metrics.add("attn_to_delimiters_by_layer", attn_delim_by_layer)
        self.metrics.add("attn_to_ordinary_by_layer", attn_ordinary_by_layer)
        self.metrics.add("attention_sink_pos_by_layer", sink_pos_by_layer)
        self.metrics.add("pos0_sink_rate_by_layer", pos0_sink_rate_by_layer)
        self.metrics.add("num_layers", num_layers)

        # Summary scalars
        self.metrics.add("avg_attn_to_bos", round(float(np.mean(attn_bos_by_layer)), 6))
        self.metrics.add("avg_attn_to_pos0", round(float(np.mean(attn_pos0_by_layer)), 6))
        self.metrics.add("avg_attn_to_last", round(float(np.mean(attn_last_by_layer)), 6))
        self.metrics.add("avg_attn_to_delimiters", round(float(np.mean(attn_delim_by_layer)), 6))
        self.metrics.add("avg_attn_to_ordinary", round(float(np.mean(attn_ordinary_by_layer)), 6))
        self.metrics.add("avg_pos0_sink_rate", round(float(np.mean(pos0_sink_rate_by_layer)), 6))
        self.metrics.add(
            "avg_pos0_to_ordinary_ratio",
            round(float(np.mean(attn_pos0_by_layer)) / max(float(np.mean(attn_ordinary_by_layer)), 1e-9), 6),
        )

        # Position 0 as sink (universal metric)
        pos0_is_sink_rate = sum(1 for s in sink_pos_by_layer if s == 0) / num_layers
        self.metrics.add("pos0_is_sink_fraction", round(pos0_is_sink_rate, 4))
        # Legacy alias (kept for backward compat with existing results/plots)
        self.metrics.add("bos_is_sink_fraction", round(pos0_is_sink_rate, 4))

    # ── Family M Part 2: decoder early-exit analysis ──────────

    def _run_decoder_early_exit(self) -> None:
        """Family M Part 2: at each layer, decode next-token prediction from
        the last-visible position and compare to the final-layer prediction.

        Reports: per-layer agreement, stabilization layer, safe-exit threshold,
        token-family breakdown.
        """
        import numpy as np
        from models.lenses.logit_lens import LogitLens
        from models.hooks.token_metadata import extract_token_metadata

        split = self._get_split()
        num_examples = (
            min(len(split.examples), self.config.batch_size)
            if self.config.batch_size
            else len(split.examples)
        )

        lens = LogitLens(self.adapter, top_k=5)

        # Per-layer aggregators (split into in-seq + gen)
        num_layers = None
        layer_agree_counts = {}   # layer -> count of agreement with final (all positions)
        layer_total = {}
        layer_inseq_agree = {}    # layer -> in-seq agreement (positions 0..N-2)
        layer_inseq_total = {}
        layer_gen_agree = {}      # layer -> gen agreement (last_visible only)
        layer_gen_total = {}
        stabilization_layers = []  # per-example: first layer after which top-1 never changes
        entropy_by_layer = {}

        for ex in split.examples[:num_examples]:
            enc = self.adapter.encode_text(ex["text"])
            input_ids = enc["input_ids"]
            att_mask = enc.get("attention_mask")
            orig_ids = input_ids.squeeze(0).tolist()
            seq_len = len(orig_ids)

            bos_pos, last_vis, _, _, sp = self._find_decoder_positions(orig_ids)

            out = self.adapter.forward(input_ids, attention_mask=att_mask)
            hidden_states = out.hidden_states
            n_layers = len(hidden_states)

            if num_layers is None:
                num_layers = n_layers
                for L in range(num_layers):
                    layer_agree_counts[L] = 0
                    layer_total[L] = 0
                    layer_inseq_agree[L] = 0
                    layer_inseq_total[L] = 0
                    layer_gen_agree[L] = 0
                    layer_gen_total[L] = 0
                    entropy_by_layer[L] = []

            # Final-layer predictions at all positions
            final_logits = out.logits.squeeze(0)  # (seq_len, vocab)
            final_pred = final_logits[last_vis].argmax().item()
            final_preds_all = final_logits.argmax(dim=-1).tolist()
            # Eval positions for in-seq: non-special positions 0..N-2
            eval_positions = [i for i in range(seq_len - 1) if (i + 1) not in sp]

            # Per-layer predictions at last_visible
            layer_preds = []
            for L in range(num_layers):
                lr = lens.decode_layer(
                    hidden_states[L], L,
                    tokenizer=self.adapter.tokenizer,
                    total_layers=num_layers,
                )
                # Get predictions at all positions
                layer_top1 = lr.top_ids[:, 0].tolist() if lr.top_ids.dim() == 2 else []
                pred_at_last = layer_top1[last_vis] if last_vis < len(layer_top1) else -1
                layer_preds.append(pred_at_last)

                # Combined agreement (backward compat)
                agrees = pred_at_last == final_pred
                layer_agree_counts[L] += int(agrees)
                layer_total[L] += 1

                # In-seq agreement: positions 0..N-2 (non-special)
                inseq_match = sum(1 for i in eval_positions
                                  if i < len(layer_top1) and layer_top1[i] == final_preds_all[i])
                layer_inseq_agree[L] += inseq_match
                layer_inseq_total[L] += len(eval_positions)

                # Gen agreement: last_visible only
                layer_gen_agree[L] += int(agrees)
                layer_gen_total[L] += 1

                # Entropy at this layer for last_vis — reuse the lens
                # decode_layer result to get logits for entropy
                import torch.nn.functional as F
                hs = hidden_states[L]
                with torch.no_grad():
                    h = lens._preprocess(hs, L, num_layers)
                    if lens.mode == "cosine":
                        logits = lens._decode_cosine(h)
                    else:
                        if lens._transform is not None:
                            h = lens._transform(h)
                        if lens._decoder is not None:
                            logits = lens._decoder(h)
                        else:
                            logits = h
                if logits.dim() == 3:
                    logits = logits.squeeze(0)
                probs = F.softmax(logits[last_vis], dim=-1)
                ent = -(probs * (probs + 1e-12).log()).sum().item()
                entropy_by_layer[L].append(ent)

            # Stabilization layer: last layer where prediction changes
            stab = num_layers - 1
            for L in range(num_layers - 2, -1, -1):
                if layer_preds[L] != final_pred:
                    stab = L + 1
                    break
            else:
                stab = 0  # agrees from the start
            stabilization_layers.append(stab)

            self.metrics.add_per_example(ex["example_id"], {
                "text": ex["text"],
                "final_pred": self.adapter.tokenizer.convert_ids_to_tokens(final_pred),
                "stabilization_layer": stab,
            })

        if num_layers is None:
            self.metrics.add("error", "No examples processed")
            return

        # Per-layer agreement rate (combined, backward compat)
        agreement_by_layer = []
        for L in range(num_layers):
            rate = layer_agree_counts[L] / max(layer_total[L], 1)
            agreement_by_layer.append(round(rate, 4))
        self.metrics.add("agreement_with_final_by_layer", agreement_by_layer)

        # Per-layer in-seq agreement (positions 0..N-2)
        inseq_agreement = []
        for L in range(num_layers):
            rate = layer_inseq_agree[L] / max(layer_inseq_total[L], 1)
            inseq_agreement.append(round(rate, 4))
        self.metrics.add("inseq_agreement_by_layer", inseq_agreement)

        # Per-layer gen agreement (last_visible only)
        gen_agreement = []
        for L in range(num_layers):
            rate = layer_gen_agree[L] / max(layer_gen_total[L], 1)
            gen_agreement.append(round(rate, 4))
        self.metrics.add("gen_agreement_by_layer", gen_agreement)

        # Per-layer entropy
        avg_entropy_by_layer = []
        for L in range(num_layers):
            avg_entropy_by_layer.append(round(float(np.mean(entropy_by_layer[L])), 4))
        self.metrics.add("avg_entropy_by_layer", avg_entropy_by_layer)

        # Stabilization stats
        self.metrics.add("median_stabilization_layer",
                         round(float(np.median(stabilization_layers)), 1))
        self.metrics.add("mean_stabilization_layer",
                         round(float(np.mean(stabilization_layers)), 2))

        # Safe-exit threshold: earliest layer with >95% in-seq agreement
        safe_exit = num_layers - 1
        for L in range(num_layers):
            if inseq_agreement[L] >= 0.95:
                safe_exit = L
                break
        self.metrics.add("safe_exit_layer_95pct", safe_exit)

        # Gen-safe-exit: earliest layer with >95% gen agreement
        gen_safe_exit = num_layers - 1
        for L in range(num_layers):
            if gen_agreement[L] >= 0.95:
                gen_safe_exit = L
                break
        self.metrics.add("gen_safe_exit_layer_95pct", gen_safe_exit)

        self.metrics.add("num_layers", num_layers)
        self.metrics.add("num_examples", len(stabilization_layers))

    # ── Family L: decoder prefix corruption ───────────────────

    def _run_decoder_prefix_corruption(self) -> None:
        """Family L: test whether decoder LMs internally correct corrupted prefix text.

        Compares hidden states at corrupted positions between clean and corrupted runs.
        Measures: cosine distance over depth, downstream drift, generation change.
        """
        import random as _random
        import numpy as np
        from models.hooks.token_metadata import extract_token_metadata

        split = self._get_split()
        num_examples = (
            min(len(split.examples), self.config.batch_size)
            if self.config.batch_size
            else len(split.examples)
        )
        self._write_progress("decoder_prefix_corruption", done=0, total=num_examples)
        rng = _random.Random(self.config.seed)

        # Aggregators
        num_layers = None
        # Per-layer cosine distance at corrupted position (clean vs corrupted)
        layer_cos_dist_at_corrupt = {}
        # Per-layer cosine distance at position after corruption
        layer_cos_dist_downstream = {}
        # Generation change: does the next-token prediction at last position change?
        gen_changed_count = 0
        total_count = 0

        for ex_idx, ex in enumerate(split.examples[:num_examples], start=1):
            enc = self.adapter.encode_text(ex["text"])
            input_ids = enc["input_ids"]
            att_mask = enc.get("attention_mask")
            orig_ids = input_ids.squeeze(0).tolist()
            seq_len = len(orig_ids)

            bos_pos, last_vis, _, ordinary_pos, sp = \
                self._find_decoder_positions(orig_ids)

            # Need at least 3 non-special positions to corrupt one
            if len(ordinary_pos) < 3:
                continue

            # Pick a position to corrupt (not first, not last, not special)
            candidates = [p for p in ordinary_pos if p > 0 and p < last_vis]
            if not candidates:
                continue
            corrupt_pos = rng.choice(candidates)

            # Generate corrupted version: replace with random non-special token
            corrupted_ids = list(orig_ids)
            vocab_size = self.adapter.vocab_size
            special_ids = set(self.adapter.tokenizer.all_special_ids)
            new_token = rng.randint(0, vocab_size - 1)
            # Avoid special tokens and same-token replacement
            while new_token in special_ids or new_token == orig_ids[corrupt_pos]:
                new_token = rng.randint(0, vocab_size - 1)
            corrupted_ids[corrupt_pos] = new_token

            corrupted_tensor = torch.tensor([corrupted_ids], device=self.adapter.device)

            # Clean forward
            clean_out = self.adapter.forward(input_ids, attention_mask=att_mask)
            clean_hs = clean_out.hidden_states

            # Corrupted forward
            corrupt_out = self.adapter.forward(corrupted_tensor, attention_mask=att_mask)
            corrupt_hs = corrupt_out.hidden_states

            n_layers = len(clean_hs)
            if num_layers is None:
                num_layers = n_layers
                for L in range(num_layers):
                    layer_cos_dist_at_corrupt[L] = []
                    layer_cos_dist_downstream[L] = []

            # Compare hidden states at corrupted position and downstream
            for L in range(num_layers):
                clean_h = clean_hs[L].squeeze(0)
                corrupt_h = corrupt_hs[L].squeeze(0)

                # Cosine distance at corrupted position
                cos_sim = torch.nn.functional.cosine_similarity(
                    clean_h[corrupt_pos].unsqueeze(0),
                    corrupt_h[corrupt_pos].unsqueeze(0),
                ).item()
                layer_cos_dist_at_corrupt[L].append(1.0 - cos_sim)

                # Downstream: position after corruption
                downstream_pos = corrupt_pos + 1
                if downstream_pos < seq_len:
                    cos_sim_ds = torch.nn.functional.cosine_similarity(
                        clean_h[downstream_pos].unsqueeze(0),
                        corrupt_h[downstream_pos].unsqueeze(0),
                    ).item()
                    layer_cos_dist_downstream[L].append(1.0 - cos_sim_ds)

            # Generation change: compare final predictions at last_visible
            clean_pred = clean_out.logits.squeeze(0)[last_vis].argmax().item()
            corrupt_pred = corrupt_out.logits.squeeze(0)[last_vis].argmax().item()
            if clean_pred != corrupt_pred:
                gen_changed_count += 1
            total_count += 1

            self.metrics.add_per_example(ex["example_id"], {
                "text": ex["text"],
                "corrupt_pos": corrupt_pos,
                "generation_changed": clean_pred != corrupt_pred,
            })
            if ex_idx == num_examples or ex_idx % 25 == 0:
                self._write_progress("decoder_prefix_corruption", done=ex_idx, total=num_examples)

        if num_layers is None:
            self.metrics.add("error", "No valid examples")
            return

        # Per-layer cosine distance at corrupted position
        cos_at_corrupt_by_layer = []
        cos_downstream_by_layer = []
        for L in range(num_layers):
            cos_at_corrupt_by_layer.append(
                round(float(np.mean(layer_cos_dist_at_corrupt[L])), 6)
                if layer_cos_dist_at_corrupt[L] else 0.0
            )
            cos_downstream_by_layer.append(
                round(float(np.mean(layer_cos_dist_downstream[L])), 6)
                if layer_cos_dist_downstream[L] else 0.0
            )

        self.metrics.add("cosine_dist_at_corrupt_by_layer", cos_at_corrupt_by_layer)
        self.metrics.add("cosine_dist_downstream_by_layer", cos_downstream_by_layer)
        self.metrics.add("generation_change_rate",
                         round(gen_changed_count / max(total_count, 1), 4))
        self.metrics.add("num_layers", num_layers)
        self.metrics.add("num_examples", total_count)

        # Self-correction signal: does cosine distance decrease in later layers?
        # Use non-overlapping slices: early = layers [1..third], late = last third
        n = len(cos_at_corrupt_by_layer)
        if n >= 4:
            third = max(1, (n - 1) // 3)  # exclude layer 0, split rest into thirds
            early_avg = float(np.mean(cos_at_corrupt_by_layer[1:1 + third]))
            late_avg = float(np.mean(cos_at_corrupt_by_layer[-third:]))
            self.metrics.add("early_cos_dist_at_corrupt", round(early_avg, 6))
            self.metrics.add("late_cos_dist_at_corrupt", round(late_avg, 6))
            self.metrics.add("self_correction_signal",
                             round(early_avg - late_avg, 6))

    # ── Family K: decoder next-token attractor analysis ───────

    def _run_decoder_attractor(self) -> None:
        """Family K: test whether decoder hidden states converge to
        next-token-resolving basins over depth.

        Analyses:
        1. Per-layer entropy tracking at last_visible
        2. Basin-entry layer detection (lens-argmax stabilization)
        3. Nearest-token tracking in embedding space (geometric basin)
        4. Cosine trajectory of last_visible hidden state across layers
        5. Late-layer DBSCAN clustering + cluster purity by next token
        6. Token-family basin-entry breakdown
        """
        import logging
        import numpy as np
        import torch.nn.functional as F
        from models.lenses.logit_lens import LogitLens
        from models.hooks.token_metadata import extract_token_metadata

        split = self._get_split()
        num_examples = (
            min(len(split.examples), self.config.batch_size)
            if self.config.batch_size
            else len(split.examples)
        )

        lens = LogitLens(self.adapter, top_k=10)

        # Get embedding matrix for nearest-token tracking
        emb_matrix = None
        model = self.adapter.model
        for attr in ("get_input_embeddings", "transformer", "model"):
            if attr == "get_input_embeddings" and hasattr(model, attr):
                emb_matrix = model.get_input_embeddings().weight.detach().float()
                break
            obj = getattr(model, attr, None)
            if obj is not None and hasattr(obj, "embed_tokens"):
                emb_matrix = obj.embed_tokens.weight.detach().float()
                break

        num_layers = None
        entropy_by_layer = {}
        basin_entry_layers = []
        nearest_basin_entry_layers = []
        late_layer_states = []
        late_layer_next_tokens = []
        agreement_by_layer = {}
        cosine_trajectories = []
        nearest_token_by_layer = {}
        basin_entry_by_family = {}

        for ex in split.examples[:num_examples]:
            enc = self.adapter.encode_text(ex["text"])
            input_ids = enc["input_ids"]
            att_mask = enc.get("attention_mask")
            orig_ids = input_ids.squeeze(0).tolist()

            bos_pos, last_vis, _, _, sp = self._find_decoder_positions(orig_ids)

            out = self.adapter.forward(input_ids, attention_mask=att_mask)
            hidden_states = out.hidden_states
            n_layers = len(hidden_states)

            if num_layers is None:
                num_layers = n_layers
                for L in range(num_layers):
                    entropy_by_layer[L] = []
                    agreement_by_layer[L] = 0
                    nearest_token_by_layer[L] = []

            final_logits = out.logits.squeeze(0)
            final_pred = final_logits[last_vis].argmax().item()

            _attractor_layer_preds = []
            _layer_cosines = []
            _nearest_tokens = []

            prev_h = None
            for L in range(num_layers):
                hs = hidden_states[L]
                cur_h = hs.squeeze(0)[last_vis].detach().float()

                # Cosine trajectory: similarity between adjacent layers
                if prev_h is not None:
                    cos_sim = F.cosine_similarity(prev_h.unsqueeze(0),
                                                  cur_h.unsqueeze(0)).item()
                    _layer_cosines.append(round(cos_sim, 6))
                prev_h = cur_h

                # Nearest token in embedding space
                if emb_matrix is not None:
                    sims = F.cosine_similarity(cur_h.unsqueeze(0), emb_matrix)
                    nearest_tid = sims.argmax().item()
                    _nearest_tokens.append(nearest_tid)
                    nearest_token_by_layer[L].append(nearest_tid)

                # Lens-based entropy and prediction
                with torch.no_grad():
                    if lens._transform is None and lens._decoder is None:
                        lens._get_projection()
                    h = lens._preprocess(hs, L, num_layers)
                    if lens.mode == "cosine":
                        logits = lens._decode_cosine(h)
                    else:
                        if lens._transform is not None:
                            h = lens._transform(h)
                        if lens._decoder is not None:
                            logits = lens._decoder(h)
                        else:
                            logits = h

                if logits.dim() == 3:
                    logits = logits.squeeze(0)

                probs = F.softmax(logits[last_vis], dim=-1)
                ent = -(probs * (probs + 1e-12).log()).sum().item()
                entropy_by_layer[L].append(ent)

                layer_pred = logits[last_vis].argmax().item()
                _attractor_layer_preds.append(layer_pred)
                if layer_pred == final_pred:
                    agreement_by_layer[L] += 1

            # Basin entry (lens-argmax stabilization, backward scan)
            basin_entry = num_layers - 1
            for L in range(num_layers - 2, -1, -1):
                if _attractor_layer_preds[L] != final_pred:
                    basin_entry = L + 1
                    break
            else:
                basin_entry = 0
            basin_entry_layers.append(basin_entry)

            # Nearest-token basin entry (embedding-space, backward scan)
            if _nearest_tokens:
                final_nearest = _nearest_tokens[-1]
                nearest_basin = num_layers - 1
                for L in range(num_layers - 2, -1, -1):
                    if _nearest_tokens[L] != final_nearest:
                        nearest_basin = L + 1
                        break
                else:
                    nearest_basin = 0
                nearest_basin_entry_layers.append(nearest_basin)

            cosine_trajectories.append(_layer_cosines)

            # Token-family of the predicted next token for basin-entry breakdown
            try:
                metas = extract_token_metadata(orig_ids, self.adapter.tokenizer)
                if last_vis < len(metas):
                    fam = metas[last_vis].family.name
                else:
                    fam = "UNKNOWN"
            except Exception:
                fam = "UNKNOWN"
            if fam not in basin_entry_by_family:
                basin_entry_by_family[fam] = []
            basin_entry_by_family[fam].append(basin_entry)

            late_state = hidden_states[-1].squeeze(0)[last_vis].detach().cpu().numpy()
            late_layer_states.append(late_state)
            late_layer_next_tokens.append(final_pred)

            per_ex = {
                "text": ex["text"],
                "basin_entry_layer": basin_entry,
                "final_pred": self.adapter.tokenizer.convert_ids_to_tokens(final_pred),
                "last_vis_family": fam,
            }
            if _nearest_tokens:
                per_ex["nearest_basin_entry_layer"] = nearest_basin
                per_ex["nearest_final_token"] = self.adapter.tokenizer.convert_ids_to_tokens(
                    _nearest_tokens[-1])
            self.metrics.add_per_example(ex["example_id"], per_ex)

        if num_layers is None:
            self.metrics.add("error", "No examples processed")
            return

        n_ex = len(basin_entry_layers)

        # ── Existing metrics (preserved) ──

        avg_entropy = [round(float(np.mean(entropy_by_layer[L])), 4)
                       for L in range(num_layers)]
        self.metrics.add("avg_entropy_by_layer", avg_entropy)

        agree_rate = [round(agreement_by_layer[L] / max(n_ex, 1), 4)
                      for L in range(num_layers)]
        self.metrics.add("agreement_with_final_by_layer", agree_rate)

        self.metrics.add("median_basin_entry_layer",
                         round(float(np.median(basin_entry_layers)), 1))
        self.metrics.add("mean_basin_entry_layer",
                         round(float(np.mean(basin_entry_layers)), 2))

        if num_layers >= 4:
            early_ent = float(np.mean(avg_entropy[1:4]))
            late_ent = float(np.mean(avg_entropy[-3:]))
            self.metrics.add("early_entropy", round(early_ent, 4))
            self.metrics.add("late_entropy", round(late_ent, 4))
            self.metrics.add("entropy_reduction", round(early_ent - late_ent, 4))

        # ── NEW: Nearest-token basin entry (geometric) ──

        if nearest_basin_entry_layers:
            self.metrics.add("nearest_median_basin_entry",
                             round(float(np.median(nearest_basin_entry_layers)), 1))
            self.metrics.add("nearest_mean_basin_entry",
                             round(float(np.mean(nearest_basin_entry_layers)), 2))

        # ── NEW: Cosine trajectory (adjacent-layer similarity) ──

        if cosine_trajectories:
            traj_len = min(len(t) for t in cosine_trajectories)
            avg_traj = []
            for i in range(traj_len):
                vals = [t[i] for t in cosine_trajectories if i < len(t)]
                avg_traj.append(round(float(np.mean(vals)), 4))
            self.metrics.add("avg_cosine_trajectory", avg_traj)

        # ── NEW: Token-family basin-entry breakdown ──

        family_basin_stats = {}
        for fam, entries in basin_entry_by_family.items():
            family_basin_stats[fam] = {
                "median": round(float(np.median(entries)), 1),
                "mean": round(float(np.mean(entries)), 2),
                "count": len(entries),
            }
        self.metrics.add("basin_entry_by_token_family", family_basin_stats)

        # ── Late-layer clustering with purity ──

        if len(late_layer_states) >= 10:
            from analysis.clustering.cluster import run_dbscan
            data = np.stack(late_layer_states)
            try:
                result = run_dbscan(data, eps=2.0, min_samples=3)
                self.metrics.add("late_layer_n_clusters", result.n_clusters)
                noise_frac = sum(1 for l in result.labels if l == -1) / len(result.labels)
                self.metrics.add("late_layer_noise_fraction", round(noise_frac, 4))

                # Cluster purity: within each cluster, what fraction shares
                # the same next-token prediction?
                if result.n_clusters > 0:
                    from collections import Counter
                    cluster_purities = []
                    for cid in range(result.n_clusters):
                        members = [late_layer_next_tokens[i]
                                   for i, l in enumerate(result.labels) if l == cid]
                        if members:
                            most_common_count = Counter(members).most_common(1)[0][1]
                            cluster_purities.append(most_common_count / len(members))
                    if cluster_purities:
                        self.metrics.add("cluster_purity_mean",
                                         round(float(np.mean(cluster_purities)), 4))
                        self.metrics.add("cluster_purity_per_cluster",
                                         [round(p, 4) for p in cluster_purities])

            except Exception as e:
                logging.warning("Family K DBSCAN clustering failed: %s", e)
                self.metrics.add("late_layer_clustering_error", str(e))

        self.metrics.add("num_layers", num_layers)
        self.metrics.add("num_examples", n_ex)

    # ── Tuned lens ────────────────────────────────────────────

    def _run_decoder_tuned_lens(self) -> None:
        """Train and evaluate a tuned lens on this decoder model.

        Trains per-layer affine probes on the first 80% of examples,
        evaluates on the remaining 20%. Reports tuned vs raw retention
        per layer, showing how much token-resolving information is present
        but rotated away from the output embedding.
        """
        from models.lenses.tuned_lens import TunedLens

        split = self._get_split()
        num_examples = (
            min(len(split.examples), self.config.batch_size)
            if self.config.batch_size
            else len(split.examples)
        )

        texts = [ex["text"] for ex in split.examples[:num_examples]]
        split_idx = max(1, int(len(texts) * 0.8))
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:]
        if not eval_texts:
            eval_texts = train_texts[-5:]  # fallback

        tuned = TunedLens(self.adapter, top_k=5)
        self._write_progress(
            "decoder_tuned_lens_train",
            done=0,
            total=max(len(train_texts), 1),
            num_train_examples=len(train_texts),
            num_eval_examples=len(eval_texts),
        )

        # Train
        train_losses = tuned.train(train_texts, epochs=50, lr=1e-3,
                                   device=self.adapter.device)
        self._write_progress(
            "decoder_tuned_lens_train",
            done=max(len(train_texts), 1),
            total=max(len(train_texts), 1),
            num_train_examples=len(train_texts),
            num_eval_examples=len(eval_texts),
        )

        # Evaluate
        self._write_progress(
            "decoder_tuned_lens_eval",
            done=0,
            total=max(len(eval_texts), 1),
            num_train_examples=len(train_texts),
            num_eval_examples=len(eval_texts),
        )
        layer_metrics = tuned.evaluate(eval_texts)
        self._write_progress(
            "decoder_tuned_lens_eval",
            done=max(len(eval_texts), 1),
            total=max(len(eval_texts), 1),
            num_train_examples=len(train_texts),
            num_eval_examples=len(eval_texts),
        )

        # Format results
        tuned_retention_by_layer = {}
        raw_retention_by_layer = {}
        improvement_by_layer = {}
        tuned_lastvis_by_layer = {}
        raw_lastvis_by_layer = {}
        tuned_lastvis_m1_by_layer = {}
        raw_lastvis_m1_by_layer = {}

        for layer_idx in sorted(layer_metrics.keys()):
            m = layer_metrics[layer_idx]
            tuned_retention_by_layer[layer_idx] = m["tuned_retention"]
            raw_retention_by_layer[layer_idx] = m["raw_retention"]
            improvement_by_layer[layer_idx] = round(
                m["tuned_retention"] - m["raw_retention"], 4
            )
            if "tuned_lastvis_retention" in m:
                tuned_lastvis_by_layer[layer_idx] = m["tuned_lastvis_retention"]
                raw_lastvis_by_layer[layer_idx] = m["raw_lastvis_retention"]
            if "tuned_lastvis_m1_retention" in m:
                tuned_lastvis_m1_by_layer[layer_idx] = m["tuned_lastvis_m1_retention"]
                raw_lastvis_m1_by_layer[layer_idx] = m["raw_lastvis_m1_retention"]

        self.metrics.add("tuned_retention_by_layer", tuned_retention_by_layer)
        self.metrics.add("raw_retention_by_layer", raw_retention_by_layer)
        self.metrics.add("improvement_by_layer", improvement_by_layer)
        self.metrics.add("train_losses_by_layer", train_losses)
        if tuned_lastvis_by_layer:
            self.metrics.add("tuned_lastvis_retention_by_layer", tuned_lastvis_by_layer)
            self.metrics.add("raw_lastvis_retention_by_layer", raw_lastvis_by_layer)
        if tuned_lastvis_m1_by_layer:
            self.metrics.add("tuned_lastvis_m1_retention_by_layer", tuned_lastvis_m1_by_layer)
            self.metrics.add("raw_lastvis_m1_retention_by_layer", raw_lastvis_m1_by_layer)

        # Summary stats (exclude layer 0)
        tuned_vals = [v for k, v in tuned_retention_by_layer.items() if k >= 1]
        raw_vals = [v for k, v in raw_retention_by_layer.items() if k >= 1]
        if tuned_vals:
            self.metrics.add("avg_tuned_retention",
                             round(sum(tuned_vals) / len(tuned_vals), 4))
            self.metrics.add("avg_raw_retention",
                             round(sum(raw_vals) / len(raw_vals), 4))
            self.metrics.add("avg_improvement",
                             round((sum(tuned_vals) - sum(raw_vals)) / len(tuned_vals), 4))

            # Late layers
            late_tuned = [v for k, v in tuned_retention_by_layer.items()
                          if k >= len(tuned_retention_by_layer) - 4 and k >= 1]
            late_raw = [v for k, v in raw_retention_by_layer.items()
                        if k >= len(raw_retention_by_layer) - 4 and k >= 1]
            if late_tuned:
                self.metrics.add("late_tuned_retention",
                                 round(sum(late_tuned) / len(late_tuned), 4))
                self.metrics.add("late_raw_retention",
                                 round(sum(late_raw) / len(late_raw), 4))

        if tuned_lastvis_by_layer:
            tuned_lastvis_vals = [v for k, v in tuned_lastvis_by_layer.items() if k >= 1]
            raw_lastvis_vals = [v for k, v in raw_lastvis_by_layer.items() if k >= 1]
            if tuned_lastvis_vals:
                self.metrics.add("avg_tuned_lastvis_retention",
                                 round(sum(tuned_lastvis_vals) / len(tuned_lastvis_vals), 4))
                self.metrics.add("avg_raw_lastvis_retention",
                                 round(sum(raw_lastvis_vals) / len(raw_lastvis_vals), 4))
                late_tuned_lastvis = [v for k, v in tuned_lastvis_by_layer.items()
                                      if k >= len(tuned_lastvis_by_layer) - 4 and k >= 1]
                late_raw_lastvis = [v for k, v in raw_lastvis_by_layer.items()
                                    if k >= len(raw_lastvis_by_layer) - 4 and k >= 1]
                if late_tuned_lastvis:
                    self.metrics.add("late_tuned_lastvis_retention",
                                     round(sum(late_tuned_lastvis) / len(late_tuned_lastvis), 4))
                    self.metrics.add("late_raw_lastvis_retention",
                                     round(sum(late_raw_lastvis) / len(late_raw_lastvis), 4))

        if tuned_lastvis_m1_by_layer:
            tuned_lastvis_m1_vals = [v for k, v in tuned_lastvis_m1_by_layer.items() if k >= 1]
            raw_lastvis_m1_vals = [v for k, v in raw_lastvis_m1_by_layer.items() if k >= 1]
            if tuned_lastvis_m1_vals:
                self.metrics.add("avg_tuned_lastvis_m1_retention",
                                 round(sum(tuned_lastvis_m1_vals) / len(tuned_lastvis_m1_vals), 4))
                self.metrics.add("avg_raw_lastvis_m1_retention",
                                 round(sum(raw_lastvis_m1_vals) / len(raw_lastvis_m1_vals), 4))
                late_tuned_lastvis_m1 = [v for k, v in tuned_lastvis_m1_by_layer.items()
                                         if k >= len(tuned_lastvis_m1_by_layer) - 4 and k >= 1]
                late_raw_lastvis_m1 = [v for k, v in raw_lastvis_m1_by_layer.items()
                                       if k >= len(raw_lastvis_m1_by_layer) - 4 and k >= 1]
                if late_tuned_lastvis_m1:
                    self.metrics.add("late_tuned_lastvis_m1_retention",
                                     round(sum(late_tuned_lastvis_m1) / len(late_tuned_lastvis_m1), 4))
                    self.metrics.add("late_raw_lastvis_m1_retention",
                                     round(sum(late_raw_lastvis_m1) / len(late_raw_lastvis_m1), 4))

        self.metrics.add("num_layers", len(layer_metrics))
        self.metrics.add("num_train_examples", len(train_texts))
        self.metrics.add("num_eval_examples", len(eval_texts))

    # ── artifact saving ────────────────────────────────────────

    def _save_artifacts(self) -> None:
        """Save metrics and config snapshot."""
        out_dir = self.base_dir / self.config.output_path
        out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics.save(out_dir / "metrics.json")

        from configs.schema import save_run_config
        save_run_config(self.config, out_dir / "config.yaml")

    def teardown(self) -> None:
        if self.adapter:
            self.adapter.unload()
