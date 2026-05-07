from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .feature_cache import load_cache_metadata
from models.lenses.logit_lens import LogitLens


def compute_encoder_lens_metrics_from_full_cache(
    cache_dir: Path,
    adapter: Any,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Recompute encoder lens-decoding metrics from a full probe cache.

    This mirrors the paper-facing MLM readout semantics:
    exact top-1 recovery of the original token at non-special positions
    across all cached examples and layers, using the model output head.
    """

    meta = load_cache_metadata(cache_dir)
    hidden_states = np.load(cache_dir / meta["files"]["hidden_states"], mmap_mode="r")
    input_ids = np.load(cache_dir / meta["files"]["input_ids"], mmap_mode="r")
    example_offsets = np.load(cache_dir / meta["files"]["example_offsets"], mmap_mode="r")
    example_ids = json.loads((cache_dir / meta["files"]["example_ids"]).read_text(encoding="utf-8"))

    total_examples = int(meta["total_examples"])
    total_tokens = int(meta["total_tokens"])
    num_layers = int(meta["num_layers"])

    special_ids = set(adapter.tokenizer.all_special_ids)
    lens = LogitLens(adapter, top_k=5, mode="normed")
    agg = defaultdict(list)

    with torch.inference_mode():
        for ex_idx in range(total_examples):
            start = int(example_offsets[ex_idx])
            end = int(example_offsets[ex_idx + 1])
            ids_np = np.asarray(input_ids[start:end], dtype=np.int64)
            if ids_np.size == 0:
                continue
            eval_positions = [i for i, tid in enumerate(ids_np.tolist()) if tid not in special_ids]
            if not eval_positions:
                continue

            hs_np = np.asarray(hidden_states[start:end], dtype=np.float32)
            orig = torch.tensor(ids_np, device=adapter.device, dtype=torch.long).unsqueeze(0)
            hs_tuple = tuple(
                torch.tensor(hs_np[:, layer_idx, :], device=adapter.device, dtype=torch.float32).unsqueeze(0)
                for layer_idx in range(num_layers)
            )
            results = lens.decode_all_layers(hs_tuple, original_ids=orig, tokenizer=None)
            for lr in results:
                matches = [
                    bool(lr.top1_matches_original[pos])
                    for pos in eval_positions
                    if pos < len(lr.top1_matches_original)
                ]
                agg[lr.layer_idx].append(sum(matches) / max(len(matches), 1))

            if progress_callback is not None:
                progress_callback(ex_idx + 1, total_examples)

    avg_by_layer = {
        int(layer_idx): round(float(sum(vals) / len(vals)), 4)
        for layer_idx, vals in sorted(agg.items())
    }
    valid = {k: v for k, v in avg_by_layer.items() if k >= 1}
    early_keys = [k for k in valid if k <= 3]
    late_keys = [k for k in valid if k >= num_layers - 4]

    return {
        "avg_retention_by_layer_ns": avg_by_layer,
        "early_layers_avg_retention_ns": round(sum(valid[k] for k in early_keys) / max(len(early_keys), 1), 4),
        "late_layers_avg_retention_ns": round(sum(valid[k] for k in late_keys) / max(len(late_keys), 1), 4),
        "all_layers_avg_retention_ns": round(sum(valid.values()) / max(len(valid), 1), 4),
        "num_layers": num_layers,
        "lens_full_cache_total_examples": total_examples,
        "lens_full_cache_total_tokens": total_tokens,
        "lens_full_cache_storage_dtype": str(meta["storage_dtype"]),
        "example_ids": example_ids,
    }
