"""Materialize full probe feature caches for CLM benchmark probe runs.

The script is idempotent: if the cache metadata already exists in the chosen
cache root, that model is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.probes.feature_cache import (
    cache_status,
    materialize_probe_feature_cache,
    recommended_probe_cache_batch_size,
    recommended_probe_cache_storage_dtype,
)
from configs.schema import load_run_config


TARGET_RUNS = [
    ROOT / "results/exploratory/phase4_benchmark/phase4_distilgpt2_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase4_benchmark/phase4_gpt2_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase4_benchmark/phase4_pythia_160m_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase4_benchmark/phase4_pythia_410m_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase4_benchmark/phase4_SmolLM2_135M_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase4_benchmark/phase4_SmolLM2_360M_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase4_benchmark/phase4_Qwen2_5_0_5B_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase4_benchmark/phase4_Llama_3_2_1B_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase4_benchmark/phase4_gemma_3_1b_pt_probe_holdout_benchmark_s42",
]

ALT_BASE_MODELS = {"google/gemma-3-1b-pt"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize full CLM probe caches if missing.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-precision", default="fp32")
    ap.add_argument("--storage-dtype", default="auto", choices=["auto", "float16", "float32"])
    ap.add_argument("--batch-size", type=int, default=0, help="Override model-specific batch size when > 0.")
    ap.add_argument("--tokenizer-batch-size", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--alt-base-dir",
        default="",
        help="Optional alternate cache root used for models in ALT_BASE_MODELS (for example when workspace disk is tight).",
    )
    return ap.parse_args()


def _cache_root_for_model(model_name: str, alt_base_dir: str) -> Path:
    if alt_base_dir and model_name in ALT_BASE_MODELS:
        return Path(alt_base_dir)
    return ROOT


def main() -> None:
    args = parse_args()
    for run_dir in TARGET_RUNS:
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            print(f"[skip] missing config: {config_path}")
            continue

        run_config = load_run_config(config_path)
        cache_root = _cache_root_for_model(run_config.model.name, args.alt_base_dir)
        status = cache_status(cache_root, run_config)
        if status.exists and not args.force:
            print(f"[skip] cache already exists for {run_config.model.name}: {status.cache_dir}")
            continue

        progress_path = status.cache_dir / "progress.json"
        status.cache_dir.mkdir(parents=True, exist_ok=True)
        storage_dtype = (
            recommended_probe_cache_storage_dtype(run_config.model.name)
            if args.storage_dtype == "auto"
            else args.storage_dtype
        )
        batch_size = args.batch_size or recommended_probe_cache_batch_size(run_config.model.name)

        def _progress(done: int, total: int) -> None:
            payload = {
                "run_id": run_config.run_id,
                "model_name": run_config.model.name,
                "done_examples": int(done),
                "total_examples": int(total),
                "cache_dir": str(status.cache_dir),
                "storage_dtype": storage_dtype,
                "batch_size": batch_size,
            }
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
            if done == total or done % 100 == 0:
                print(f"[progress] {run_config.model.name}: {done}/{total}")

        print(f"[start] {run_config.model.name} root={cache_root} dtype={storage_dtype} batch={batch_size}")
        meta = materialize_probe_feature_cache(
            run_config,
            cache_root,
            device_override=args.device,
            model_precision_override=args.model_precision,
            storage_dtype=storage_dtype,
            batch_size=batch_size,
            tokenizer_batch_size=args.tokenizer_batch_size,
            force=args.force,
            progress_callback=_progress,
        )
        print(
            f"[done] {run_config.model.name}: "
            f"{meta['total_examples']} examples, {meta['total_tokens']} tokens, "
            f"{meta['num_layers']} layers, {meta['hidden_size']} hidden, "
            f"dtype={meta['storage_dtype']}, root={cache_root}"
        )


if __name__ == "__main__":
    main()
