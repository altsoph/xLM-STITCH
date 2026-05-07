"""Materialize full probe feature caches for MLM benchmark probe runs.

Initial target models:
  - answerdotai/ModernBERT-base
  - xlm-roberta-base

The script is idempotent: if the cache metadata already exists in the run
directory, it skips that model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.probes.feature_cache import cache_status, materialize_probe_feature_cache
from configs.schema import load_run_config

TARGET_RUNS = [
    ROOT / "results" / "exploratory" / "phase2_benchmark" / "phase2_ModernBERT_base_probe_holdout_benchmark_s42",
    ROOT / "results" / "exploratory" / "phase2_benchmark" / "phase2_xlm_roberta_base_probe_holdout_benchmark_s42",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize full MLM probe caches if missing.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-precision", default="fp32")
    ap.add_argument("--storage-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tokenizer-batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    for run_dir in TARGET_RUNS:
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            print(f"[skip] missing config: {config_path}")
            continue

        run_config = load_run_config(config_path)
        status = cache_status(ROOT, run_config)
        if status.exists and not args.force:
            print(f"[skip] cache already exists for {run_config.model.name}: {status.cache_dir}")
            continue

        progress_path = status.cache_dir / "progress.json"
        status.cache_dir.mkdir(parents=True, exist_ok=True)

        def _progress(done: int, total: int) -> None:
            payload = {
                "run_id": run_config.run_id,
                "model_name": run_config.model.name,
                "done_examples": int(done),
                "total_examples": int(total),
                "cache_dir": str(status.cache_dir),
            }
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
            print(f"[progress] {run_config.model.name}: {done}/{total}")

        print(f"[start] {run_config.model.name}")
        meta = materialize_probe_feature_cache(
            run_config,
            ROOT,
            device_override=args.device,
            model_precision_override=args.model_precision,
            storage_dtype=args.storage_dtype,
            batch_size=args.batch_size,
            tokenizer_batch_size=args.tokenizer_batch_size,
            force=args.force,
            progress_callback=_progress,
        )
        print(
            f"[done] {run_config.model.name}: "
            f"{meta['total_examples']} examples, {meta['total_tokens']} tokens, "
            f"{meta['num_layers']} layers, {meta['hidden_size']} hidden, "
            f"dtype={meta['storage_dtype']}"
        )


if __name__ == "__main__":
    main()
