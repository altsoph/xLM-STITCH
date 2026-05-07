from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.schema import load_run_config
from analysis.probes.feature_cache import LayerProbeCacheDataset, cache_status, load_cache_metadata
from models.probes.linear_probe import MLPProbeTrainer


RUN_DIRS = [
    ROOT / "results/exploratory/phase1_benchmark/phase1_albert_base_v2_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase1_benchmark/phase1_bert_base_uncased_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase1_benchmark/phase1_roberta_base_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase2_benchmark/phase2_ModernBERT_base_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase2_benchmark/phase2_xlm_roberta_base_probe_holdout_benchmark_s42",
]
REPORT_JSON = ROOT / "reports/run_summaries/mlm_probe_holdout_mlp_full_cache_refresh_20260506.json"
REPORT_MD = ROOT / "reports/run_summaries/mlm_probe_holdout_mlp_full_cache_refresh_20260506.md"


def _load_metrics(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_metrics(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def _load_head_metrics(path: Path) -> dict:
    rel_path = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=ROOT,
        text=True,
    )
    return json.loads(raw)


def _refresh_run(run_dir: Path) -> dict:
    cfg = load_run_config(run_dir / "config.yaml")
    metrics_payload = _load_metrics(run_dir / "metrics.json")
    head_metrics_payload = _load_head_metrics(run_dir / "metrics.json")
    baseline_acc = float(head_metrics_payload["summary_metrics"]["holdout_layer_mlp_acc"])

    cache = cache_status(ROOT, cfg)
    if not cache.exists:
        raise RuntimeError(f"Missing full cache for {cfg.run_id}: {cache.cache_dir}")
    cache_meta = load_cache_metadata(cache.cache_dir)

    train_ds = LayerProbeCacheDataset(cache.cache_dir, split="train")
    test_ds = LayerProbeCacheDataset(cache.cache_dir, split="test")

    probe_device = cfg.model.device
    batch_size = 4096 if str(probe_device).startswith("cuda") else 1024
    trainer = MLPProbeTrainer(
        int(cache_meta["hidden_size"]),
        int(cache_meta["num_layers"]),
        epochs=30,
        batch_size=batch_size,
        device=probe_device,
        seed=cfg.seed,
    )
    trainer.fit_dataset(train_ds)
    new_acc = round(float(trainer.score_dataset(test_ds)), 4)

    summary = metrics_payload["summary_metrics"]
    summary["holdout_layer_mlp_acc"] = new_acc
    summary["baseline_holdout_layer_mlp_acc"] = round(baseline_acc, 4)
    summary["mlp_probe_source"] = "full_cache_torch"
    summary["mlp_full_cache_total_examples"] = int(cache_meta["total_examples"])
    summary["mlp_full_cache_total_tokens"] = int(cache_meta["total_tokens"])
    summary["mlp_full_cache_train_examples"] = int(train_ds.train_examples)
    summary["mlp_full_cache_test_examples"] = int(test_ds.test_examples)
    summary["mlp_full_cache_train_samples"] = int(len(train_ds))
    summary["mlp_full_cache_test_samples"] = int(len(test_ds))
    summary["mlp_full_cache_storage_dtype"] = str(cache_meta["storage_dtype"])
    summary["mlp_full_cache_refresh_time"] = datetime.now(timezone.utc).isoformat()

    _save_metrics(run_dir / "metrics.json", metrics_payload)

    return {
        "run_id": cfg.run_id,
        "model_name": cfg.model.name,
        "baseline_mlp_holdout": round(baseline_acc, 4),
        "new_mlp_holdout": new_acc,
        "delta_vs_head": round(new_acc - baseline_acc, 4),
        "device": probe_device,
        "cache_total_examples": int(cache_meta["total_examples"]),
        "cache_total_tokens": int(cache_meta["total_tokens"]),
        "cache_train_examples": int(train_ds.train_examples),
        "cache_test_examples": int(test_ds.test_examples),
        "cache_train_samples": int(len(train_ds)),
        "cache_test_samples": int(len(test_ds)),
    }


def _write_report(results: list[dict]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": "mlm_probe_holdout_mlp_full_cache_refresh",
        "results": results,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")

    lines = [
        "# MLM Probe Holdout MLP Full-Cache Refresh",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "| Model | Baseline MLP holdout | Refreshed MLP holdout | Delta vs HEAD | Train/Test examples |",
        "|---|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['model_name']} | {item['baseline_mlp_holdout']:.4f} | {item['new_mlp_holdout']:.4f} | "
            f"{item['delta_vs_head']:+.4f} | {item['cache_train_examples']}/{item['cache_test_examples']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `new_mlp_holdout` is trained from `probe_feature_cache_full_v1` with the torch minibatch MLP path.",
            "- Existing linear/QDA/transfer metrics in the same run directories are left untouched.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = [_refresh_run(run_dir) for run_dir in RUN_DIRS]
    _write_report(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
