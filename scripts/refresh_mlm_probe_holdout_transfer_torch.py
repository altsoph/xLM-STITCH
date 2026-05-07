from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.probes.feature_cache import LayerProbeCacheDataset, cache_status, load_cache_metadata
from configs.schema import load_run_config
from models.hooks.token_metadata import TokenFamily
from models.probes.linear_probe import TorchLinearProbeTrainer


RUN_DIRS = [
    ROOT / "results/exploratory/phase1_benchmark/phase1_albert_base_v2_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase1_benchmark/phase1_bert_base_uncased_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase1_benchmark/phase1_roberta_base_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase2_benchmark/phase2_ModernBERT_base_probe_holdout_benchmark_s42",
    ROOT / "results/exploratory/phase2_benchmark/phase2_xlm_roberta_base_probe_holdout_benchmark_s42",
]
REPORT_JSON = ROOT / "reports/run_summaries/mlm_probe_holdout_transfer_torch_refresh_20260506.json"
REPORT_MD = ROOT / "reports/run_summaries/mlm_probe_holdout_transfer_torch_refresh_20260506.md"


def _load_metrics(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_metrics(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def _load_head_metrics(path: Path) -> dict:
    rel_path = path.relative_to(ROOT).as_posix()
    try:
        raw = subprocess.check_output(
            ["git", "show", f"HEAD:{rel_path}"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return json.loads(raw)
    except Exception:
        return _load_metrics(path)


def _transfer_batch_size(hidden_size: int, device: str) -> int:
    if isinstance(device, str) and device.startswith("cuda"):
        return 32768 if hidden_size <= 1024 else 16384
    return 4096


def _fit_and_score(train_ds, test_ds, *, hidden_size: int, num_layers: int, device: str, seed: int) -> float:
    trainer = TorchLinearProbeTrainer(
        hidden_size,
        num_layers,
        epochs=12,
        batch_size=_transfer_batch_size(hidden_size, device),
        device=device,
        seed=seed,
    )
    trainer.fit_dataset(train_ds)
    return round(float(trainer.score_dataset(test_ds)), 4)


def _refresh_run(run_dir: Path) -> dict:
    cfg = load_run_config(run_dir / "config.yaml")
    metrics_payload = _load_metrics(run_dir / "metrics.json")
    head_metrics_payload = _load_head_metrics(run_dir / "metrics.json")

    cache = cache_status(ROOT, cfg)
    if not cache.exists:
        raise RuntimeError(f"Missing full cache for {cfg.run_id}: {cache.cache_dir}")
    cache_meta = load_cache_metadata(cache.cache_dir)

    hidden_size = int(cache_meta["hidden_size"])
    num_layers = int(cache_meta["num_layers"])
    device = cfg.model.device

    content_value = TokenFamily.CONTENT.value
    function_value = TokenFamily.FUNCTION.value

    content_train_ds = LayerProbeCacheDataset(cache.cache_dir, split="train", family_values=[content_value])
    function_train_ds = LayerProbeCacheDataset(cache.cache_dir, split="train", family_values=[function_value])
    content_test_ds = LayerProbeCacheDataset(cache.cache_dir, split="test", family_values=[content_value])
    function_test_ds = LayerProbeCacheDataset(cache.cache_dir, split="test", family_values=[function_value])

    c2f = _fit_and_score(
        content_train_ds,
        function_test_ds,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device,
        seed=cfg.seed,
    )
    f2c = _fit_and_score(
        function_train_ds,
        content_test_ds,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device,
        seed=cfg.seed,
    )

    summary = metrics_payload["summary_metrics"]
    summary["baseline_content_to_function_transfer_acc"] = float(
        head_metrics_payload["summary_metrics"]["content_to_function_transfer_acc"]
    )
    summary["baseline_function_to_content_transfer_acc"] = float(
        head_metrics_payload["summary_metrics"]["function_to_content_transfer_acc"]
    )
    summary["content_to_function_transfer_acc"] = c2f
    summary["function_to_content_transfer_acc"] = f2c
    summary["transfer_probe_source"] = "full_cache_torch_linear"
    summary["transfer_full_cache_total_examples"] = int(cache_meta["total_examples"])
    summary["transfer_full_cache_total_tokens"] = int(cache_meta["total_tokens"])
    summary["transfer_full_cache_train_examples"] = int(content_train_ds.train_examples)
    summary["transfer_full_cache_test_examples"] = int(content_train_ds.test_examples)
    summary["transfer_full_cache_storage_dtype"] = str(cache_meta["storage_dtype"])
    summary["content_train_transfer_samples"] = int(len(content_train_ds))
    summary["function_train_transfer_samples"] = int(len(function_train_ds))
    summary["content_test_transfer_samples"] = int(len(content_test_ds))
    summary["function_test_transfer_samples"] = int(len(function_test_ds))
    summary["transfer_probe_refresh_time"] = datetime.now(timezone.utc).isoformat()

    _save_metrics(run_dir / "metrics.json", metrics_payload)

    return {
        "run_id": cfg.run_id,
        "model_name": cfg.model.name,
        "baseline_content_to_function_transfer_acc": float(
            head_metrics_payload["summary_metrics"]["content_to_function_transfer_acc"]
        ),
        "new_content_to_function_transfer_acc": c2f,
        "delta_content_to_function_vs_head": round(
            c2f - float(head_metrics_payload["summary_metrics"]["content_to_function_transfer_acc"]),
            4,
        ),
        "baseline_function_to_content_transfer_acc": float(
            head_metrics_payload["summary_metrics"]["function_to_content_transfer_acc"]
        ),
        "new_function_to_content_transfer_acc": f2c,
        "delta_function_to_content_vs_head": round(
            f2c - float(head_metrics_payload["summary_metrics"]["function_to_content_transfer_acc"]),
            4,
        ),
        "cache_train_examples": int(content_train_ds.train_examples),
        "cache_test_examples": int(content_train_ds.test_examples),
        "content_train_samples": int(len(content_train_ds)),
        "function_train_samples": int(len(function_train_ds)),
        "content_test_samples": int(len(content_test_ds)),
        "function_test_samples": int(len(function_test_ds)),
        "cache_storage_dtype": str(cache_meta["storage_dtype"]),
    }


def _write_report(results: list[dict]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": "mlm_probe_holdout_transfer_torch_refresh",
        "results": results,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")

    lines = [
        "# MLM Probe Holdout Transfer Torch Refresh",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "| Model | C->F baseline | C->F refreshed | Delta | F->C baseline | F->C refreshed | Delta | Train/Test examples |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['model_name']} | {item['baseline_content_to_function_transfer_acc']:.4f} | "
            f"{item['new_content_to_function_transfer_acc']:.4f} | "
            f"{item['delta_content_to_function_vs_head']:+.4f} | "
            f"{item['baseline_function_to_content_transfer_acc']:.4f} | "
            f"{item['new_function_to_content_transfer_acc']:.4f} | "
            f"{item['delta_function_to_content_vs_head']:+.4f} | "
            f"{item['cache_train_examples']}/{item['cache_test_examples']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Transfer metrics are trained from `probe_feature_cache_full_v1` with a torch linear probe.",
            "- `C->F` = train on content tokens from the train split, test on function tokens from the test split.",
            "- `F->C` = train on function tokens from the train split, test on content tokens from the test split.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = [_refresh_run(run_dir) for run_dir in RUN_DIRS]
    _write_report(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
