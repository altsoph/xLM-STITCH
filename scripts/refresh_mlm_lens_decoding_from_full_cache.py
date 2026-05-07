from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.probes.feature_cache import cache_status
from analysis.probes.lens_cache import compute_encoder_lens_metrics_from_full_cache
from configs.schema import load_run_config
from models.adapters.factory import create_adapter


RUN_PAIRS = [
    (
        ROOT / "results/exploratory/phase1_benchmark/phase1_albert_base_v2_lens_decoding_benchmark_s42",
        ROOT / "results/exploratory/phase1_benchmark/phase1_albert_base_v2_probe_holdout_benchmark_s42",
    ),
    (
        ROOT / "results/exploratory/phase1_benchmark/phase1_bert_base_uncased_lens_decoding_benchmark_s42",
        ROOT / "results/exploratory/phase1_benchmark/phase1_bert_base_uncased_probe_holdout_benchmark_s42",
    ),
    (
        ROOT / "results/exploratory/phase1_benchmark/phase1_roberta_base_lens_decoding_benchmark_s42",
        ROOT / "results/exploratory/phase1_benchmark/phase1_roberta_base_probe_holdout_benchmark_s42",
    ),
    (
        ROOT / "results/exploratory/phase2_benchmark/phase2_ModernBERT_base_lens_decoding_benchmark_s42",
        ROOT / "results/exploratory/phase2_benchmark/phase2_ModernBERT_base_probe_holdout_benchmark_s42",
    ),
    (
        ROOT / "results/exploratory/phase2_benchmark/phase2_xlm_roberta_base_lens_decoding_benchmark_s42",
        ROOT / "results/exploratory/phase2_benchmark/phase2_xlm_roberta_base_probe_holdout_benchmark_s42",
    ),
]

REPORT_JSON = ROOT / "reports/run_summaries/mlm_lens_decoding_full_cache_refresh_20260506.json"
REPORT_MD = ROOT / "reports/run_summaries/mlm_lens_decoding_full_cache_refresh_20260506.md"


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
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


def _resolve_adapter(cfg) -> object:
    family_override = None
    if cfg.model.family != "auto":
        from models.adapters.base import ModelFamily

        family_override = ModelFamily[cfg.model.family.upper()]
    adapter = create_adapter(
        cfg.model.name,
        device="cuda",
        precision="fp32",
        output_attentions=False,
        family=family_override,
    )
    adapter.load()
    return adapter


def _refresh_run(lens_run_dir: Path, probe_run_dir: Path) -> dict:
    lens_cfg = load_run_config(lens_run_dir / "config.yaml")
    probe_cfg = load_run_config(probe_run_dir / "config.yaml")
    metrics_payload = _load_json(lens_run_dir / "metrics.json")
    head_metrics_payload = _load_head_metrics(lens_run_dir / "metrics.json")

    cache = cache_status(ROOT, probe_cfg)
    if not cache.exists:
        raise RuntimeError(f"Missing full cache for {probe_cfg.run_id}: {cache.cache_dir}")

    adapter = _resolve_adapter(lens_cfg)
    try:
        refreshed = compute_encoder_lens_metrics_from_full_cache(cache.cache_dir, adapter)
    finally:
        adapter.unload()

    summary = metrics_payload["summary_metrics"]
    baseline_by_layer = head_metrics_payload["summary_metrics"]["avg_retention_by_layer_ns"]
    summary["baseline_avg_retention_by_layer_ns"] = baseline_by_layer
    summary["avg_retention_by_layer_ns"] = {str(k): v for k, v in refreshed["avg_retention_by_layer_ns"].items()}
    summary["early_layers_avg_retention_ns"] = refreshed["early_layers_avg_retention_ns"]
    summary["late_layers_avg_retention_ns"] = refreshed["late_layers_avg_retention_ns"]
    summary["all_layers_avg_retention_ns"] = refreshed["all_layers_avg_retention_ns"]
    summary["num_layers"] = refreshed["num_layers"]
    summary["lens_decode_source"] = "full_cache_output_head"
    summary["lens_full_cache_total_examples"] = refreshed["lens_full_cache_total_examples"]
    summary["lens_full_cache_total_tokens"] = refreshed["lens_full_cache_total_tokens"]
    summary["lens_full_cache_storage_dtype"] = refreshed["lens_full_cache_storage_dtype"]
    summary["lens_full_cache_refresh_time"] = datetime.now(timezone.utc).isoformat()
    summary.pop("effective_lens_examples", None)
    summary.pop("lens_example_cap_reason", None)

    metrics_payload["per_example_metrics"] = [
        {"example_id": example_id, "num_layers": refreshed["num_layers"]}
        for example_id in refreshed["example_ids"]
    ]

    _save_json(lens_run_dir / "metrics.json", metrics_payload)

    new_last_key = str(max(int(k) for k in summary["avg_retention_by_layer_ns"].keys()))
    old_last_key = str(max(int(k) for k in baseline_by_layer.keys()))
    baseline_last = float(baseline_by_layer[old_last_key])
    new_last = float(summary["avg_retention_by_layer_ns"][new_last_key])

    return {
        "run_id": lens_cfg.run_id,
        "model_name": lens_cfg.model.name,
        "baseline_final_readout": round(baseline_last, 4),
        "new_final_readout": round(new_last, 4),
        "delta_final_readout_vs_head": round(new_last - baseline_last, 4),
        "cache_total_examples": refreshed["lens_full_cache_total_examples"],
        "cache_total_tokens": refreshed["lens_full_cache_total_tokens"],
        "cache_storage_dtype": refreshed["lens_full_cache_storage_dtype"],
        "num_layers": refreshed["num_layers"],
    }


def _write_report(results: list[dict]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": "mlm_lens_decoding_full_cache_refresh",
        "results": results,
    }
    _save_json(REPORT_JSON, payload)

    lines = [
        "# MLM Lens Decoding Full-Cache Refresh",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "| Model | Baseline final readout | Refreshed final readout | Delta | Cache examples |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['model_name']} | {item['baseline_final_readout']:.4f} | "
            f"{item['new_final_readout']:.4f} | {item['delta_final_readout_vs_head']:+.4f} | "
            f"{item['cache_total_examples']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Layerwise MLM readout is recomputed from `probe_feature_cache_full_v1` using the model output head.",
            "- This bypasses the encoder benchmark lens guard in `experiments/runner.py`.",
            "- The refreshed metric matches the paper-facing `Final readout` / `fig:mlm-readout-depth` semantics.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = [_refresh_run(lens_dir, probe_dir) for lens_dir, probe_dir in RUN_PAIRS]
    _write_report(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
