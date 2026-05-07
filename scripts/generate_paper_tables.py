"""Generate paper-facing tables and figure-data CSVs from canonical artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "exploratory"
TABLE_DIR = ROOT / "paper" / "tables"
FIG_DIR = ROOT / "paper" / "figure_data"


MLM_MODELS = [
    {
        "name": "albert-base-v2",
        "display": "ALBERT",
        "phase": "phase1",
        "layers": 12,
        "params_m": 11.3,
        "result_splits": {
            "lens_decoding": "benchmark",
            "clean_passthrough": "benchmark",
            "single_corrupt_repair": "benchmark",
            "swap_repair": "benchmark",
            "distant_swap_repair": "benchmark",
            "swap_independence": "benchmark",
            "probe_holdout": "benchmark",
            "special_token_intervention": "benchmark",
        },
    },
    {
        "name": "bert-base-uncased",
        "display": "BERT",
        "phase": "phase1",
        "layers": 12,
        "params_m": 109.5,
        "result_splits": {
            "lens_decoding": "benchmark",
            "clean_passthrough": "benchmark",
            "single_corrupt_repair": "benchmark",
            "swap_repair": "benchmark",
            "distant_swap_repair": "benchmark",
            "swap_independence": "benchmark",
            "probe_holdout": "benchmark",
            "special_token_intervention": "benchmark",
        },
    },
    {
        "name": "roberta-base",
        "display": "RoBERTa",
        "phase": "phase1",
        "layers": 12,
        "params_m": 124.7,
        "result_splits": {
            "lens_decoding": "benchmark",
            "clean_passthrough": "benchmark",
            "single_corrupt_repair": "benchmark",
            "swap_repair": "benchmark",
            "distant_swap_repair": "benchmark",
            "swap_independence": "benchmark",
            "probe_holdout": "benchmark",
            "special_token_intervention": "benchmark",
        },
    },
    {
        "name": "answerdotai/ModernBERT-base",
        "display": "ModernBERT",
        "phase": "phase2",
        "layers": 22,
        "params_m": 149.7,
        "result_splits": {
            "lens_decoding": "benchmark",
            "clean_passthrough": "benchmark",
            "single_corrupt_repair": "benchmark",
            "swap_repair": "benchmark",
            "distant_swap_repair": "benchmark",
            "swap_independence": "benchmark",
            "probe_holdout": "benchmark",
            "special_token_intervention": "benchmark",
        },
    },
    {
        "name": "xlm-roberta-base",
        "display": "XLM-R",
        "phase": "phase2",
        "layers": 12,
        "params_m": 278.3,
        "result_splits": {
            "lens_decoding": "benchmark",
            "clean_passthrough": "benchmark",
            "single_corrupt_repair": "benchmark",
            "swap_repair": "benchmark",
            "distant_swap_repair": "benchmark",
            "swap_independence": "benchmark",
            "probe_holdout": "benchmark",
            "special_token_intervention": "benchmark",
        },
    },
]


CLM_MODELS = [
    {"name": "distilgpt2", "display": "distilgpt2", "layers": 6, "params_m": 81.9},
    {"name": "gpt2", "display": "gpt2", "layers": 12, "params_m": 124.4},
    {"name": "EleutherAI/pythia-160m", "display": "pythia-160m", "layers": 12, "params_m": 162.3},
    {"name": "EleutherAI/pythia-410m", "display": "pythia-410m", "layers": 24, "params_m": 405.3},
    {"name": "HuggingFaceTB/SmolLM2-135M", "display": "SmolLM2-135M", "layers": 30, "params_m": 134.5},
    {"name": "HuggingFaceTB/SmolLM2-360M", "display": "SmolLM2-360M", "layers": 32, "params_m": 361.8},
    {"name": "Qwen/Qwen2.5-0.5B", "display": "Qwen2.5-0.5B", "layers": 24, "params_m": 494.0},
    {"name": "meta-llama/Llama-3.2-1B", "display": "Llama-3.2-1B", "layers": 16, "params_m": 1235.8},
    {"name": "google/gemma-3-1b-pt", "display": "gemma-3-1b-pt", "layers": 26, "params_m": 999.9},
]


def short_model(name: str) -> str:
    return name.split("/")[-1]


def model_run_token(name: str) -> str:
    return short_model(name).replace("-", "_").replace(".", "_")


def load_metrics(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["summary_metrics"]


def run_metrics(run_dir: Path) -> dict:
    return load_metrics(run_dir / "metrics.json")


def find_run(split_dir: str, pattern: str) -> Path:
    matches = list((RESULTS / split_dir).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {split_dir}/{pattern}, got {len(matches)}")
    return matches[0]


def fmt_pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def fmt_params(x: float) -> str:
    return f"{x:.1f}M"


def max_layer_value(layer_map: dict[str, float]) -> float:
    last_key = max(int(k) for k in layer_map)
    return float(layer_map[str(last_key)])


def first_present(metrics: dict, *keys: str):
    for key in keys:
        if key in metrics:
            return metrics[key]
    raise KeyError(keys[0])


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def md_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    rows = [[str(c) for c in row] for row in rows]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def mlm_metrics(model: dict, task: str) -> dict:
    split = model["result_splits"][task]
    phase = model["phase"]
    pattern = f"{phase}_{model_run_token(model['name'])}_{task}_{split}_s42"
    return run_metrics(find_run(f"{phase}_{split}", pattern))


def clm_metrics(model: dict, task: str) -> dict:
    pattern = f"phase4_{model_run_token(model['name'])}_{task}_benchmark_s42"
    return run_metrics(find_run("phase4_benchmark", pattern))


def build_mlm_inventory() -> tuple[list[str], list[list[object]]]:
    headers = ["Family", "Model", "Layers", "Params"]
    rows = []
    for model in MLM_MODELS:
        rows.append([
            "MLM",
            model["display"],
            model["layers"],
            fmt_params(model["params_m"]),
        ])
    for model in CLM_MODELS:
        rows.append([
            "CLM",
            model["display"],
            model["layers"],
            fmt_params(model["params_m"]),
        ])
    return headers, rows


def build_mlm_readability_and_repair() -> tuple[list[str], list[list[object]]]:
    headers = [
        "Model",
        "Final Readout",
        "Clean Change",
        "Single Repair",
        "Adj. Swap Both",
        "Dist-8 Both",
        "Sim. Indep. Both",
    ]
    rows = []
    for model in MLM_MODELS:
        lens = mlm_metrics(model, "lens_decoding")
        clean = mlm_metrics(model, "clean_passthrough")
        single = mlm_metrics(model, "single_corrupt_repair")
        swap = mlm_metrics(model, "swap_repair")
        distant = mlm_metrics(model, "distant_swap_repair")
        p_single = single["unmasked_restored_rate"]
        sim_both = p_single ** 2
        rows.append([
            model["display"],
            fmt_pct(max_layer_value(lens["avg_retention_by_layer_ns"])),
            fmt_pct(clean["avg_change_rate"]),
            fmt_pct(p_single),
            fmt_pct(swap["both_swap_positions_restored_rate"]),
            fmt_pct(distant["dist8_both_restored_rate"]),
            fmt_pct(sim_both),
        ])
    return headers, rows


def build_mlm_depth_and_control() -> tuple[list[str], list[list[object]]]:
    headers = [
        "Model",
        "MLP Holdout",
        "Content->Function",
        "Function->Content",
        "Freeze Both",
        "Ordinary Ctrl",
    ]
    rows = []
    for model in MLM_MODELS:
        probe = mlm_metrics(model, "probe_holdout")
        spec = mlm_metrics(model, "special_token_intervention")
        rows.append([
            model["display"],
            fmt_pct(probe["holdout_layer_mlp_acc"]),
            fmt_pct(probe["content_to_function_transfer_acc"]),
            fmt_pct(probe["function_to_content_transfer_acc"]),
            fmt_pct(spec["avg_retention_freeze_both"]),
            fmt_pct(spec["avg_retention_zero_ordinary_mean"]),
        ])
    return headers, rows


def build_clm_depth_and_readout() -> tuple[list[str], list[list[object]]]:
    headers = [
        "Model",
        "MLP Holdout",
        "Late Next-Token Readout",
        "Late Right-Edge Recovery",
        "Final Shifted-Input Recovery",
        "Freeze First",
        "Ordinary Ctrl",
    ]
    rows = []
    for model in CLM_MODELS:
        probe = clm_metrics(model, "probe_holdout")
        readout = clm_metrics(model, "decoder_tuned_lens")
        lens = clm_metrics(model, "lens_decoding")
        control = clm_metrics(model, "decoder_control_intervention")
        rows.append([
            model["display"],
            fmt_pct(probe["holdout_layer_mlp_acc"]),
            fmt_pct(readout["late_tuned_lastvis_retention"]),
            fmt_pct(readout["late_tuned_lastvis_m1_retention"]),
            fmt_pct(max_layer_value(lens["avg_retention_by_layer_ns"])),
            fmt_pct(control["gen_freeze_first"]),
            fmt_pct(control["gen_zero_ordinary_mean"]),
        ])
    return headers, rows


def build_clm_repair_and_control() -> tuple[list[str], list[list[object]]]:
    headers = [
        "Model",
        "Single-Token Recovery",
        "Adjacent Two-Token Recovery",
        "Distance-8 Two-Token Recovery",
        "Simulated Independent Double Recovery",
    ]
    rows = []
    for model in CLM_MODELS:
        single = clm_metrics(model, "single_corrupt_repair")
        swap = clm_metrics(model, "swap_repair")
        distant = clm_metrics(model, "distant_swap_repair")
        p_single = single["unmasked_restored_rate"]
        rows.append([
            model["display"],
            fmt_pct(p_single),
            fmt_pct(swap["both_swap_positions_restored_rate"]),
            fmt_pct(distant["dist8_both_restored_rate"]),
            fmt_pct(p_single ** 2),
        ])
    return headers, rows


def build_mlm_distance_curve_rows() -> list[list[object]]:
    rows = []
    for model in MLM_MODELS:
        swap = mlm_metrics(model, "swap_repair")
        distant = mlm_metrics(model, "distant_swap_repair")
        rows.extend([
            [model["display"], model["result_splits"]["swap_repair"], 1, swap["both_swap_positions_restored_rate"]],
            [model["display"], model["result_splits"]["distant_swap_repair"], 3, distant["dist3_both_restored_rate"]],
            [model["display"], model["result_splits"]["distant_swap_repair"], 5, distant["dist5_both_restored_rate"]],
            [model["display"], model["result_splits"]["distant_swap_repair"], 8, distant["dist8_both_restored_rate"]],
        ])
    return rows


def build_mlm_independence_rows() -> list[list[object]]:
    rows = []
    for model in MLM_MODELS:
        single = mlm_metrics(model, "single_corrupt_repair")
        swap = mlm_metrics(model, "swap_repair")
        indep = mlm_metrics(model, "swap_independence")
        p_single = single["unmasked_restored_rate"]
        rows.append([
            model["display"],
            model["result_splits"]["single_corrupt_repair"],
            p_single,
            p_single ** 2,
            swap["both_swap_positions_restored_rate"],
            indep["random_random_any_restored_rate"],
            indep["correct_random_any_restored_rate"],
        ])
    return rows


def build_clm_distance_curve_rows() -> list[list[object]]:
    rows = []
    for model in CLM_MODELS:
        swap = clm_metrics(model, "swap_repair")
        distant = clm_metrics(model, "distant_swap_repair")
        rows.extend([
            [model["display"], 1, swap["both_swap_positions_restored_rate"]],
            [model["display"], 3, distant["dist3_both_restored_rate"]],
            [model["display"], 5, distant["dist5_both_restored_rate"]],
            [model["display"], 8, distant["dist8_both_restored_rate"]],
        ])
    return rows


def build_clm_independence_rows() -> list[list[object]]:
    rows = []
    for model in CLM_MODELS:
        single = clm_metrics(model, "single_corrupt_repair")
        swap = clm_metrics(model, "swap_repair")
        indep = clm_metrics(model, "swap_independence")
        p_single = single["unmasked_restored_rate"]
        rows.append([
            model["display"],
            "benchmark",
            p_single,
            p_single ** 2,
            swap["both_swap_positions_restored_rate"],
            indep["random_random_any_restored_rate"],
            indep["correct_random_any_restored_rate"],
        ])
    return rows


def build_clm_control_heatmap_rows() -> list[list[object]]:
    rows = []
    conditions = [
        "freeze_bos",
        "freeze_delimiters",
        "freeze_first",
        "zero_bos",
        "zero_delimiters",
        "zero_first",
        "zero_ordinary_mean",
    ]
    for model in CLM_MODELS:
        control = clm_metrics(model, "decoder_control_intervention")
        baseline = max(control["gen_baseline"], 1e-9)
        for condition in conditions:
            rows.append([
                model["display"],
                condition,
                control[f"gen_{condition}"],
                control[f"gen_{condition}"] / baseline,
            ])
    return rows


def build_clm_readout_curve_rows() -> list[list[object]]:
    rows = []
    for model in CLM_MODELS:
        m = clm_metrics(model, "decoder_tuned_lens")
        for layer in sorted(int(k) for k in m["tuned_lastvis_retention_by_layer"]):
            ks = str(layer)
            rows.append([
                model["display"],
                layer,
                m["tuned_lastvis_retention_by_layer"][ks],
                m["raw_lastvis_retention_by_layer"][ks],
                m["tuned_lastvis_m1_retention_by_layer"][ks],
                m["raw_lastvis_m1_retention_by_layer"][ks],
            ])
    return rows


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sections: list[tuple[str, list[str], list[list[object]], str]] = [
        (
            "Table 1. Model inventory",
            *build_mlm_inventory(),
            "Model inventory for all encoder MLMs and decoder CLMs included in the paper.",
        ),
        (
            "Table 2. MLM readability and local repair",
            *build_mlm_readability_and_repair(),
            "All rows use the benchmark split. Final readout is last-layer LM-head retention against the original token. Clean change is the fraction of clean non-special positions changed under direct forward prediction. Single repair is unmasked restoration after replacing one token with another token from the same sentence. Adjacent swap both reports distance-1 swaps where both corrupted positions are restored. Dist-8 both is the same metric for distance 8. Simulated independent both is `p^2` from the single-token repair rate `p`.",
        ),
        (
            "Table 3. MLM depth code and control positions",
            *build_mlm_depth_and_control(),
            "All rows use the benchmark split. Depth code comes from `probe_holdout`; control-position metrics come from `special_token_intervention`.",
        ),
        (
            "Table 4. CLM depth code, readability, and control position",
            *build_clm_depth_and_readout(),
            "All rows use the benchmark split. `MLP Holdout` comes from `probe_holdout`. `Late Next-Token Readout` is the average over the last four non-embedding layers of tuned-lens agreement with the final-layer top-1 prediction at output position `last_visible`. `Late Right-Edge Recovery` is the same quantity at output position `last_visible-1`. `Final Shifted-Input Recovery` is final-layer exact-match under the plain logit lens against the shifted ground-truth target over all evaluated non-special output positions. `Freeze First` and `Ordinary Ctrl` come from `decoder_control_intervention`.",
        ),
        (
            "Table 5. CLM local recovery and locality",
            *build_clm_repair_and_control(),
            "All rows use the benchmark split. `Single-Token Recovery` corrupts one visible non-special token at position `i>0` and counts recovery when the model output at position `i-1` predicts the original pre-corruption token at position `i`. The adjacent and distance-8 columns require both corrupted positions to be restored under the same causal-shift target definition. `Simulated Independent Double Recovery` is `p^2` from the single-token recovery rate `p`.",
        ),
    ]

    md_parts = ["# Paper Tables", "", "Generated from canonical artifacts."]
    for idx, (title, headers, rows, note) in enumerate(sections, start=1):
        write_csv(TABLE_DIR / f"table_{idx:02d}.csv", headers, rows)
        md_parts.extend(["", f"## {title}", "", note, "", md_table(headers, rows)])

    with open(TABLE_DIR / "paper_tables.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_parts) + "\n")

    write_csv(
        FIG_DIR / "mlm_distance_curve.csv",
        ["model", "split", "distance", "any_restored_rate"],
        build_mlm_distance_curve_rows(),
    )
    write_csv(
        FIG_DIR / "mlm_independence.csv",
        [
            "model",
            "split",
            "single_restore_rate",
            "simulated_independent_any",
            "adjacent_swap_any",
            "random_random_any",
            "correct_random_any",
        ],
        build_mlm_independence_rows(),
    )
    write_csv(
        FIG_DIR / "clm_distance_curve.csv",
        ["model", "distance", "any_restored_rate"],
        build_clm_distance_curve_rows(),
    )
    write_csv(
        FIG_DIR / "clm_independence.csv",
        [
            "model",
            "split",
            "single_restore_rate",
            "simulated_independent_any",
            "adjacent_swap_any",
            "random_random_any",
            "correct_random_any",
        ],
        build_clm_independence_rows(),
    )
    write_csv(
        FIG_DIR / "clm_readout_curves.csv",
        ["model", "layer", "tuned_lastvis", "raw_lastvis", "tuned_lastvis_m1", "raw_lastvis_m1"],
        build_clm_readout_curve_rows(),
    )
    write_csv(
        FIG_DIR / "clm_control_heatmap.csv",
        ["model", "condition", "gen_retention", "retention_vs_baseline"],
        build_clm_control_heatmap_rows(),
    )

    print(f"Wrote {TABLE_DIR / 'paper_tables.md'}")


if __name__ == "__main__":
    main()
