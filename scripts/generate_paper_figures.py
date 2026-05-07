"""Generate paper-facing figures from canonical experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "exploratory"
FIG_DIR = ROOT / "paper" / "figures"


MLM_MODELS = [
    {"name": "albert-base-v2", "display": "ALBERT", "phase": "phase1"},
    {"name": "bert-base-uncased", "display": "BERT", "phase": "phase1"},
    {"name": "roberta-base", "display": "RoBERTa", "phase": "phase1"},
    {"name": "answerdotai/ModernBERT-base", "display": "ModernBERT", "phase": "phase2"},
    {"name": "xlm-roberta-base", "display": "XLM-R", "phase": "phase2"},
]


CLM_MODELS = [
    {"name": "distilgpt2", "display": "distilgpt2"},
    {"name": "gpt2", "display": "gpt2"},
    {"name": "EleutherAI/pythia-160m", "display": "pythia-160m"},
    {"name": "EleutherAI/pythia-410m", "display": "pythia-410m"},
    {"name": "HuggingFaceTB/SmolLM2-135M", "display": "SmolLM2-135M"},
    {"name": "HuggingFaceTB/SmolLM2-360M", "display": "SmolLM2-360M"},
    {"name": "Qwen/Qwen2.5-0.5B", "display": "Qwen2.5-0.5B"},
    {"name": "meta-llama/Llama-3.2-1B", "display": "Llama-3.2-1B"},
    {"name": "google/gemma-3-1b-pt", "display": "gemma-3-1b"},
]


CLM_PCA_IMAGE_PATHS = {
    "distilgpt2": ROOT / "reports" / "plots" / "phase4_family_g_benchmark" / "distilgpt2" / "pca3d_all_layers_by_depth.png",
    "gpt2": ROOT / "reports" / "plots" / "phase4_family_g_benchmark" / "gpt2" / "pca3d_all_layers_by_depth.png",
    "pythia-160m": ROOT / "reports" / "plots" / "phase4_family_g_benchmark" / "pythia-160m" / "pca3d_all_layers_by_depth.png",
    "pythia-410m": ROOT / "reports" / "plots" / "phase4_family_g_benchmark" / "pythia-410m" / "pca3d_all_layers_by_depth.png",
    "SmolLM2-135M": ROOT / "reports" / "plots" / "phase4_family_g_benchmark" / "SmolLM2-135M" / "pca3d_all_layers_by_depth.png",
    "SmolLM2-360M": ROOT / "reports" / "plots" / "phase4_family_g_benchmark" / "SmolLM2-360M" / "pca3d_all_layers_by_depth.png",
    "Qwen2.5-0.5B": ROOT / "reports" / "plots" / "phase4_family_g_benchmark" / "Qwen2.5-0.5B" / "pca3d_all_layers_by_depth.png",
    "Llama-3.2-1B": ROOT / "reports" / "plots" / "phase4_family_g_benchmark" / "Llama-3.2-1B" / "pca3d_all_layers_by_depth.png",
    "gemma-3-1b": ROOT / "reports" / "plots" / "phase4_family_g_benchmark" / "gemma-3-1b-pt" / "pca3d_all_layers_by_depth.png",
}


MLM_TASK_SPLITS = {
    "phase1": {
        "lens_decoding": "benchmark",
        "single_corrupt_repair": "benchmark",
        "swap_repair": "benchmark",
        "distant_swap_repair": "benchmark",
        "swap_independence": "benchmark",
        "special_token_intervention": "benchmark",
    },
    "phase2": {
        "lens_decoding": "benchmark",
        "single_corrupt_repair": "benchmark",
        "swap_repair": "benchmark",
        "distant_swap_repair": "benchmark",
        "swap_independence": "benchmark",
        "special_token_intervention": "benchmark",
    },
}


PCA_IMAGE_PATHS = {
    "ALBERT": ROOT / "reports" / "plots" / "phase1_family_g_benchmark" / "albert-base-v2" / "pca3d_all_layers_by_depth.png",
    "BERT": ROOT / "reports" / "plots" / "phase1_family_g_benchmark" / "bert-base-uncased" / "pca3d_all_layers_by_depth.png",
    "RoBERTa": ROOT / "reports" / "plots" / "phase1_family_g_benchmark" / "roberta-base" / "pca3d_all_layers_by_depth.png",
    "ModernBERT": ROOT / "reports" / "plots" / "phase2_family_g_benchmark" / "ModernBERT-base" / "pca3d_all_layers_by_depth.png",
    "XLM-R": ROOT / "reports" / "plots" / "phase2_family_g_benchmark" / "xlm-roberta-base" / "pca3d_all_layers_by_depth.png",
}


def short_model(name: str) -> str:
    return name.split("/")[-1]


def model_run_token(name: str) -> str:
    return short_model(name).replace("-", "_").replace(".", "_")


def find_run(split_dir: str, pattern: str) -> Path:
    matches = list((RESULTS / split_dir).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {split_dir}/{pattern}, got {len(matches)}")
    return matches[0]


def run_metrics(run_dir: Path) -> dict:
    with open(run_dir / "metrics.json", encoding="utf-8") as f:
        return json.load(f)["summary_metrics"]


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def cropped_image(
    path: Path,
    top_crop: float = 0.08,
    bottom_crop: float = 0.0,
    left_crop: float = 0.0,
    right_crop: float = 0.0,
) -> np.ndarray:
    img = mpimg.imread(path)
    h, w = img.shape[0], img.shape[1]
    top = int(h * top_crop)
    bottom = h - int(h * bottom_crop)
    left = int(w * left_crop)
    right = w - int(w * right_crop)
    return img[top:bottom, left:right, ...]


def mlm_metrics(model: dict, task: str) -> dict:
    phase = model["phase"]
    split = MLM_TASK_SPLITS[phase][task]
    pattern = f"{phase}_{model_run_token(model['name'])}_{task}_{split}_s42"
    return run_metrics(find_run(f"{phase}_{split}", pattern))


def clm_metrics(model: dict, task: str) -> dict:
    pattern = f"phase4_{model_run_token(model['name'])}_{task}_benchmark_s42"
    return run_metrics(find_run("phase4_benchmark", pattern))


def fig01_mlm_local_repair() -> None:
    colors = sns.color_palette("tab10", len(MLM_MODELS))
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.8))

    for color, model in zip(colors, MLM_MODELS):
        swap = mlm_metrics(model, "swap_repair")
        dist = mlm_metrics(model, "distant_swap_repair")
        axes[0].plot(
            [1, 3, 5, 8],
            [
                swap["both_swap_positions_restored_rate"],
                dist["dist3_both_restored_rate"],
                dist["dist5_both_restored_rate"],
                dist["dist8_both_restored_rate"],
            ],
            linewidth=2.0,
            color=color,
            label=model["display"],
        )

    x = np.arange(len(MLM_MODELS)) * 1.3
    width = 0.18
    single_vals = []
    sim_vals = []
    actual_vals = []
    dist8_vals = []
    for model in MLM_MODELS:
        single = mlm_metrics(model, "single_corrupt_repair")["unmasked_restored_rate"]
        actual = mlm_metrics(model, "swap_repair")["both_swap_positions_restored_rate"]
        dist8 = mlm_metrics(model, "distant_swap_repair")["dist8_both_restored_rate"]
        single_vals.append(single)
        sim_vals.append(single ** 2)
        actual_vals.append(actual)
        dist8_vals.append(dist8)

    axes[1].bar(x - 1.5 * width, single_vals, width=width, color="#4c72b0", label="single-token repair")
    axes[1].bar(x - 0.5 * width, sim_vals, width=width, color="#55a868", label="simulated independent both")
    axes[1].bar(x + 0.5 * width, actual_vals, width=width, color="#c44e52", label="actual adjacent swap both")
    axes[1].bar(x + 1.5 * width, dist8_vals, width=width, color="#8172b2", label="actual dist-8 swap both")

    axes[0].set_xticks([1, 3, 5, 8])
    axes[0].set_xlabel("Distance between swapped tokens")
    axes[0].set_ylabel("Both swapped tokens restored")
    axes[0].set_ylim(0, 0.5)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m["display"] for m in MLM_MODELS], rotation=35, ha="right")
    axes[1].set_ylabel("Restoration probability")
    axes[1].set_ylim(0, 0.5)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    save_fig(fig, FIG_DIR / "fig01_mlm_local_repair.png")


def fig02_mlm_special_token_heatmap() -> None:
    conditions = [
        "freeze_cls",
        "freeze_sep",
        "freeze_both",
        "zero_cls",
        "zero_sep",
        "zero_both",
        "zero_ordinary_mean",
    ]
    label_map = {
        "freeze_cls": "freeze CLS",
        "freeze_sep": "freeze SEP",
        "freeze_both": "freeze both",
        "zero_cls": "zero CLS",
        "zero_sep": "zero SEP",
        "zero_both": "zero both",
        "zero_ordinary_mean": "zero ordinary mean",
    }
    data = []
    for cond in conditions:
        row = []
        for model in MLM_MODELS:
            metrics = mlm_metrics(model, "special_token_intervention")
            row.append(metrics[f"avg_retention_{cond}"] / max(metrics["avg_retention_baseline"], 1e-9))
        data.append(row)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    sns.heatmap(
        np.array(data),
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Retention relative to baseline"},
        xticklabels=[m["display"] for m in MLM_MODELS],
        yticklabels=[label_map[c] for c in conditions],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, FIG_DIR / "fig02_mlm_special_token_heatmap.png")


def fig03_mlm_pca_depth_panel() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0))
    for ax, model in zip(axes.flat, MLM_MODELS):
        ax.imshow(mpimg.imread(PCA_IMAGE_PATHS[model["display"]]))
        ax.set_title(model["display"], fontsize=16, pad=8)
        ax.axis("off")
    axes.flat[-1].axis("off")
    save_fig(fig, FIG_DIR / "fig03_mlm_pca_depth_panel.png")


def fig12_mlm_readout_depth() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4), sharey=True)
    main_models = [m for m in MLM_MODELS if m["display"] != "ModernBERT"]
    modernbert = next(m for m in MLM_MODELS if m["display"] == "ModernBERT")

    colors = {
        "ALBERT": "#4c72b0",
        "BERT": "#55a868",
        "RoBERTa": "#c44e52",
        "XLM-R": "#8172b2",
        "ModernBERT": "#dd8452",
    }

    for model in main_models:
        metrics = mlm_metrics(model, "lens_decoding")
        layers = sorted(int(k) for k in metrics["avg_retention_by_layer_ns"])
        values = [metrics["avg_retention_by_layer_ns"][str(k)] for k in layers]
        axes[0].plot(layers, values, linewidth=2.2, color=colors[model["display"]], label=model["display"])

    metrics = mlm_metrics(modernbert, "lens_decoding")
    layers = sorted(int(k) for k in metrics["avg_retention_by_layer_ns"])
    values = [metrics["avg_retention_by_layer_ns"][str(k)] for k in layers]
    axes[1].plot(layers, values, linewidth=2.4, color=colors["ModernBERT"], label="ModernBERT")

    axes[0].set_title("ALBERT, BERT, RoBERTa, XLM-R", fontsize=14, pad=8)
    axes[1].set_title("ModernBERT", fontsize=14, pad=8)
    axes[0].set_ylabel("Exact-match readout")
    for ax in axes:
        ax.set_xlabel("Layer")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    save_fig(fig, FIG_DIR / "fig12_mlm_readout_depth.png")


def _grid_axes(nrows: int = 3, ncols: int = 3, figsize: tuple[float, float] = (15.0, 11.0)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    return fig, axes


def fig04_clm_next_token_readout() -> None:
    fig, axes = _grid_axes()
    for idx, (ax, model) in enumerate(zip(axes.flat, CLM_MODELS)):
        metrics = clm_metrics(model, "decoder_tuned_lens")
        layers = sorted(int(k) for k in metrics["tuned_lastvis_retention_by_layer"])
        tuned = [metrics["tuned_lastvis_retention_by_layer"][str(k)] for k in layers]
        raw = [metrics["raw_lastvis_retention_by_layer"][str(k)] for k in layers]
        ax.plot(layers, tuned, linewidth=1.8, label="tuned @ last", color="#1f77b4")
        ax.plot(layers, raw, linewidth=1.6, linestyle="--", label="raw @ last", color="#d62728")
        ax.set_title(model["display"], fontsize=10)
        row_idx = idx // 3
        ax.set_xlabel("Layer" if row_idx == 2 else "")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("Retention")
    axes[1, 0].set_ylabel("Retention")
    axes[2, 0].set_ylabel("Retention")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.subplots_adjust(hspace=0.34)
    save_fig(fig, FIG_DIR / "fig04_clm_next_token_readout.png")


def fig05_clm_shifted_recovery() -> None:
    fig, axes = _grid_axes()
    for idx, (ax, model) in enumerate(zip(axes.flat, CLM_MODELS)):
        metrics = clm_metrics(model, "decoder_tuned_lens")
        layers = sorted(int(k) for k in metrics["tuned_lastvis_retention_by_layer"])
        lv = [metrics["tuned_lastvis_retention_by_layer"][str(k)] for k in layers]
        lvm1 = [metrics["tuned_lastvis_m1_retention_by_layer"][str(k)] for k in layers]
        ax.plot(layers, lv, linewidth=1.8, label="tuned @ last", color="#1f77b4")
        ax.plot(layers, lvm1, linewidth=1.8, label="tuned @ last-1", color="#2ca02c")
        ax.set_title(model["display"], fontsize=10)
        row_idx = idx // 3
        ax.set_xlabel("Layer" if row_idx == 2 else "")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("Retention")
    axes[1, 0].set_ylabel("Retention")
    axes[2, 0].set_ylabel("Retention")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.subplots_adjust(hspace=0.26)
    save_fig(fig, FIG_DIR / "fig05_clm_shifted_recovery.png")


def fig06_clm_local_repair() -> None:
    colors = sns.color_palette("tab10", len(CLM_MODELS))
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.4))

    for color, model in zip(colors, CLM_MODELS):
        swap = clm_metrics(model, "swap_repair")
        dist = clm_metrics(model, "distant_swap_repair")
        axes[0].plot(
            [1, 3, 5, 8],
            [
                swap["both_swap_positions_restored_rate"],
                dist["dist3_both_restored_rate"],
                dist["dist5_both_restored_rate"],
                dist["dist8_both_restored_rate"],
            ],
            linewidth=2.0,
            color=color,
            label=model["display"],
        )

    x = np.arange(len(CLM_MODELS)) * 1.2
    width = 0.16
    single_vals = []
    sim_vals = []
    actual_vals = []
    dist8_vals = []
    for model in CLM_MODELS:
        single = clm_metrics(model, "single_corrupt_repair")["unmasked_restored_rate"]
        actual = clm_metrics(model, "swap_repair")["both_swap_positions_restored_rate"]
        dist8 = clm_metrics(model, "distant_swap_repair")["dist8_both_restored_rate"]
        single_vals.append(single)
        sim_vals.append(single ** 2)
        actual_vals.append(actual)
        dist8_vals.append(dist8)

    axes[1].bar(x - 1.5 * width, single_vals, width=width, color="#4c72b0", label="single-token repair")
    axes[1].bar(x - 0.5 * width, sim_vals, width=width, color="#55a868", label="simulated independent both")
    axes[1].bar(x + 0.5 * width, actual_vals, width=width, color="#c44e52", label="actual adjacent swap both")
    axes[1].bar(x + 1.5 * width, dist8_vals, width=width, color="#8172b2", label="actual dist-8 swap both")

    axes[0].set_xlabel("Distance between swapped tokens")
    axes[0].set_ylabel("Both swapped tokens restored")
    axes[0].set_xticks([1, 3, 5, 8])
    axes[0].set_ylim(0, 0.5)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=3)

    axes[1].set_ylabel("Restoration probability")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m["display"] for m in CLM_MODELS], rotation=35, ha="right")
    axes[1].set_ylim(0, 0.5)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    save_fig(fig, FIG_DIR / "fig06_clm_local_repair.png")


def fig07_clm_sink_intervention() -> None:
    labels = [m["display"] for m in CLM_MODELS]
    pos0_attn = []
    ordinary_attn = []
    gen_freeze_first = []
    gen_zero_ordinary = []
    for model in CLM_MODELS:
        attn = clm_metrics(model, "decoder_attention_centrality")
        ctl = clm_metrics(model, "decoder_control_intervention")
        pos0_attn.append(attn["avg_attn_to_pos0"])
        ordinary_attn.append(attn["avg_attn_to_ordinary"])
        gen_freeze_first.append(ctl["gen_freeze_first"])
        gen_zero_ordinary.append(ctl["gen_zero_ordinary_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.8))
    x = np.arange(len(labels))
    width = 0.38
    axes[0].bar(x - width / 2, pos0_attn, width=width, color="#4c72b0", label="avg attn to pos0")
    axes[0].bar(x + width / 2, ordinary_attn, width=width, color="#55a868", label="avg attn to ordinary")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("Average incoming attention")
    axes[0].grid(True, axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    axes[1].bar(x - width / 2, gen_freeze_first, width=width, color="#c44e52", label="freeze first")
    axes[1].bar(x + width / 2, gen_zero_ordinary, width=width, color="#55a868", label="ordinary control")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("Generation retention")
    axes[1].grid(True, axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    save_fig(fig, FIG_DIR / "fig07_clm_sink_intervention.png")


def fig08_clm_prefix_self_correction() -> None:
    labels = []
    values = []
    for model in CLM_MODELS:
        metrics = clm_metrics(model, "decoder_prefix_corruption")
        labels.append(model["display"])
        values.append(metrics["self_correction_signal"])

    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.bar(np.arange(len(labels)), values, color="#2ca02c")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Prefix self-correction across CLMs")
    ax.set_ylabel("Self-correction signal")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.2)
    save_fig(fig, FIG_DIR / "fig08_clm_prefix_self_correction.png")


def fig09_clm_local_repair_mirror() -> None:
    colors = sns.color_palette("tab10", len(CLM_MODELS))
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.4))

    for color, model in zip(colors, CLM_MODELS):
        swap = clm_metrics(model, "swap_repair")
        dist = clm_metrics(model, "distant_swap_repair")
        axes[0].plot(
            [1, 3, 5, 8],
            [
                swap["both_swap_positions_restored_rate"],
                dist["dist3_both_restored_rate"],
                dist["dist5_both_restored_rate"],
                dist["dist8_both_restored_rate"],
            ],
            linewidth=2.0,
            color=color,
            label=model["display"],
        )

    x = np.arange(len(CLM_MODELS)) * 1.2
    width = 0.16
    single_vals = []
    sim_vals = []
    actual_vals = []
    dist8_vals = []
    for model in CLM_MODELS:
        single = clm_metrics(model, "single_corrupt_repair")["unmasked_restored_rate"]
        actual = clm_metrics(model, "swap_repair")["both_swap_positions_restored_rate"]
        dist8 = clm_metrics(model, "distant_swap_repair")["dist8_both_restored_rate"]
        single_vals.append(single)
        sim_vals.append(single ** 2)
        actual_vals.append(actual)
        dist8_vals.append(dist8)

    axes[1].bar(x - 1.5 * width, single_vals, width=width, color="#4c72b0", label="single-token repair")
    axes[1].bar(x - 0.5 * width, sim_vals, width=width, color="#55a868", label="simulated independent both")
    axes[1].bar(x + 0.5 * width, actual_vals, width=width, color="#c44e52", label="actual adjacent swap both")
    axes[1].bar(x + 1.5 * width, dist8_vals, width=width, color="#8172b2", label="actual dist-8 swap both")

    axes[0].set_xlabel("Distance between swapped tokens")
    axes[0].set_ylabel("Both swapped tokens restored")
    axes[0].set_xticks([1, 3, 5, 8])
    axes[0].set_ylim(0, 0.5)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1].set_ylabel("Restoration probability")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m["display"] for m in CLM_MODELS], rotation=35, ha="right")
    axes[1].set_ylim(0, 0.5)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)

    save_fig(fig, FIG_DIR / "fig09_clm_local_repair_mirror.png")


def fig10_clm_control_heatmap() -> None:
    conditions = [
        "freeze_bos",
        "freeze_delimiters",
        "freeze_first",
        "zero_bos",
        "zero_delimiters",
        "zero_first",
        "zero_ordinary_mean",
    ]
    label_map = {
        "freeze_bos": "freeze BOS",
        "freeze_delimiters": "freeze delimiters",
        "freeze_first": "freeze position 0",
        "zero_bos": "zero BOS",
        "zero_delimiters": "zero delimiters",
        "zero_first": "zero position 0",
        "zero_ordinary_mean": "zero ordinary mean",
    }
    data = []
    for cond in conditions:
        row = []
        for model in CLM_MODELS:
            metrics = clm_metrics(model, "decoder_control_intervention")
            baseline = max(metrics["gen_baseline"], 1e-9)
            row.append(metrics[f"gen_{cond}"] / baseline)
        data.append(row)

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    sns.heatmap(
        np.array(data),
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Generation retention relative to baseline"},
        xticklabels=[m["display"] for m in CLM_MODELS],
        yticklabels=[label_map[c] for c in conditions],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, FIG_DIR / "fig10_clm_control_heatmap.png")


def fig11_clm_pca_depth_panel() -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 13.2))
    for ax, model in zip(axes.flat, CLM_MODELS):
        ax.imshow(mpimg.imread(CLM_PCA_IMAGE_PATHS[model["display"]]))
        ax.set_title(model["display"], fontsize=14, pad=8)
        ax.axis("off")
    save_fig(fig, FIG_DIR / "fig11_clm_pca_depth_panel.png")


def write_captions() -> None:
    path = FIG_DIR / "figure_captions.md"
    text = """# Paper Figures

Generated from canonical artifacts on `2026-04-30`.

## Figure 1

`fig01_mlm_local_repair.png`

Left: all five MLM families show much stronger two-token recovery when the corrupted tokens are farther apart. Right: single-token repair, the simulated independent-both baseline, actual adjacent swap repair with both positions restored, and actual distance-8 swap repair with both positions restored. Model order in the right panel is ALBERT, BERT, RoBERTa, ModernBERT, and XLM-R.

## Figure 2

`fig02_mlm_special_token_heatmap.png`

Special-token intervention is strongly architecture-dependent across the full MLM model set.

## Figure 3

`fig03_mlm_pca_depth_panel.png`

Each cell shows a PCA-3D depth plot. Depth separation remains clear for all MLM families included in the paper.

## Figure 12

`fig12_mlm_readout_depth.png`

Layerwise MLM exact-match readout curves on the benchmark split. ALBERT, BERT, RoBERTa, and XLM-R are shown together, while ModernBERT is separated into its own panel. All five models retain substantial token-decoding structure through depth.

## Figure 4

`fig04_clm_next_token_readout.png`

All nine CLMs show measurable next-token readout across depth.

## Figure 5

`fig05_clm_shifted_recovery.png`

All nine CLMs show strong tuned-lens readout both at `last` and at `last-1`, the decoder position that predicts the token at the visible right edge.

## Figure 6

`fig06_clm_local_repair.png`

Left: all nine CLMs show much stronger two-token recovery when the corrupted tokens are farther apart. Right: single-token repair, the simulated independent-both baseline, actual adjacent swap repair with both positions restored, and actual distance-8 swap repair with both positions restored.

## Figure 7

`fig07_clm_sink_intervention.png`

Average incoming attention to position 0 is compared directly with average incoming attention to ordinary positions, and freezing position 0 is compared directly with an ordinary-position control. Model order is distilgpt2, gpt2, pythia-160m, pythia-410m, SmolLM2-135M, SmolLM2-360M, Qwen2.5-0.5B, Llama-3.2-1B, and gemma-3-1b.

## Figure 8

`fig08_clm_prefix_self_correction.png`

All nine CLMs show positive prefix self-correction signal.

## Figure 9

`fig09_clm_local_repair_mirror.png`

CLMs show the same broad local-repair pattern as MLMs: adjacent corruption is harder than distant corruption, and actual adjacent swap repair with both positions restored stays below the simulated independent-both baseline.

## Figure 10

`fig10_clm_control_heatmap.png`

Generation retention under decoder control-position interventions shows that position 0 is usually much more causally important than ordinary-position controls.

## Figure 11

`fig11_clm_pca_depth_panel.png`

Each cell shows a PCA-3D depth plot. Depth separation remains clear for all paper-facing CLMs on the benchmark split.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    fig01_mlm_local_repair()
    fig02_mlm_special_token_heatmap()
    fig03_mlm_pca_depth_panel()
    fig12_mlm_readout_depth()
    fig04_clm_next_token_readout()
    fig05_clm_shifted_recovery()
    fig06_clm_local_repair()
    fig07_clm_sink_intervention()
    fig08_clm_prefix_self_correction()
    fig09_clm_local_repair_mirror()
    fig10_clm_control_heatmap()
    fig11_clm_pca_depth_panel()
    write_captions()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
