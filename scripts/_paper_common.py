from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.schema import DatasetConfig, ModelConfig, RunConfig
RESULTS = ROOT / "results" / "exploratory"
PAPER_DIR = ROOT / "paper"
TABLE_DIR = PAPER_DIR / "tables"
FIG_DIR = PAPER_DIR / "figures"
FIG_DATA_DIR = PAPER_DIR / "figure_data"
REPORT_DIR = ROOT / "reports" / "drift"
REFERENCE_DIR = ROOT / "reference"

MLM_MODELS = [
    {"name": "albert-base-v2", "display": "ALBERT", "phase": "phase1", "layers": 12, "params_m": 11.3},
    {"name": "bert-base-uncased", "display": "BERT", "phase": "phase1", "layers": 12, "params_m": 109.5},
    {"name": "roberta-base", "display": "RoBERTa", "phase": "phase1", "layers": 12, "params_m": 124.7},
    {"name": "answerdotai/ModernBERT-base", "display": "ModernBERT", "phase": "phase2", "layers": 22, "params_m": 149.7},
    {"name": "xlm-roberta-base", "display": "XLM-R", "phase": "phase2", "layers": 12, "params_m": 278.3},
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
    {"name": "google/gemma-3-1b-pt", "display": "gemma-3-1b", "layers": 26, "params_m": 999.9},
]

MLM_TASKS = [
    "probe_holdout",
    "lens_decoding",
    "clean_passthrough",
    "single_corrupt_repair",
    "swap_repair",
    "distant_swap_repair",
    "special_token_intervention",
]

CLM_TASKS = [
    "probe_holdout",
    "lens_decoding",
    "decoder_tuned_lens",
    "single_corrupt_repair",
    "swap_repair",
    "distant_swap_repair",
    "decoder_control_intervention",
]

SMOKE_MLM_MODELS = ["bert-base-uncased"]
SMOKE_CLM_MODELS = ["gpt2"]


def model_run_token(name: str) -> str:
    return name.split("/")[-1].replace("-", "_").replace(".", "_")


def mlm_run_dir(model: dict, task: str, split: str) -> Path:
    phase = model["phase"]
    return RESULTS / f"{phase}_{split}" / f"{phase}_{model_run_token(model['name'])}_{task}_{split}_s42"


def clm_run_dir(model: dict, task: str, split: str) -> Path:
    return RESULTS / f"phase4_{split}" / f"phase4_{model_run_token(model['name'])}_{task}_{split}_s42"


def build_mlm_configs(split: str, device: str) -> list[RunConfig]:
    out = []
    for model in MLM_MODELS:
        for task in MLM_TASKS:
            out.append(RunConfig(
                run_id=f"{model['phase']}_{model_run_token(model['name'])}_{task}_{split}_s42",
                phase=model["phase"],
                model=ModelConfig(name=model["name"], device=device, output_attentions=False),
                dataset=DatasetConfig(name="bookcorpusopen", split=split),
                task_family=task,
                seed=42,
                batch_size=0,
                output_dir=f"results/exploratory/{model['phase']}_{split}",
            ))
    return out


def build_clm_configs(split: str, device: str) -> list[RunConfig]:
    out = []
    for model in CLM_MODELS:
        for task in CLM_TASKS:
            out.append(RunConfig(
                run_id=f"phase4_{model_run_token(model['name'])}_{task}_{split}_s42",
                phase="phase4",
                model=ModelConfig(name=model["name"], device=device, output_attentions=False),
                dataset=DatasetConfig(
                    name="bookcorpusopen",
                    split=split,
                    variant="clm_prefix_trimmed",
                    trim_min_tokens=2,
                    trim_max_tokens=5,
                    min_visible_tokens=4,
                ),
                task_family=task,
                seed=42,
                batch_size=0,
                output_dir=f"results/exploratory/phase4_{split}",
            ))
    return out


def filter_for_smoke(configs: list[RunConfig]) -> list[RunConfig]:
    keep = set(SMOKE_MLM_MODELS + SMOKE_CLM_MODELS)
    return [cfg for cfg in configs if cfg.model.name in keep]


def load_metrics(path: Path) -> dict:
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)["summary_metrics"]


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def max_layer_value(layer_map: dict[str, float]) -> float:
    last_key = max(int(k) for k in layer_map)
    return float(layer_map[str(last_key)])
