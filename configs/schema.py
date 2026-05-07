"""Config schema for declarative run specifications.

Each experiment run is defined by a YAML manifest. This module
provides the dataclasses and loaders for those configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str
    split: str = "smoke"
    path: str = ""
    max_examples: int = 0
    language: str = "en"
    variant: str = ""
    trim_min_tokens: int = 0
    trim_max_tokens: int = 0
    min_visible_tokens: int = 0


@dataclass
class ModelConfig:
    name: str
    family: str = "auto"
    device: str = "cpu"
    precision: str = "fp32"
    output_attentions: bool = False
    lens_mode: str = "auto"  # "auto" = use PREFERRED_LENS_MODES lookup


# Per-model optimal untrained lens mode, determined by benchmark.
# "auto" in ModelConfig.lens_mode resolves to these defaults.
# Keys are short model names (after split("/")[-1]).
PREFERRED_LENS_MODES: Dict[str, Dict[str, str]] = {
    # {short_name: {"standard": mode, "weighted": mode}}
    # Determined by scripts/benchmark_lens_modes.py on smoke (50 examples).
    "distilgpt2":       {"standard": "raw",     "weighted": "raw"},
    "gpt2":             {"standard": "raw",     "weighted": "normed"},
    "pythia-160m":      {"standard": "normed",  "weighted": "normed"},
    "pythia-410m":      {"standard": "normed",  "weighted": "normed"},
    "SmolLM2-135M":     {"standard": "normed",  "weighted": "normed"},
    "SmolLM2-360M":     {"standard": "normed",  "weighted": "normed"},
    "Qwen2.5-0.5B":     {"standard": "normed",  "weighted": "normed"},
    "TinyLlama_v1.1":   {"standard": "cosine",  "weighted": "cosine"},
    "Llama-3.2-1B":     {"standard": "raw",     "weighted": "raw"},
    "gemma-3-1b-pt":    {"standard": "normed",  "weighted": "normed"},
}


def resolve_lens_mode(model_name: str, metric: str = "standard") -> str:
    """Resolve 'auto' lens mode for a given model.

    Parameters
    ----------
    model_name : str
        Full or short model name.
    metric : str
        Which metric to optimise for: 'standard' or 'weighted'.
    """
    short = model_name.split("/")[-1]
    entry = PREFERRED_LENS_MODES.get(short, {})
    return entry.get(metric, "normed")  # fallback: normed


@dataclass
class InterventionConfig:
    enabled: bool = False
    operations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RunConfig:
    """Full specification for a single experiment run."""
    run_id: str
    phase: str
    model: ModelConfig
    dataset: DatasetConfig
    task_family: str
    seed: int = 42
    batch_size: int = 8
    precision: str = "fp32"
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    output_dir: str = "results/exploratory"
    artifact_level: str = "exploratory"
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.run_id


def load_run_config(path: str | Path) -> RunConfig:
    """Load a RunConfig from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    model = ModelConfig(**raw.pop("model"))
    dataset = DatasetConfig(**raw.pop("dataset"))
    intervention = InterventionConfig(**raw.pop("intervention", {}))

    return RunConfig(
        model=model,
        dataset=dataset,
        intervention=intervention,
        **raw,
    )


def save_run_config(config: RunConfig, path: str | Path) -> None:
    """Save a RunConfig to YAML."""
    from dataclasses import asdict
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
