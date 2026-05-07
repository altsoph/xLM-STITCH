"""Materialize full probe feature caches for later CPU-side probe training.

The cache stores untruncated per-token hidden states across all layers for a
fixed dataset split. Because sequence lengths vary, the on-disk layout is a
packed token-major tensor plus metadata arrays that preserve example and token
boundaries:

  hidden_states: [total_tokens, num_layers, hidden_size]
  input_ids: [total_tokens]
  family_ids: [total_tokens]
  position_in_example: [total_tokens]
  example_offsets: [num_examples + 1]

This keeps the full examples × layers × tokens information without padding
everything to a global max length.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.schema import RunConfig
from datasets.data_manager import get_corpus_split, make_causal_prefix_trimmed_split
from models.adapters.factory import create_adapter
from models.hooks.token_metadata import extract_token_metadata


CACHE_VERSION = 1
CACHE_DIRNAME = "probe_feature_cache_full_v1"

PROBE_CACHE_STORAGE_DTYPE_OVERRIDES: dict[str, str] = {
    "google/gemma-3-1b-pt": "float32",
}

PROBE_CACHE_BATCH_SIZE_HINTS: dict[str, int] = {
    "distilgpt2": 8,
    "gpt2": 8,
    "EleutherAI/pythia-160m": 8,
    "EleutherAI/pythia-410m": 4,
    "HuggingFaceTB/SmolLM2-135M": 4,
    "HuggingFaceTB/SmolLM2-360M": 4,
    "Qwen/Qwen2.5-0.5B": 4,
    "meta-llama/Llama-3.2-1B": 2,
    "google/gemma-3-1b-pt": 2,
}


@dataclass
class CacheStatus:
    exists: bool
    cache_dir: Path
    metadata_path: Path


def cache_dir_for_run(base_dir: Path, run_config: RunConfig) -> Path:
    return base_dir / run_config.output_path / CACHE_DIRNAME


def cache_status(base_dir: Path, run_config: RunConfig) -> CacheStatus:
    cache_dir = cache_dir_for_run(base_dir, run_config)
    metadata_path = cache_dir / "metadata.json"
    return CacheStatus(
        exists=metadata_path.exists(),
        cache_dir=cache_dir,
        metadata_path=metadata_path,
    )


def load_cache_metadata(cache_dir: Path) -> dict[str, Any]:
    with open(cache_dir / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def recommended_probe_cache_storage_dtype(model_name: str, default: str = "float16") -> str:
    return PROBE_CACHE_STORAGE_DTYPE_OVERRIDES.get(model_name, default)


def recommended_probe_cache_batch_size(model_name: str, default: int = 8) -> int:
    return PROBE_CACHE_BATCH_SIZE_HINTS.get(model_name, default)


class LayerProbeCacheDataset(Dataset):
    """Lazy token×layer view over a full probe feature cache."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        split: str = "train",
        train_ratio: float = 0.7,
        layer_indices: Sequence[int] | None = None,
        family_values: Sequence[int] | None = None,
    ):
        meta = load_cache_metadata(cache_dir)
        self.meta = meta
        self.hidden_states = np.load(cache_dir / meta["files"]["hidden_states"], mmap_mode="r")
        self.example_index = np.load(cache_dir / meta["files"]["example_index"], mmap_mode="r")
        self.family_ids = np.load(cache_dir / meta["files"]["family_ids"], mmap_mode="r")
        self.num_layers = int(meta["num_layers"])
        self.hidden_size = int(meta["hidden_size"])
        self.total_examples = int(meta["total_examples"])
        self.train_examples = max(1, int(self.total_examples * train_ratio))
        self.test_examples = self.total_examples - self.train_examples
        self.layer_indices = np.array(
            list(layer_indices) if layer_indices is not None else list(range(self.num_layers)),
            dtype=np.int16,
        )
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if split == "train":
            token_mask = self.example_index < self.train_examples
        else:
            token_mask = self.example_index >= self.train_examples
        if family_values is not None:
            family_mask = np.isin(self.family_ids, np.array(list(family_values), dtype=self.family_ids.dtype))
            token_mask = np.logical_and(token_mask, family_mask)
        self.token_indices = np.flatnonzero(token_mask).astype(np.int64)
        self._layers_per_token = len(self.layer_indices)

    def __len__(self) -> int:
        return int(len(self.token_indices) * self._layers_per_token)

    def __getitem__(self, idx: int):
        token_slot = idx // self._layers_per_token
        layer_slot = idx % self._layers_per_token
        token_idx = int(self.token_indices[token_slot])
        layer_idx = int(self.layer_indices[layer_slot])
        feat = np.asarray(self.hidden_states[token_idx, layer_idx], dtype=np.float32)
        return torch.from_numpy(feat), layer_idx


def _load_split(run_config: RunConfig, tokenizer: Any):
    split = get_corpus_split(
        split=run_config.dataset.split,
        max_examples=run_config.dataset.max_examples,
    )
    if run_config.dataset.variant == "clm_prefix_trimmed":
        split = make_causal_prefix_trimmed_split(
            split=split,
            tokenizer=tokenizer,
            seed=run_config.seed,
            trim_min_tokens=max(1, run_config.dataset.trim_min_tokens or 2),
            trim_max_tokens=max(1, run_config.dataset.trim_max_tokens or 5),
            min_visible_tokens=max(1, run_config.dataset.min_visible_tokens or 4),
        )
    return split


def _batched(seq: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def _token_lengths(texts: Sequence[str], tokenizer: Any, batch_size: int) -> list[int]:
    lengths: list[int] = []
    for batch in _batched(texts, batch_size):
        encoded = tokenizer(
            list(batch),
            return_attention_mask=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        attn = encoded["attention_mask"].numpy()
        lengths.extend(int(attn[i].sum()) for i in range(attn.shape[0]))
    return lengths


def _dtype_name(dtype: np.dtype) -> str:
    return np.dtype(dtype).name


def materialize_probe_feature_cache(
    run_config: RunConfig,
    base_dir: Path,
    *,
    device_override: str | None = None,
    model_precision_override: str | None = None,
    storage_dtype: str = "float16",
    batch_size: int = 8,
    tokenizer_batch_size: int = 64,
    force: bool = False,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Create a full probe feature cache if missing and return its metadata."""

    status = cache_status(base_dir, run_config)
    if status.exists and not force:
        return load_cache_metadata(status.cache_dir)

    cache_dir = status.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    family_override = None
    if run_config.model.family != "auto":
        from models.adapters.base import ModelFamily
        family_override = ModelFamily[run_config.model.family.upper()]

    model_device = device_override or run_config.model.device
    model_precision = model_precision_override or run_config.model.precision

    adapter = create_adapter(
        run_config.model.name,
        device=model_device,
        precision=model_precision,
        output_attentions=False,
        family=family_override,
    )
    adapter.load()

    try:
        split = _load_split(run_config, adapter.tokenizer)
        examples = split.examples
        texts = [ex["text"] for ex in examples]
        example_ids = [ex["example_id"] for ex in examples]

        token_lengths = _token_lengths(texts, adapter.tokenizer, tokenizer_batch_size)
        total_examples = len(texts)
        total_tokens = int(sum(token_lengths))

        sample_batch = texts[: min(batch_size, total_examples)]
        sample_enc = adapter.encode_text(list(sample_batch))
        sample_out = adapter.forward(**sample_enc)
        num_layers = len(sample_out.hidden_states)
        hidden_size = int(sample_out.hidden_states[0].shape[-1])

        storage_np_dtype = np.float16 if storage_dtype == "float16" else np.float32
        hidden_path = cache_dir / "hidden_states.npy"
        input_ids_path = cache_dir / "input_ids.npy"
        family_ids_path = cache_dir / "family_ids.npy"
        position_path = cache_dir / "position_in_example.npy"
        example_index_path = cache_dir / "example_index.npy"
        offsets_path = cache_dir / "example_offsets.npy"
        example_ids_path = cache_dir / "example_ids.json"

        hidden_mm = np.lib.format.open_memmap(
            hidden_path,
            mode="w+",
            dtype=storage_np_dtype,
            shape=(total_tokens, num_layers, hidden_size),
        )
        input_ids_mm = np.lib.format.open_memmap(
            input_ids_path,
            mode="w+",
            dtype=np.int32,
            shape=(total_tokens,),
        )
        family_ids_mm = np.lib.format.open_memmap(
            family_ids_path,
            mode="w+",
            dtype=np.uint8,
            shape=(total_tokens,),
        )
        position_mm = np.lib.format.open_memmap(
            position_path,
            mode="w+",
            dtype=np.int16,
            shape=(total_tokens,),
        )
        example_index_mm = np.lib.format.open_memmap(
            example_index_path,
            mode="w+",
            dtype=np.int32,
            shape=(total_tokens,),
        )
        example_offsets = np.zeros(total_examples + 1, dtype=np.int64)

        cursor = 0
        processed = 0
        with torch.inference_mode():
            for batch_start in range(0, total_examples, batch_size):
                batch_examples = examples[batch_start:batch_start + batch_size]
                batch_texts = [ex["text"] for ex in batch_examples]
                enc = adapter.encode_text(batch_texts)
                out = adapter.forward(**enc)
                input_ids = enc["input_ids"].detach().cpu().numpy()
                attention_mask = enc["attention_mask"].detach().cpu().numpy()
                layer_stack = np.stack(
                    [hs.detach().cpu().numpy() for hs in out.hidden_states],
                    axis=1,  # [batch, layers, seq, hidden]
                )

                for b_idx, ex in enumerate(batch_examples):
                    valid_len = int(attention_mask[b_idx].sum())
                    token_ids = input_ids[b_idx, :valid_len].tolist()
                    metas = extract_token_metadata(token_ids, adapter.tokenizer)
                    family_ids = np.array([m.family.value for m in metas], dtype=np.uint8)
                    positions = np.arange(valid_len, dtype=np.int16)
                    packed = np.transpose(layer_stack[b_idx, :, :valid_len, :], (1, 0, 2))

                    hidden_mm[cursor:cursor + valid_len] = packed.astype(storage_np_dtype, copy=False)
                    input_ids_mm[cursor:cursor + valid_len] = np.array(token_ids, dtype=np.int32)
                    family_ids_mm[cursor:cursor + valid_len] = family_ids
                    position_mm[cursor:cursor + valid_len] = positions
                    example_index_mm[cursor:cursor + valid_len] = batch_start + b_idx
                    example_offsets[batch_start + b_idx] = cursor
                    cursor += valid_len
                    processed += 1

                if progress_callback is not None:
                    progress_callback(processed, total_examples)

        example_offsets[-1] = cursor
        np.save(offsets_path, example_offsets)
        with open(example_ids_path, "w", encoding="utf-8") as f:
            json.dump(example_ids, f, ensure_ascii=True, indent=2)

        metadata = {
            "version": CACHE_VERSION,
            "model_name": run_config.model.name,
            "dataset_split": run_config.dataset.split,
            "dataset_variant": run_config.dataset.variant,
            "total_examples": total_examples,
            "total_tokens": int(cursor),
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "storage_dtype": _dtype_name(storage_np_dtype),
            "model_precision": model_precision,
            "device": model_device,
            "batch_size": batch_size,
            "tokenizer_batch_size": tokenizer_batch_size,
            "files": {
                "hidden_states": hidden_path.name,
                "input_ids": input_ids_path.name,
                "family_ids": family_ids_path.name,
                "position_in_example": position_path.name,
                "example_index": example_index_path.name,
                "example_offsets": offsets_path.name,
                "example_ids": example_ids_path.name,
            },
        }
        with open(status.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2)
        return metadata
    finally:
        adapter.unload()
