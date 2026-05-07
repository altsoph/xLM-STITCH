"""Canonical data pipeline and dataset management.

Manages loading, splitting, and caching of all dataset types:
  - corpus_base / corpus_large
  - coref_structured
  - handcrafted_debug
  - multilingual_eval
  - decoder_eval
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DataSplit:
    name: str
    examples: List[Dict[str, Any]]

    def __len__(self) -> int:
        return len(self.examples)


SPLIT_SIZES = {
    "smoke": (50, 200),
    "dev": (200, 500),
    "benchmark": (500, 5000),
    "stress": (1000, 10000),
}

HANDCRAFTED_SMOKE = [
    "The cat sat on the mat.",
    "She handed him the book that was on the table.",
    "The doctor told the nurse that she would be late.",
    "After the storm, the river flooded the valley below.",
    "John gave Mary the letter, but she didn't read it.",
    "The trophy would not fit in the brown suitcase because it was too big.",
    "The city council refused the demonstrators a permit because they feared violence.",
    "The lawyer asked the witness a question, and he answered truthfully.",
    "I saw the man on the hill with the telescope.",
    "The horse raced past the barn fell.",
    "Time flies like an arrow; fruit flies like a banana.",
    "The old man the boats.",
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.",
    "They are hunting dogs.",
    "Visiting relatives can be boring.",
    "The professor said the student failed.",
    "We saw her duck.",
    "The chicken is ready to eat.",
    "Flying planes can be dangerous.",
    "The complex houses married and single soldiers and their families.",
]


def load_sentences_from_file(path: str | Path) -> List[str]:
    """Load one sentence per line from a text file."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def make_splits(
    sentences: List[str],
    seed: int = 42,
) -> Dict[str, DataSplit]:
    """Create canonical splits from a list of sentences."""
    rng = random.Random(seed)
    shuffled = list(sentences)
    rng.shuffle(shuffled)

    splits: Dict[str, DataSplit] = {}
    offset = 0
    for split_name in ["smoke", "dev", "benchmark", "stress"]:
        min_size, max_size = SPLIT_SIZES[split_name]
        target = min(max_size, max(min_size, len(shuffled) - offset))
        chunk = shuffled[offset : offset + target]
        offset += target
        examples = []
        for i, sent in enumerate(chunk):
            examples.append({
                "example_id": f"{split_name}_{i}",
                "text": sent,
                "source_hash": hashlib.sha256(sent.encode()).hexdigest()[:16],
            })
        splits[split_name] = DataSplit(name=split_name, examples=examples)
        if offset >= len(shuffled):
            break

    return splits


def get_handcrafted_debug() -> DataSplit:
    """Return the fixed handcrafted debug dataset."""
    examples = []
    for i, sent in enumerate(HANDCRAFTED_SMOKE):
        examples.append({
            "example_id": f"debug_{i}",
            "text": sent,
            "source_hash": hashlib.sha256(sent.encode()).hexdigest()[:16],
        })
    return DataSplit(name="handcrafted_debug", examples=examples)


def save_split(split: DataSplit, output_dir: str | Path) -> Path:
    """Save a split to a JSONL manifest file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split.name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ex in split.examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return path


def load_split(path: str | Path) -> DataSplit:
    """Load a split from a JSONL file."""
    path = Path(path)
    name = path.stem
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return DataSplit(name=name, examples=examples)


def get_corpus_split(
    split: str = "smoke",
    max_examples: int = 0,
) -> DataSplit:
    """Load a canonical corpus split (smoke/dev/benchmark/stress).

    Falls back to handcrafted_debug if split files don't exist yet.
    """
    splits_dir = Path(__file__).resolve().parent / "splits"
    path = splits_dir / f"{split}.jsonl"
    if path.exists():
        ds = load_split(path)
        if max_examples > 0:
            ds = DataSplit(name=ds.name, examples=ds.examples[:max_examples])
        return ds
    return get_handcrafted_debug()


def make_causal_prefix_trimmed_split(
    split: DataSplit,
    tokenizer: Any,
    seed: int = 42,
    trim_min_tokens: int = 2,
    trim_max_tokens: int = 5,
    min_visible_tokens: int = 4,
) -> DataSplit:
    """Derive a CLM-only split with a hidden suffix.

    For each example, deterministically remove 2-5 tokenizer tokens from the
    sentence end so the decoder's next-token target is usually internal rather
    than sentence-final. Trimming is bounded to leave at least
    ``min_visible_tokens`` in the visible prefix.
    """
    derived_examples: List[Dict[str, Any]] = []

    trim_applied = 0
    trim_total = 0
    trim_examples = 0

    for ex in split.examples:
        text = ex["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        visible_ids = list(token_ids)
        trimmed_suffix_ids: List[int] = []

        available_trim = max(0, min(trim_max_tokens, len(token_ids) - min_visible_tokens))
        if available_trim > 0:
            trim_low = min(trim_min_tokens, available_trim)
            trim_high = available_trim
            key = f"{seed}:{ex.get('source_hash', ex.get('example_id', text))}"
            rng_seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
            rng = random.Random(rng_seed)
            trim_n = rng.randint(trim_low, trim_high)
            if trim_n > 0:
                visible_ids = token_ids[:-trim_n]
                trimmed_suffix_ids = token_ids[-trim_n:]

        if visible_ids:
            visible_text = tokenizer.decode(
                visible_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        else:
            visible_text = text

        if not visible_text:
            visible_text = text
            visible_ids = list(token_ids)
            trimmed_suffix_ids = []

        trim_n = len(trimmed_suffix_ids)
        trim_applied += trim_n
        trim_total += len(token_ids)
        trim_examples += int(trim_n > 0)

        new_ex = dict(ex)
        new_ex["text"] = visible_text
        new_ex["clm_trimmed_suffix_tokens"] = trim_n
        new_ex["clm_visible_token_count"] = len(visible_ids)
        new_ex["clm_original_token_count"] = len(token_ids)
        if trimmed_suffix_ids:
            new_ex["clm_hidden_suffix_text"] = tokenizer.decode(
                trimmed_suffix_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        derived_examples.append(new_ex)

    derived = DataSplit(name=f"{split.name}_clm_trimmed", examples=derived_examples)
    derived.trim_metadata = {  # type: ignore[attr-defined]
        "variant": "clm_prefix_trimmed",
        "trim_examples": trim_examples,
        "avg_trimmed_tokens": round(trim_applied / max(len(split.examples), 1), 4),
        "avg_original_tokens": round(trim_total / max(len(split.examples), 1), 4),
        "trim_min_tokens": trim_min_tokens,
        "trim_max_tokens": trim_max_tokens,
        "min_visible_tokens": min_visible_tokens,
    }
    return derived
