"""Task generators for input perturbation benchmarks.

Generates corruption tasks from clean sentences:
  - local pair swaps
  - cyclic permutation
  - full shuffle
  - grammar corruption
  - punctuation corruption
  - named-entity corruption
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from models.hooks.token_metadata import TokenFamily, TokenMeta


@dataclass
class TaskExample:
    """Universal task-example schema (Section 6.3 of plan)."""
    example_id: str
    source_dataset: str
    text: str
    tokenized_text: List[str]
    language: str = "en"
    task_family: str = ""
    seed: int = 0
    generator_version: str = "1.0"
    tokenizer_id: str = ""
    source_hash: str = ""
    corruption_recipe: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _non_special_positions(metas: List[TokenMeta]) -> List[int]:
    return [m.position for m in metas if not m.is_special]


def generate_swap_task(
    text: str,
    tokens: List[str],
    token_ids: List[int],
    metas: List[TokenMeta],
    source_dataset: str = "corpus_base",
    seed: int = 42,
    tokenizer_id: str = "",
    distance: int = 1,
    exclude_positions: Optional[Set[int]] = None,
) -> Optional[TaskExample]:
    """Generate a local pair-swap corruption task."""
    rng = random.Random(seed)
    excluded = exclude_positions or set()
    positions = [p for p in _non_special_positions(metas) if p not in excluded]
    if len(positions) < 2:
        return None

    valid_pairs = [(p, p + distance) for p in positions
                   if (p + distance) in positions and p + distance < len(tokens)]
    if not valid_pairs:
        return None

    a, b = rng.choice(valid_pairs)
    corrupted_ids = list(token_ids)
    corrupted_ids[a], corrupted_ids[b] = corrupted_ids[b], corrupted_ids[a]
    corrupted_tokens = list(tokens)
    corrupted_tokens[a], corrupted_tokens[b] = corrupted_tokens[b], corrupted_tokens[a]

    return TaskExample(
        example_id=f"swap_{_hash_text(text)}_{seed}",
        source_dataset=source_dataset,
        text=text,
        tokenized_text=corrupted_tokens,
        task_family="swap_repair",
        seed=seed,
        tokenizer_id=tokenizer_id,
        source_hash=_hash_text(text),
        corruption_recipe=f"swap_d{distance}_pos{a}_{b}",
        metadata={
            "original_tokens": tokens,
            "original_ids": token_ids,
            "corrupted_ids": corrupted_ids,
            "swap_positions": [a, b],
            "swap_distance": distance,
            "token_types": [m.family.name for m in metas],
        },
    )


def generate_cyclic_permutation_task(
    text: str,
    tokens: List[str],
    token_ids: List[int],
    metas: List[TokenMeta],
    source_dataset: str = "corpus_base",
    seed: int = 42,
    tokenizer_id: str = "",
    cycle_length: int = 3,
) -> Optional[TaskExample]:
    """Generate a cyclic permutation corruption."""
    rng = random.Random(seed)
    positions = _non_special_positions(metas)
    if len(positions) < cycle_length:
        return None

    selected = rng.sample(positions, cycle_length)
    selected.sort()

    corrupted_ids = list(token_ids)
    corrupted_tokens = list(tokens)

    vals_ids = [corrupted_ids[p] for p in selected]
    vals_tok = [corrupted_tokens[p] for p in selected]
    shifted_ids = [vals_ids[-1]] + vals_ids[:-1]
    shifted_tok = [vals_tok[-1]] + vals_tok[:-1]
    for i, p in enumerate(selected):
        corrupted_ids[p] = shifted_ids[i]
        corrupted_tokens[p] = shifted_tok[i]

    return TaskExample(
        example_id=f"cyclic_{_hash_text(text)}_{seed}",
        source_dataset=source_dataset,
        text=text,
        tokenized_text=corrupted_tokens,
        task_family="cyclic_permutation",
        seed=seed,
        tokenizer_id=tokenizer_id,
        source_hash=_hash_text(text),
        corruption_recipe=f"cyclic_{cycle_length}_pos{'_'.join(map(str, selected))}",
        metadata={
            "original_tokens": tokens,
            "original_ids": token_ids,
            "corrupted_ids": corrupted_ids,
            "cycle_positions": selected,
            "cycle_length": cycle_length,
        },
    )


def generate_shuffle_task(
    text: str,
    tokens: List[str],
    token_ids: List[int],
    metas: List[TokenMeta],
    source_dataset: str = "corpus_base",
    seed: int = 42,
    tokenizer_id: str = "",
) -> Optional[TaskExample]:
    """Generate a full-shuffle corruption of non-special tokens."""
    rng = random.Random(seed)
    positions = _non_special_positions(metas)
    if len(positions) < 3:
        return None

    corrupted_ids = list(token_ids)
    corrupted_tokens = list(tokens)

    pairs = [(corrupted_ids[p], corrupted_tokens[p]) for p in positions]
    rng.shuffle(pairs)
    for i, p in enumerate(positions):
        corrupted_ids[p] = pairs[i][0]
        corrupted_tokens[p] = pairs[i][1]

    return TaskExample(
        example_id=f"shuffle_{_hash_text(text)}_{seed}",
        source_dataset=source_dataset,
        text=text,
        tokenized_text=corrupted_tokens,
        task_family="full_shuffle",
        seed=seed,
        tokenizer_id=tokenizer_id,
        source_hash=_hash_text(text),
        corruption_recipe="full_shuffle",
        metadata={
            "original_tokens": tokens,
            "original_ids": token_ids,
            "corrupted_ids": corrupted_ids,
            "shuffled_positions": positions,
        },
    )


def generate_swap_control_conditions(
    text: str,
    tokens: List[str],
    token_ids: List[int],
    metas: List[TokenMeta],
    swap_a: int,
    swap_b: int,
    source_dataset: str = "corpus_base",
    seed: int = 42,
    tokenizer_id: str = "",
) -> Dict[str, TaskExample]:
    """Generate 4 control conditions for swap-recovery independence (Family D).

    All random tokens are sampled sentence-locally (from non-special positions
    in the same sentence), preserving the old work's methodology.

    Conditions:
      correct_duplicate:  pos_a = orig[a], pos_b = orig[a]
      correct_random:     pos_a = orig[a], pos_b = random sentence-local token
      swapped_random:     pos_a = orig[b] (swapped), pos_b = random sentence-local
      random_random:      pos_a = random, pos_b = random
    """
    rng = random.Random(seed)
    ns_positions = _non_special_positions(metas)
    pool_ids = [token_ids[p] for p in ns_positions if p not in (swap_a, swap_b)]
    if len(pool_ids) < 2:
        return {}

    def _make(condition: str, id_a: int, id_b: int) -> TaskExample:
        c_ids = list(token_ids)
        c_ids[swap_a] = id_a
        c_ids[swap_b] = id_b
        c_tok = list(tokens)
        c_tok[swap_a] = tokens[token_ids.index(id_a)] if id_a in token_ids else f"[id:{id_a}]"
        c_tok[swap_b] = tokens[token_ids.index(id_b)] if id_b in token_ids else f"[id:{id_b}]"
        return TaskExample(
            example_id=f"swapctrl_{condition}_{_hash_text(text)}_{seed}",
            source_dataset=source_dataset,
            text=text,
            tokenized_text=c_tok,
            task_family="swap_independence",
            seed=seed,
            tokenizer_id=tokenizer_id,
            source_hash=_hash_text(text),
            corruption_recipe=f"{condition}_pos{swap_a}_{swap_b}",
            metadata={
                "original_ids": token_ids,
                "corrupted_ids": c_ids,
                "condition": condition,
                "swap_positions": [swap_a, swap_b],
            },
        )

    rand_a = rng.choice(pool_ids)
    rand_b = rng.choice(pool_ids)

    return {
        "correct_duplicate": _make("correct_duplicate", token_ids[swap_a], token_ids[swap_a]),
        "correct_random": _make("correct_random", token_ids[swap_a], rng.choice(pool_ids)),
        "swapped_random": _make("swapped_random", token_ids[swap_b], rng.choice(pool_ids)),
        "random_random": _make("random_random", rand_a, rand_b),
    }


def generate_clean_passthrough(
    text: str,
    tokens: List[str],
    token_ids: List[int],
    metas: List[TokenMeta],
    source_dataset: str = "corpus_base",
    seed: int = 0,
    tokenizer_id: str = "",
) -> TaskExample:
    """Generate a clean passthrough task (no corruption, baseline)."""
    return TaskExample(
        example_id=f"clean_{_hash_text(text)}_{seed}",
        source_dataset=source_dataset,
        text=text,
        tokenized_text=tokens,
        task_family="clean_passthrough",
        seed=seed,
        tokenizer_id=tokenizer_id,
        source_hash=_hash_text(text),
        corruption_recipe="none",
        metadata={
            "original_tokens": tokens,
            "original_ids": token_ids,
            "token_types": [m.family.name for m in metas],
        },
    )
