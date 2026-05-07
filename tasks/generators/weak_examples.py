"""Weak-example mining for masked-token steering experiments (Family H).

Finds masked examples where the model's top prediction is uncertain,
enabling replay/forcing experiments.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class WeakExample:
    """A masked example where the model is not confident."""
    example_id: str
    text: str
    mask_position: int
    original_token_id: int
    original_token: str
    top_k_predictions: List[Tuple[str, float]]
    top1_prob: float
    entropy: float
    source_dataset: str = ""
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "text": self.text,
            "mask_position": self.mask_position,
            "original_token_id": self.original_token_id,
            "original_token": self.original_token,
            "top_k_predictions": self.top_k_predictions,
            "top1_prob": self.top1_prob,
            "entropy": self.entropy,
            "source_dataset": self.source_dataset,
            "seed": self.seed,
        }


def mine_weak_examples(
    adapter: Any,
    sentences: List[str],
    source_dataset: str = "corpus_base",
    top1_threshold: float = 0.5,
    min_entropy: float = 1.0,
    top_k: int = 10,
    max_examples: int = 1000,
    seed: int = 42,
) -> List[WeakExample]:
    """Mine weak masked examples from sentences.

    For each sentence, mask each non-special token and check if the
    model's top-1 prediction probability is below *top1_threshold*.
    """
    tokenizer = adapter.tokenizer
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer has no mask token; cannot mine weak examples for this model.")

    results: List[WeakExample] = []
    for sent in sentences:
        if len(results) >= max_examples:
            break

        enc = tokenizer(sent, return_tensors="pt", truncation=True)
        input_ids = enc["input_ids"].to(adapter.device)
        att_mask = enc.get("attention_mask", None)
        if att_mask is not None:
            att_mask = att_mask.to(adapter.device)

        orig_ids = input_ids.squeeze(0).tolist()
        special_ids = set(tokenizer.all_special_ids)

        for pos in range(len(orig_ids)):
            if orig_ids[pos] in special_ids:
                continue
            if len(results) >= max_examples:
                break

            masked_ids = input_ids.clone()
            masked_ids[0, pos] = mask_id

            out = adapter.forward(masked_ids, attention_mask=att_mask)
            logits = out.logits[0, pos]
            probs = F.softmax(logits, dim=-1)

            top1_prob = probs.max().item()
            entropy = -(probs * probs.clamp(min=1e-10).log()).sum().item()

            if top1_prob < top1_threshold and entropy > min_entropy:
                topk_probs, topk_ids = probs.topk(top_k)
                topk_list = [
                    (tokenizer.convert_ids_to_tokens(tid.item()), p.item())
                    for tid, p in zip(topk_ids, topk_probs)
                ]

                eid = hashlib.sha256(f"{sent}_{pos}_{seed}".encode()).hexdigest()[:16]
                results.append(WeakExample(
                    example_id=f"weak_{eid}",
                    text=sent,
                    mask_position=pos,
                    original_token_id=orig_ids[pos],
                    original_token=tokenizer.convert_ids_to_tokens(orig_ids[pos]),
                    top_k_predictions=topk_list,
                    top1_prob=top1_prob,
                    entropy=entropy,
                    source_dataset=source_dataset,
                    seed=seed,
                ))
    return results
