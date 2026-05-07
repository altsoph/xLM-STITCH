"""Core metrics pipeline.

Every experiment emits a machine-readable metrics dict.

Core metrics:
  - token retention rate
  - exact restoration rate
  - masked-token accuracy
  - change rate by position
  - probe accuracy
  - cross-token transfer accuracy
  - steering success rate
  - target-candidate promotion rate
  - KL divergence between decoded distributions
  - degradation slope under repeated depth
  - intervention effect size by token family and layer
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class MetricsBundle:
    """Container for metrics from a single experiment run."""
    run_id: str
    phase: str
    model_name: str
    task_family: str
    dataset_split: str
    seed: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    per_example: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def add_per_example(self, example_id: str, metrics: Dict[str, Any]) -> None:
        self.per_example.append({"example_id": example_id, **metrics})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "phase": self.phase,
            "model_name": self.model_name,
            "task_family": self.task_family,
            "dataset_split": self.dataset_split,
            "seed": self.seed,
            "summary_metrics": self.metrics,
            "per_example_metrics": self.per_example,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def token_retention_rate(
    original_ids: List[int],
    output_ids: List[int],
    ignore_positions: Optional[set] = None,
) -> float:
    """Fraction of tokens that remain unchanged."""
    matches = 0
    total = 0
    for i, (o, p) in enumerate(zip(original_ids, output_ids)):
        if ignore_positions and i in ignore_positions:
            continue
        total += 1
        if o == p:
            matches += 1
    return matches / max(total, 1)


def exact_restoration_rate(
    original_ids: List[List[int]],
    output_ids: List[List[int]],
) -> float:
    """Fraction of examples where the full sequence is exactly restored."""
    exact = sum(1 for o, p in zip(original_ids, output_ids) if o == p)
    return exact / max(len(original_ids), 1)


def masked_token_accuracy(
    predicted_ids: List[int],
    target_ids: List[int],
    mask_positions: List[int],
) -> float:
    """Accuracy at masked positions."""
    if not mask_positions:
        return 0.0
    correct = sum(1 for p in mask_positions if predicted_ids[p] == target_ids[p])
    return correct / len(mask_positions)


def change_rate_by_position(
    original_ids: List[int],
    output_ids: List[int],
) -> List[bool]:
    """Per-position boolean indicating whether the token changed."""
    return [o != p for o, p in zip(original_ids, output_ids)]


def kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(P || Q) from logits."""
    p = torch.nn.functional.softmax(p_logits, dim=-1)
    log_p = torch.nn.functional.log_softmax(p_logits, dim=-1)
    log_q = torch.nn.functional.log_softmax(q_logits, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


def degradation_slope(
    metrics_by_depth: Dict[int, float],
) -> float:
    """Compute linear slope of metric degradation over repeated depth."""
    if len(metrics_by_depth) < 2:
        return 0.0
    depths = sorted(metrics_by_depth.keys())
    x = np.array(depths, dtype=float)
    y = np.array([metrics_by_depth[d] for d in depths], dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def weighted_in_sequence_retention(
    per_layer_per_position_matches: Dict[int, List[bool]],
    num_layers: int,
    seq_len: int,
) -> float:
    """Weighted retention that emphasizes later positions and deeper layers.

    Parameters
    ----------
    per_layer_per_position_matches : dict
        ``{layer_idx: [bool per eval position]}``.  Eval positions are
        ordered left-to-right (causal shift already applied by caller).
    num_layers : int
        Total number of layers (including embedding layer 0).
    seq_len : int
        Total sequence length (used for position weighting).

    Weighting scheme
    ----------------
    * Position weight: ``pos / (seq_len - 1)`` — position 0 gets weight 0,
      last position gets weight 1, linear ramp between.
    * Layer weight: ``layer / (num_layers - 1)`` — layer 0 gets weight 0,
      last layer gets weight 1.
    * Final weight for each (layer, position) cell is the product
      ``pos_weight * layer_weight``.  The metric is the weighted-mean of
      match indicators.

    Returns 0.0 when there are no valid cells.
    """
    if num_layers < 2 or seq_len < 2:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for layer_idx, matches in per_layer_per_position_matches.items():
        layer_w = layer_idx / (num_layers - 1)
        if layer_w == 0.0:
            continue
        for pos_idx, m in enumerate(matches):
            # pos_idx is the index within eval_positions;
            # we use it directly as position offset (0-based)
            pos_w = (pos_idx + 1) / seq_len  # +1 so first eval pos > 0
            w = layer_w * pos_w
            total_weight += w
            if m:
                weighted_sum += w

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def intervention_effect_size(
    baseline_metric: float,
    intervention_metric: float,
) -> float:
    """Absolute effect size of an intervention relative to baseline."""
    return intervention_metric - baseline_metric


def steering_success_rate(
    target_token_ids: List[int],
    predictions_before: List[int],
    predictions_after: List[int],
    positions: List[int],
) -> Dict[str, float]:
    """Compute promotion and steering success rates."""
    any_promotion = 0
    target_promotion = 0
    total = len(positions)
    for pos in positions:
        before = predictions_before[pos]
        after = predictions_after[pos]
        target = target_token_ids[pos] if pos < len(target_token_ids) else -1
        if after != before:
            any_promotion += 1
        if after == target:
            target_promotion += 1
    return {
        "any_change_rate": any_promotion / max(total, 1),
        "target_promotion_rate": target_promotion / max(total, 1),
    }
