"""Intervention engine for causal experiments.

Supports: patch, replay, freeze, replace, zero, swap,
          repeat-layer, and low-rank correction.

Each operation can be scoped to:
  - one position
  - selected token family
  - selected layer range
  - all non-special tokens
  - control positions only (CLS/SEP/BOS/EOS)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


class InterventionType(Enum):
    PATCH = auto()
    REPLAY = auto()
    FREEZE = auto()
    REPLACE = auto()
    ZERO = auto()
    SWAP = auto()
    REPEAT_LAYER = auto()
    LOW_RANK_CORRECT = auto()


class InterventionScope(Enum):
    SINGLE_POSITION = auto()
    TOKEN_FAMILY = auto()
    LAYER_RANGE = auto()
    ALL_NON_SPECIAL = auto()
    CONTROL_ONLY = auto()
    CUSTOM = auto()


@dataclass
class InterventionSpec:
    """Declarative specification of a single intervention."""
    op: InterventionType
    scope: InterventionScope
    layer_indices: List[int]
    positions: Optional[Set[int]] = None
    source_states: Optional[torch.Tensor] = None
    swap_pairs: Optional[List[Tuple[int, int]]] = None
    rank: int = 8
    correction_matrix: Optional[torch.Tensor] = None
    repeat_count: int = 1
    label: str = ""


def _apply_to_positions(
    hidden: torch.Tensor,
    positions: Set[int],
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply *fn* to hidden states at specified positions only."""
    out = hidden.clone()
    pos_list = sorted(positions)
    for p in pos_list:
        if p < out.shape[1]:
            out[:, p, :] = fn(out[:, p, :])
    return out


def make_freeze_hook(
    positions: Set[int],
    frozen_states: Dict[int, torch.Tensor],
    layer_idx: int,
) -> Callable:
    """Hook that overwrites positions with previously frozen values."""
    def hook_fn(module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        if layer_idx in frozen_states:
            for p in positions:
                if p < h.shape[1]:
                    h[:, p, :] = frozen_states[layer_idx][:, p, :].to(h.device)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook_fn


def make_zero_hook(positions: Set[int]) -> Callable:
    """Hook that zeros hidden states at specified positions."""
    def hook_fn(module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        for p in positions:
            if p < h.shape[1]:
                h[:, p, :] = 0.0
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook_fn


def make_swap_hook(pairs: List[Tuple[int, int]]) -> Callable:
    """Hook that swaps hidden states between position pairs."""
    def hook_fn(module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            h = output[0].clone()
        else:
            h = output.clone()
        for a, b in pairs:
            if a < h.shape[1] and b < h.shape[1]:
                tmp = h[:, a, :].clone()
                h[:, a, :] = h[:, b, :]
                h[:, b, :] = tmp
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook_fn


def make_replace_hook(positions: Set[int], source_states: torch.Tensor) -> Callable:
    """Hook that replaces positions with values from source_states."""
    def hook_fn(module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        for p in positions:
            if p < h.shape[1] and p < source_states.shape[1]:
                h[:, p, :] = source_states[:, p, :].to(h.device)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook_fn


def make_patch_hook(positions: Set[int], patch_states: torch.Tensor) -> Callable:
    """Hook that patches positions with external hidden states (e.g. from a donor)."""
    def hook_fn(module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        for p in positions:
            if p < h.shape[1] and p < patch_states.shape[1]:
                h[:, p, :] = patch_states[:, p, :].to(h.device)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook_fn


def make_low_rank_correction_hook(
    positions: Set[int],
    correction_matrix: torch.Tensor,
) -> Callable:
    """Hook that applies a low-rank additive correction at specified positions."""
    def hook_fn(module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        for p in positions:
            if p < h.shape[1]:
                h[:, p, :] = h[:, p, :] + (h[:, p, :] @ correction_matrix.to(h.device))
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook_fn


class InterventionEngine:
    """Applies intervention specs to a model adapter via hooks."""

    def __init__(self, adapter: Any):
        self.adapter = adapter
        self._active_hooks: List[Any] = []

    def apply(self, spec: InterventionSpec) -> None:
        """Register hooks for a single intervention spec."""
        for layer_idx in spec.layer_indices:
            hook_fn = self._make_hook(spec, layer_idx)
            hh = self.adapter.register_hook(layer_idx, hook_fn, hook_type="forward")
            self._active_hooks.append(hh)

    def apply_many(self, specs: List[InterventionSpec]) -> None:
        for s in specs:
            self.apply(s)

    def clear(self) -> None:
        self.adapter.remove_all_hooks()
        self._active_hooks.clear()

    def _make_hook(self, spec: InterventionSpec, layer_idx: int) -> Callable:
        positions = spec.positions or set()

        if spec.op == InterventionType.ZERO:
            return make_zero_hook(positions)
        elif spec.op == InterventionType.SWAP:
            return make_swap_hook(spec.swap_pairs or [])
        elif spec.op == InterventionType.REPLACE:
            assert spec.source_states is not None
            return make_replace_hook(positions, spec.source_states)
        elif spec.op == InterventionType.PATCH:
            assert spec.source_states is not None
            return make_patch_hook(positions, spec.source_states)
        elif spec.op == InterventionType.FREEZE:
            assert spec.source_states is not None
            frozen = {layer_idx: spec.source_states}
            return make_freeze_hook(positions, frozen, layer_idx)
        elif spec.op == InterventionType.LOW_RANK_CORRECT:
            assert spec.correction_matrix is not None
            return make_low_rank_correction_hook(positions, spec.correction_matrix)
        elif spec.op == InterventionType.REPLAY:
            assert spec.source_states is not None
            return make_patch_hook(positions, spec.source_states)
        elif spec.op == InterventionType.REPEAT_LAYER:
            return _identity_hook
        else:
            raise ValueError(f"Unsupported intervention: {spec.op}")


def _identity_hook(module: nn.Module, input: Any, output: Any) -> Any:
    """No-op hook used as placeholder for REPEAT_LAYER (handled externally)."""
    return output
