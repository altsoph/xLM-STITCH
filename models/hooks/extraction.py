"""Hidden-state extraction hooks.

Registers forward hooks on transformer layers to capture intermediate
hidden states, with optional per-position filtering and caching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


@dataclass
class ExtractionCache:
    """Stores captured hidden states per layer."""
    states: Dict[int, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def clear(self) -> None:
        self.states.clear()
        self.metadata.clear()

    def get_trajectory(self, position: int) -> torch.Tensor:
        """Return hidden-state trajectory for *position* across stored layers: (num_layers, hidden_dim)."""
        layers = sorted(self.states.keys())
        vecs = [self.states[l][:, position, :] for l in layers]
        return torch.stack(vecs, dim=1).squeeze(0)


def make_extraction_hook(
    cache: ExtractionCache,
    layer_idx: int,
    positions: Optional[Set[int]] = None,
    detach: bool = True,
) -> Callable:
    """Create a forward hook that captures the layer's output hidden state."""

    def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        if detach:
            h = h.detach().cpu()
        if positions is not None:
            pos_list = sorted(positions)
            h = h[:, pos_list, :]
        cache.states[layer_idx] = h
        cache.metadata[layer_idx] = {"positions": positions}

    return hook_fn


def attach_extraction_hooks(
    adapter: Any,
    cache: ExtractionCache,
    layer_indices: Optional[List[int]] = None,
    positions: Optional[Set[int]] = None,
) -> List[Any]:
    """Attach extraction hooks to specified layers and return handles."""
    if layer_indices is None:
        layer_indices = list(range(adapter.num_layers))
    handles = []
    for idx in layer_indices:
        fn = make_extraction_hook(cache, idx, positions=positions)
        hh = adapter.register_hook(idx, fn, hook_type="forward")
        handles.append(hh)
    return handles
