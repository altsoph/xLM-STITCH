"""Unified model adapter API.

Supports three model families:
  1. Encoder MLMs (BERT, ALBERT, RoBERTa, DeBERTa, ModernBERT, XLM-R)
  2. Encoder-decoder / seq2seq (BART, T5)
  3. Decoder-only causal LMs (GPT-2, Pythia, SmolLM)

Every adapter exposes the same interface regardless of family.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


class ModelFamily(Enum):
    ENCODER_MLM = auto()
    ENCODER_DECODER = auto()
    DECODER_ONLY = auto()


@dataclass
class ForwardOutput:
    """Standardised output from any adapter forward pass."""

    logits: torch.Tensor
    hidden_states: Tuple[torch.Tensor, ...]
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    token_metadata: Optional[List[Dict[str, Any]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookHandle:
    """Bookkeeping for a registered hook."""

    name: str
    layer_idx: int
    handle: Any
    hook_type: str  # "forward" | "forward_pre"


class BaseAdapter(abc.ABC):
    """Abstract adapter that every model-specific class must implement."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        precision: str = "fp32",
        output_attentions: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.output_attentions = output_attentions

        self._hooks: List[HookHandle] = []
        self._intervention_fns: Dict[str, Callable] = {}

        self.model: nn.Module = None  # type: ignore[assignment]
        self.tokenizer = None
        self.config = None
        self.family: ModelFamily = ModelFamily.ENCODER_MLM

    # ── lifecycle ──────────────────────────────────────────────────

    @abc.abstractmethod
    def load(self) -> None:
        """Load model, tokenizer, and config onto *self.device*."""

    def unload(self) -> None:
        self.remove_all_hooks()
        del self.model
        self.model = None  # type: ignore[assignment]
        torch.cuda.empty_cache()

    # ── forward ────────────────────────────────────────────────────

    @abc.abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> ForwardOutput:
        """Run a full forward pass and return standardised output."""

    def encode_text(self, text: str | List[str], **tokenizer_kwargs: Any) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call .load() first.")
        defaults = dict(return_tensors="pt", padding=True, truncation=True)
        defaults.update(tokenizer_kwargs)
        batch = self.tokenizer(text, **defaults)
        return {k: v.to(self.device) for k, v in batch.items()}

    # ── hooks and interventions ────────────────────────────────────

    @abc.abstractmethod
    def get_layer_modules(self) -> List[nn.Module]:
        """Return an ordered list of transformer layer modules."""

    @abc.abstractmethod
    def get_embedding_module(self) -> nn.Module:
        """Return the embedding / input-projection module."""

    @abc.abstractmethod
    def get_head_module(self) -> nn.Module:
        """Return the output / LM-head module."""

    @property
    def num_layers(self) -> int:
        return len(self.get_layer_modules())

    def register_hook(
        self,
        layer_idx: int,
        fn: Callable[[nn.Module, Any, Any], Optional[Any]],
        hook_type: str = "forward",
    ) -> HookHandle:
        mod = self.get_layer_modules()[layer_idx]
        if hook_type == "forward":
            h = mod.register_forward_hook(fn)
        elif hook_type == "forward_pre":
            h = mod.register_forward_pre_hook(fn)
        else:
            raise ValueError(f"Unknown hook_type: {hook_type}")
        hh = HookHandle(name=f"layer_{layer_idx}_{hook_type}", layer_idx=layer_idx, handle=h, hook_type=hook_type)
        self._hooks.append(hh)
        return hh

    def remove_all_hooks(self) -> None:
        for hh in self._hooks:
            hh.handle.remove()
        self._hooks.clear()
        self._intervention_fns.clear()

    def set_intervention(self, name: str, fn: Callable) -> None:
        self._intervention_fns[name] = fn

    def clear_interventions(self) -> None:
        self.remove_all_hooks()
        self._intervention_fns.clear()

    # ── info helpers ───────────────────────────────────────────────

    def get_final_norm(self) -> Optional[nn.Module]:
        """Return the final norm applied before the LM head, if any.

        For decoder models, HuggingFace hidden_states[-1] includes this norm
        but intermediate layers do not. Override in subclass to provide it.
        """
        return None

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size  # type: ignore[union-attr]

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size  # type: ignore[union-attr]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, family={self.family.name}, device={self.device!r})"
