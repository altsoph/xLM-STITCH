"""Adapter for encoder-only masked-language models.

Covers: BERT, ALBERT, RoBERTa, ModernBERT, XLM-RoBERTa.

QUARANTINED (do not use for MLM tasks):
  - microsoft/deberta-v3-base: TWO issues. (1) MLM head checkpoint key mismatch
    (lm_predictions.lm_head.* vs cls.predictions.*). (2) More fundamentally,
    HuggingFace's disentangled attention implementation diverges from Microsoft's
    original code — encoder weights load perfectly (198/198 match) but the forward
    pass loses token identity by layer 4 (cos to input drops from 0.77 at L0 to
    0.15 at L4, nearest tokens become random CJK characters). This is an upstream
    HF bug in the attention computation. Tested 2026-04-10.
  - microsoft/deberta-base: same issues (v1 checkpoint).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from .base import BaseAdapter, ForwardOutput, ModelFamily


_LAYER_ATTR_CANDIDATES = [
    ("encoder", "layer"),          # BERT, RoBERTa, XLM-R
    ("albert", "albert_layer_groups"),  # ALBERT (weight-shared)
    ("deberta", "encoder", "layer"),    # DeBERTa-v3
    ("model", "encoder", "layers"),     # ModernBERT
]

QUARANTINED_MODELS = frozenset({
    "microsoft/deberta-v3-base",
    "microsoft/deberta-base",
})


def _resolve_layers(model: nn.Module, config: Any) -> List[nn.Module]:
    """Walk common attribute paths to find the transformer layer list."""
    model_type = getattr(config, "model_type", "")

    if model_type == "albert":
        groups = model.albert.encoder.albert_layer_groups
        inner = groups[0].albert_layers[0]
        n = config.num_hidden_layers
        return [inner] * n

    for path in [
        ["encoder", "layer"],
        ["deberta", "encoder", "layer"],
        ["roberta", "encoder", "layer"],
        ["model", "encoder", "layers"],
        ["model", "layers"],             # ModernBERT
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue

    base = getattr(model, "base_model", model)
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        return list(base.encoder.layer)
    if hasattr(base, "layers"):
        return list(base.layers)

    raise RuntimeError(f"Cannot locate transformer layers for model_type={model_type}")


def _resolve_embeddings(model: nn.Module, config: Any) -> nn.Module:
    model_type = getattr(config, "model_type", "")
    for attr in ["embeddings", "albert.embeddings", "roberta.embeddings",
                 "deberta.embeddings", "model.embeddings"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    base = getattr(model, "base_model", model)
    if hasattr(base, "embeddings"):
        return base.embeddings
    raise RuntimeError(f"Cannot locate embeddings for model_type={model_type}")


class EncoderMLMAdapter(BaseAdapter):
    """Adapter for masked-language encoder models."""

    def __init__(self, model_name: str, device: str = "cpu", precision: str = "fp32", output_attentions: bool = False):
        super().__init__(model_name, device, precision, output_attentions)
        self.family = ModelFamily.ENCODER_MLM

    def load(self) -> None:
        if self.model_name in QUARANTINED_MODELS:
            raise RuntimeError(
                f"Model {self.model_name} is quarantined: MLM head checkpoint "
                f"is incompatible with HuggingFace. All predictions are garbage. "
                f"See module docstring for details."
            )
        local_files_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        self.config = AutoConfig.from_pretrained(self.model_name, local_files_only=local_files_only)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision, torch.float32)
        extra_kwargs: Dict[str, Any] = {}
        if self.output_attentions:
            extra_kwargs["attn_implementation"] = "eager"
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name,
            config=self.config,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            **extra_kwargs,
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_files_only)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> ForwardOutput:
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=self.output_attentions,
                **kwargs,
            )
        return ForwardOutput(
            logits=out.logits,
            hidden_states=out.hidden_states,
            attentions=getattr(out, "attentions", None),
        )

    def get_layer_modules(self) -> List[nn.Module]:
        return _resolve_layers(self.model, self.config)

    def get_embedding_module(self) -> nn.Module:
        return _resolve_embeddings(self.model, self.config)

    def get_head_module(self) -> nn.Module:
        if hasattr(self.model, "cls"):
            return self.model.cls
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        if hasattr(self.model, "predictions"):
            return self.model.predictions
        # ModernBERT: head + decoder
        if hasattr(self.model, "head") and hasattr(self.model, "decoder"):
            return self.model.decoder
        raise RuntimeError("Cannot locate LM head")
