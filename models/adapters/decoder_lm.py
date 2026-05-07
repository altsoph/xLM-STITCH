"""Adapter for decoder-only causal language models.

Covers: GPT-2, distilgpt2, Pythia, SmolLM2, Qwen2, TinyLlama, Llama-3, Gemma-3.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .base import BaseAdapter, ForwardOutput, ModelFamily


def _resolve_decoder_layers(model: nn.Module, config: Any) -> List[nn.Module]:
    model_type = getattr(config, "model_type", "")

    for path in [
        ["transformer", "h"],       # GPT-2
        ["gpt_neox", "layers"],     # Pythia
        ["model", "layers"],        # LLaMA-style / SmolLM
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue

    raise RuntimeError(f"Cannot locate decoder layers for model_type={model_type}")


def _resolve_decoder_embeddings(model: nn.Module, config: Any) -> nn.Module:
    for path in [
        ["transformer", "wte"],     # GPT-2
        ["gpt_neox", "embed_in"],   # Pythia
        ["model", "embed_tokens"],  # LLaMA-style / SmolLM
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Cannot locate decoder embeddings")


class DecoderLMAdapter(BaseAdapter):
    """Adapter for decoder-only causal LMs."""

    def __init__(self, model_name: str, device: str = "cpu", precision: str = "fp32", output_attentions: bool = False):
        super().__init__(model_name, device, precision, output_attentions)
        self.family = ModelFamily.DECODER_ONLY

    def load(self) -> None:
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.output_attentions:
            self.config.output_attentions = True
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        return _resolve_decoder_layers(self.model, self.config)

    def get_embedding_module(self) -> nn.Module:
        return _resolve_decoder_embeddings(self.model, self.config)

    def get_head_module(self) -> nn.Module:
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        if hasattr(self.model, "embed_out"):  # Pythia
            return self.model.embed_out
        raise RuntimeError("Cannot locate LM head")

    def get_final_norm(self) -> Optional[nn.Module]:
        """Return the final layer norm applied before the LM head.

        HuggingFace hidden_states[-1] already includes this norm, but
        intermediate hidden_states[0..N-2] do not. Callers (e.g. LogitLens)
        should apply this to intermediate layers before decoding.

        Returns None for encoder models where the head chain already
        includes its own normalization.
        """
        # GPT-2 / distilgpt2
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            return self.model.transformer.ln_f
        # Pythia (GPT-NeoX)
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "final_layer_norm"):
            return self.model.gpt_neox.final_layer_norm
        # LLaMA / Qwen / SmolLM2 / Gemma
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        return None
