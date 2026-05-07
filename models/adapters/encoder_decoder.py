"""Adapter for encoder-decoder / seq2seq models.

Covers: BART, T5.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from .base import BaseAdapter, ForwardOutput, ModelFamily


def _resolve_enc_layers(model: nn.Module, config: Any) -> List[nn.Module]:
    model_type = getattr(config, "model_type", "")
    for path in [
        ["model", "encoder", "layers"],  # BART
        ["encoder", "block"],            # T5
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot locate encoder layers for {model_type}")


def _resolve_dec_layers(model: nn.Module, config: Any) -> List[nn.Module]:
    model_type = getattr(config, "model_type", "")
    for path in [
        ["model", "decoder", "layers"],  # BART
        ["decoder", "block"],            # T5
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot locate decoder layers for {model_type}")


class EncoderDecoderAdapter(BaseAdapter):
    """Adapter for encoder-decoder seq2seq models."""

    def __init__(self, model_name: str, device: str = "cpu", precision: str = "fp32", output_attentions: bool = False):
        super().__init__(model_name, device, precision, output_attentions)
        self.family = ModelFamily.ENCODER_DECODER

    def load(self) -> None:
        self.config = AutoConfig.from_pretrained(self.model_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision, torch.float32)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            config=self.config,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> ForwardOutput:
        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (input_ids.shape[0], 1),
                self.config.decoder_start_token_id or self.tokenizer.pad_token_id or 0,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                output_attentions=self.output_attentions,
                **kwargs,
            )
        enc_hidden = out.encoder_hidden_states if hasattr(out, "encoder_hidden_states") else ()
        dec_hidden = out.decoder_hidden_states if hasattr(out, "decoder_hidden_states") else ()
        return ForwardOutput(
            logits=out.logits,
            hidden_states=enc_hidden,
            attentions=getattr(out, "encoder_attentions", None),
            extra={"decoder_hidden_states": dec_hidden, "decoder_attentions": getattr(out, "decoder_attentions", None)},
        )

    def get_layer_modules(self) -> List[nn.Module]:
        return _resolve_enc_layers(self.model, self.config)

    def get_decoder_layer_modules(self) -> List[nn.Module]:
        return _resolve_dec_layers(self.model, self.config)

    def get_embedding_module(self) -> nn.Module:
        if hasattr(self.model.model, "encoder"):
            return self.model.model.encoder.embed_tokens
        if hasattr(self.model, "encoder"):
            emb = getattr(self.model.encoder, "embed_tokens", None)
            if emb:
                return emb
        if hasattr(self.model, "shared"):
            return self.model.shared
        raise RuntimeError("Cannot locate encoder embeddings")

    def get_head_module(self) -> nn.Module:
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        raise RuntimeError("Cannot locate LM head")
