"""Model adapter factory.

Usage:
    adapter = create_adapter("bert-base-uncased", device="cuda")
    adapter.load()
    out = adapter.forward(input_ids)
"""

from __future__ import annotations

from transformers import AutoConfig

from .base import BaseAdapter, ModelFamily
from .decoder_lm import DecoderLMAdapter
from .encoder_decoder import EncoderDecoderAdapter
from .encoder_mlm import EncoderMLMAdapter

_ENCODER_MLM_TYPES = {
    "bert", "albert", "roberta", "xlm-roberta", "deberta", "deberta-v2",
    "modernbert", "electra", "distilbert",
}

_ENCODER_DECODER_TYPES = {"bart", "t5", "mt5", "mbart"}

_DECODER_ONLY_TYPES = {"gpt2", "gpt_neox", "llama", "mistral", "phi", "qwen2", "gemma", "gemma2", "gemma3_text"}


def detect_family(model_name: str) -> ModelFamily:
    """Infer model family from HF config."""
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    mt = getattr(cfg, "model_type", "").lower().replace("-", "_")

    if mt in _ENCODER_MLM_TYPES or mt.replace("_", "-") in _ENCODER_MLM_TYPES:
        return ModelFamily.ENCODER_MLM
    if mt in _ENCODER_DECODER_TYPES:
        return ModelFamily.ENCODER_DECODER
    if mt in _DECODER_ONLY_TYPES:
        return ModelFamily.DECODER_ONLY

    if getattr(cfg, "is_encoder_decoder", False):
        return ModelFamily.ENCODER_DECODER
    if hasattr(cfg, "num_labels") and not hasattr(cfg, "decoder_start_token_id"):
        return ModelFamily.ENCODER_MLM

    return ModelFamily.DECODER_ONLY


def create_adapter(
    model_name: str,
    device: str = "cpu",
    precision: str = "fp32",
    output_attentions: bool = False,
    family: ModelFamily | None = None,
) -> BaseAdapter:
    """Create and return an adapter for *model_name*."""
    if family is None:
        family = detect_family(model_name)

    cls_map = {
        ModelFamily.ENCODER_MLM: EncoderMLMAdapter,
        ModelFamily.ENCODER_DECODER: EncoderDecoderAdapter,
        ModelFamily.DECODER_ONLY: DecoderLMAdapter,
    }
    return cls_map[family](model_name, device=device, precision=precision, output_attentions=output_attentions)
