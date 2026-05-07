"""Logit lens: project intermediate hidden states through the output head.

Supports multiple projection modes for decoder models:

  - ``raw``:  direct projection, no preprocessing
  - ``normed``:  apply final layer norm to intermediate layers before projection
  - ``centered_normed``:  subtract position-mean, then apply final norm
  - ``cosine``:  cosine similarity between hidden states and embedding rows

HuggingFace ``hidden_states[-1]`` already includes the final norm; all
modes skip re-norming the last layer automatically.

Encoder models ignore the mode (their LM head chain already includes
normalization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Valid lens mode strings
LENS_MODES = ("raw", "normed", "centered_normed", "cosine")


@dataclass
class LensOutput:
    """Output from a single lens decoding pass."""
    layer_idx: int
    top_tokens: List[List[Tuple[str, float]]]  # per-position top-k (token_str, prob)
    top_ids: torch.Tensor           # (seq_len, k)
    kl_to_final: torch.Tensor       # (seq_len,)
    top1_matches_original: List[bool]


class LogitLens:
    """Project hidden states through the unembedding head.

    Parameters
    ----------
    adapter : BaseAdapter
        Model adapter with ``get_head_module()`` and ``get_final_norm()``.
    top_k : int
        Number of top tokens to return per position.
    mode : str
        Projection mode.  One of ``LENS_MODES``.
    """

    def __init__(self, adapter: Any, top_k: int = 10,
                 mode: str = "normed",
                 # legacy compat
                 apply_final_norm: Optional[bool] = None):
        self.adapter = adapter
        self.top_k = top_k

        # legacy: translate old boolean flag
        if apply_final_norm is not None:
            self.mode = "normed" if apply_final_norm else "raw"
        else:
            self.mode = mode

        if self.mode not in LENS_MODES:
            raise ValueError(f"Unknown lens mode {self.mode!r}, "
                             f"expected one of {LENS_MODES}")

        self._transform = None
        self._decoder = None
        self._final_norm = None
        self._head_weight = None  # for cosine mode

    # ------------------------------------------------------------------
    # Projection discovery
    # ------------------------------------------------------------------

    def _get_projection(self) -> None:
        """Discover the LM head pipeline and optional final norm."""
        model = self.adapter.model

        self._transform = None
        self._decoder = None

        chains = [
            (["head"], ["decoder"]),
            (["cls", "predictions", "transform"],
             ["cls", "predictions", "decoder"]),
            (["lm_head"], None),
        ]

        for transform_path, decoder_path in chains:
            try:
                obj = model
                for attr in transform_path:
                    obj = getattr(obj, attr)
                transform = obj
            except AttributeError:
                continue

            decoder = None
            if decoder_path is not None:
                try:
                    obj = model
                    for attr in decoder_path:
                        obj = getattr(obj, attr)
                    decoder = obj
                except AttributeError:
                    continue

            self._transform = transform
            self._decoder = decoder
            break
        else:
            self._decoder = self.adapter.get_head_module()

        # Resolve final norm
        if self.mode in ("normed", "centered_normed"):
            self._final_norm = self.adapter.get_final_norm()

        # Cache head weight matrix for cosine mode
        if self.mode == "cosine":
            head = self.adapter.get_head_module()
            self._head_weight = head.weight.detach()  # (vocab, hidden)

    # ------------------------------------------------------------------
    # Pre-processing helpers
    # ------------------------------------------------------------------

    def _maybe_norm(self, h: torch.Tensor, layer_idx: int,
                    total_layers: int) -> torch.Tensor:
        """Apply final norm to intermediate layers if the mode requires it."""
        if (self._final_norm is not None
                and layer_idx < total_layers - 1):
            with torch.no_grad():
                h = self._final_norm(h)
        return h

    def _preprocess(self, h: torch.Tensor, layer_idx: int,
                    total_layers: int) -> torch.Tensor:
        """Apply mode-specific preprocessing before the head projection."""
        is_last = (layer_idx == total_layers - 1)

        if self.mode == "raw":
            return h

        if self.mode == "normed":
            return self._maybe_norm(h, layer_idx, total_layers)

        if self.mode == "centered_normed":
            if not is_last:
                # Subtract position-mean (remove DC component), then norm
                if h.dim() == 3:
                    h = h - h.mean(dim=1, keepdim=True)
                else:
                    h = h - h.mean(dim=0, keepdim=True)
            return self._maybe_norm(h, layer_idx, total_layers)

        # cosine: handled separately in decode_layer
        return h

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_layer(
        self,
        hidden_state: torch.Tensor,
        layer_idx: int,
        original_ids: Optional[torch.Tensor] = None,
        final_logits: Optional[torch.Tensor] = None,
        tokenizer: Any = None,
        total_layers: Optional[int] = None,
    ) -> LensOutput:
        """Decode a single layer's hidden state."""
        if self._transform is None and self._decoder is None:
            self._get_projection()

        h = hidden_state
        _tl = total_layers or (layer_idx + 1)  # fallback

        with torch.no_grad():
            if self.mode == "cosine":
                logits = self._decode_cosine(h)
            else:
                h = self._preprocess(h, layer_idx, _tl)
                if self._transform is not None:
                    h = self._transform(h)
                if self._decoder is not None:
                    logits = self._decoder(h)
                elif hasattr(h, "logits"):
                    logits = h
                else:
                    logits = h

        if logits.dim() == 3:
            logits = logits.squeeze(0)

        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(self.top_k, dim=-1)

        top_tokens: List[List[Tuple[str, float]]] = []
        if tokenizer is not None:
            for pos in range(top_ids.shape[0]):
                pairs = []
                for k in range(self.top_k):
                    tid = top_ids[pos, k].item()
                    p = top_probs[pos, k].item()
                    pairs.append((tokenizer.convert_ids_to_tokens(tid), p))
                top_tokens.append(pairs)

        kl = torch.zeros(logits.shape[0])
        if final_logits is not None:
            fl = final_logits.squeeze(0) if final_logits.dim() == 3 else final_logits
            final_log_p = F.log_softmax(fl, dim=-1)
            layer_log_p = F.log_softmax(logits, dim=-1)
            layer_p = F.softmax(logits, dim=-1)
            kl = (layer_p * (layer_log_p - final_log_p)).sum(dim=-1)

        top1_matches: List[bool] = []
        if original_ids is not None:
            orig = original_ids.squeeze(0) if original_ids.dim() == 2 else original_ids
            for pos in range(top_ids.shape[0]):
                top1_matches.append(top_ids[pos, 0].item() == orig[pos].item())
        else:
            top1_matches = [False] * top_ids.shape[0]

        return LensOutput(
            layer_idx=layer_idx,
            top_tokens=top_tokens,
            top_ids=top_ids,
            kl_to_final=kl,
            top1_matches_original=top1_matches,
        )

    def _decode_cosine(self, h: torch.Tensor) -> torch.Tensor:
        """Cosine-similarity decoding: normalize both h and W rows."""
        if self._head_weight is None:
            head = self.adapter.get_head_module()
            self._head_weight = head.weight.detach()
        if h.dim() == 3:
            h = h.squeeze(0)
        h_n = F.normalize(h, dim=-1)
        W_n = F.normalize(self._head_weight, dim=-1)
        return h_n @ W_n.T  # (seq_len, vocab)

    # ------------------------------------------------------------------
    # Batch decode
    # ------------------------------------------------------------------

    def decode_all_layers(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        original_ids: Optional[torch.Tensor] = None,
        tokenizer: Any = None,
    ) -> List[LensOutput]:
        """Decode every layer's hidden state."""
        if self._transform is None and self._decoder is None:
            self._get_projection()

        total_layers = len(hidden_states)

        # Compute final-layer logits for KL reference
        with torch.no_grad():
            h = hidden_states[-1]
            h = self._preprocess(h, total_layers - 1, total_layers)
            if self.mode == "cosine":
                final_logits = self._decode_cosine(h)
            else:
                if self._transform is not None:
                    h = self._transform(h)
                if self._decoder is not None:
                    final_logits = self._decoder(h)
                else:
                    final_logits = h

        results = []
        for i, hs in enumerate(hidden_states):
            results.append(self.decode_layer(
                hs, i,
                original_ids=original_ids,
                final_logits=final_logits,
                tokenizer=tokenizer,
                total_layers=total_layers,
            ))
        return results
