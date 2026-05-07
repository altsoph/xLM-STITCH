"""Tuned lens: learned per-layer affine transformation for intermediate decoding.

For decoder models, the raw logit lens gives very low retention (7-17%)
because intermediate hidden states are not aligned with the output embedding.
The tuned lens learns a per-layer affine probe (Wx + b) that maps each
layer's hidden state into the final-layer logit space, revealing that
decoders DO encode token-resolving information — it's just rotated.

**Important**: intermediate hidden states must be normalized with the
model's final layer norm before the learned probe.  Without this,
RMSNorm models (Qwen, gemma, SmolLM2) get ~0% tuned retention because
the probe cannot compensate for the enormous norm difference.  The
``apply_final_norm`` flag (default True) handles this automatically.

Reference: nostalgebraist (2020), Belrose et al. (2023) "Eliciting Latent
Predictions from Transformers with the Tuned Lens".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TunedLensOutput:
    """Output from a tuned lens decoding pass at one layer."""
    layer_idx: int
    top1_retention: float   # fraction of positions where tuned top-1 matches final top-1
    kl_to_final: float      # mean KL divergence from tuned distribution to final
    top_ids: Optional[torch.Tensor] = None  # (seq_len, k)


class TunedLensProbe(nn.Module):
    """Per-layer affine transformation: out = Wx + b."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as identity + zero bias (start close to raw lens)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TunedLens:
    """Tuned lens: train and apply per-layer affine probes.

    Parameters
    ----------
    adapter : BaseAdapter
        Model adapter.
    top_k : int
        Number of top tokens to return per position.
    apply_final_norm : bool
        If True (default), apply the model's final layer norm to
        intermediate hidden states before the learned probe.  This is
        critical for RMSNorm models (Qwen, gemma, SmolLM2) where
        intermediate norms differ by 100-1000x from the final layer.

    Usage::

        tuned = TunedLens(adapter)
        tuned.train(examples, epochs=50, lr=1e-3)
        results = tuned.evaluate(examples)
    """

    def __init__(self, adapter: Any, top_k: int = 5,
                 apply_final_norm: bool = True):
        self.adapter = adapter
        self.top_k = top_k
        self.apply_final_norm = apply_final_norm
        self.probes: Dict[int, TunedLensProbe] = {}
        self._head = None
        self._final_norm = None

    def _get_head(self) -> nn.Module:
        """Get the LM head for decoding."""
        if self._head is None:
            self._head = self.adapter.get_head_module()
        return self._head

    def _get_final_norm(self) -> Optional[nn.Module]:
        """Get the final layer norm, cached."""
        if self._final_norm is None and self.apply_final_norm:
            self._final_norm = self.adapter.get_final_norm()
        return self._final_norm

    def _norm_intermediate(self, hs: torch.Tensor, layer_idx: int,
                           total_layers: int) -> torch.Tensor:
        """Apply final norm to intermediate layers (not the last one)."""
        norm = self._get_final_norm()
        if norm is not None and layer_idx < total_layers - 1:
            with torch.no_grad():
                return norm(hs)
        return hs

    def train(
        self,
        texts: List[str],
        epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> Dict[int, float]:
        """Train per-layer affine probes to map intermediate hidden states
        to final-layer hidden states (MSE loss in hidden space).

        If ``apply_final_norm`` is True, the final norm is applied to
        intermediate layers before collecting training data, so the probe
        learns to map norm(h_L) → h_final rather than raw h_L → h_final.

        Returns per-layer final training loss.
        """
        all_hidden: Dict[int, List[torch.Tensor]] = {}
        all_final_hs: List[torch.Tensor] = []

        with torch.no_grad():
            for text in texts:
                enc = self.adapter.encode_text(text)
                out = self.adapter.forward(**enc)

                total_layers = len(out.hidden_states)

                # Target: final-layer hidden states (before the head)
                final_hs = out.hidden_states[-1].squeeze(0)  # (seq_len, hidden)
                all_final_hs.append(final_hs.detach())

                for layer_idx, hs in enumerate(out.hidden_states[:-1]):
                    # Apply final norm to intermediate layers
                    normed = self._norm_intermediate(hs, layer_idx, total_layers)
                    if layer_idx not in all_hidden:
                        all_hidden[layer_idx] = []
                    all_hidden[layer_idx].append(normed.squeeze(0).detach())

        if not all_hidden:
            return {}

        target_hs = torch.cat(all_final_hs, dim=0)  # (N, hidden)

        hidden_size = self.adapter.hidden_size
        losses_by_layer: Dict[int, float] = {}

        for layer_idx in sorted(all_hidden.keys()):
            hs = torch.cat(all_hidden[layer_idx], dim=0)  # (N, hidden)

            probe = TunedLensProbe(hidden_size).to(device)
            optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

            for epoch in range(epochs):
                optimizer.zero_grad()
                transformed = probe(hs)
                loss = F.mse_loss(transformed, target_hs.detach())
                loss.backward()
                optimizer.step()

            losses_by_layer[layer_idx] = round(loss.item(), 6)
            probe.eval()
            self.probes[layer_idx] = probe

        return losses_by_layer

    def decode_layer(
        self,
        hidden_state: torch.Tensor,
        layer_idx: int,
        final_logits: Optional[torch.Tensor] = None,
        total_layers: Optional[int] = None,
    ) -> TunedLensOutput:
        """Decode a layer's hidden state through the tuned probe + LM head."""
        # Apply norm to intermediate layers
        h = hidden_state
        if total_layers is not None:
            h = self._norm_intermediate(h, layer_idx, total_layers)

        if layer_idx not in self.probes:
            # No probe for this layer — fall back to identity (raw lens)
            transformed = h
        else:
            with torch.no_grad():
                transformed = self.probes[layer_idx](h)

        head = self._get_head()
        with torch.no_grad():
            logits = head(transformed)

        if logits.dim() == 3:
            logits = logits.squeeze(0)

        top_ids = logits.topk(self.top_k, dim=-1).indices

        # Retention: compare top-1 to final top-1
        retention = 0.0
        kl_val = 0.0
        if final_logits is not None:
            fl = final_logits.squeeze(0) if final_logits.dim() == 3 else final_logits
            final_top1 = fl.argmax(dim=-1)
            tuned_top1 = logits.argmax(dim=-1)
            retention = (tuned_top1 == final_top1).float().mean().item()

            # KL divergence
            p = F.softmax(logits, dim=-1)
            log_p = F.log_softmax(logits, dim=-1)
            log_q = F.log_softmax(fl, dim=-1)
            kl_val = (p * (log_p - log_q)).sum(dim=-1).mean().item()

        return TunedLensOutput(
            layer_idx=layer_idx,
            top1_retention=round(retention, 4),
            kl_to_final=round(kl_val, 4),
            top_ids=top_ids,
        )

    def evaluate(
        self,
        texts: List[str],
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate tuned lens on a set of texts.

        Returns per-layer metrics: tuned_retention, raw_retention, kl_to_final.

        For decoder models, also reports position-specific agreement with the
        final-layer top-1 prediction at:
          - `lastvis_retention`: generation position (last visible token)
          - `lastvis_m1_retention`: final in-sequence position (last_visible - 1)
        """
        from .logit_lens import LogitLens
        raw_lens = LogitLens(self.adapter, top_k=self.top_k,
                             mode="normed" if self.apply_final_norm else "raw")

        results_by_layer: Dict[int, Dict[str, List[float]]] = {}

        with torch.no_grad():
            for text in texts:
                enc = self.adapter.encode_text(text)
                out = self.adapter.forward(**enc)

                total_layers = len(out.hidden_states)
                head = self._get_head()
                final_logits = head(out.hidden_states[-1])

                for layer_idx, hs in enumerate(out.hidden_states):
                    if layer_idx not in results_by_layer:
                        results_by_layer[layer_idx] = {
                            "tuned_retention": [],
                            "raw_retention": [],
                            "kl_to_final": [],
                            "tuned_lastvis_retention": [],
                            "raw_lastvis_retention": [],
                            "tuned_lastvis_m1_retention": [],
                            "raw_lastvis_m1_retention": [],
                        }

                    # Tuned lens (norm applied internally)
                    tuned = self.decode_layer(
                        hs, layer_idx,
                        final_logits=final_logits,
                        total_layers=total_layers,
                    )
                    results_by_layer[layer_idx]["tuned_retention"].append(
                        tuned.top1_retention)
                    results_by_layer[layer_idx]["kl_to_final"].append(
                        tuned.kl_to_final)

                    # Raw/normed lens for comparison
                    lr = raw_lens.decode_layer(
                        hs, layer_idx,
                        final_logits=final_logits,
                        total_layers=total_layers,
                    )
                    fl = final_logits.squeeze(0) if final_logits.dim() == 3 else final_logits
                    raw_top1 = lr.top_ids[:, 0]
                    final_top1 = fl.argmax(dim=-1)
                    raw_ret = (raw_top1 == final_top1).float().mean().item()
                    results_by_layer[layer_idx]["raw_retention"].append(raw_ret)

                    # Decoder position-specific metrics: agreement with final
                    # next-token predictions at the generation position and the
                    # immediately preceding in-sequence position.
                    seq_len = final_top1.shape[0]
                    last_vis_pos = seq_len - 1
                    last_vis_m1_pos = seq_len - 2 if seq_len > 1 else None
                    tuned_top1 = tuned.top_ids[:, 0]

                    if 0 <= last_vis_pos < seq_len:
                        results_by_layer[layer_idx]["tuned_lastvis_retention"].append(
                            float((tuned_top1[last_vis_pos] == final_top1[last_vis_pos]).item())
                        )
                        results_by_layer[layer_idx]["raw_lastvis_retention"].append(
                            float((raw_top1[last_vis_pos] == final_top1[last_vis_pos]).item())
                        )

                    if last_vis_m1_pos is not None and 0 <= last_vis_m1_pos < seq_len:
                        results_by_layer[layer_idx]["tuned_lastvis_m1_retention"].append(
                            float((tuned_top1[last_vis_m1_pos] == final_top1[last_vis_m1_pos]).item())
                        )
                        results_by_layer[layer_idx]["raw_lastvis_m1_retention"].append(
                            float((raw_top1[last_vis_m1_pos] == final_top1[last_vis_m1_pos]).item())
                        )

        # Average
        avg = {}
        for layer_idx, metrics in results_by_layer.items():
            avg[layer_idx] = {
                "tuned_retention": round(float(np.mean(metrics["tuned_retention"])), 4),
                "raw_retention": round(float(np.mean(metrics["raw_retention"])), 4),
                "kl_to_final": round(float(np.mean(metrics["kl_to_final"])), 4),
            }
            if metrics["tuned_lastvis_retention"]:
                avg[layer_idx]["tuned_lastvis_retention"] = round(
                    float(np.mean(metrics["tuned_lastvis_retention"])), 4
                )
                avg[layer_idx]["raw_lastvis_retention"] = round(
                    float(np.mean(metrics["raw_lastvis_retention"])), 4
                )
            if metrics["tuned_lastvis_m1_retention"]:
                avg[layer_idx]["tuned_lastvis_m1_retention"] = round(
                    float(np.mean(metrics["tuned_lastvis_m1_retention"])), 4
                )
                avg[layer_idx]["raw_lastvis_m1_retention"] = round(
                    float(np.mean(metrics["raw_lastvis_m1_retention"])), 4
                )
        return avg
