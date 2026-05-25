"""SoftAttnPool — non-zonal learned attention pooling.

Drop-in replacement for ZonalSoftOR. Designed to fix the M24 atypical-TB
failure mode by removing the hard zone-prior partition that caused mid-lung
consolidative TB (M24 cases #6, #7, #9, #10) to be attenuated when the
backbone clearly encoded patch-level evidence.

Architecture:
  - Input: (B, T, d_in) patch tokens, T = patch_grid^2 (typically 64 for 8x8)
  - Single learnable attention layer:
      a_t = softmax( w^T tanh( W x_t + b ) )    over t = 1..T
      out = sum_t a_t * x_t                     -> (B, d_in)
  - Returns (out, attn_weights) so downstream code can visualize attention
    similar to the M24 BoxEvidence heatmap (the attention map IS interpretable).

No fixed zones, no SoftOR, no LogSumExp pooling. The same fusion+box+distill
heads of the original TBHeadT2 consume the pooled (B, d_in) vector unchanged.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttnPool(nn.Module):
    """Learned non-zonal soft-attention pooling over patch tokens.

    Total params: d_in * d_hidden + d_hidden (W,b) + d_hidden (w) = O(d_in * d_hidden).
    With d_in=768 and d_hidden=128 that's ~98k params — same order as the existing
    ZonalSoftOR (which is bigger because of zone-conditional gating).
    """

    def __init__(self, d_in: int, d_hidden: int = 128) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(d_in, d_hidden)
        self.attn_weight = nn.Parameter(torch.empty(d_hidden))
        nn.init.normal_(self.attn_weight, mean=0.0, std=0.02)

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """patches: (B, T, d_in). Returns (pooled (B, d_in), attn (B, T))."""
        # (B, T, d_hidden) -> (B, T) via attn_weight dot product
        h = torch.tanh(self.attn_proj(patches))
        scores = (h * self.attn_weight).sum(dim=-1)  # (B, T)
        attn = F.softmax(scores, dim=-1)             # (B, T)
        pooled = (attn.unsqueeze(-1) * patches).sum(dim=1)  # (B, d_in)
        return pooled, attn
