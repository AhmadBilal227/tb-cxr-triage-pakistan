"""MixStyle feature-space augmentation for domain generalization.

Reference: Zhou et al., ICLR 2021, "Domain Generalization with MixStyle."
Adapted from the official PyTorch implementation for feature-space tokens
(rather than CNN spatial features). Per Wave 1.4 ranking #2 (TB-targeted
Hwang 2025), MixStyle on the feature side is orthogonal to architectural
changes and to LoRA-style adapter training.

How it works:
  - Each forward in training mode, with probability p, mixes channel-wise
    mean/std between random pairs of batch items, using a Beta(alpha, alpha)
    interpolation weight. This introduces feature-style variability that
    simulates cross-source shifts.
  - Identity in eval mode.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """Feature-side MixStyle augmentation.

    Applied to (B, T, d) patch-token features. Mixes channel-wise stats
    along the (B, T) axes for each channel d.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        assert 0.0 <= p <= 1.0
        assert alpha > 0
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self._beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) or (B, d). Returns same shape with mixed stats in training."""
        if not self.training or self.p == 0.0:
            return x
        if torch.rand(1).item() > self.p:
            return x

        if x.dim() == 2:
            x_3d = x.unsqueeze(1)  # (B, 1, d)
            squeeze = True
        else:
            x_3d = x
            squeeze = False

        B, T, d = x_3d.shape
        # channel-wise mean/std over T
        mu = x_3d.mean(dim=1, keepdim=True)            # (B, 1, d)
        var = x_3d.var(dim=1, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_norm = (x_3d - mu) / sig

        # random permutation of batch
        perm = torch.randperm(B, device=x.device)
        mu_perm = mu[perm]
        sig_perm = sig[perm]

        # interpolation weight per item
        lam = self._beta.sample((B,)).to(x.device).view(B, 1, 1)
        mu_mix = lam * mu + (1 - lam) * mu_perm
        sig_mix = lam * sig + (1 - lam) * sig_perm

        out = x_norm * sig_mix + mu_mix
        if squeeze:
            out = out.squeeze(1)
        return out
