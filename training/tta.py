"""Test-time augmentation, CXR-safe.

CXR-safe constraint (per Wave 1.4 literature survey, ranked #3 by ROI):
  - H-flip OK (handled by anatomic-symmetry assumption + sequelae head's left/right structure)
  - Brightness +/- 10% OK
  - Contrast +/- 10% OK
  - NO rotation (left-right diagnostic anatomy)
  - NO random crop (apices + costophrenic angles are diagnostic landmarks)
  - NO elastic deformation
  - NO color jitter (CXR is grayscale)

Inference cost: K_PASSES x baseline. With K=5 the wall-time becomes ~1.5x
of single-pass (the seg-crop preprocessing dominates, not the backbone).
"""
from __future__ import annotations
from typing import Iterator
import torch

K_PASSES = 5
AUG_NAMES = ('identity', 'hflip', 'brighten', 'darken', 'contrast_up')


def tta_passes(img: torch.Tensor) -> Iterator[torch.Tensor]:
    """Yield K_PASSES augmented variants of a single CHW image tensor.

    First yield is identity (so K=1 collapses to no-TTA cleanly).
    """
    assert img.dim() == 3, f'expected CHW, got {img.shape}'
    yield img  # 0: identity
    yield torch.flip(img, dims=(-1,))  # 1: H-flip
    yield (img + 0.10).clamp(0.0, 1.0)  # 2: brighten
    yield (img - 0.10).clamp(0.0, 1.0)  # 3: darken
    yield ((img - 0.5) * 1.10 + 0.5).clamp(0.0, 1.0)  # 4: contrast up


def tta_average_probs(probs_per_pass: list[float]) -> float:
    """Average a list of TTA probabilities into a single calibrated probability."""
    assert len(probs_per_pass) == K_PASSES, f'expected {K_PASSES} passes, got {len(probs_per_pass)}'
    return float(sum(probs_per_pass) / len(probs_per_pass))
