from __future__ import annotations
import numpy as np
import torch

from tta import tta_passes, K_PASSES, AUG_NAMES  # type: ignore[import-not-found]


def test_tta_produces_K_passes_per_image() -> None:
    """tta_passes returns K image variants for a given single CHW input."""
    img = torch.rand(3, 224, 224)
    variants = list(tta_passes(img))
    assert len(variants) == K_PASSES
    for v in variants:
        assert v.shape == img.shape


def test_tta_first_pass_is_identity() -> None:
    """First pass MUST be the unmodified image so single-pass eval is recoverable."""
    img = torch.rand(3, 100, 100)
    variants = list(tta_passes(img))
    assert torch.allclose(variants[0], img), 'first variant must be identity (no augmentation)'


def test_tta_includes_hflip_not_rotation() -> None:
    """CXR-safe: H-flip is OK, rotation is NOT (left-right anatomy distinction matters)."""
    assert 'hflip' in AUG_NAMES
    assert 'rotate' not in AUG_NAMES
    assert 'rotation' not in AUG_NAMES


def test_tta_brightness_change_is_bounded() -> None:
    """Brightness shift must be small (±10% per Wave 1.4 CXR-safe constraint)."""
    img = torch.full((3, 50, 50), 0.5)
    variants = list(tta_passes(img))
    for v in variants:
        # max +/- 0.10 change from baseline; tolerate 0.0001 numerical
        assert v.min() >= 0.5 - 0.10 - 1e-3
        assert v.max() <= 0.5 + 0.10 + 1e-3


def _run_all() -> None:
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {fn_name}')


if __name__ == '__main__':
    _run_all()
