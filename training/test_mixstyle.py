from __future__ import annotations
import torch
from mixstyle import MixStyle  # type: ignore[import-not-found]


def test_mixstyle_identity_when_disabled() -> None:
    """MixStyle(p=0) must be identity — for inference."""
    mod = MixStyle(p=0.0)
    mod.train()
    x = torch.randn(4, 64, 768)
    out = mod(x)
    assert torch.allclose(out, x)


def test_mixstyle_identity_in_eval_mode() -> None:
    """MixStyle never fires during eval."""
    mod = MixStyle(p=1.0)
    mod.eval()
    x = torch.randn(4, 64, 768)
    out = mod(x)
    assert torch.allclose(out, x)


def test_mixstyle_changes_stats_when_active() -> None:
    """When training and p=1.0, MixStyle mixes channel mean/std between batch items."""
    mod = MixStyle(p=1.0, alpha=0.3)
    mod.train()
    torch.manual_seed(0)
    x = torch.randn(8, 64, 768) * 2.0 + 5.0  # nontrivial mean/std
    out = mod(x)
    # Output and input cannot be byte-identical
    assert not torch.allclose(out, x)
    # But shape must match
    assert out.shape == x.shape


def test_mixstyle_preserves_dtype_and_device() -> None:
    mod = MixStyle(p=1.0)
    mod.train()
    x = torch.randn(4, 64, 768).float()
    out = mod(x)
    assert out.dtype == x.dtype
    assert out.device == x.device


def _run_all() -> None:
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {fn_name}')


if __name__ == '__main__':
    _run_all()
