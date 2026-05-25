from __future__ import annotations
import torch

from heads.soft_attn_pool import SoftAttnPool  # type: ignore[import-not-found]


def test_soft_attn_pool_forward_shape() -> None:
    """SoftAttnPool(d_in=768) takes (B, T, 768) patches and emits (B, 768)."""
    mod = SoftAttnPool(d_in=768, d_hidden=128)
    x = torch.randn(4, 64, 768)
    out, attn = mod(x)
    assert out.shape == (4, 768), out.shape
    assert attn.shape == (4, 64), attn.shape


def test_soft_attn_pool_attention_sums_to_one() -> None:
    """Attention weights must form a probability distribution over patch tokens."""
    torch.manual_seed(0)
    mod = SoftAttnPool(d_in=768, d_hidden=128)
    x = torch.randn(2, 64, 768)
    _, attn = mod(x)
    sums = attn.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), sums


def test_soft_attn_pool_no_hard_zone_priors() -> None:
    """SoftAttnPool has no fixed-zone partition — only a learnable attention layer."""
    mod = SoftAttnPool(d_in=768, d_hidden=128)
    # All parameters must be learnable; no buffer of zone indices like ZonalSoftOR has
    n_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    n_buffers = sum(b.numel() for b in mod.buffers())
    assert n_params > 0
    assert n_buffers == 0, f'SoftAttnPool must have no zone-index buffers, got {n_buffers}'


def test_soft_attn_pool_gradient_flows() -> None:
    """Backward pass through SoftAttnPool produces non-zero gradients on params."""
    mod = SoftAttnPool(d_in=768, d_hidden=128)
    x = torch.randn(2, 64, 768)
    out, _ = mod(x)
    out.sum().backward()
    for name, p in mod.named_parameters():
        assert p.grad is not None, f'no grad on {name}'
        assert p.grad.abs().sum().item() > 0, f'zero grad on {name}'


def _run_all() -> None:
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {fn_name}')


if __name__ == '__main__':
    _run_all()
