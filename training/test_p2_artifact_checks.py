"""Unit tests for the non-trivial pooling helpers in p2_artifact_checks.py.

These verify the robust-pool readouts behave as intended on tiny synthetic tensors
(no cached features needed), so a silent off-axis or off-by-one bug in the readout
can't masquerade as an artifact verdict.

  HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    training/.venv/bin/python -m pytest training/test_p2_artifact_checks.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_artifact_checks import pool_patches, _l2norm_tokens  # noqa: E402


def _toy():
    # N=2, T=4 tokens, D=3 channels. Channel 0 has one big outlier in sample 0.
    p = np.array([
        [[10.0, 1.0, 0.0],
         [0.1, 2.0, 0.0],
         [0.2, 3.0, 1.0],
         [0.3, 4.0, 2.0]],
        [[1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0]],
    ], dtype="float64")
    return p


def test_max_pool_is_channelwise_over_tokens():
    p = _toy()
    out = pool_patches(p, "max")
    assert out.shape == (2, 3)
    # sample 0: per-channel max over the 4 tokens
    np.testing.assert_allclose(out[0], [10.0, 4.0, 2.0])
    np.testing.assert_allclose(out[1], [1.0, 1.0, 1.0])


def test_mean_pool():
    p = _toy()
    out = pool_patches(p, "mean")
    np.testing.assert_allclose(out[0], p[0].mean(axis=0))
    np.testing.assert_allclose(out[1], [1.0, 1.0, 1.0])


def test_robust_pools_suppress_the_outlier_vs_raw_max():
    # The whole point of check 3: robust pools must NOT track the lone 10.0 outlier
    # in sample-0 channel-0 the way raw max does.
    p = _toy()
    raw = pool_patches(p, "max")[0, 0]            # == 10.0
    p95 = pool_patches(p, "p95")[0, 0]
    top8 = pool_patches(p, "top8mean")[0, 0]      # T<8 so mean of all 4
    wmax = pool_patches(p, "wmax")[0, 0]
    lse = pool_patches(p, "lse")[0, 0]
    assert raw == 10.0
    # every robust pool should sit strictly below the raw max for an outlier-dominated channel
    for v in (p95, top8, wmax, lse):
        assert v < raw, (v, raw)
    # top8mean with T=4 is just the mean over all four tokens
    np.testing.assert_allclose(top8, p[0, :, 0].mean())
    # winsorized max clips at the 98th pct then maxes -> below the raw 10.0 outlier
    assert wmax < raw


def test_lse_low_temp_approaches_max_for_flat_channel():
    # For a channel with no outlier (sample 1, all ones) LSE should be very close to 1.0.
    p = _toy()
    lse = pool_patches(p, "lse")[1]
    np.testing.assert_allclose(lse, [1.0, 1.0, 1.0], atol=1e-6)


def test_lse_is_between_mean_and_max():
    # log-sum-exp is bounded: mean <= lse <= max for any temperature. With our low tau it
    # should sit much closer to max than to mean for an outlier-dominated channel.
    p = _toy()
    ch0 = p[0, :, 0]  # [10, 0.1, 0.2, 0.3] -> max 10, mean ~2.65
    lse = pool_patches(p, "lse")[0, 0]
    assert ch0.mean() <= lse <= ch0.max()
    assert lse > ch0.mean()  # low tau pushes it toward max, not mean


def test_top8mean_equals_topk_mean_when_T_large():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(3, 32, 5))
    out = pool_patches(p, "top8mean")
    # brute-force reference: mean of the 8 largest per (sample, channel)
    ref = np.sort(p, axis=1)[:, -8:, :].mean(axis=1)
    np.testing.assert_allclose(out, ref, rtol=1e-10)


def test_l2norm_tokens_unit_norm():
    p = _toy()
    n = _l2norm_tokens(p)
    norms = np.linalg.norm(n, axis=2)
    # every token (except a true zero-vector, none here) becomes unit norm
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-9)


def test_l2norm_handles_zero_token():
    p = np.zeros((1, 2, 3))
    p[0, 1] = [3.0, 0.0, 4.0]  # norm 5
    n = _l2norm_tokens(p)
    # zero token stays zero (no div-by-zero), nonzero token becomes unit
    np.testing.assert_allclose(n[0, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(n[0, 1]), 1.0)


def test_p95_below_max_with_outlier():
    p = _toy()
    assert pool_patches(p, "p95")[0, 0] < pool_patches(p, "max")[0, 0]


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
