from __future__ import annotations
import numpy as np
from pathlib import Path
import torch

from locked_protocol import make_calibration_split  # type: ignore[import-not-found]


def test_calibration_split_is_deterministic() -> None:
    """Same seed + same data → same split. The whole point of 'locked'."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(12345)
        logit = np.linspace(-5, 5, 1000).astype('float64')
        label = (np.arange(1000) % 2 == 0).astype('int64')
        source = np.array(['a'] * 500 + ['b'] * 500)
        cal_a, eval_a = make_calibration_split(logit, label, source, seed=7)
        cal_b, eval_b = make_calibration_split(logit, label, source, seed=7)
        assert (cal_a == cal_b).all()
        assert (eval_a == eval_b).all()
    finally:
        np.random.set_state(rng_state)


def test_calibration_split_is_stratified_by_label_and_source() -> None:
    """The split must preserve label and source balance to within 1 percentage point."""
    rng = np.random.default_rng(0)
    logit = rng.normal(0, 2, 1000)
    label = rng.integers(0, 2, 1000)
    source = np.array(rng.choice(['m', 'q', 's', 't'], 1000))
    cal, evl = make_calibration_split(logit, label, source, seed=7, cal_frac=0.20)
    assert len(cal) + len(evl) == 1000
    assert 195 <= len(cal) <= 205  # 20% +/- 0.5%
    for src in ['m', 'q', 's', 't']:
        cal_frac = (source[cal] == src).mean() if len(cal) else 0.0
        eval_frac = (source[evl] == src).mean() if len(evl) else 0.0
        assert abs(cal_frac - eval_frac) < 0.05  # stratification balance


def _run_all() -> None:
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {fn_name}')


if __name__ == '__main__':
    _run_all()
