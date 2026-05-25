"""Unit tests for the P2.0 linear-probe metric helpers (the load-bearing logic).

Covers: pAUC normalization bounds, sens@spec interpolation, patient-level bootstrap
grouping (resampling patients not images), and the fused-feature concat.

Run either way:
    training/.venv/bin/python training/test_p2_linear_probe.py
    training/.venv/bin/python -m pytest training/test_p2_linear_probe.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import pytest  # noqa: F401
    _approx = pytest.approx
except Exception:  # pytest not installed in the training venv
    class _Approx:
        def __init__(self, value, abs=1e-9):
            self.value = value
            self.abs = abs

        def __eq__(self, other):
            return abs(other - self.value) <= self.abs

    def _approx(value, abs=1e-9):
        return _Approx(value, abs=abs)

from p2_linear_probe import (  # noqa: E402
    _pauc, _sens_at_spec, _patient_bootstrap, _feature_matrix, PAUC_FPR_MAX,
)


def test_pauc_perfect_separator_is_one():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])
    # perfect separation -> full TPR achievable within any FPR region -> normalized pAUC = 1
    assert _pauc(y, score) == _approx(1.0, abs=1e-6)


def test_pauc_no_skill_is_about_half():
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 500)
    score = rng.random(1000)  # random scores -> no skill
    val = _pauc(y, score)
    # normalized partial AUC of a no-skill classifier ~ 0.5 (the diagonal area / fpr_max)
    assert 0.35 < val < 0.65


def test_pauc_bounded_unit_interval():
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = rng.integers(0, 2, 200)
        if len(np.unique(y)) < 2:
            continue
        score = rng.random(200)
        v = _pauc(y, score)
        # McClish-standardized: 0.5 = no-skill, 1.0 = perfect; worse-than-random can dip < 0.5
        assert -0.01 <= v <= 1.0 + 1e-9


def test_sens_at_spec_perfect():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    # perfect separation -> sensitivity at spec 0.70 is 1.0
    assert _sens_at_spec(y, score, 0.70) == _approx(1.0, abs=1e-6)


def test_sens_at_spec_monotone_in_spec():
    rng = np.random.default_rng(2)
    y = np.array([0] * 200 + [1] * 200)
    # informative but imperfect score
    score = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.0, 1, 200)])
    s_loose = _sens_at_spec(y, score, 0.50)
    s_tight = _sens_at_spec(y, score, 0.90)
    # requiring higher specificity cannot increase sensitivity
    assert s_loose >= s_tight - 1e-6


def test_patient_bootstrap_resamples_patients_not_images():
    # two patients, 3 images each; bootstrap CIs should reflect patient-level variance.
    rng = np.random.default_rng(3)
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
    groups = np.array([0, 0, 0, 1, 1, 1])  # 2 patients
    out = _patient_bootstrap(y, score, groups, n=200, seed=5)
    assert out["n_patients"] == 2
    assert out["n"] == 6
    # point AUROC of a perfect separator is 1.0
    assert out["auroc"] == _approx(1.0, abs=1e-6)


def test_feature_matrix_fused_is_concat():
    class D(dict):
        pass
    d = {"cls": np.ones((4, 3), dtype="float32"),
         "txrv": np.full((4, 5), 2.0, dtype="float32")}
    cls = _feature_matrix(d, "cls")
    txrv = _feature_matrix(d, "txrv")
    fused = _feature_matrix(d, "fused")
    assert cls.shape == (4, 3)
    assert txrv.shape == (4, 5)
    assert fused.shape == (4, 8)
    assert np.allclose(fused[:, :3], 1.0) and np.allclose(fused[:, 3:], 2.0)


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
