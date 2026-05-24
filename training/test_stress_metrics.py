"""CPU-only unit tests for training/stress_metrics.py.

The load-bearing test: `stratified_fpr` must reproduce the KNOWN scar false-positive rate
(SeqFPR = 117/139 = 0.8417, EXPERIMENT_LOG §C "T2 SEQUELAE") when wired to the TBX11K healed-scar
probe with the deployed T2 head + its frozen threshold/temperature. If this drifts, either the
metric path or the deployed artifacts changed — STOP and investigate.

The remaining tests pin the pluggable-subgroup contract, the Brier/ECE math, and the input
validation, so a later NIH per-finding wiring can't silently regress the table.

    training/.venv/bin/python training/test_stress_metrics.py
(or)  training/.venv/bin/python -m pytest training/test_stress_metrics.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_metrics import (  # noqa: E402
    KNOWN_SCAR_FPR,
    brier,
    build_nih_subgroups,
    stratified_fpr,
    _load_deployed_t2,
    _seq_arrs,
)

TOL = 0.01  # the reproduction must land within 1 percentage point of the logged 0.8417


def test_brier_pure_negatives_is_mean_p_squared():
    """For an all-negative subgroup, Brier reduces to mean(p**2)."""
    p = np.array([0.0, 0.5, 1.0], dtype="float64")
    y = np.zeros(3, dtype="int64")
    assert abs(brier(y, p) - float(np.mean(p ** 2))) < 1e-12, brier(y, p)


def test_brier_perfect_calibration_is_zero():
    """Perfect predictions -> Brier 0 (sanity on the label-aware branch)."""
    y = np.array([0, 1, 0, 1], dtype="int64")
    p = y.astype("float64")
    assert abs(brier(y, p)) < 1e-12, brier(y, p)


def test_stratified_fpr_reproduces_known_scar_seqfpr():
    """LOAD-BEARING: stratified_fpr on the TBX11K healed-scar probe with the deployed T2 head +
    frozen threshold/temperature reproduces the known SeqFPR 0.8417 (117/139) within tolerance."""
    model, T, thr = _load_deployed_t2()
    arrs = _seq_arrs()
    n = arrs["cls"].shape[0]
    assert n == 139, f"sequelae probe should have 139 images, got {n}"
    rows = stratified_fpr(model, arrs, {"tbx11k_scar": np.arange(n)}, thr, T)
    assert len(rows) == 1
    scar = rows[0]
    assert scar["name"] == "tbx11k_scar"
    assert scar["n"] == 139, scar["n"]
    # the headline reproduction
    assert abs(float(scar["fpr"]) - KNOWN_SCAR_FPR) <= TOL, (
        f"FPR {scar['fpr']:.4f} drifted from known SeqFPR {KNOWN_SCAR_FPR:.4f} (> {TOL})"
    )
    assert scar["n_fp"] == 117, f"expected 117 false positives, got {scar['n_fp']}"
    # the CI must bracket the point estimate and be a valid 2-tuple in [0,1]
    lo, hi = scar["fpr_ci"]
    assert 0.0 <= lo <= float(scar["fpr"]) <= hi <= 1.0, scar["fpr_ci"]
    # on a confidently-wrong scar set Brier is high (mean p**2 ~ 0.7) and the >=.95 tail is large
    assert scar["brier"] > 0.5, scar["brier"]
    assert scar["frac_high"] > 0.5, scar["frac_high"]


def test_empty_subgroup_returns_nan_row_not_crash():
    """An empty index array yields a graceful NaN row (e.g. an NIH finding with 0 examples), not a
    crash — so a sparse per-finding NIH dict can't kill the whole table."""
    model, T, thr = _load_deployed_t2()
    arrs = _seq_arrs()
    rows = stratified_fpr(model, arrs, {"empty": np.array([], dtype="int64")}, thr, T)
    assert rows[0]["n"] == 0
    assert np.isnan(rows[0]["fpr"])  # type: ignore[arg-type]


def test_out_of_range_index_raises():
    """An index past the feature-set end is a wiring bug -> ValueError, never a silent wrong number."""
    model, T, thr = _load_deployed_t2()
    arrs = _seq_arrs()
    n = arrs["cls"].shape[0]
    raised = False
    try:
        stratified_fpr(model, arrs, {"bad": np.array([n])}, thr, T)
    except ValueError:
        raised = True
    assert raised, "expected ValueError for out-of-range subgroup index"


def test_build_nih_subgroups_contract():
    """The pluggable NIH builder: per-finding index arrays + a no-finding bucket; None -> {}."""
    assert build_nih_subgroups(None) == {}
    fm = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype="int64")  # 4 imgs, 2 findings
    sg = build_nih_subgroups(fm, ("fibrosis", "nodule"))
    assert set(sg.keys()) == {"nih_fibrosis", "nih_nodule", "nih_no_finding"}
    assert sg["nih_fibrosis"].tolist() == [0, 3]
    assert sg["nih_nodule"].tolist() == [1, 3]
    assert sg["nih_no_finding"].tolist() == [2]


def _run_all() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed} tests passed")


if __name__ == "__main__":
    _run_all()
