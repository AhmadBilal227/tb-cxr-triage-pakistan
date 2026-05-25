"""CPU-only unit tests for training/fairness_audit_m27a.py.

These tests pin the M27a audit's contract — load-bearing properties that, if they
drift, mean the audit is silently mis-stating subgroup fairness:

  1. The rule constants in training/fairness_audit_m27a.py MIRROR the canonical
     thresholds in src/lib/pipeline/asymmetricEvidence.ts. If the production
     constants change, this test detects the drift before the audit lies about
     fairness.

  2. `_abstain_mask` correctly implements the AND-gate (every condition must
     hold; any single condition failing means the rule does NOT fire).

  3. `_rule_inputs` returns finite, in-[0,1] tb_prob / box_max / pathology_max
     values for a small index slice — proves the forward pass and the
     sigmoid/extraction math don't drift.

  4. `audit_per_finding_nih` reproduces a sane baseline (No_Finding subgroup
     ABSTAIN rate is the divisor for all multiplier numbers) and returns one
     row per finding column.

  5. `gating_verdict` correctly returns FAIL when a NIH subgroup with >5x
     baseline and zero catch is constructed, and PASS when everything sits
     under 3x.

  6. The full `run_audit()` pipeline runs end-to-end and writes the JSON the
     case study + drift log expect.

    training/.venv/bin/python training/test_subgroup_fairness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairness_audit_m27a import (  # noqa: E402
    BOX_EVIDENCE_HIGH_THRESHOLD,
    FINDING_COLS,
    GATE_FAIL_MIN_MULTIPLIER,
    GATE_PASS_MAX_MULTIPLIER,
    PATHOLOGY_HIGH_THRESHOLD,
    TB_PROB_LOW_THRESHOLD,
    TB_RELEVANT_TXRV_IDX,
    TB_RELEVANT_TXRV_LABELS,
    TXRV_LABELS_18,
    _abstain_mask,
    _load_calibration,
    _load_deployed_head,
    _load_nih_arrs,
    _load_train_arrs,
    _rule_inputs,
    audit_per_finding_nih,
    audit_per_source,
    gating_verdict,
    run_audit,
)


def test_rule_constants_mirror_typescript_source_of_truth():
    """The Python constants in fairness_audit_m27a.py MIRROR the canonical
    thresholds in src/lib/pipeline/asymmetricEvidence.ts. If the production
    rule moves, the audit MUST follow — otherwise we're auditing a stale rule.
    """
    ts_path = Path(__file__).resolve().parents[1] / "src" / "lib" / "pipeline" / "asymmetricEvidence.ts"
    src = ts_path.read_text()
    assert f"TB_PROB_LOW_THRESHOLD = {TB_PROB_LOW_THRESHOLD}" in src, (
        f"TB_PROB_LOW_THRESHOLD={TB_PROB_LOW_THRESHOLD} not found in {ts_path} — rule drifted"
    )
    assert f"BOX_EVIDENCE_HIGH_THRESHOLD = {BOX_EVIDENCE_HIGH_THRESHOLD}" in src, (
        f"BOX_EVIDENCE_HIGH_THRESHOLD={BOX_EVIDENCE_HIGH_THRESHOLD} not found in {ts_path}"
    )
    assert f"PATHOLOGY_HIGH_THRESHOLD = {PATHOLOGY_HIGH_THRESHOLD}" in src, (
        f"PATHOLOGY_HIGH_THRESHOLD={PATHOLOGY_HIGH_THRESHOLD} not found in {ts_path}"
    )
    # The 5 TB-relevant TXRV labels MUST match the production list.
    for label in TB_RELEVANT_TXRV_LABELS:
        assert f"'{label}'" in src, f"label {label!r} not declared in {ts_path}"


def test_txrv_label_indices_resolve_within_18_class_block():
    """TB_RELEVANT_TXRV_IDX MUST resolve to valid indices into the 18-class TXRV
    block AND the labels MUST round-trip to the canonical names."""
    assert len(TB_RELEVANT_TXRV_IDX) == len(TB_RELEVANT_TXRV_LABELS) == 5
    for idx, label in zip(TB_RELEVANT_TXRV_IDX, TB_RELEVANT_TXRV_LABELS):
        assert 0 <= idx < 18, f"idx {idx} out of [0,18) for {label!r}"
        assert TXRV_LABELS_18[idx] == label, f"index {idx} maps to {TXRV_LABELS_18[idx]!r}, not {label!r}"


def test_abstain_mask_and_gate_all_conditions_required():
    """_abstain_mask is an AND-gate: any single condition failing means NO fire.
    Build a tiny tabular case set and verify each row's expected behaviour."""
    thr_high = 0.6105
    # rows:  tb_prob              box_max                              path_max                              expected
    rows = [
        # All conditions met -> fire.
        (0.05,                  BOX_EVIDENCE_HIGH_THRESHOLD + 0.01,  PATHOLOGY_HIGH_THRESHOLD + 0.01,    True),
        # tb_prob too high -> no fire (above the very-low band).
        (TB_PROB_LOW_THRESHOLD, BOX_EVIDENCE_HIGH_THRESHOLD + 0.01,  PATHOLOGY_HIGH_THRESHOLD + 0.01,    False),
        # box_max just below threshold -> no fire.
        (0.05,                  BOX_EVIDENCE_HIGH_THRESHOLD - 0.01,  PATHOLOGY_HIGH_THRESHOLD + 0.01,    False),
        # path_max just below threshold -> no fire.
        (0.05,                  BOX_EVIDENCE_HIGH_THRESHOLD + 0.01,  PATHOLOGY_HIGH_THRESHOLD - 0.01,    False),
        # tb_prob is positive call (>= thr_high) -> no fire (base verdict TB).
        (thr_high + 0.05,       BOX_EVIDENCE_HIGH_THRESHOLD + 0.01,  PATHOLOGY_HIGH_THRESHOLD + 0.01,    False),
    ]
    tb = np.array([r[0] for r in rows], dtype="float64")
    bx = np.array([r[1] for r in rows], dtype="float64")
    pa = np.array([r[2] for r in rows], dtype="float64")
    expected = np.array([r[3] for r in rows], dtype="bool")
    got = _abstain_mask(tb, bx, pa, thr_high)
    assert (got == expected).all(), f"expected {expected.tolist()}, got {got.tolist()}"


def test_rule_inputs_returns_finite_probabilities_in_unit_interval():
    """Forward pass must produce tb_prob/box_max/path_max in [0,1] and finite —
    sanity that the sigmoid/extraction math hasn't drifted."""
    T, thr = _load_calibration()
    arrs = _load_train_arrs()
    model = _load_deployed_head(arrs)
    idx = np.arange(50)  # a small slice keeps the test fast
    sig = _rule_inputs(model, arrs, idx, T)
    for k in ("tb_prob", "box_max", "pathology_max"):
        v = sig[k]
        assert v.shape == (50,), f"{k} shape {v.shape} != (50,)"
        assert np.all(np.isfinite(v)), f"{k} has non-finite values"
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0, (
            f"{k} out of [0,1]: min={v.min()}, max={v.max()}"
        )


def test_audit_per_finding_nih_one_row_per_column_plus_sane_baseline():
    """The NIH audit must emit one row per FINDING_COLS and the No_Finding row's
    ABSTAIN rate IS the baseline used as the multiplier divisor."""
    T, thr = _load_calibration()
    arrs = _load_train_arrs()
    model = _load_deployed_head(arrs)
    nih_arrs, findings = _load_nih_arrs()
    rows, baseline = audit_per_finding_nih(model, nih_arrs, findings, T, thr)
    names = [r["name"] for r in rows]
    assert names == list(FINDING_COLS), f"rows out of order: {names}"
    assert 0.0 <= baseline <= 1.0, f"baseline {baseline} out of [0,1]"
    # The No_Finding row's abstain_rate IS the baseline (the multiplier divisor).
    nofinding = next(r for r in rows if r["name"] == "No_Finding")
    assert abs(float(nofinding["abstain_rate"]) - baseline) < 1e-12, (  # type: ignore[arg-type]
        f"baseline {baseline} != No_Finding.abstain_rate {nofinding['abstain_rate']}"
    )
    # No_Finding's multiplier IS 1.0 by definition.
    assert abs(float(nofinding["multiplier_vs_no_finding"]) - 1.0) < 1e-9, (  # type: ignore[arg-type]
        f"No_Finding multiplier != 1.0: {nofinding['multiplier_vs_no_finding']}"
    )


def test_audit_per_source_emits_one_row_per_training_source():
    """The per-source audit must emit one row per training source (4 sources:
    montgomery, qatar, shenzhen, tbx11k) and reasonable strict-sens/spec."""
    T, thr = _load_calibration()
    arrs = _load_train_arrs()
    model = _load_deployed_head(arrs)
    rows = audit_per_source(model, arrs, T, thr)
    src_names = {str(r["source"]) for r in rows}
    assert src_names == {"montgomery", "qatar", "shenzhen", "tbx11k"}, src_names
    for r in rows:
        n = int(r["n"])
        assert n > 0
        # Strict sens/spec must be in [0,1].
        for k in ("strict_sensitivity", "strict_specificity",
                  "effective_sensitivity_with_abstain", "abstain_rate",
                  "abstain_rate_tb_pos", "abstain_rate_tb_neg"):
            v = float(r[k])  # type: ignore[arg-type]
            assert 0.0 <= v <= 1.0, f"{r['source']}.{k} = {v} out of [0,1]"


def test_gating_verdict_pass_when_all_subgroups_within_threshold():
    """Synthetic PASS: every subgroup under the WATCH/FAIL gates AND positive-
    containing sources have catch/cost >= 1.0."""
    source_rows = [
        {"source": "a", "n": 100, "n_pos": 50, "n_neg": 50,
         "n_abstain": 3, "n_abstain_pos": 2, "n_abstain_neg": 1,
         "abstain_rate": 0.03, "abstain_rate_tb_pos": 0.04, "abstain_rate_tb_neg": 0.02,
         "strict_sensitivity": 0.95, "strict_specificity": 0.98,
         "effective_sensitivity_with_abstain": 0.97,
         "catch": 2, "cost": 1, "catch_cost_ratio": 2.0},
    ]
    nih_rows = [
        {"name": "No_Finding", "tag": "easy-neg", "n": 1000, "n_abstain": 20,
         "abstain_rate": 0.02, "multiplier_vs_no_finding": 1.0,
         "mean_tb_prob": 0.05, "mean_box_max": 0.5, "mean_pathology_max": 0.2},
        {"name": "Fibrosis", "tag": "TB-mimic", "n": 100, "n_abstain": 4,
         "abstain_rate": 0.04, "multiplier_vs_no_finding": 2.0,
         "mean_tb_prob": 0.1, "mean_box_max": 0.7, "mean_pathology_max": 0.4},
    ]
    v = gating_verdict(source_rows, nih_rows, 0.02)
    assert v["verdict"] == "PASS", v


def test_gating_verdict_fail_when_subgroup_exceeds_5x_with_zero_catch():
    """Synthetic FAIL: a NIH subgroup at 6x baseline trips the FAIL gate (NIH
    has no TB positives so catch is zero by construction — Clever-Hans for
    ABSTAIN)."""
    source_rows = [
        {"source": "a", "n": 100, "n_pos": 50, "n_neg": 50,
         "n_abstain": 1, "n_abstain_pos": 0, "n_abstain_neg": 1,
         "abstain_rate": 0.01, "abstain_rate_tb_pos": 0.0, "abstain_rate_tb_neg": 0.02,
         "strict_sensitivity": 0.95, "strict_specificity": 0.98,
         "effective_sensitivity_with_abstain": 0.95,
         "catch": 0, "cost": 1, "catch_cost_ratio": 0.0},
    ]
    nih_rows = [
        {"name": "No_Finding", "tag": "easy-neg", "n": 1000, "n_abstain": 20,
         "abstain_rate": 0.02, "multiplier_vs_no_finding": 1.0,
         "mean_tb_prob": 0.05, "mean_box_max": 0.5, "mean_pathology_max": 0.2},
        {"name": "Fibrosis", "tag": "TB-mimic", "n": 100, "n_abstain": 12,
         "abstain_rate": 0.12, "multiplier_vs_no_finding": 6.0,
         "mean_tb_prob": 0.1, "mean_box_max": 0.7, "mean_pathology_max": 0.4},
    ]
    v = gating_verdict(source_rows, nih_rows, 0.02)
    assert v["verdict"] == "FAIL", v
    assert any("Fibrosis" in e["subgroup"] for e in v["failing_subgroups"]), v["failing_subgroups"]  # type: ignore[union-attr]


def test_run_audit_writes_full_json_with_expected_keys():
    """End-to-end: run_audit() returns a JSON-serialisable dict with the keys
    the case study + drift log downstream consume."""
    result = run_audit()
    for k in ("audit", "calibration", "rule_thresholds", "scope",
              "dimension_1_per_source", "dimension_2_per_finding_nih",
              "gating_verdict"):
        assert k in result, f"missing key: {k}"
    assert result["gating_verdict"]["verdict"] in {"PASS", "WATCH", "FAIL"}
    # Calibration values must match the deployed JSON literals.
    T, thr = _load_calibration()
    assert abs(result["calibration"]["temperature"] - T) < 1e-6
    assert abs(result["calibration"]["threshold_95sens"] - thr) < 1e-6


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
