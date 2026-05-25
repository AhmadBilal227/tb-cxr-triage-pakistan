"""Deterministic smoke tests for TriageEngine (Milestone 22).

Run pattern matches training/test_stress_metrics.py + test_onnx_parity.py: a plain
`_run_all()` driver with PASS/FAIL prints, no pytest dependency (the venv has it but
the existing pair stays portable, and the CI on a fresh machine should not need
pytest just for these).

LOAD-BEARING ASSERTIONS:
  1. The engine reproduces the calibrated `tb_prob` on the committed sample
     `public/samples/tb-sample-1.jpg` to within ±0.02 across runs.
     The expected value is the headline number from the M22 live run (which IS the
     EXPERIMENT_LOG row). If this drifts: STOP, investigate. Either the calibration
     constants drifted, the preprocessing diverged from extract_features.py, or the
     deployed head .pt was replaced.
  2. All audit fields are populated and non-empty.
  3. Warm latency total < 2000 ms on M4 (lets seg pre-warm fluctuate without flagging).

    training/.venv/bin/python training/test_triage_engine.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

# Force offline mode before we import triage_core (which imports transformers).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from triage_core import TriageEngine, _platt_sigmoid, _verdict_from  # noqa: E402

# Headline value measured live at M22 commit time (see CASE_STUDY + EXPERIMENT_LOG).
# tb-sample-1.jpg calibrated tb_prob is ~0.9999 under T=1.5915 — well above the
# 0.95-sensitivity threshold 0.6105, verdict = "tb".
EXPECTED_TB_PROB_TB_SAMPLE = 0.9999769
TB_PROB_TOL = 0.02  # generous to absorb fp32/MPS drift between runs

EXPECTED_VERDICT_TB_SAMPLE = "tb"

LATENCY_BUDGET_MS_TOTAL = 2000  # warm M4; cold first run may exceed seg budget — test only after warmup


def _read_bytes(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()


def test_platt_sigmoid_matches_torch_sigmoid_under_T1():
    """T=1 should be standard sigmoid (sanity on the calibration math)."""
    import math

    for z in (-3.0, -0.5, 0.0, 0.7, 2.4):
        expected = 1.0 / (1.0 + math.exp(-z))
        got = _platt_sigmoid(z, 1.0)
        assert abs(got - expected) < 1e-9, (z, got, expected)


def test_verdict_rule_matches_sequelae_escalation_constants():
    """Spot-check the verdict gate at every named threshold. THIS MUST MATCH
    src/lib/pipeline/sequelaeEscalation.ts — they implement the same rule."""
    # high tb_prob: never escalates regardless of s_inactive
    v, sn = _verdict_from(p_tb=0.95, s_inactive=0.95, borderline_high=0.6105)
    assert v == "tb" and sn is None, (v, sn)

    # exactly at threshold: tb
    v, _ = _verdict_from(p_tb=0.6105, s_inactive=0.0, borderline_high=0.6105)
    assert v == "tb"

    # below threshold, in borderline band, with scar pattern: ABSTAIN
    v, sn = _verdict_from(p_tb=0.5, s_inactive=0.9, borderline_high=0.6105)
    assert v == "abstain" and sn is not None, (v, sn)

    # below threshold, in borderline band, no scar: still no_tb
    v, sn = _verdict_from(p_tb=0.5, s_inactive=0.5, borderline_high=0.6105)
    assert v == "no_tb" and sn is None, (v, sn)

    # below BORDERLINE_LOW=0.35: no escalation even with high s_inactive
    v, sn = _verdict_from(p_tb=0.2, s_inactive=0.95, borderline_high=0.6105)
    assert v == "no_tb" and sn is None, (v, sn)


def test_engine_reproduces_calibrated_tb_prob_on_tb_sample():
    """LOAD-BEARING: the deployed head + canonical preprocessing must produce a
    calibrated `tb_prob` within ±0.02 of the M22 headline number on the committed
    sample. Drift outside this band is the M22 EXPERIMENT_LOG §B "doc claim
    exceeded by evidence" tripwire."""
    eng = TriageEngine()
    sample = REPO / "public" / "samples" / "tb-sample-1.jpg"
    image_bytes = _read_bytes(sample)

    # First run = warmup (seg lazy-loads). Don't latency-budget this one.
    res0 = eng.run(image_bytes)
    assert abs(res0.tb_prob - EXPECTED_TB_PROB_TB_SAMPLE) < TB_PROB_TOL, (
        f"tb_prob {res0.tb_prob:.6f} drifted from M22 baseline {EXPECTED_TB_PROB_TB_SAMPLE:.6f} "
        f"by more than {TB_PROB_TOL} — either calibration, preprocessing, or weights changed"
    )
    assert res0.verdict == EXPECTED_VERDICT_TB_SAMPLE, res0.verdict

    # Determinism: a second run on the same bytes should produce ~the same number.
    # (MPS fp32 noise can wobble in the 1e-5 range; we assert within tolerance, not bit-exact.)
    res1 = eng.run(image_bytes)
    assert abs(res1.tb_prob - res0.tb_prob) < 1e-3, (res0.tb_prob, res1.tb_prob)

    # Latency budget — second run is warm
    assert res1.latency_ms["total"] < LATENCY_BUDGET_MS_TOTAL, res1.latency_ms


def test_engine_runs_normal_sample_to_no_tb():
    """The committed normal sample should screen-negative under the validated threshold.
    This is the other half of the EXPERIMENT_LOG row — we measure both numbers honestly."""
    eng = TriageEngine()
    sample = REPO / "public" / "samples" / "normal-sample-1.jpg"
    res = eng.run(_read_bytes(sample))
    # the headline measured number is ~0.0034; any value < 0.35 (BORDERLINE_LOW) is
    # safely a no_tb verdict regardless of s_inactive
    assert res.tb_prob < 0.35, res.tb_prob
    assert res.verdict == "no_tb", res.verdict


def test_audit_fields_present_and_well_formed():
    """No empty / None audit field — the audit trail is part of the contract."""
    eng = TriageEngine()
    sample = REPO / "public" / "samples" / "tb-sample-1.jpg"
    res = eng.run(_read_bytes(sample))
    a = res.audit
    assert a.model_id == "tb_head_t2", a.model_id
    assert a.model_sha.startswith("sha256:") and len(a.model_sha) == len("sha256:") + 64, a.model_sha
    assert isinstance(a.git_sha, str) and len(a.git_sha) > 0, a.git_sha
    assert a.version == 1, a.version
    assert "T" in a.calibration and a.calibration["T"] > 0, a.calibration
    assert "thr_at_95sens" in a.calibration, a.calibration
    assert "T_sequelae" in a.calibration, a.calibration
    # timestamp is ISO 8601 UTC
    assert a.timestamp.endswith("Z") and "T" in a.timestamp, a.timestamp


def test_latency_dict_has_all_stage_keys():
    """The per-stage latency keys are part of the wire schema."""
    eng = TriageEngine()
    sample = REPO / "public" / "samples" / "tb-sample-1.jpg"
    res = eng.run(_read_bytes(sample))
    for k in ("harmonize", "seg", "rad_dino", "txrv", "heads", "total"):
        assert k in res.latency_ms, (k, res.latency_ms)
        assert res.latency_ms[k] >= 0, (k, res.latency_ms[k])


def test_engine_supports_tta_mode() -> None:
    """When use_tta=True the verdict is computed by averaging K_PASSES probs."""
    from triage_core import TriageEngine
    from tta import K_PASSES

    eng = TriageEngine(use_tta=True)
    sample_path = Path(__file__).parent.parent / 'public' / 'samples' / 'tb-sample-1.jpg'
    if not sample_path.exists():
        print('SKIP test_engine_supports_tta_mode — sample missing')
        return
    img_bytes = sample_path.read_bytes()
    result = eng.run(img_bytes)
    # tta_passes attribute should be populated to K_PASSES probabilities
    assert hasattr(result, 'tta_passes')
    assert result.tta_passes is not None
    assert len(result.tta_passes) == K_PASSES
    # baseline tb_prob should equal the TTA-averaged tb_prob (it IS the average)
    assert abs(result.tb_prob - sum(result.tta_passes) / K_PASSES) < 1e-6


def test_engine_uses_locked_calibration_when_available() -> None:
    """If data/p0_locked_calibration.json exists, the engine uses its T and thr."""
    import json
    from pathlib import Path
    from triage_core import TriageEngine

    locked_path = Path(__file__).parent.parent / 'data' / 'p0_locked_calibration.json'
    if not locked_path.exists():
        print('SKIP test_engine_uses_locked_calibration_when_available — locked JSON missing')
        return
    with locked_path.open() as f:
        locked = json.load(f)
    eng = TriageEngine(use_locked_protocol=True)
    assert abs(eng.T - locked['T']) < 1e-9
    assert abs(eng.thr - locked['thr_at_95sens']) < 1e-9


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed} tests passed")


if __name__ == "__main__":
    _run_all()
