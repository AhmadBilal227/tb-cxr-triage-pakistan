"""M27a — subgroup fairness audit on the M26 asymmetric-evidence ABSTAIN rule.

WHY THIS EXISTS. arXiv 2010.14134 ("Selective Classification Can Magnify Disparities
Across Groups") shows that learning-to-abstain rules can amplify per-subgroup
disparities — exactly the failure mode that hurts under-represented populations
(HIV+, elderly, immunocompromised, all of whom carry the same atypical-TB
morphology that the M26 rule was designed to catch). M26's rule fires on
borderline tb_prob AND high box-evidence AND high TXRV-pathology: by design it
fires more often on radiographs that look like TB mimics. The fairness question
is whether the elevation is PROPORTIONAL TO THE CATCH BENEFIT, or whether one
subgroup gets disproportionately routed to ABSTAIN with no rescue benefit.

THE AUDIT — three dimensions, all reported per subgroup:

  DIMENSION 1: per-source ABSTAIN rates (LODO data — SCOPE-LIMITED, see below)
    For each training source (montgomery, shenzhen, qatar, tbx11k): apply the M26
    rule to that source's predictions, report total / TB+ / TB- abstain rates,
    strict and effective sensitivity / specificity.

  DIMENSION 2: per-finding ABSTAIN rates on NIH (10k images, all NO_TB by NIH)
    For each ChestX-ray14 finding (15 subgroups including No_Finding): the
    fraction of images the rule abstains on.

  DIMENSION 3: catch-vs-cost ratio per subgroup with >=10 cases
    catch = TB-positives the rule rescued from confident-miss to ABSTAIN.
    cost  = TB-negatives the rule false-abstained.
    ratio = catch / max(cost, 1). For NIH (all negatives) catch=0 by construction;
    we report only the ABSTAIN rate for those subgroups.

SCOPE LIMITATION (honest read, recorded here AND in the case study):
  The LODO cache at `data/image_oof_logits.npz` carries only image_logit + label +
  source — NOT the box_evidence_max + top_pathology_score the M26 rule reasons on.
  Those signals were NOT cached during the original LODO run (which predated M24
  enrichment), so a strict held-out-fold fairness audit on the full 13k OOF
  predictions is NOT possible without re-running LODO end-to-end.

  This audit's DIMENSION 1 instead runs the DEPLOYED T2 head on `data/features.npz`
  (the 13k training-set features that DO carry the inputs the rule needs) and
  applies the rule. This is a TRAINING-SET PROBE, NOT a held-out LODO probe:
  positives will score high (the head saw them in training), so the per-source
  numbers characterize "the rule's behaviour on the deployment-distribution
  inputs", NOT "the rule's behaviour on out-of-fold OOF predictions". A future
  drift-log row can replace this with a true OOF audit once box_evidence and
  txrv_pathologies are added to the LODO cache.

  DIMENSION 2 has NO such limitation: NIH was never in training data; the per-
  finding ABSTAIN rates on 10k NIH images are a genuine off-distribution
  fairness probe.

GATING (PASS / WATCH / FAIL):
  PASS  : no subgroup has >3x baseline ABSTAIN rate AND catch/cost ratio >=1
          on all positive-containing subgroups.
  WATCH : 1-2 subgroups slightly elevated (2-3x) but defensible (e.g. Fibrosis is
          the rule's literal catch class by design).
  FAIL  : any subgroup with >5x ABSTAIN rate AND no catch (Clever-Hans for
          ABSTAIN — a subgroup disproportionately routed to ABSTAIN with no
          rescue benefit).

The intellectual-honesty discipline: this verdict is recorded, not tuned. If
Fibrosis fires at 8x baseline AND has positive catch — that is the rule WORKING
as designed (its catch class). If it fires at 8x baseline AND catch is zero —
that is a fairness FAIL.

    PYTORCH_ENABLE_MPS_FALLBACK=1 training/.venv/bin/python training/fairness_audit_m27a.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

import train_tb  # noqa: E402  (DEVICE pin must happen before predict_t2 uses it)

# CPU-ONLY: stress_metrics convention. The forward-only audit is trivial on CPU.
train_tb.DEVICE = "cpu"

from train_tb import (  # noqa: E402  (after DEVICE pin)
    TBHeadT2,
    _batches,
    _gather,
)

DATA = REPO / "data"
FEATURES_NPZ = DATA / "features.npz"
NIH_NPZ = DATA / "features_nih14.npz"
HEAD_PT = DATA / "tb_head_t2.pt"
CAL_PATH = DATA / "tb_threshold_t2.json"
DEVICE = "cpu"

# M26 rule constants — MIRRORED VERBATIM from
# src/lib/pipeline/asymmetricEvidence.ts so the audit reasons on the same numbers
# the production rule uses. If those constants move, update both files (the
# header in asymmetricEvidence.ts is the canonical derivation site).
TB_PROB_LOW_THRESHOLD = 0.20
BOX_EVIDENCE_HIGH_THRESHOLD = 0.88
PATHOLOGY_HIGH_THRESHOLD = 0.44

# The 5 TB-relevant TXRV labels the rule's pathology signal pools over (the
# `TB_RELEVANT_TXRV_LABELS` list in asymmetricEvidence.ts). The order of all 18
# TXRV columns in the txrv feature block is fixed by `TXRV_LABELS` in
# training/triage_core.py (canonical DenseNet pathology order). Indices below
# are derived from that order and pinned here for audit clarity.
TXRV_LABELS_18: tuple[str, ...] = (
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture",
    "Lung Opacity", "Enlarged Cardiomediastinum",
)
TB_RELEVANT_TXRV_LABELS: tuple[str, ...] = (
    "Lung Opacity", "Effusion", "Lung Lesion", "Infiltration", "Consolidation",
)
TB_RELEVANT_TXRV_IDX: tuple[int, ...] = tuple(
    TXRV_LABELS_18.index(name) for name in TB_RELEVANT_TXRV_LABELS
)

# Per-finding NIH columns — same order/contract as training/nih_stress_run.py.
FINDING_COLS: tuple[str, ...] = (
    "No_Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
)
# Mimicness tag — driven by the same MIMIC frozenset nih_stress_run.py uses;
# the M26 rule was designed to catch the TB-mimic class, so elevated ABSTAIN
# on TB-mimic findings is the rule WORKING as designed.
MIMIC_FINDINGS: frozenset[str] = frozenset({
    "Fibrosis", "Nodule", "Consolidation", "Mass",
    "Pleural_Thickening", "Infiltration", "Atelectasis",
})
EASY_FINDINGS: frozenset[str] = frozenset({"No_Finding"})

MIN_N_REPORT = 10  # the catch/cost ratio only makes sense when n >= 10 (CI width)

# Gating thresholds — see file header.
GATE_PASS_MAX_MULTIPLIER = 3.0
GATE_FAIL_MIN_MULTIPLIER = 5.0


# ---------------------------------------------------------------------------
# Calibration + head loading (mirrors training/nih_stress_run.py exactly).
# ---------------------------------------------------------------------------
def _load_calibration() -> tuple[float, float]:
    """Read the DEPLOYED temperature and 0.95-sensitivity threshold from the saved
    JSON (the one source of truth — the deployed head was trained against these)."""
    cfg = json.loads(CAL_PATH.read_text())
    return float(cfg["temperature"]), float(cfg["threshold"])


def _load_deployed_head(arrs: dict[str, np.ndarray]) -> TBHeadT2:
    """Load the deployed T2 head (fusion+zonal+box) from `tb_head_t2.pt`."""
    model = TBHeadT2(
        arrs["patches"].shape[2], arrs["cls"].shape[1], arrs["txrv"].shape[1],
        frozenset({"fusion", "zonal", "box"}),
    ).to(DEVICE)
    if not HEAD_PT.exists():
        raise FileNotFoundError(
            f"deployed head {HEAD_PT} missing — run training/nih_stress_run.py first to cache it"
        )
    model.load_state_dict(torch.load(HEAD_PT, map_location=DEVICE))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Forward pass: produce (tb_prob, box_evidence_max, top_pathology_score) for an
# index set against an arrs dict (the 4 keys cls/patches/txrv/zones the head
# consumes). The CLOSEST analogue is `train_tb.predict_logits_t2` + the
# `evidence_maps_t2` helper, but we want BOTH outputs in one pass — fused here
# to avoid two forward passes over 13k rows.
# ---------------------------------------------------------------------------
def _platt(z: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z / max(T, 1e-3)))


def _rule_inputs(
    model: TBHeadT2,
    arrs: dict[str, np.ndarray],
    idx: np.ndarray,
    T: float,
) -> dict[str, np.ndarray]:
    """One forward pass -> (tb_prob, box_evidence_max, top_pathology_score) for
    `idx` rows. Matches the production rule's signal extraction exactly:

      tb_prob              = sigmoid(image_logit / T) under deployed T
      box_evidence_max     = max over the 8x8 sigmoid(evidence) cells
      top_pathology_score  = max over the 5 TB-relevant TXRV sigmoid scores

    The TXRV sigmoid scores are computed from the LAST 18 columns of the txrv
    feature block — the same RAW logits the production engine sigmoids (see
    triage_core.run's `txrv_pathologies` block). This mirrors the production
    contract, NOT the calibrated head temperature; the rule's 0.44 threshold
    was anchored against these raw-sigmoid scores.
    """
    model.eval()
    logits_out: list[np.ndarray] = []
    box_max_out: list[np.ndarray] = []
    pathology_max_out: list[np.ndarray] = []
    head_arrs = {k: arrs[k] for k in ("cls", "patches", "txrv", "zones")}
    with torch.no_grad():
        for b in _batches(len(idx), shuffle=False):
            ix = idx[b]
            g = _gather(head_arrs, ix)
            model._zones = g["zones"]
            out = model(g["cls"], g["patches"], g["txrv"])
            logits_out.append(out["logit"].cpu().numpy())
            ev = out["evidence"]
            if ev is None:
                raise RuntimeError(
                    "evidence head is off on the loaded model — fairness audit requires the "
                    "box lever (the M26 rule reasons on box_evidence_max)"
                )
            # ev is [B, 64] logits; sigmoid + per-row max over 64 cells.
            ev_probs = torch.sigmoid(ev).cpu().numpy()
            box_max_out.append(ev_probs.max(axis=1))
            # txrv [B, 1042] — the last 18 are TXRV logits per `triage_core.TXRV_LABELS`.
            txrv_logits = g["txrv"][:, -18:].cpu().numpy()
            txrv_probs = 1.0 / (1.0 + np.exp(-txrv_logits))
            relevant = txrv_probs[:, list(TB_RELEVANT_TXRV_IDX)]
            pathology_max_out.append(relevant.max(axis=1))
    logits = np.concatenate(logits_out)
    return {
        "tb_prob": _platt(logits, T),
        "box_max": np.concatenate(box_max_out),
        "pathology_max": np.concatenate(pathology_max_out),
    }


# ---------------------------------------------------------------------------
# The rule itself, vectorised — mirrors `applyAsymmetricEvidence` in
# src/lib/pipeline/asymmetricEvidence.ts. Returns a boolean mask "abstain due to
# M26 rule" given the three signals and the BASE verdict (we approximate the
# base verdict as `tb_prob >= thr_at_95sens`, which is the M22 verdict rule
# pre-sequelae; sequelae escalates NO_TB->ABSTAIN earlier in the chain, but for
# this audit we want to count just the M26 firings, so we apply M26 only on the
# rows the base rule classifies as NO_TB).
# ---------------------------------------------------------------------------
def _abstain_mask(
    tb_prob: np.ndarray, box_max: np.ndarray, pathology_max: np.ndarray, thr_high: float
) -> np.ndarray:
    """Boolean mask: True where the M26 rule would fire (escalate NO_TB to ABSTAIN).

    Conditions ALL must hold (AND-gate):
      - tb_prob < thr_high                          (base verdict is NO_TB)
      - tb_prob < TB_PROB_LOW_THRESHOLD             (very-low TB band)
      - box_max >= BOX_EVIDENCE_HIGH_THRESHOLD      (model saw something)
      - pathology_max >= PATHOLOGY_HIGH_THRESHOLD   (backbone reports pathology)
    """
    base_no_tb = tb_prob < thr_high
    return (
        base_no_tb
        & (tb_prob < TB_PROB_LOW_THRESHOLD)
        & (box_max >= BOX_EVIDENCE_HIGH_THRESHOLD)
        & (pathology_max >= PATHOLOGY_HIGH_THRESHOLD)
    )


# ---------------------------------------------------------------------------
# Audit functions — one per dimension.
# ---------------------------------------------------------------------------
def audit_per_source(
    model: TBHeadT2, arrs: dict[str, np.ndarray], T: float, thr_high: float,
) -> list[dict[str, object]]:
    """DIMENSION 1: per-source ABSTAIN rates on the training-set features.

    SCOPE: training-set features (NOT held-out LODO OOF — see file header). For
    each source, reports total / TB+ / TB- abstain rates, strict sens/spec, and
    the effective sensitivity if ABSTAIN-on-TB+ is counted as a rescue.
    """
    y = arrs["y"].astype("int64")
    sources = arrs["source"]
    all_idx = np.arange(len(y))
    sig = _rule_inputs(model, arrs, all_idx, T)
    abstain = _abstain_mask(sig["tb_prob"], sig["box_max"], sig["pathology_max"], thr_high)
    pred_tb = sig["tb_prob"] >= thr_high
    rows: list[dict[str, object]] = []
    for src in sorted(set(sources.tolist())):
        m = sources == src
        n = int(m.sum())
        n_pos = int((m & (y == 1)).sum())
        n_neg = int((m & (y == 0)).sum())
        n_abs = int((m & abstain).sum())
        n_abs_pos = int((m & abstain & (y == 1)).sum())
        n_abs_neg = int((m & abstain & (y == 0)).sum())
        n_tp = int((m & pred_tb & (y == 1)).sum())
        n_fn = int((m & ~pred_tb & ~abstain & (y == 1)).sum())  # strict miss (not abstained)
        n_fp = int((m & pred_tb & (y == 0)).sum())
        n_tn = int((m & ~pred_tb & ~abstain & (y == 0)).sum())
        # Strict sens/spec exclude abstained rows; effective sens counts an
        # ABSTAIN on a TB+ as a "rescue" (the case will not be wrongly cleared).
        strict_sens = n_tp / max(n_tp + n_fn, 1)
        strict_spec = n_tn / max(n_tn + n_fp, 1)
        effective_sens = (n_tp + n_abs_pos) / max(n_pos, 1)
        rows.append({
            "source": src,
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "abstain_rate": n_abs / max(n, 1),
            "abstain_rate_tb_pos": n_abs_pos / max(n_pos, 1),
            "abstain_rate_tb_neg": n_abs_neg / max(n_neg, 1),
            "strict_sensitivity": float(strict_sens),
            "strict_specificity": float(strict_spec),
            "effective_sensitivity_with_abstain": float(effective_sens),
            "n_abstain": n_abs,
            "n_abstain_pos": n_abs_pos,
            "n_abstain_neg": n_abs_neg,
            "catch": n_abs_pos,
            "cost": n_abs_neg,
            "catch_cost_ratio": n_abs_pos / max(n_abs_neg, 1),
        })
    return rows


def audit_per_finding_nih(
    model: TBHeadT2, arrs: dict[str, np.ndarray], findings: dict[str, np.ndarray],
    T: float, thr_high: float,
) -> tuple[list[dict[str, object]], float]:
    """DIMENSION 2: per-finding ABSTAIN rates on the 10k NIH images.

    Returns (rows, baseline_no_finding_rate). All 10k rows are non-TB by NIH
    labels, so catch is 0 by construction for every subgroup; the headline
    fairness signal is the ABSTAIN rate relative to the No_Finding baseline.
    """
    n = int(arrs["cls"].shape[0])
    all_idx = np.arange(n)
    sig = _rule_inputs(model, arrs, all_idx, T)
    abstain = _abstain_mask(sig["tb_prob"], sig["box_max"], sig["pathology_max"], thr_high)
    # Baseline: ABSTAIN rate on No_Finding rows (the "easy negatives").
    nofinding_idx = np.where(findings["No_Finding"] == 1)[0]
    base_rate = float(abstain[nofinding_idx].mean()) if nofinding_idx.size else 0.0
    rows: list[dict[str, object]] = []
    for f in FINDING_COLS:
        idx = np.where(findings[f] == 1)[0]
        n_f = int(idx.size)
        if n_f == 0:
            rows.append({
                "name": f, "n": 0, "n_abstain": 0, "abstain_rate": float("nan"),
                "multiplier_vs_no_finding": float("nan"),
                "mean_tb_prob": float("nan"), "mean_box_max": float("nan"),
                "mean_pathology_max": float("nan"),
                "tag": "TB-mimic" if f in MIMIC_FINDINGS else ("easy-neg" if f in EASY_FINDINGS else "other-abn"),
            })
            continue
        n_abs = int(abstain[idx].sum())
        rate = n_abs / n_f
        rows.append({
            "name": f,
            "n": n_f,
            "n_abstain": n_abs,
            "abstain_rate": float(rate),
            "multiplier_vs_no_finding": float(rate / base_rate) if base_rate > 0 else float("inf") if n_abs > 0 else 0.0,
            "mean_tb_prob": float(sig["tb_prob"][idx].mean()),
            "mean_box_max": float(sig["box_max"][idx].mean()),
            "mean_pathology_max": float(sig["pathology_max"][idx].mean()),
            "tag": "TB-mimic" if f in MIMIC_FINDINGS else ("easy-neg" if f in EASY_FINDINGS else "other-abn"),
        })
    return rows, base_rate


def gating_verdict(
    source_rows: list[dict[str, object]],
    nih_rows: list[dict[str, object]],
    baseline_rate: float,
) -> dict[str, object]:
    """Compute PASS / WATCH / FAIL given the audit rows.

    Returns the verdict + the rationale + which subgroups (if any) tripped a
    gate. No threshold-fudging — the gate constants are at the top of this file.
    """
    elevated: list[dict[str, object]] = []
    failing: list[dict[str, object]] = []
    # Source dimension: catch/cost on TB-positive-containing subgroups; flag
    # ratios below 1.0 (more cost than catch).
    for r in source_rows:
        if int(r["n_pos"]) >= MIN_N_REPORT:  # type: ignore[arg-type]
            ratio = float(r["catch_cost_ratio"])  # type: ignore[arg-type]
            if ratio < 1.0:
                elevated.append({
                    "subgroup": f"source/{r['source']}",
                    "reason": f"catch/cost ratio {ratio:.2f} < 1.0 (cost exceeds rescue)",
                    "catch": int(r["catch"]),  # type: ignore[arg-type]
                    "cost": int(r["cost"]),  # type: ignore[arg-type]
                })
    # NIH dimension: flag elevated multipliers; FAIL on >5x with zero catch.
    for r in nih_rows:
        if int(r["n"]) < MIN_N_REPORT:
            continue
        mult = float(r["multiplier_vs_no_finding"])  # type: ignore[arg-type]
        if mult > GATE_FAIL_MIN_MULTIPLIER:
            # NIH is all-negative -> catch is 0 by construction. A subgroup with
            # >5x ABSTAIN and no catch is a Clever-Hans for ABSTAIN.
            failing.append({
                "subgroup": f"nih/{r['name']}",
                "reason": f"NIH ABSTAIN multiplier {mult:.2f}x > {GATE_FAIL_MIN_MULTIPLIER}x baseline "
                          f"(tag={r['tag']}, catch=0 because NIH has no TB positives)",
                "multiplier": mult,
            })
        elif mult > GATE_PASS_MAX_MULTIPLIER:
            elevated.append({
                "subgroup": f"nih/{r['name']}",
                "reason": f"NIH ABSTAIN multiplier {mult:.2f}x > {GATE_PASS_MAX_MULTIPLIER}x baseline (tag={r['tag']})",
                "multiplier": mult,
            })
    if failing:
        verdict = "FAIL"
        summary = (
            f"{len(failing)} subgroup(s) tripped the >{GATE_FAIL_MIN_MULTIPLIER}x-with-no-catch gate "
            "(Clever-Hans for ABSTAIN — disproportionate routing with no rescue benefit)"
        )
    elif elevated:
        verdict = "WATCH"
        summary = (
            f"{len(elevated)} subgroup(s) elevated above {GATE_PASS_MAX_MULTIPLIER}x baseline OR "
            "below 1.0 catch/cost ratio (defensible if the elevated subgroups are the rule's "
            "designed TB-mimic catch class)"
        )
    else:
        verdict = "PASS"
        summary = (
            f"no subgroup above {GATE_PASS_MAX_MULTIPLIER}x baseline AND all positive-containing "
            "source subgroups have catch/cost >= 1.0"
        )
    return {
        "verdict": verdict,
        "summary": summary,
        "baseline_rate": float(baseline_rate),
        "gate_pass_max_multiplier": float(GATE_PASS_MAX_MULTIPLIER),
        "gate_fail_min_multiplier": float(GATE_FAIL_MIN_MULTIPLIER),
        "elevated_subgroups": elevated,
        "failing_subgroups": failing,
    }


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------
def _load_train_arrs() -> dict[str, np.ndarray]:
    z = np.load(FEATURES_NPZ, allow_pickle=True)
    return {
        "cls": z["cls"].astype("float32"),
        "patches": z["patches"].astype("float32"),
        "txrv": z["txrv"].astype("float32"),
        "zones": z["zones"].astype("float32"),
        "y": z["y"].astype("int64"),
        "source": z["source"],
    }


def _load_nih_arrs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    z = np.load(NIH_NPZ, allow_pickle=True)
    arrs: dict[str, np.ndarray] = {
        "cls": z["cls"].astype("float32"),
        "patches": z["patches"].astype("float32"),
        "txrv": z["txrv"].astype("float32"),
        "zones": z["zones"].astype("float32"),
    }
    findings: dict[str, np.ndarray] = {f: z[f].astype("int64") for f in FINDING_COLS}
    return arrs, findings


# ---------------------------------------------------------------------------
# Pretty-printers.
# ---------------------------------------------------------------------------
def _print_per_source(rows: list[dict[str, object]]) -> None:
    print("\n=== DIMENSION 1: per-source ABSTAIN rates "
          "(TRAINING-SET features — NOT held-out OOF; see file header for scope) ===")
    print(f"{'source':>12s} {'n':>5s} {'pos':>4s} {'neg':>5s} "
          f"{'abs_all':>8s} {'abs_TB+':>8s} {'abs_TB-':>8s} "
          f"{'sens_strict':>11s} {'spec_strict':>11s} {'sens_eff':>9s} "
          f"{'catch':>6s} {'cost':>6s} {'c/c':>6s}")
    for r in rows:
        print(f"{r['source']:>12s} {r['n']:>5d} {r['n_pos']:>4d} {r['n_neg']:>5d} "
              f"{r['abstain_rate']:>8.3f} {r['abstain_rate_tb_pos']:>8.3f} {r['abstain_rate_tb_neg']:>8.3f} "
              f"{r['strict_sensitivity']:>11.3f} {r['strict_specificity']:>11.3f} {r['effective_sensitivity_with_abstain']:>9.3f} "
              f"{r['catch']:>6d} {r['cost']:>6d} {r['catch_cost_ratio']:>6.2f}")


def _print_per_finding(rows: list[dict[str, object]], baseline_rate: float) -> None:
    print(f"\n=== DIMENSION 2: per-finding NIH ABSTAIN rates ===")
    print(f"  baseline (No_Finding ABSTAIN rate) = {baseline_rate:.4f}")
    rows_sorted = sorted(rows, key=lambda r: -float(r["abstain_rate"]) if not np.isnan(float(r["abstain_rate"])) else -1.0)
    print(f"{'finding':>22s} {'tag':>10s} {'n':>5s} {'n_abs':>6s} "
          f"{'abs_rate':>9s} {'x_baseline':>11s} "
          f"{'mean_tb':>8s} {'mean_box':>9s} {'mean_path':>10s}")
    for r in rows_sorted:
        if int(r["n"]) == 0:
            continue
        print(f"{r['name']:>22s} {r['tag']:>10s} {r['n']:>5d} {r['n_abstain']:>6d} "
              f"{r['abstain_rate']:>9.4f} {r['multiplier_vs_no_finding']:>11.2f} "
              f"{r['mean_tb_prob']:>8.3f} {r['mean_box_max']:>9.3f} {r['mean_pathology_max']:>10.3f}")


def _print_verdict(verdict: dict[str, object]) -> None:
    print(f"\n=== GATING VERDICT: {verdict['verdict']} ===")
    print(f"  {verdict['summary']}")
    if verdict["elevated_subgroups"]:  # type: ignore[truthy-bool]
        print("  Elevated (WATCH):")
        for e in verdict["elevated_subgroups"]:  # type: ignore[union-attr]
            print(f"    - {e['subgroup']}: {e['reason']}")
    if verdict["failing_subgroups"]:  # type: ignore[truthy-bool]
        print("  Failing (FAIL):")
        for e in verdict["failing_subgroups"]:  # type: ignore[union-attr]
            print(f"    - {e['subgroup']}: {e['reason']}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def run_audit() -> dict[str, object]:
    """Run the full audit; returns the JSON-serialisable result + prints tables."""
    torch.manual_seed(0)
    np.random.seed(0)

    print("loading deployed calibration (data/tb_threshold_t2.json)…")
    T, thr_high = _load_calibration()
    print(f"  T={T:.4f}  thr@95sens={thr_high:.4f}")

    print("loading training-set features (data/features.npz)…")
    train_arrs = _load_train_arrs()
    print(f"  rows={len(train_arrs['y'])}  sources={sorted(set(train_arrs['source'].tolist()))}")

    print("loading NIH features (data/features_nih14.npz)…")
    nih_arrs, findings = _load_nih_arrs()
    print(f"  rows={int(nih_arrs['cls'].shape[0])}  findings={len(findings)}")

    print("loading deployed T2 head…")
    model = _load_deployed_head(train_arrs)

    source_rows = audit_per_source(model, train_arrs, T, thr_high)
    _print_per_source(source_rows)

    nih_rows, baseline_rate = audit_per_finding_nih(model, nih_arrs, findings, T, thr_high)
    _print_per_finding(nih_rows, baseline_rate)

    verdict = gating_verdict(source_rows, nih_rows, baseline_rate)
    _print_verdict(verdict)

    return {
        "audit": "M27a — subgroup fairness on M26 asymmetric-evidence rule",
        "calibration": {"temperature": T, "threshold_95sens": thr_high},
        "rule_thresholds": {
            "tb_prob_low": TB_PROB_LOW_THRESHOLD,
            "box_evidence_high": BOX_EVIDENCE_HIGH_THRESHOLD,
            "pathology_high": PATHOLOGY_HIGH_THRESHOLD,
            "tb_relevant_txrv_labels": list(TB_RELEVANT_TXRV_LABELS),
        },
        "scope": {
            "dimension_1_per_source": (
                "training-set features (data/features.npz) — NOT held-out LODO OOF. "
                "The LODO cache (data/image_oof_logits.npz) does not carry box_evidence or "
                "txrv_pathologies, which the rule reasons on; a strict OOF fairness audit "
                "requires re-running LODO end-to-end with the M24-enriched outputs cached."
            ),
            "dimension_2_per_finding_nih": (
                "10k NIH ChestX-ray14 images via data/features_nih14.npz — genuine "
                "off-distribution probe (NIH was never in training data)."
            ),
        },
        "dimension_1_per_source": source_rows,
        "dimension_2_per_finding_nih": {
            "baseline_no_finding_rate": baseline_rate,
            "rows": nih_rows,
        },
        "gating_verdict": verdict,
    }


def main() -> None:
    result = run_audit()
    out = DATA / "m27a_fairness_audit.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
