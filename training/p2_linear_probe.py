"""P2.0 — the cheapest backbone discriminator (ZERO new feature extraction).

Trains L2-logistic-regression probes on the ALREADY-CACHED 4-source training corpus
(data/features.npz: rad-dino CLS 768 + TXRV 1042) and evaluates on the held-out
Pakistani cohort (data/features_mendeley_pk.npz, same schema). Three branches:

  (a) rad-dino CLS only      — does the frozen DINO representation separate PK TB?
  (b) TXRV only              — the fixed pathology-feature branch on its own
  (c) CLS ⊕ TXRV fused       — both, the way the deployed head consumes them

For each branch we report, on BOTH surfaces (held-out Pakistani + the seed=7 LODO
80% eval slice), under PATIENT-LEVEL bootstrap (group by patient_id/group, not image):
  - AUROC (95% CI)
  - pAUC in the high-specificity region (FPR <= 0.2), normalized to [0,1]
  - sensitivity at specificity = 0.70

The headline is THRESHOLD-FREE (AUROC / pAUC). A threshold, if reported, is fit ONCE
on the seed=7 cal slice — never on the eval surface.

DECISION GATE P2.0 (pre-registered, in the plan + EXPERIMENT_LOG):
  - rad-dino CLS or fused PK AUROC ~= 0.78 (within CI of the deployed head's external
    0.781) -> the representation caps at the external ceiling -> PROCEED to P2.1.
  - PK AUROC materially HIGHER (>= +0.05 over 0.78, CIs apart) -> rad-dino already
    holds more external signal than the head extracts -> the lever is the head/
    training/calibration, NOT a new backbone -> STOP the backbone spike.

  HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    training/.venv/bin/python training/p2_linear_probe.py --output data/p2_linear_probe.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from locked_protocol import make_calibration_split  # noqa: E402

DATA = REPO / "data"
ORIG_SOURCES = ("montgomery", "shenzhen", "qatar", "tbx11k")
BOOTSTRAP_N = 2000
DEPLOYED_PK_AUROC = 0.781  # frozen-baseline external number (P1 meta / M27)
GATE_MARGIN = 0.05         # "materially higher" = >= +0.05 over the deployed ceiling
PAUC_FPR_MAX = 0.20        # high-specificity region (spec >= 0.80)
SENS_AT_SPEC = 0.70        # report sensitivity at this specificity

# Pre-registered gate branches (CLS / TXRV / fused — the deployed head's NON-spatial inputs).
BRANCHES = ("cls", "txrv", "fused")

# POST-HOC diagnostic branches (NOT in the pre-registered P2.0 gate). Added after the gate branches
# collapsed externally, to test whether the rad-dino external signal lives in the SPATIAL patch
# tokens (which the head pools via zonal-soft-OR and the CLS probe discards). patch-MAX is a
# nonlinear MIL/OR-style readout — closer to the deployed head's signal path than a CLS probe. These
# are HYPOTHESIS-GENERATING (they consult the Pakistani set), reported separately, and do NOT change
# the literal gate decision; they DO qualify its interpretation. See the CASE_STUDY P2 entry.
DIAG_BRANCHES = ("patch_mean", "patch_max", "cls_patchmax_txrv")


def _feature_matrix(d, branch: str) -> np.ndarray:
    cls = d["cls"].astype("float64")
    txrv = d["txrv"].astype("float64")
    if branch == "cls":
        return cls
    if branch == "txrv":
        return txrv
    if branch == "fused":
        return np.concatenate([cls, txrv], axis=1)
    # ---- post-hoc patch-token diagnostics ----
    if branch in ("patch_mean", "patch_max", "cls_patchmax_txrv"):
        patches = d["patches"].astype("float64")  # [N, 64, 768]
        patch_mean = patches.mean(axis=1)
        patch_max = patches.max(axis=1)
        if branch == "patch_mean":
            return patch_mean
        if branch == "patch_max":
            return patch_max
        return np.concatenate([cls, patch_max, txrv], axis=1)
    raise ValueError(f"unknown branch {branch}")


def _pauc(y: np.ndarray, score: np.ndarray, fpr_max: float = PAUC_FPR_MAX) -> float:
    """Partial AUC over FPR in [0, fpr_max], McClish-standardized to [0.5, 1] so a no-skill
    classifier maps to 0.5 and a perfect classifier to 1.0.

    raw_area = integral of TPR over FPR in [0, fpr_max].
    no-skill raw_area (diagonal) = fpr_max^2 / 2; max raw_area = fpr_max.
    standardized = 0.5 * (1 + (raw_area - min) / (max - min)),
    where min = fpr_max^2/2 (no-skill) and max = fpr_max (perfect)."""
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    # clip the curve at fpr_max, interpolating the crossing point
    keep = fpr <= fpr_max
    if not keep.any():
        return float("nan")
    fx = fpr[keep]
    tx = tpr[keep]
    if fx[-1] < fpr_max:
        # interpolate tpr at fpr_max using the next point
        nxt = np.searchsorted(fpr, fpr_max)
        if nxt < len(fpr):
            f0, f1 = fpr[nxt - 1], fpr[nxt]
            t0, t1 = tpr[nxt - 1], tpr[nxt]
            t_at = t0 + (t1 - t0) * (fpr_max - f0) / (f1 - f0) if f1 > f0 else t0
            fx = np.append(fx, fpr_max)
            tx = np.append(tx, t_at)
    raw_area = np.trapezoid(tx, fx)
    min_area = 0.5 * fpr_max * fpr_max  # no-skill (diagonal) area in [0, fpr_max]
    max_area = fpr_max                  # perfect-classifier area in [0, fpr_max]
    std = 0.5 * (1.0 + (raw_area - min_area) / (max_area - min_area))
    return float(std)


def _sens_at_spec(y: np.ndarray, score: np.ndarray, spec_target: float = SENS_AT_SPEC) -> float:
    """Sensitivity (TPR) at the threshold giving specificity == spec_target (FPR = 1-spec)."""
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    fpr_target = 1.0 - spec_target
    # tpr at the operating point where fpr first reaches fpr_target (interpolate)
    return float(np.interp(fpr_target, fpr, tpr))


def _patient_bootstrap(
    y: np.ndarray, score: np.ndarray, groups: np.ndarray,
    n: int = BOOTSTRAP_N, seed: int = 11,
) -> dict:
    """PATIENT-LEVEL bootstrap: resample unique groups (patients) with replacement, take all
    images of each resampled patient. Returns AUROC/pAUC/sens@spec point estimates + 95% CIs."""
    auroc_pt = roc_auc_score(y, score) if len(np.unique(y)) > 1 else float("nan")
    pauc_pt = _pauc(y, score)
    sens_pt = _sens_at_spec(y, score)

    uniq = np.unique(groups)
    g_to_rows: dict = {}
    for g in uniq:
        g_to_rows[g] = np.where(groups == g)[0]
    rng = np.random.default_rng(seed)
    aurocs, paucs, sens_vals = [], [], []
    for _ in range(n):
        picks = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([g_to_rows[g] for g in picks])
        yy, ss = y[rows], score[rows]
        if len(np.unique(yy)) < 2:
            continue
        aurocs.append(roc_auc_score(yy, ss))
        paucs.append(_pauc(yy, ss))
        sens_vals.append(_sens_at_spec(yy, ss))

    def ci(vals):
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return [float("nan"), float("nan")]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    return {
        "auroc": float(auroc_pt), "auroc_ci": ci(aurocs),
        "pauc_fpr20": float(pauc_pt), "pauc_fpr20_ci": ci(paucs),
        "sens_at_spec70": float(sens_pt), "sens_at_spec70_ci": ci(sens_vals),
        "n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "n_patients": int(len(uniq)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DATA / "p2_linear_probe.json"))
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength for the probe")
    args = ap.parse_args()

    # ---- training corpus (4 sources only; the probe trains here, never on Pakistani) ----
    dtr = np.load(DATA / "features.npz", allow_pickle=True)
    src = dtr["source"].astype(str)
    orig_mask = np.isin(src, list(ORIG_SOURCES))
    y_all = dtr["y"].astype("int64")
    groups_all = dtr["group"].astype("int64")

    # ---- LODO eval slice: seed=7 80% complement of the cal slice on the 4-source rows ----
    # We need per-source OOF logits to mirror p1, but for a linear PROBE the honest
    # in-distribution reference is: train the probe on the cal-slice complement and eval on
    # the seed=7 eval slice. To avoid train/eval leakage on the SAME corpus we hold out the
    # eval slice from probe training entirely.
    y_orig = y_all[orig_mask]
    src_orig = src[orig_mask]
    groups_orig = groups_all[orig_mask]
    # deterministic seed=7 split on the 4-source rows (same util as the locked protocol)
    cal_idx, eval_idx = make_calibration_split(
        np.zeros(len(y_orig)), y_orig, src_orig)  # logit arg unused by the split
    # probe-training rows = everything in the 4-source corpus that is NOT in the LODO eval slice
    is_eval = np.zeros(len(y_orig), dtype=bool)
    is_eval[eval_idx] = True
    train_rows_orig = np.where(~is_eval)[0]  # cal slice -> probe train (in-distribution)

    # ---- held-out Pakistani ----
    dpk = np.load(DATA / "features_mendeley_pk.npz", allow_pickle=True)
    y_pk = dpk["y"].astype("int64")
    if "group" in dpk.files:
        groups_pk = dpk["group"].astype("int64")
    else:
        # PK is one image per patient_id; group by patient_id string
        pid = dpk["patient_id"].astype(str)
        _, groups_pk = np.unique(pid, return_inverse=True)

    print(f"Training corpus (4 src): n={int(orig_mask.sum())} "
          f"probe-train(cal)={len(train_rows_orig)} lodo-eval={len(eval_idx)}")
    print(f"Pakistani holdout: n={len(y_pk)} pos={int((y_pk==1).sum())} neg={int((y_pk==0).sum())} "
          f"patients={len(np.unique(groups_pk))}")
    print(f"Deployed-head external ceiling for the gate: PK AUROC={DEPLOYED_PK_AUROC} "
          f"(materially-higher margin = +{GATE_MARGIN})\n")

    results = {
        "meta": {
            "deployed_pk_auroc": DEPLOYED_PK_AUROC,
            "gate_margin": GATE_MARGIN,
            "pauc_fpr_max": PAUC_FPR_MAX,
            "sens_at_spec": SENS_AT_SPEC,
            "probe": f"LogisticRegression(L2, C={args.C}, standardized)",
            "bootstrap_n": BOOTSTRAP_N,
            "n_train_corpus_orig": int(orig_mask.sum()),
            "n_probe_train_cal": int(len(train_rows_orig)),
            "n_lodo_eval": int(len(eval_idx)),
            "n_pakistani": int(len(y_pk)),
            "lodo_eval_caveat": (
                "The 'lodo_eval' slice here is a SAME-SITE seed=7 holdout (both the probe-train cal "
                "slice and this eval slice contain all 4 sources), NOT true leave-one-DATASET-out. "
                "Its near-perfect AUROC (~0.99) is an IN-DISTRIBUTION reference only and must NOT be "
                "compared against the deployed head's true-LODO 0.923 (held-out SITE). The honest "
                "external number is 'pakistani'."
            ),
        },
        "branches": {},
    }

    def _run_branch(b: str) -> dict:
        Xtr = _feature_matrix(dtr, b)[orig_mask]
        Xpk_b = _feature_matrix(dpk, b)
        scaler = StandardScaler().fit(Xtr[train_rows_orig])
        clf = LogisticRegression(
            C=args.C, max_iter=5000, class_weight="balanced", solver="lbfgs")
        clf.fit(scaler.transform(Xtr[train_rows_orig]), y_orig[train_rows_orig])
        # LODO eval slice (in-distribution reference) — held out from probe training
        s_lodo = clf.decision_function(scaler.transform(Xtr[eval_idx]))
        lodo = _patient_bootstrap(y_orig[eval_idx], s_lodo, groups_orig[eval_idx], seed=7)
        # held-out Pakistani (external, never seen in probe training)
        s_pk = clf.decision_function(scaler.transform(Xpk_b))
        pk = _patient_bootstrap(y_pk, s_pk, groups_pk, seed=11)
        print(f"  PAKISTANI: AUROC={pk['auroc']:.3f} {pk['auroc_ci']} "
              f"pAUC(FPR<=.2)={pk['pauc_fpr20']:.3f} sens@spec.70={pk['sens_at_spec70']:.3f}")
        print(f"  LODO-EVAL: AUROC={lodo['auroc']:.3f} {lodo['auroc_ci']} "
              f"pAUC(FPR<=.2)={lodo['pauc_fpr20']:.3f} sens@spec.70={lodo['sens_at_spec70']:.3f}\n")
        return {"pakistani": pk, "lodo_eval": lodo}

    for b in BRANCHES:
        print(f"===== GATE BRANCH {b} =====")
        results["branches"][b] = _run_branch(b)

    # ---- POST-HOC patch-token diagnostics (NOT part of the pre-registered gate) ----
    results["diagnostics_posthoc"] = {
        "note": (
            "These branches were added AFTER the gate branches collapsed externally, to test "
            "whether rad-dino's external signal lives in the SPATIAL patch tokens (the head pools "
            "them via zonal-soft-OR; the CLS probe discards them). patch-MAX is a nonlinear MIL/OR "
            "readout, closer to the deployed head's signal path. They CONSULT the Pakistani set, so "
            "they are HYPOTHESIS-GENERATING and DO NOT change the literal gate decision; they "
            "qualify its interpretation (the CLS gate is a misspecified discriminator for an "
            "OR-pooling head)."
        ),
        "branches": {},
    }
    for b in DIAG_BRANCHES:
        print(f"===== DIAGNOSTIC BRANCH {b} (post-hoc) =====")
        results["diagnostics_posthoc"]["branches"][b] = _run_branch(b)

    # ---- evaluate the gate ----
    cls_pk = results["branches"]["cls"]["pakistani"]
    fused_pk = results["branches"]["fused"]["pakistani"]
    best_auroc = max(cls_pk["auroc"], fused_pk["auroc"])
    best_branch = "cls" if cls_pk["auroc"] >= fused_pk["auroc"] else "fused"
    best_ci_lo = (cls_pk if best_branch == "cls" else fused_pk)["auroc_ci"][0]
    # "materially higher" requires BOTH: point >= ceiling+margin AND lower-CI above ceiling
    materially_higher = (best_auroc >= DEPLOYED_PK_AUROC + GATE_MARGIN) and (best_ci_lo > DEPLOYED_PK_AUROC)
    decision = "STOP" if materially_higher else "PROCEED"
    reasoning = (
        f"best probe PK AUROC={best_auroc:.3f} (branch={best_branch}, "
        f"CI_lo={best_ci_lo:.3f}) vs deployed ceiling {DEPLOYED_PK_AUROC}. "
    )
    if materially_higher:
        reasoning += (
            f"Materially higher (>= +{GATE_MARGIN} AND CI_lo above ceiling): rad-dino "
            "features hold untapped external signal -> lever is head/training/calibration, "
            "NOT a new backbone. STOP the backbone spike."
        )
    else:
        reasoning += (
            "Not materially higher: the rad-dino representation caps near the deployed "
            "external ceiling -> backbone is a candidate lever -> PROCEED to P2.1."
        )
    # ---- interpretation caveat: is the CLS/fused gate a VALID discriminator? ----
    # The deployed head's signal path is the SPATIAL patch tokens under zonal-soft-OR (an OR/max
    # readout), NOT the CLS token. If a patch-MAX probe reaches >= the deployed external AUROC while
    # CLS/fused collapse, the gate branches under-read the backbone -> the literal PROCEED is on a
    # misspecified discriminator, and the honest label is INCONCLUSIVE (lever is plausibly the
    # POOLING/head, not the backbone family). Flag this for the coordinator.
    pmax_pk = results["diagnostics_posthoc"]["branches"]["patch_max"]["pakistani"]
    gate_branches_collapsed = best_auroc < DEPLOYED_PK_AUROC - GATE_MARGIN
    patchmax_recovers = pmax_pk["auroc"] >= DEPLOYED_PK_AUROC
    gate_valid = not (gate_branches_collapsed and patchmax_recovers)
    interp = (
        f"patch_max PK AUROC={pmax_pk['auroc']:.3f} {pmax_pk['auroc_ci']}. "
    )
    if not gate_valid:
        interp += (
            "GATE MISSPECIFIED: the CLS/fused gate branches collapsed externally "
            f"(best {best_auroc:.3f} << deployed {DEPLOYED_PK_AUROC}) while a patch-MAX (OR-style) "
            f"probe recovers >= the deployed external AUROC ({pmax_pk['auroc']:.3f}). The deployed "
            "head ALREADY pools patch tokens via soft-OR, so the CLS probe is the wrong "
            "discriminator. The literal PROCEED rests on a misspecified probe; the honest label is "
            "INCONCLUSIVE. The rad-dino representation DOES hold ~deployed-level external signal in "
            "its spatial tokens -> the lever is plausibly POOLING/head/training, not the backbone "
            "FAMILY. maira-2 (same DINO family) is therefore LOWER-priority than head/pooling work. "
            "Surface to the coordinator before extracting maira-2."
        )
    else:
        interp += "Gate branches are a valid discriminator for this head; no misspecification flag."

    results["gate_p2_0"] = {
        "decision": decision,
        "best_branch": best_branch,
        "best_pk_auroc": float(best_auroc),
        "best_pk_auroc_ci_lo": float(best_ci_lo),
        "deployed_ceiling": DEPLOYED_PK_AUROC,
        "margin": GATE_MARGIN,
        "materially_higher": bool(materially_higher),
        "reasoning": reasoning,
        "gate_valid_discriminator": bool(gate_valid),
        "interpretation_caveat": interp,
        "honest_label": "PROCEED" if gate_valid else "INCONCLUSIVE (gate misspecified)",
    }
    print("=" * 70)
    print(f"GATE P2.0 (literal): {decision}")
    print(f"  {reasoning}")
    print(f"GATE P2.0 (honest):  {results['gate_p2_0']['honest_label']}")
    print(f"  {interp}")
    print("=" * 70)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
