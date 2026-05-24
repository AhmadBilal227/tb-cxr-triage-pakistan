"""Stratified / mimic-FPR stress metrics for the frozen TB CXR triage head.

WHY THIS EXISTS (deployment-failure honesty, GPT strategy review). Our real failure mode is NOT
missing easy normals — it is FALSE-POSITIVES on TB MIMICS: healed scars, fibrosis, calcified
granulomas, nodules. The active-vs-scar AUROC is only ~0.72 and the scar false-positive rate
(SeqFPR) is ~0.842 (the main head confidently over-calls 84% of old scars as active). A single
"mean benchmark AUROC" hides this entirely: it averages easy negatives in with the hard mimics and
reports a flattering number. Per the project ethos, we report the deployment failure mode, not the
average.

WHAT THIS MODULE DOES. For each named negative SUBGROUP (an index array into a negatives feature
set) it reports, at the DEPLOYED operating point (the frozen 0.95-sensitivity threshold + the
frozen temperature):
  - n                 — subgroup size (CI width context)
  - FPR               — fraction scored >= threshold, with a Clopper-Pearson 95% CI
  - mean_score        — mean calibrated p_active over the subgroup (how confidently wrong)
  - frac_high         — fraction scored >= 0.95 (confidently-wrong tail)
  - Brier             — mean squared error of the calibrated prob vs the TRUE label (0 for negatives)
  - ECE               — expected calibration error computed ON THAT SUBGROUP

All metrics are vs the RADIOGRAPHIC reference label — NOT bacteriologically-confirmed active TB
(endpoint honesty, train_tb.py header / CASE_STUDY.md). A "false positive" here means the head
flagged a radiographic mimic that the source labeled non-active.

SUBGROUPS ARE PLUGGABLE. `stratified_fpr` accepts an arbitrary `dict[str, np.ndarray]` of named
index arrays into the negatives feature set, so when NIH ChestX-ray14 per-finding features land
(another agent owns that extraction) they slot in as e.g. {"nih_no_finding": idx, "nih_fibrosis":
idx, "nih_nodule": idx, ...} with NO change here. This module does NOT depend on NIH data existing
now; it is validated today against the TBX11K inactive/healed-scar probe ("tbx11k_scar").

    python training/stress_metrics.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

import train_tb

# CPU-ONLY by contract (do not touch the GPU). train_tb places input tensors on its module-level
# DEVICE (which auto-selects MPS); pin it to CPU BEFORE importing the predict helpers so _gather and
# the loaded head agree on device. The frozen head scores only 139 + N images — CPU is ample.
train_tb.DEVICE = "cpu"

from train_tb import (  # noqa: E402  (import after the DEVICE pin above, intentionally)
    TBHeadT2,
    clopper_pearson,
    ece,
    predict_t2,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
DEVICE = "cpu"

# Known reference value: the main T2 head over-calls 117/139 = 0.8417 of the TBX11K healed-scar
# probe as active at the deployed 0.95-sensitivity threshold (EXPERIMENT_LOG §C "T2 SEQUELAE" row).
KNOWN_SCAR_FPR = 117.0 / 139.0  # 0.8417

# A "confidently wrong" tail threshold: a calibrated probability this high on a NEGATIVE is an
# alarming false positive (a radiologist would not get a useful "low-confidence" hedge).
HIGH_SCORE = 0.95


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score (mean squared error of the calibrated probability vs the true label). For a pure-
    negative subgroup (y all 0) this reduces to mean(p**2): it rewards LOW scores and punishes
    confident false positives more than the bare FPR does."""
    y = y.astype("float64")
    p = p.astype("float64")
    return float(np.mean((p - y) ** 2))


def stratified_fpr(
    model: TBHeadT2,
    arrs: Mapping[str, np.ndarray],
    subgroups: Mapping[str, np.ndarray],
    thr: float,
    T: float,
    *,
    y_true: np.ndarray | None = None,
    predict_fn: Callable[[TBHeadT2, Mapping[str, np.ndarray], np.ndarray, float], np.ndarray] | None = None,
    high_score: float = HIGH_SCORE,
) -> list[dict[str, object]]:
    """Per-subgroup deployment-failure-mode table for a frozen head at a FIXED operating point.

    Parameters
    ----------
    model
        The frozen deployed head (e.g. the T2 head loaded from `data/tb_head_t2.pt`).
    arrs
        Feature arrays for the NEGATIVES set this subgroup dict indexes into (keys the predict
        function needs, e.g. "cls"/"patches"/"txrv"/"zones" for the T2 head).
    subgroups
        Named index arrays into `arrs` — e.g. {"tbx11k_scar": np.arange(139)} or, when NIH lands,
        {"nih_no_finding": idx0, "nih_fibrosis": idx1, "nih_nodule": idx2, ...}. Each value is a
        1-D integer index array; subgroups MAY overlap (e.g. a multi-label NIH finding).
    thr
        The DEPLOYED decision threshold (the frozen train-derived 0.95-sensitivity threshold). A
        prediction p >= thr is a positive call; on a negative subgroup that is a FALSE positive.
    T
        The DEPLOYED temperature. Predictions are temperature-scaled before thresholding so the
        scores, Brier, and ECE all reflect the actual deployed probabilities.
    y_true
        Optional per-image true labels aligned to `arrs`. Defaults to all-zeros (the subgroups are
        negatives, so a positive call is a false positive). Pass real labels only if a subgroup
        feature set legitimately mixes classes; for a pure-negatives mimic set leave it None.
    predict_fn
        Pluggable scorer `(model, arrs, idx, T) -> probs`. Defaults to the T2 head's `predict_t2`.
        Lets the same table back a different head later without changing this signature.
    high_score
        Tail threshold for `frac_high` (default 0.95) — the confidently-wrong fraction.

    Returns
    -------
    A list of per-subgroup dicts (one row per named subgroup), each with: name, n, n_fp, fpr,
    fpr_ci (Clopper-Pearson 95%), mean_score, frac_high, brier, ece. Pure-Python/numpy scalars
    (JSON-serialisable). FPR is reported with a binomial CI, never as a bare point estimate
    (small mimic subgroups have wide CIs — that uncertainty must be visible, per ethos).
    """
    if predict_fn is None:
        predict_fn = predict_t2  # type: ignore[assignment]
    n_total = int(np.asarray(next(iter(arrs.values()))).shape[0])
    if y_true is None:
        y_full = np.zeros(n_total, dtype="int64")  # subgroups are negatives by construction
    else:
        y_full = np.asarray(y_true).astype("int64")
        if y_full.shape[0] != n_total:
            raise ValueError(f"y_true length {y_full.shape[0]} != feature-set length {n_total}")

    rows: list[dict[str, object]] = []
    for name, idx in subgroups.items():
        idx = np.asarray(idx).astype("int64")
        if idx.ndim != 1:
            raise ValueError(f"subgroup {name!r} index must be 1-D, got shape {idx.shape}")
        if idx.size == 0:
            rows.append({"name": name, "n": 0, "n_fp": 0, "fpr": float("nan"),
                         "fpr_ci": (float("nan"), float("nan")), "mean_score": float("nan"),
                         "frac_high": float("nan"), "brier": float("nan"), "ece": float("nan")})
            continue
        if int(idx.min()) < 0 or int(idx.max()) >= n_total:
            raise ValueError(f"subgroup {name!r} index out of range for feature set of size {n_total}")
        p = predict_fn(model, arrs, idx, T)  # calibrated, temperature-scaled deployed probabilities
        y_sub = y_full[idx]
        neg = y_sub == 0  # FPR is defined over the negatives within the subgroup
        n_neg = int(neg.sum())
        n_fp = int(((p >= thr) & neg).sum())
        fpr = n_fp / n_neg if n_neg else float("nan")
        ci = clopper_pearson(n_fp, n_neg) if n_neg else (float("nan"), float("nan"))
        rows.append({
            "name": name,
            "n": int(idx.size),
            "n_fp": n_fp,
            "fpr": float(fpr),
            "fpr_ci": (float(ci[0]), float(ci[1])),
            "mean_score": float(np.mean(p)),
            "frac_high": float(np.mean(p >= high_score)),
            "brier": brier(y_sub, p),
            "ece": float(ece(y_sub, p)),
        })
    return rows


def print_stratified_table(rows: list[dict[str, object]], thr: float, T: float) -> None:
    """Print the per-mimic stress table. Leads with FPR + binomial CI (the safety-critical failure
    mode for a screen is a FALSE positive on a mimic, which drives confirmatory-test burden)."""
    print(f"\n--- stratified mimic-FPR stress (deployed thr={thr:.3f}, T={T:.3f}; "
          f"radiographic endpoint, NOT bacteriologically-confirmed) ---")
    print(f"{'subgroup':>16s} {'n':>5s} {'FPR':>7s} {'95% CI':>15s} "
          f"{'mean':>6s} {'frac>=.95':>9s} {'Brier':>7s} {'ECE':>6s}")
    for r in rows:
        lo, hi = r["fpr_ci"]  # type: ignore[misc]
        ci = f"{lo:.3f}-{hi:.3f}" if not np.isnan(lo) else "n/a"  # type: ignore[arg-type]
        fpr = r["fpr"]
        fpr_s = f"{fpr:.3f}" if not np.isnan(fpr) else "n/a"  # type: ignore[arg-type]
        print(f"{r['name']:>16s} {r['n']:>5d} {fpr_s:>7s} {ci:>15s} "
              f"{r['mean_score']:>6.3f} {r['frac_high']:>9.3f} {r['brier']:>7.3f} {r['ece']:>6.3f}")
    print("  NOTE: a 'false positive' = the head flagged a radiographic mimic the source labeled "
          "non-active.\n  Brier/ECE are computed ON each subgroup; small mimic sets have WIDE CIs "
          "(the uncertainty is intentional).")


def _seq_arrs() -> dict[str, np.ndarray]:
    """Feature arrays for the TBX11K inactive/healed-scar probe (the 'tbx11k_scar' mimic subgroup).
    All 139 rows are negatives for the ACTIVE-TB decision (an old scar is not active disease), so the
    whole set is one negative subgroup indexed 0..138."""
    s = np.load(DATA / "features_sequelae.npz", allow_pickle=True)
    return {
        "cls": s["cls"].astype("float32"),
        "patches": s["patches"].astype("float32"),
        "txrv": s["txrv"].astype("float32"),
        "zones": s["zones"].astype("float32"),
    }


def _load_deployed_t2() -> tuple[TBHeadT2, float, float]:
    """Load the frozen deployed T2 head + its deployed temperature and 0.95-sensitivity threshold."""
    cfg = json.loads((DATA / "tb_threshold_t2.json").read_text())
    arrs = _seq_arrs()
    model = TBHeadT2(arrs["patches"].shape[2], arrs["cls"].shape[1], arrs["txrv"].shape[1],
                     frozenset({"fusion", "zonal", "box"})).to(DEVICE)
    model.load_state_dict(torch.load(DATA / "tb_head_t2.pt", map_location=DEVICE))
    model.eval()
    return model, float(cfg["temperature"]), float(cfg["threshold"])


def build_nih_subgroups(
    nih_findings: np.ndarray | None,
    finding_names: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Build a pluggable NIH per-finding subgroup dict for `stratified_fpr` — used when NIH
    ChestX-ray14 features land (another agent owns that extraction; this does NOT run today).

    `nih_findings` is a [N, n_findings] binary multi-label matrix aligned to the NIH negatives
    feature set (1 = finding present). Returns {"nih_<finding>": idx_array, ...} plus
    "nih_no_finding" for rows with no finding flagged. Subgroups MAY overlap (multi-label). If
    `nih_findings` is None this returns {} (graceful: NIH not present yet)."""
    if nih_findings is None:
        return {}
    fm = np.asarray(nih_findings)
    if fm.ndim != 2:
        raise ValueError(f"nih_findings must be 2-D [N, n_findings], got shape {fm.shape}")
    names = finding_names or tuple(f"f{i}" for i in range(fm.shape[1]))
    if len(names) != fm.shape[1]:
        raise ValueError(f"finding_names length {len(names)} != n_findings {fm.shape[1]}")
    subgroups: dict[str, np.ndarray] = {}
    for j, fname in enumerate(names):
        idx = np.where(fm[:, j] > 0)[0]
        subgroups[f"nih_{fname}"] = idx
    subgroups["nih_no_finding"] = np.where(fm.sum(axis=1) == 0)[0]
    return subgroups


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    model, T, thr = _load_deployed_t2()
    arrs = _seq_arrs()
    # Subgroup set TODAY: just the TBX11K healed-scar probe. Designed so NIH per-finding subgroups
    # (build_nih_subgroups) slot in unchanged when their features land. Each entry is an index array
    # into the negatives feature set passed alongside it; here every row is a negative.
    subgroups: dict[str, np.ndarray] = {"tbx11k_scar": np.arange(arrs["cls"].shape[0])}
    print("ENDPOINT: radiographic-TB-pattern (NOT bacteriologically-confirmed active TB). "
          "Research preview.")
    print(f"deployed T2 head: temperature={T:.3f}, 0.95-sensitivity threshold={thr:.3f}")
    rows = stratified_fpr(model, arrs, subgroups, thr, T)
    print_stratified_table(rows, thr, T)
    scar = next(r for r in rows if r["name"] == "tbx11k_scar")
    print(f"\nCORRECTNESS CHECK: tbx11k_scar FPR through stratified_fpr = "
          f"{scar['fpr']:.4f} ({scar['n_fp']}/{scar['n']})  vs known SeqFPR {KNOWN_SCAR_FPR:.4f}  "
          f"-> {'MATCH' if abs(float(scar['fpr']) - KNOWN_SCAR_FPR) <= 0.01 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
