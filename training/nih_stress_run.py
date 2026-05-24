"""NIH ChestX-ray14 per-finding FPR stress test.

WHY THIS EXISTS. The scar-subgroup catastrophic mis-calibration (Brier 0.806 / ECE 0.846 on the
139-image TBX11K healed-scar probe; FPR 0.842 at the deployed 0.95-sens threshold) raised an open
question: is the decision-layer broken ONLY on healed scars (in which case the sequelae head is
the targeted fix) or does it generalize to all radiographic TB mimics (fibrosis, nodule, consolid-
ation, mass, pleural thickening, infiltration, atelectasis) — i.e. a broader decision-layer
problem? NIH ChestX-ray14 (10k images, per-finding labels) lets us stratify the FPR by finding and
settle that question with sample-rich CIs.

WHAT IT DOES.
  1. Loads `data/features.npz` (training-set, 13,092 dual-backbone rows).
  2. Loads the cached deployed T2 head from `data/tb_head_t2.pt` (zonal+box+fusion, prior-fixed
     config — same as train_tb.py's final save). RE-DERIVES temperature T (fit_temperature on the
     0.15 stratified val split, SEED=0) and the 0.95-sensitivity threshold from that val split.
     This is the exact procedure train_tb.py used to write `tb_threshold_t2.json`, so the values
     should land at T~1.591, thr~0.610 (the EXPERIMENT_LOG row). The user explicitly asked the
     data to redrive these, not to read the JSON literal — so the audit trail is consistent.
  3. Loads `data/features_nih14.npz` (10k images, all non-TB by NIH labels).
  4. For each ChestX-ray14 finding column (15 of them) builds the subgroup index, drops subgroups
     with n<50 (marked n/a), and runs `training.stress_metrics.stratified_fpr` at the DEPLOYED
     (T, thr). Also reports the overall NIH FPR (the naive binary baseline that hides the per-
     finding gap).
  5. Prints a single ranked table (sorted by FPR descending) with the mimicness tag for each row:
        TB-mimic     = Fibrosis, Nodule, Consolidation, Mass, Pleural_Thickening, Infiltration, Atelectasis
        easy negative = No_Finding (the deceptive baseline a single AUROC averages over)
        other-abnormal = the rest (Cardiomegaly, Effusion, Pneumonia, Pneumothorax, Edema, Emphysema, Hernia)

NOT RUN HERE (intentional). Re-training the T2 head from features.npz takes minutes-to-hours and
the saved weights already match the deployed config; we load. If `tb_head_t2.pt` is missing or
the val-split sensitivity check fails, we DO retrain with SEED=0 (the user said that is fine).

    PYTORCH_ENABLE_MPS_FALLBACK=1 training/.venv/bin/python training/nih_stress_run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

import train_tb  # noqa: E402  (DEVICE pin must happen before predict_t2 uses it)

# CPU-ONLY: stress_metrics convention. 10k images * frozen-head MLP is trivial on CPU and avoids
# the MPS adaptive_avg_pool2d fallback chatter for a forward-only run.
train_tb.DEVICE = "cpu"

from train_tb import (  # noqa: E402  (after DEVICE pin)
    SEED,
    TARGET_SENS,
    TBHeadT2,
    ece,
    fit_temperature,
    predict_logits_t2,
    predict_t2,
    threshold_for_sensitivity,
    train_head_t2,
)
from stress_metrics import brier, print_stratified_table, stratified_fpr  # noqa: E402

DATA = REPO / "data"
FEATURES_NPZ = DATA / "features.npz"
NIH_NPZ = DATA / "features_nih14.npz"
HEAD_PT = DATA / "tb_head_t2.pt"
DEVICE = "cpu"

# Locked from extract_features.FINDING semantics; matches the sidecar column order.
FINDING_COLS: tuple[str, ...] = (
    "No_Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
)

# Mimicness tags. Driven by what the literature and our own SeqFPR finding identify as the failure-mode
# classes — calcified granulomas, fibrosis, nodules, consolidation, mass, pleural thickening,
# infiltration, atelectasis all radiographically resemble active TB and are the deployment-relevant
# false-positive sources. No_Finding is the easy-negative baseline. Cardiomegaly / Effusion / etc.
# are abnormalities BUT not TB-mimic-shaped.
MIMIC: frozenset[str] = frozenset({
    "Fibrosis", "Nodule", "Consolidation", "Mass",
    "Pleural_Thickening", "Infiltration", "Atelectasis",
})
EASY: frozenset[str] = frozenset({"No_Finding"})
MIN_N = 50  # subgroup-size floor for "report" (smaller groups have prohibitively wide CIs)


def _tag(name: str) -> str:
    if name in MIMIC:
        return "TB-mimic"
    if name in EASY:
        return "easy-neg"
    return "other-abn"


def _train_features() -> dict[str, np.ndarray]:
    """Load training-set features with the keys the T2 head needs (cls, patches, txrv, zones).
    Returns float32 copies (numpy) — `_gather` will move them onto DEVICE (here, CPU)."""
    z = np.load(FEATURES_NPZ, allow_pickle=True)
    return {
        "cls": z["cls"].astype("float32"),
        "patches": z["patches"].astype("float32"),
        "txrv": z["txrv"].astype("float32"),
        "zones": z["zones"].astype("float32"),
        "y": z["y"].astype("int64"),
    }


def _nih_features() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load NIH features. Returns (arrs_for_head, finding_arrays). `arrs_for_head` has the keys the
    T2 head needs; `finding_arrays` is the per-finding 0/1 label arrays."""
    z = np.load(NIH_NPZ, allow_pickle=True)
    arrs: dict[str, np.ndarray] = {
        "cls": z["cls"].astype("float32"),
        "patches": z["patches"].astype("float32"),
        "txrv": z["txrv"].astype("float32"),
        "zones": z["zones"].astype("float32"),
    }
    findings: dict[str, np.ndarray] = {f: z[f].astype("int64") for f in FINDING_COLS}
    return arrs, findings


def _load_or_train_head(arrs: dict[str, np.ndarray], y: np.ndarray) -> TBHeadT2:
    """Load cached deployed T2 head; if it does not load cleanly, retrain with SEED=0."""
    model = TBHeadT2(arrs["patches"].shape[2], arrs["cls"].shape[1], arrs["txrv"].shape[1],
                     frozenset({"fusion", "zonal", "box"})).to(DEVICE)
    if HEAD_PT.exists():
        try:
            model.load_state_dict(torch.load(HEAD_PT, map_location=DEVICE))
            model.eval()
            print(f"loaded cached head: {HEAD_PT}")
            return model
        except Exception as e:
            print(f"cached head failed to load ({e!r}); retraining with SEED={SEED}")
    print(f"training T2 head from scratch (SEED={SEED}, levers=fusion+zonal+box)")
    allidx = np.arange(len(y))
    tr, va = train_test_split(allidx, test_size=0.15, stratify=y, random_state=SEED)
    model = train_head_t2({k: arrs[k] for k in ("cls", "patches", "txrv", "zones")}, y, tr, va,
                          frozenset({"fusion", "zonal", "box"}))
    torch.save(model.state_dict(), HEAD_PT)
    return model


def _fit_T_and_threshold(model: TBHeadT2, arrs: dict[str, np.ndarray],
                         y: np.ndarray) -> tuple[float, float]:
    """Same val slice as train_tb.py final save: 0.15 stratified split with SEED=0. Fit T on val
    logits, then threshold-at-0.95-sensitivity on val probabilities. Reproducible by seed."""
    allidx = np.arange(len(y))
    _, va = train_test_split(allidx, test_size=0.15, stratify=y, random_state=SEED)
    arrs_head = {k: arrs[k] for k in ("cls", "patches", "txrv", "zones")}
    logits_va = predict_logits_t2(model, arrs_head, va)
    T = fit_temperature(logits_va, y[va])
    probs_va = predict_t2(model, arrs_head, va, T)
    thr = threshold_for_sensitivity(y[va], probs_va, TARGET_SENS)
    return float(T), float(thr)


def _overall_naive_fpr(model: TBHeadT2, arrs: dict[str, np.ndarray], thr: float,
                       T: float) -> dict[str, float]:
    """The 'naive binary' baseline GPT flagged: aggregate FPR over ALL 10k NIH images
    (treating every row as a negative). Computed Brier/ECE on the same set."""
    idx = np.arange(int(arrs["cls"].shape[0]))
    p = predict_t2(model, {k: arrs[k] for k in ("cls", "patches", "txrv", "zones")}, idx, T)
    y0 = np.zeros_like(idx, dtype="int64")
    fp = int((p >= thr).sum())
    n = int(idx.size)
    return {
        "n": n,
        "n_fp": fp,
        "fpr": fp / n,
        "mean_score": float(np.mean(p)),
        "frac_high": float(np.mean(p >= 0.95)),
        "brier": brier(y0, p),
        "ece": float(ece(y0, p)),
    }


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"loading training features: {FEATURES_NPZ}")
    train = _train_features()
    print(f"loading NIH features:      {NIH_NPZ}")
    nih_arrs, findings = _nih_features()
    n_nih = int(nih_arrs["cls"].shape[0])
    print(f"  training rows={len(train['y'])}  pos={int(train['y'].sum())}  "
          f"NIH rows={n_nih}")

    print("\n=== step 1: load or train deployed T2 head (fusion+zonal+box) ===")
    model = _load_or_train_head(train, train["y"])

    print("\n=== step 2: refit temperature + 0.95-sensitivity threshold on val split (SEED=0) ===")
    T, thr = _fit_T_and_threshold(model, train, train["y"])
    # cross-check with the saved JSON if present (audit trail)
    saved = json.loads((DATA / "tb_threshold_t2.json").read_text()) if (DATA / "tb_threshold_t2.json").exists() else {}
    saved_T = saved.get("temperature")
    saved_thr = saved.get("threshold")
    print(f"  refit T={T:.4f}  thr@{TARGET_SENS:.0%}sens={thr:.4f}")
    if saved_T is not None and saved_thr is not None:
        dT = abs(T - float(saved_T))
        dthr = abs(thr - float(saved_thr))
        print(f"  vs saved tb_threshold_t2.json T={saved_T:.4f}  thr={saved_thr:.4f}  "
              f"(|dT|={dT:.4f}, |dthr|={dthr:.4f})")

    print("\n=== step 3: NIH overall FPR (the deceptive naive baseline) ===")
    overall = _overall_naive_fpr(model, nih_arrs, thr, T)
    print(f"  NIH all 10k: n={overall['n']}  n_fp={overall['n_fp']}  FPR={overall['fpr']:.4f}  "
          f"mean_score={overall['mean_score']:.3f}  frac>=.95={overall['frac_high']:.3f}  "
          f"Brier={overall['brier']:.3f}  ECE={overall['ece']:.3f}")

    print("\n=== step 4: per-finding stratified FPR ===")
    subgroups: dict[str, np.ndarray] = {}
    too_small: dict[str, int] = {}
    for f in FINDING_COLS:
        idx = np.where(findings[f] == 1)[0]
        if idx.size < MIN_N:
            too_small[f] = int(idx.size)
            continue
        subgroups[f] = idx
    if too_small:
        print(f"  skipped (n<{MIN_N}): {too_small}")

    rows = stratified_fpr(model, nih_arrs, subgroups, thr=thr, T=T)
    # Decorate with mimicness tag and sort by FPR descending.
    for r in rows:
        r["tag"] = _tag(str(r["name"]))
    rows_sorted = sorted(rows, key=lambda r: -float(r["fpr"]) if not np.isnan(float(r["fpr"])) else 0.0)
    print_stratified_table(rows_sorted, thr, T)

    print("\n=== step 5: ranked summary (mimicness tag) ===")
    print(f"{'finding':>20s} {'tag':>10s} {'n':>5s} {'FPR':>7s} {'95% CI':>16s} "
          f"{'mean':>6s} {'Brier':>7s} {'ECE':>6s}")
    for r in rows_sorted:
        lo, hi = r["fpr_ci"]  # type: ignore[misc]
        ci = f"{lo:.3f}-{hi:.3f}" if not np.isnan(float(lo)) else "n/a"  # type: ignore[arg-type]
        print(f"{r['name']:>20s} {r['tag']:>10s} {r['n']:>5d} {r['fpr']:>7.3f} {ci:>16s} "
              f"{r['mean_score']:>6.3f} {r['brier']:>7.3f} {r['ece']:>6.3f}")

    # Summary stats by tag for the headline.
    by_tag: dict[str, list[float]] = {"TB-mimic": [], "other-abn": [], "easy-neg": []}
    by_tag_brier: dict[str, list[float]] = {"TB-mimic": [], "other-abn": [], "easy-neg": []}
    by_tag_ece: dict[str, list[float]] = {"TB-mimic": [], "other-abn": [], "easy-neg": []}
    for r in rows_sorted:
        t = str(r["tag"])
        by_tag[t].append(float(r["fpr"]))
        by_tag_brier[t].append(float(r["brier"]))
        by_tag_ece[t].append(float(r["ece"]))
    print("\n=== headline: mean FPR / Brier / ECE by tag ===")
    for t in ("TB-mimic", "other-abn", "easy-neg"):
        if by_tag[t]:
            print(f"  {t:>10s} (k={len(by_tag[t])})  FPR={np.mean(by_tag[t]):.3f}  "
                  f"Brier={np.mean(by_tag_brier[t]):.3f}  ECE={np.mean(by_tag_ece[t]):.3f}")

    # Cache the rows as JSON for downstream audit + the experiment-log row.
    out_json = DATA / "nih_stress_rows.json"
    serial = {
        "T": T, "thr": thr,
        "saved_T": float(saved_T) if saved_T is not None else None,
        "saved_thr": float(saved_thr) if saved_thr is not None else None,
        "overall": overall,
        "rows": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in rows_sorted],
        "skipped_small": too_small,
    }
    out_json.write_text(json.dumps(serial, indent=2, default=float))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
