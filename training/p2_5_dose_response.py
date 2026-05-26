"""P2.5 Pakistani dose-response under the preregistered protocol.

Tests the pivotal alternative to the P0->P2 finding (image-only model surgery on a frozen Rad-DINO
does NOT close the Pakistani external gap -> domain shift): does adding IN-DOMAIN Pakistani data to
TRAINING close the gap, and HOW MUCH is needed?

PREREGISTERED (docs/baselines/2026-05-26-p2.5-preregistration.md, committed before any run):
  - FIXED Pakistani test set = 40% of patients, stratified by label, seed=7 (one-image-per-patient
    confirmed -> patient-disjoint == image-disjoint). NEVER trained on at any dose.
  - Training pool = the other 60%. Nested doses {0, 150, 300, 600, ALL} drawn seed=7, stratified to
    the pool's TB+/normal ratio, nested so larger doses contain smaller ones.
  - Each dose trains ONLY the deployed zonal-soft-OR head (levers {fusion,zonal,box}, PRIOR-FIXED
    knobs, seed fixed) on the 4 source datasets + d Pakistani rows. Pakistani rows: source
    'mendeley_pk_train', fresh group ids past the existing max, grid_label=zeros/has_box=False
    (synthesized exactly as load_nih14_normals()).
  - PER-CONFIG calibration: T + thr@95sens fitted ONCE per dose on THAT model's 4-source LODO OOF
    seed=7 cal slice (make_calibration_split). NEVER reused across configs; NEVER fit on the PK test.
  - Metrics: PK-test patient-bootstrap AUROC (95% CI) + spec@sens=0.90 + sens@spec=0.70 + ECE;
    4-source LODO AUROC + sens@thr (the FORGETTING check).

LEAD CAVEAT: once PK is in training the held-out PK test set is WITHIN-COHORT (same-site), NOT a fresh
external site. This measures DATA-CLOSABILITY + dose-response, NOT new-site generalization.

  HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    training/.venv/bin/python training/p2_5_dose_response.py --output data/p2.5_dose_response.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from train_tb import (  # noqa: E402
    SEED, REMIX_SOURCES, _load_arrs, train_head_t2, predict_logits_t2, ece,
)
from locked_protocol import (  # noqa: E402
    make_calibration_split, _sigmoid, _fit_temperature, _threshold_for_sensitivity,
)

DATA = REPO / "data"
FULL_LEVERS = frozenset({"fusion", "zonal", "box"})
ORIG_SOURCES = ("montgomery", "shenzhen", "qatar", "tbx11k")
PK_TEST_FRAC = 0.40          # preregistered FIXED test fraction
PK_SPLIT_SEED = 7            # preregistered split seed
DOSE_SEED = 7                # preregistered nested-dose draw seed
DOSES = [0, 150, 300, 600, "ALL"]
BOOTSTRAP_N = 2000
TARGET_SENS_CAL = 0.95       # thr@95sens fitted on the 4-source OOF cal slice


def _load_source_corpus():
    """The 4-source training corpus (montgomery/shenzhen/qatar/tbx11k) + group ids."""
    d = np.load(DATA / "features.npz", allow_pickle=True)
    arrs = _load_arrs(d)
    y = d["y"].astype("int64")
    src = d["source"].astype(object)
    groups = d["group"].astype("int64")
    return arrs, y, src, groups


def _load_pakistani():
    d = np.load(DATA / "features_mendeley_pk.npz", allow_pickle=True)
    arrs = {
        "cls": d["cls"].astype("float32"),
        "patches": d["patches"].astype("float32"),
        "txrv": d["txrv"].astype("float32"),
        "zones": d["zones"].astype("float32"),
        # PK rows carry no box supervision (has_box all False, grid_label all zeros) -> synthesize
        # exactly as load_nih14_normals() does, so the box-lever masking treats them as no-box.
        "grid_label": np.zeros((len(d["y"]), 8, 8), dtype="float32"),
        "has_box": np.zeros(len(d["y"]), dtype=bool),
    }
    y = d["y"].astype("int64")
    return arrs, y


def _pk_test_pool_split(pk_y):
    """FIXED 40% stratified test set (seed=7). Returns (pool_idx, test_idx) into the PK arrays."""
    idx = np.arange(len(pk_y))
    pool_idx, test_idx = train_test_split(idx, test_size=PK_TEST_FRAC, stratify=pk_y,
                                          random_state=PK_SPLIT_SEED)
    assert len(set(pool_idx.tolist()) & set(test_idx.tolist())) == 0, "PK pool/test overlap"
    return np.sort(pool_idx), np.sort(test_idx)


def _nested_dose_rows(pk_y, pool_idx, dose):
    """Nested, stratified, seed=7 selection of `dose` pool rows. Larger doses contain smaller ones
    because the per-class permutation is drawn once from a fixed seed and we take prefixes."""
    pool_pos = pool_idx[pk_y[pool_idx] == 1]
    pool_neg = pool_idx[pk_y[pool_idx] == 0]
    rng = np.random.default_rng(DOSE_SEED)
    perm_pos = pool_pos[rng.permutation(len(pool_pos))]
    perm_neg = pool_neg[rng.permutation(len(pool_neg))]
    pr = float((pk_y[pool_idx] == 1).mean())  # pool TB+ fraction
    if dose == "ALL":
        sel = np.concatenate([perm_pos, perm_neg])
        return np.sort(sel)
    npos = min(int(round(dose * pr)), len(perm_pos))
    nneg = min(dose - npos, len(perm_neg))
    sel = np.concatenate([perm_pos[:npos], perm_neg[:nneg]])
    return np.sort(sel)


def _build_training_corpus(src_arrs, src_y, src_src, src_groups, pk_arrs, pk_y, pk_rows):
    """4-source corpus + the selected Pakistani training-pool rows (source 'mendeley_pk_train',
    fresh group ids past the existing max). Returns (arrs, y, src, groups)."""
    if len(pk_rows) == 0:
        return ({k: v.copy() for k, v in src_arrs.items()}, src_y.copy(),
                src_src.copy(), src_groups.copy())
    pk_sub = {k: pk_arrs[k][pk_rows] for k in src_arrs}
    arrs = {k: np.concatenate([src_arrs[k], pk_sub[k]], axis=0) for k in src_arrs}
    y = np.concatenate([src_y, pk_y[pk_rows]])
    src = np.concatenate([src_src, np.array(["mendeley_pk_train"] * len(pk_rows), dtype=object)])
    new_groups = np.arange(len(pk_rows)) + int(src_groups.max()) + 1
    groups = np.concatenate([src_groups, new_groups])
    return arrs, y, src, groups


def _gen_oof_logits(arrs, y, src, groups):
    """LODO out-of-fold RAW image logits over the 4 ORIGINAL sources only (the fixed eval surface).
    mendeley_pk_train rows, when present, stay in TRAIN for every fold (PK is never a held-out fold
    and never an OOF eval row). Returns (oof_logit, oof_y, oof_src) over the 4-source rows."""
    orig_mask = np.isin(src.astype(str), list(ORIG_SOURCES))
    orig_positions = np.where(orig_mask)[0]
    oof = np.full(len(orig_positions), np.nan, dtype="float64")
    pos_lookup = {int(g): i for i, g in enumerate(orig_positions)}
    for fold_index, ho in enumerate(ORIG_SOURCES):
        te = np.where(src.astype(str) == ho)[0]       # held-out site rows (one of the 4 originals)
        tr_all = np.where(src.astype(str) != ho)[0]   # everything else INCLUDING mendeley_pk_train
        te_groups = set(int(x) for x in groups[te] if x >= 0)
        leak = np.array([g in te_groups and g >= 0 for g in groups[tr_all]])
        tr_all = tr_all[~leak]
        tr, va = train_test_split(tr_all, test_size=0.2, stratify=y[tr_all], random_state=SEED)
        model = train_head_t2(arrs, y, tr, va, FULL_LEVERS, seed=SEED + fold_index,
                              head_kind="zonal-softor", mixstyle_p=0.0)
        logits_te = predict_logits_t2(model, arrs, te)
        for j, abs_idx in enumerate(te):
            oof[pos_lookup[int(abs_idx)]] = logits_te[j]
        flag = "[LEAKAGE-PRONE re-mix]" if ho in REMIX_SOURCES else "[cleaner external]"
        print(f"    OOF holdout={ho:12s} n={len(te):5d} filled {flag}")
    if np.isnan(oof).any():
        raise RuntimeError("OOF logits incomplete -- a 4-source fold was not covered")
    oy = y[orig_mask]
    osrc = src.astype(str)[orig_mask]
    return oof, oy, osrc


def _fit_per_config_calibration(oof_logit, oof_y, oof_src):
    """Fit T + thr@95sens on THAT model's 4-source OOF seed=7 cal slice (NEVER the PK test). Returns
    (T, thr, n_cal, n_eval, eval_idx)."""
    cal_idx, eval_idx = make_calibration_split(oof_logit, oof_y, oof_src)
    T = _fit_temperature(oof_logit[cal_idx], oof_y[cal_idx])
    p_cal = _sigmoid(oof_logit[cal_idx] / T)
    thr = _threshold_for_sensitivity(oof_y[cal_idx], p_cal, target=TARGET_SENS_CAL)
    return T, thr, int(len(cal_idx)), int(len(eval_idx)), eval_idx


def _train_full_and_score_pk(arrs, y, pk_arrs, pk_test_idx):
    """Train ONE deployed head on the ENTIRE training corpus (4 sources + the dose's PK rows), score
    the FIXED PK test set. Returns RAW PK-test logits."""
    allidx = np.arange(len(y))
    tr, va = train_test_split(allidx, test_size=0.15, stratify=y, random_state=SEED)
    model = train_head_t2(arrs, y, tr, va, FULL_LEVERS, seed=SEED,
                          head_kind="zonal-softor", mixstyle_p=0.0)
    pk_logit = predict_logits_t2(model, pk_arrs, pk_test_idx)
    return pk_logit.astype("float64")


def _auroc_ci(p, y, n=BOOTSTRAP_N, seed=1):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], p[b]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _spec_at_sens(y, p, target_sens=0.90):
    """Specificity at the threshold that yields >= target sensitivity (PK test's own ROC)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target_sens)[0]
    if len(ok) == 0:
        return float("nan"), float("nan")
    i = ok[0]
    return float(1.0 - fpr[i]), float(thr[i])


def _sens_at_spec(y, p, target_spec=0.70):
    """Sensitivity at the threshold that yields >= target specificity (PK test's own ROC)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where((1.0 - fpr) >= target_spec)[0]
    if len(ok) == 0:
        return float("nan"), float("nan")
    i = ok[-1]  # the most permissive thr still meeting the spec floor -> highest sens
    return float(tpr[i]), float(thr[i])


def _lodo_sens_at_thr(oof_logit, oof_y, eval_idx, T, thr):
    """Sensitivity on the 4-source LODO eval slice at the per-config fitted thr@95sens."""
    p = _sigmoid(oof_logit[eval_idx] / T)
    ye = oof_y[eval_idx]
    pred = (p >= thr).astype(int)
    n_pos = int((ye == 1).sum())
    tp = int(((ye == 1) & (pred == 1)).sum())
    return tp / n_pos if n_pos else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DATA / "p2.5_dose_response.json"))
    ap.add_argument("--only", default="", help="comma-separated doses to run (default all)")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None

    print("P2.5 PAKISTANI DOSE-RESPONSE (within-cohort holdout; NOT new-site generalization)")
    print("ENDPOINT: radiographic-TB-pattern (NOT bacteriologically-confirmed). Research preview.")

    src_arrs, src_y, src_src, src_groups = _load_source_corpus()
    print(f"4-source corpus: n={len(src_y)} pos={int((src_y==1).sum())} neg={int((src_y==0).sum())} "
          f"sources={ {s:int((src_src.astype(str)==s).sum()) for s in ORIG_SOURCES} }")

    pk_arrs, pk_y = _load_pakistani()
    pool_idx, test_idx = _pk_test_pool_split(pk_y)
    print(f"PK: total={len(pk_y)} | FIXED test n={len(test_idx)} "
          f"(pos={int((pk_y[test_idx]==1).sum())} neg={int((pk_y[test_idx]==0).sum())}) | "
          f"pool n={len(pool_idx)} (pos={int((pk_y[pool_idx]==1).sum())} "
          f"neg={int((pk_y[pool_idx]==0).sum())})")
    assert len(set(pool_idx.tolist()) & set(test_idx.tolist())) == 0

    pk_test_y = pk_y[test_idx]

    results = {
        "meta": {
            "preregistration": "docs/baselines/2026-05-26-p2.5-preregistration.md",
            "caveat": ("WITHIN-COHORT (same-site) holdout once PK is in training; measures "
                       "DATA-CLOSABILITY + dose-response, NOT new-site generalization."),
            "pk_test_n": int(len(test_idx)),
            "pk_test_pos": int((pk_test_y == 1).sum()),
            "pk_test_neg": int((pk_test_y == 0).sum()),
            "pk_pool_n": int(len(pool_idx)),
            "doses": [str(d) for d in DOSES],
            "head": "zonal-softor", "levers": ["fusion", "zonal", "box"],
            "calibration": "per-config T + thr@95sens on the 4-source OOF seed=7 cal slice",
            "bootstrap_n": BOOTSTRAP_N,
            "lodo_baseline_sens_floor": 0.846,
        },
        "doses": {},
    }

    for dose in DOSES:
        dkey = str(dose)
        if only and dkey not in only:
            continue
        pk_rows = _nested_dose_rows(pk_y, pool_idx, dose)
        print(f"\n===== DOSE {dkey}: +{len(pk_rows)} PK train rows "
              f"(pos={int((pk_y[pk_rows]==1).sum())} neg={int((pk_y[pk_rows]==0).sum())}) =====")
        arrs, y, src, groups = _build_training_corpus(
            src_arrs, src_y, src_src, src_groups, pk_arrs, pk_y, pk_rows)
        print(f"  training corpus n={len(y)} pos={int((y==1).sum())} neg={int((y==0).sum())}")

        print("  -- 4-source LODO OOF (per-config calibration surface; forgetting check) --")
        oof_logit, oof_y, oof_src = _gen_oof_logits(arrs, y, src, groups)
        T, thr, n_cal, n_eval, eval_idx = _fit_per_config_calibration(oof_logit, oof_y, oof_src)
        print(f"  per-config calibration: T={T:.4f} thr@95sens={thr:.4f} "
              f"(n_cal={n_cal} n_eval={n_eval})")

        # LODO eval-slice metrics at the per-config calibration
        p_lodo = _sigmoid(oof_logit[eval_idx] / T)
        lodo_y = oof_y[eval_idx]
        lodo_auroc = float(roc_auc_score(lodo_y, p_lodo))
        lodo_sens = _lodo_sens_at_thr(oof_logit, oof_y, eval_idx, T, thr)

        print("  -- full-corpus deployed head -> score FIXED PK test --")
        pk_logit = _train_full_and_score_pk(arrs, y, pk_arrs, test_idx)
        pk_p = _sigmoid(pk_logit / T)  # per-config T

        pk_auroc = float(roc_auc_score(pk_test_y, pk_p))
        pk_ci = list(_auroc_ci(pk_p, pk_test_y))
        spec90, thr_spec90 = _spec_at_sens(pk_test_y, pk_p, 0.90)
        sens70, thr_sens70 = _sens_at_spec(pk_test_y, pk_p, 0.70)
        pk_ece = float(ece(pk_test_y, pk_p))

        results["doses"][dkey] = {
            "pk_train_rows": int(len(pk_rows)),
            "pk_train_pos": int((pk_y[pk_rows] == 1).sum()),
            "pk_train_neg": int((pk_y[pk_rows] == 0).sum()),
            "calibration": {"T": T, "thr_at_95sens": thr, "n_cal": n_cal, "n_eval": n_eval},
            "pakistani_test": {
                "auroc": pk_auroc, "auroc_ci": pk_ci,
                "spec_at_sens0.90": spec90, "sens_at_spec0.70": sens70,
                "ece": pk_ece,
                "n": int(len(test_idx)),
            },
            "lodo_eval": {"auroc": lodo_auroc, "sens_at_thr": lodo_sens, "n": int(len(eval_idx))},
        }

        print(f"  PK-TEST: AUROC={pk_auroc:.3f} {[round(x,3) for x in pk_ci]} "
              f"spec@sens0.90={spec90:.3f} sens@spec0.70={sens70:.3f} ece={pk_ece:.3f}")
        print(f"  LODO-EVAL: AUROC={lodo_auroc:.3f} sens@thr={lodo_sens:.3f} "
              f"(floor 0.846; {'OK' if lodo_sens >= 0.80 else 'CRATER WATCH'})")

        # incremental write so a crash mid-sweep keeps completed doses
        Path(args.output).write_text(json.dumps(results, indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")

    # dose-response summary table
    print("\n=== DOSE-RESPONSE (within-cohort; NOT new-site generalization) ===")
    print(f"{'dose':>6} {'PKrows':>7} | {'PK AUROC':>9} {'95% CI':>17} {'spec@.90':>9} "
          f"{'sens@.70':>9} {'ECE':>6} | {'LODO AUC':>9} {'LODO sens':>10}")
    for dose in DOSES:
        dkey = str(dose)
        if dkey not in results["doses"]:
            continue
        r = results["doses"][dkey]
        pk = r["pakistani_test"]; lo = r["lodo_eval"]
        ci = f"[{pk['auroc_ci'][0]:.3f},{pk['auroc_ci'][1]:.3f}]"
        print(f"{dkey:>6} {r['pk_train_rows']:>7} | {pk['auroc']:>9.3f} {ci:>17} "
              f"{pk['spec_at_sens0.90']:>9.3f} {pk['sens_at_spec0.70']:>9.3f} {pk['ece']:>6.3f} | "
              f"{lo['auroc']:>9.3f} {lo['sens_at_thr']:>10.3f}")


if __name__ == "__main__":
    main()
