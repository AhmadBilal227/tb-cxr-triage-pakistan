"""P2.0b — artifact-discrimination battery for the patch-MAX external lift (ZERO new extraction).

P2.0 found that a channel-wise MAX-pool over rad-dino's 64 patch tokens, linear-probed
(trained on the 4 source datasets, evaluated on the held-out Pakistani cohort), reaches
PK AUROC ~0.853 — beating the deployed head (0.781) and the CLS/mean probes (~0.60). BUT
that number was SELECTED by consulting the Pakistani set, and channel-wise MAX is exactly
the readout most prone to an ARTIFACT shortcut: a single hot/high-norm patch from burned-in
text, side/view markers, collimation edges, or exposure cues can dominate the max, and DINO
patch features encode all of those. The 0.853 may be an artifact-detector, not TB signal.

This battery runs five CHEAP checks on the ALREADY-CACHED features to decide whether the
external lift is REAL transferable TB signal or a shortcut. NOTHING here re-extracts features;
everything is computed from data/features.npz (4 sources) + data/features_mendeley_pk.npz (PK).

Probe everywhere = the SAME estimator as P2.0: StandardScaler + L2 LogisticRegression
(C=1.0, class_weight='balanced'), fit ONLY on source data, never on Pakistani.

CHECKS:
  1. LODO-replay (the decisive CLEAN test, NO Pakistani). Leave-one-SOURCE-out: train on 3
     source datasets, test on the held-out 4th, rotate over all 4. Done for max / mean / cls.
     KEY: does max beat mean/cls on held-out SOURCE folds too, or ONLY on Pakistani?
  2. Channel ablation. Rank the max probe's 768 channels by |coef|; drop top 1/5/10/20; refit
     on source; re-eval on PK. Collapse when top-~5 removed => few-channel/outlier-driven.
  3. Alternative pools (outlier sensitivity). Replace raw max with p95-over-tokens, top-8-mean,
     log-sum-exp (low temp), winsorized-max. Re-train + re-eval on PK. If only raw max ~0.85
     and robust pools fall toward ~0.78 or below => outlier/artifact-driven.
  4. Token L2-normalization before pooling. L2-normalize each of the 64 tokens (kills magnitude/
     exposure/quality cues, keeps direction), THEN max-pool, re-train + re-eval on PK. Lift
     vanishes => norm/quality driven (shortcut). Survives => directional (more likely semantic).
  5. Negative-control: predict SITE not TB. Train the same max probe to predict source-dataset
     membership (4-way) and Pakistani-vs-source (binary). High site-separability raises the
     shortcut prior (interpret jointly with 2-4; site-predictability alone does not kill the TB
     claim).

VERDICT is driven primarily by check 1 (LODO-replay) and checks 3-4 (robust pools + token-norm).

  HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    training/.venv/bin/python training/p2_artifact_checks.py \
      --output data/p2_artifact_checks.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

# reuse P2.0's patient-level bootstrap + pAUC + sens@spec helpers verbatim
from p2_linear_probe import (  # noqa: E402
    _patient_bootstrap,
    DEPLOYED_PK_AUROC,
    ORIG_SOURCES,
)
from locked_protocol import make_calibration_split  # noqa: E402

DATA = REPO / "data"
PROBE_C = 1.0
PROBE_KW = dict(C=PROBE_C, max_iter=5000, class_weight="balanced", solver="lbfgs")


# ---------------------------------------------------------------------------
# pooling readouts over the [N, 64, 768] patch-token tensor
# ---------------------------------------------------------------------------
def pool_patches(patches: np.ndarray, kind: str) -> np.ndarray:
    """Pool [N, T, D] patch tokens over T -> [N, D]. All readouts the battery needs.

    raw / robust pools operate per-channel over the T=64 tokens:
      max          channel-wise max (the P2.0 readout under test)
      mean         channel-wise mean
      p95          channel-wise 95th percentile over tokens (robust max)
      top8mean     mean of the 8 largest per channel (robust max)
      lse          log-sum-exp with low temperature tau=4 (smooth max; small tau -> closer to max)
      wmax         winsorized max: clip each channel at its per-channel 98th pct, then max
    """
    p = patches.astype("float64")
    T = p.shape[1]
    if kind == "max":
        return p.max(axis=1)
    if kind == "mean":
        return p.mean(axis=1)
    if kind == "p95":
        return np.percentile(p, 95, axis=1)
    if kind == "top8mean":
        k = min(8, T)
        # partial-sort the top-k per (N, channel) along the token axis
        idx = np.argpartition(p, T - k, axis=1)[:, T - k:, :]
        topk = np.take_along_axis(p, idx, axis=1)
        return topk.mean(axis=1)
    if kind == "lse":
        # log-sum-exp over tokens with LOW temperature tau; subtract per-channel max for stability.
        # tau small -> sharp (near max); tau large -> near mean. We use a small tau so LSE is a
        # genuine SOFT-MAX (an outlier-robustified max), not a soft-mean. rad-dino patch channels
        # have small per-token spread (median std ~0.26, range ~1.2), so tau must be WELL below
        # that for LSE to track max; tau=0.1 keeps it close to (but smoother than) the hard max.
        # lse = max + tau*log(mean(exp((x-max)/tau)))
        tau = 0.1
        m = p.max(axis=1, keepdims=True)
        z = np.exp((p - m) / tau)
        return (m[:, 0, :] + tau * np.log(z.mean(axis=1)))
    if kind == "wmax":
        cap = np.percentile(p, 98, axis=1, keepdims=True)  # per (N, channel) 98th pct over tokens
        return np.minimum(p, cap).max(axis=1)
    raise ValueError(f"unknown pool {kind}")


def cls_feature(d, kind: str) -> np.ndarray:
    """Non-patch readouts: 'cls' = the CLS token (768)."""
    if kind == "cls":
        return d["cls"].astype("float64")
    raise ValueError(f"unknown cls feature {kind}")


# ---------------------------------------------------------------------------
# probe fit/eval seam (fit on source rows only)
# ---------------------------------------------------------------------------
def fit_probe(Xtr: np.ndarray, ytr: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(**PROBE_KW)
    clf.fit(scaler.transform(Xtr), ytr)
    return scaler, clf


def score(scaler: StandardScaler, clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return clf.decision_function(scaler.transform(X))


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def load() -> dict:
    dtr = np.load(DATA / "features.npz", allow_pickle=True)
    dpk = np.load(DATA / "features_mendeley_pk.npz", allow_pickle=True)
    src = dtr["source"].astype(str)
    orig = np.isin(src, list(ORIG_SOURCES))
    out = {
        "patches_tr": dtr["patches"][orig],         # [Ns, 64, 768] float16
        "cls_tr": dtr["cls"][orig],                 # [Ns, 768]
        "y_tr": dtr["y"].astype("int64")[orig],
        "src_tr": src[orig],
        "grp_tr": dtr["group"].astype("int64")[orig],
        "patches_pk": dpk["patches"],
        "cls_pk": dpk["cls"],
        "y_pk": dpk["y"].astype("int64"),
    }
    pid = dpk["patient_id"].astype(str)
    _, out["grp_pk"] = np.unique(pid, return_inverse=True)

    # P2.0 trained its diagnostic patch_max probe on the seed=7 20% cal slice of the source
    # corpus (n=2619), NOT the full corpus. To test the SAME estimator that produced the 0.853
    # external lift, every PK-eval check below fits the probe on this cal slice (fit_rows). The
    # LODO-replay (check 1) is a different, cleaner protocol that legitimately trains on 3 full
    # source datasets, so it does NOT use fit_rows. We also record the full-corpus PK AUROC so a
    # reader can see how much of the 0.853 was an artifact of the small fit population.
    cal_idx, _ = make_calibration_split(
        np.zeros(len(out["y_tr"])), out["y_tr"], out["src_tr"])
    out["fit_rows"] = cal_idx
    return out


# ---------------------------------------------------------------------------
# CHECK 1 — LODO-replay (leave-one-SOURCE-out), no Pakistani
# ---------------------------------------------------------------------------
def check1_lodo_replay(D: dict) -> dict:
    """For pool in {max, mean, cls}: leave-one-source-out, train on 3, test on the 4th.
    Patient-level bootstrap AUROC per fold + a sample-weighted mean across folds."""
    pools = {
        "max": lambda d_patches, d_cls: pool_patches(d_patches, "max"),
        "mean": lambda d_patches, d_cls: pool_patches(d_patches, "mean"),
        "cls": lambda d_patches, d_cls: d_cls.astype("float64"),
    }
    # precompute pooled source matrices once per pool (avoid repooling per fold)
    res: dict = {}
    for name, fn in pools.items():
        Xsrc = fn(D["patches_tr"], D["cls_tr"])
        folds = {}
        fold_auroc, fold_n = [], []
        for held in ORIG_SOURCES:
            test_m = D["src_tr"] == held
            train_m = ~test_m
            if test_m.sum() == 0 or len(np.unique(D["y_tr"][test_m])) < 2:
                folds[held] = {"skipped": True, "reason": "no positives/negatives in fold"}
                continue
            scaler, clf = fit_probe(Xsrc[train_m], D["y_tr"][train_m])
            s = clf.decision_function(scaler.transform(Xsrc[test_m]))
            boot = _patient_bootstrap(
                D["y_tr"][test_m], s, D["grp_tr"][test_m], seed=hash(held) % 10_000)
            folds[held] = boot
            fold_auroc.append(boot["auroc"])
            fold_n.append(boot["n_patients"])
        # macro mean (unweighted over folds) + sample-weighted (by patients) mean
        macro = float(np.mean(fold_auroc)) if fold_auroc else float("nan")
        wmean = (float(np.average(fold_auroc, weights=fold_n))
                 if fold_auroc else float("nan"))
        res[name] = {"folds": folds, "macro_mean_auroc": macro,
                     "patient_weighted_mean_auroc": wmean}
    return res


# ---------------------------------------------------------------------------
# CHECK 2 — channel ablation (drop top-|coef| channels of the max probe, refit, re-eval PK)
# ---------------------------------------------------------------------------
def check2_channel_ablation(D: dict) -> dict:
    fr = D["fit_rows"]
    yfit = D["y_tr"][fr]
    Xsrc = pool_patches(D["patches_tr"], "max")[fr]
    Xpk = pool_patches(D["patches_pk"], "max")
    # base probe: rank channels by |coef| on the STANDARDIZED features (scale-free ranking)
    scaler, clf = fit_probe(Xsrc, yfit)
    coef = clf.coef_.ravel()
    order = np.argsort(-np.abs(coef))  # most important first
    rows = []
    for drop in (0, 1, 5, 10, 20):
        keep = np.ones(Xsrc.shape[1], dtype=bool)
        if drop > 0:
            keep[order[:drop]] = False
        sc, cf = fit_probe(Xsrc[:, keep], yfit)
        s_pk = cf.decision_function(sc.transform(Xpk[:, keep]))
        auroc = roc_auc_score(D["y_pk"], s_pk)
        rows.append({"dropped_top": int(drop), "n_channels": int(keep.sum()),
                     "pk_auroc": float(auroc),
                     "dropped_channel_ids": [int(c) for c in order[:drop]]})
    base = rows[0]["pk_auroc"]
    return {"curve": rows, "base_pk_auroc": float(base),
            "drop5_delta": float(rows[2]["pk_auroc"] - base),
            "drop5_retained_frac": float(rows[2]["pk_auroc"] / base) if base else float("nan")}


# ---------------------------------------------------------------------------
# CHECK 3 — alternative pools (outlier sensitivity), patient-level PK
# ---------------------------------------------------------------------------
def check3_alt_pools(D: dict) -> dict:
    fr = D["fit_rows"]
    yfit = D["y_tr"][fr]
    res = {}
    for kind in ("max", "p95", "top8mean", "lse", "wmax", "mean"):
        Xsrc = pool_patches(D["patches_tr"], kind)[fr]
        Xpk = pool_patches(D["patches_pk"], kind)
        scaler, clf = fit_probe(Xsrc, yfit)
        s_pk = clf.decision_function(scaler.transform(Xpk))
        res[kind] = _patient_bootstrap(D["y_pk"], s_pk, D["grp_pk"], seed=11)
    return res


# ---------------------------------------------------------------------------
# CHECK 4 — token L2-normalization before pooling, then max
# ---------------------------------------------------------------------------
def _l2norm_tokens(patches: np.ndarray) -> np.ndarray:
    p = patches.astype("float64")
    n = np.linalg.norm(p, axis=2, keepdims=True)
    n = np.where(n < 1e-8, 1.0, n)
    return p / n


def check4_token_norm(D: dict) -> dict:
    fr = D["fit_rows"]
    yfit = D["y_tr"][fr]
    res = {}
    # raw max (reference) vs L2-normalized-then-max
    Xsrc_raw = pool_patches(D["patches_tr"], "max")[fr]
    Xpk_raw = pool_patches(D["patches_pk"], "max")
    sc, cf = fit_probe(Xsrc_raw, yfit)
    res["raw_max"] = _patient_bootstrap(
        D["y_pk"], cf.decision_function(sc.transform(Xpk_raw)), D["grp_pk"], seed=11)

    Xsrc_n = _l2norm_tokens(D["patches_tr"]).max(axis=1)[fr]
    Xpk_n = _l2norm_tokens(D["patches_pk"]).max(axis=1)
    sc2, cf2 = fit_probe(Xsrc_n, yfit)
    res["l2norm_then_max"] = _patient_bootstrap(
        D["y_pk"], cf2.decision_function(sc2.transform(Xpk_n)), D["grp_pk"], seed=11)
    res["lift_retained_auroc_delta"] = float(
        res["l2norm_then_max"]["auroc"] - 0.5) / max(1e-9, (res["raw_max"]["auroc"] - 0.5))
    return res


# ---------------------------------------------------------------------------
# CHECK 5 — negative control: predict SITE not TB (max-pool probe)
# ---------------------------------------------------------------------------
def check5_site_prediction(D: dict) -> dict:
    """How separable is SITE from the max-pool readout? Two probes:
      (a) 4-way source membership among the source datasets (5-fold patient-grouped CV accuracy)
      (b) Pakistani-vs-source binary AUROC (train on a balanced source/PK split, held-out eval).
    High separability raises the shortcut prior; interpret jointly with checks 2-4."""
    from sklearn.model_selection import StratifiedGroupKFold

    out: dict = {}

    # (a) 4-way source membership, grouped CV
    Xs = pool_patches(D["patches_tr"], "max")
    src = D["src_tr"]
    src_codes = {s: i for i, s in enumerate(ORIG_SOURCES)}
    y_site = np.array([src_codes[s] for s in src])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    accs, chance = [], float((np.bincount(y_site).max()) / len(y_site))
    for tr, te in sgkf.split(Xs, y_site, groups=D["grp_tr"]):
        sc = StandardScaler().fit(Xs[tr])
        # lbfgs defaults to multinomial for multiclass in current sklearn (the multi_class
        # kwarg was removed); keep it explicit-by-default.
        clf = LogisticRegression(C=PROBE_C, max_iter=3000, class_weight="balanced",
                                 solver="lbfgs")
        clf.fit(sc.transform(Xs[tr]), y_site[tr])
        accs.append(float((clf.predict(sc.transform(Xs[te])) == y_site[te]).mean()))
    out["site_4way"] = {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "majority_class_chance": chance,
        "n_folds": len(accs),
    }

    # (b) Pakistani-vs-source binary. Split BOTH corpora into train/test by patient so the
    # eval is on held-out patients. Label PK=1, source=0. Patient-grouped 50/50 split.
    Xpk = pool_patches(D["patches_pk"], "max")
    rng = np.random.default_rng(0)

    def grouped_half(groups):
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        half = uniq[: len(uniq) // 2]
        is_tr = np.isin(groups, half)
        return is_tr

    src_tr_mask = grouped_half(D["grp_tr"])
    pk_tr_mask = grouped_half(D["grp_pk"])
    Xtr = np.concatenate([Xs[src_tr_mask], Xpk[pk_tr_mask]])
    ytr = np.concatenate([np.zeros(src_tr_mask.sum()), np.ones(pk_tr_mask.sum())])
    Xte = np.concatenate([Xs[~src_tr_mask], Xpk[~pk_tr_mask]])
    yte = np.concatenate([np.zeros((~src_tr_mask).sum()), np.ones((~pk_tr_mask).sum())])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(**PROBE_KW)
    clf.fit(sc.transform(Xtr), ytr)
    s = clf.decision_function(sc.transform(Xte))
    out["pakistani_vs_source"] = {
        "auroc": float(roc_auc_score(yte, s)),
        "n_test": int(len(yte)),
        "note": "held-out-patient AUROC for PK-vs-source membership from the max-pool readout",
    }
    return out


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
def synthesize(c1: dict, c2: dict, c3: dict, c4: dict, c5: dict) -> dict:
    max_lodo = c1["max"]["patient_weighted_mean_auroc"]
    mean_lodo = c1["mean"]["patient_weighted_mean_auroc"]
    cls_lodo = c1["cls"]["patient_weighted_mean_auroc"]
    max_wins_lodo = (max_lodo > mean_lodo + 0.02) and (max_lodo > cls_lodo + 0.02)

    # Robust-MAX family = readouts that approximate max while SUPPRESSING the single hottest
    # token (p95 / top8mean / winsorized-max / low-temp LSE). If these hold near the ceiling, the
    # signal is NOT carried by one lone outlier patch. (Plain `mean` is a separate test of whether
    # averaging works at all, not a robustness test, so it is NOT in this set.)
    robust = [c3[k]["auroc"] for k in ("p95", "top8mean", "lse", "wmax")]
    robust_min = float(np.min(robust))
    robust_holds = robust_min >= DEPLOYED_PK_AUROC - 0.03  # robust pools stay near/above ceiling

    norm_auroc = c4["l2norm_then_max"]["auroc"]
    norm_survives = norm_auroc >= DEPLOYED_PK_AUROC - 0.03

    ablation_graceful = c2["drop5_retained_frac"] >= 0.96  # <4% relative drop on top-5 removal

    # Decision is driven PRIMARILY by check 1 (clean source-LODO, no Pakistani) and checks 3-4
    # (robust pools + token-norm). The KEY question for the P2.0 hypothesis was "is MAX a generally
    # better readout, or only on Pakistani?" If max LOSES to mean/cls on clean source-LODO, the
    # "max generalizes better" claim is false and the PK lift is a PK-specific / selection effect.
    max_is_best_readout_anywhere = max_wins_lodo  # max provably better than mean AND cls on LODO
    signals_for = sum([max_is_best_readout_anywhere, robust_holds, norm_survives, ablation_graceful])

    if max_is_best_readout_anywhere and robust_holds and norm_survives:
        verdict = "REAL"
    elif (not robust_holds) and (not norm_survives):
        verdict = "ARTIFACT"
    elif not max_is_best_readout_anywhere:
        # max is NOT a better readout on clean held-out SOURCE folds -> the specific 0.853 MAX lift
        # on Pakistani is not a general "max transfers better" effect. Whether any TB signal
        # survives at all is read off robust pools + token-norm.
        if robust_holds and norm_survives:
            verdict = ("MIXED — the MAX-readout claim does NOT survive (mean/CLS beat max on clean "
                       "source-LODO); some directional signal survives robust pooling + token-norm, "
                       "but the MAX-specific external lift is a PK-specific / selection effect, not "
                       "a general property of max-pooling")
        else:
            verdict = ("ARTIFACT-LEANING — MAX loses to mean/CLS on clean source-LODO AND the lift "
                       "degrades under robust pooling/token-norm; the 0.853 looks PK-specific / "
                       "outlier-driven, not transferable TB signal")
    else:
        verdict = "MIXED"

    return {
        "verdict": verdict,
        "signals_for_real_count": int(signals_for),
        "decisive_check1_lodo": {
            "max_weighted": max_lodo, "mean_weighted": mean_lodo, "cls_weighted": cls_lodo,
            "max_wins_on_source_lodo": bool(max_wins_lodo),
        },
        "check3_robust_pools": {"min_robust_pk_auroc": robust_min,
                                "robust_holds_near_ceiling": bool(robust_holds)},
        "check4_token_norm": {"l2norm_then_max_pk_auroc": norm_auroc,
                              "survives": bool(norm_survives)},
        "check2_ablation": {"drop5_retained_frac": c2["drop5_retained_frac"],
                            "graceful": bool(ablation_graceful)},
        "check5_site": {
            "site_4way_acc": c5["site_4way"]["cv_accuracy_mean"],
            "site_4way_chance": c5["site_4way"]["majority_class_chance"],
            "pk_vs_source_auroc": c5["pakistani_vs_source"]["auroc"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DATA / "p2_artifact_checks.json"))
    args = ap.parse_args()

    D = load()
    print(f"source corpus n={len(D['y_tr'])} pos={(D['y_tr']==1).sum()} "
          f"neg={(D['y_tr']==0).sum()}")
    print(f"PK holdout    n={len(D['y_pk'])} pos={(D['y_pk']==1).sum()} "
          f"neg={(D['y_pk']==0).sum()}")
    print(f"probe fit population = seed=7 cal slice (n={len(D['fit_rows'])}), matching P2.0\n")

    # fit-population sensitivity: the same raw-max probe trained on the FULL source corpus
    # vs the seed=7 cal slice that produced P2.0's 0.853. A big gap => the 0.853 partly rode
    # the small fit sample (a fragility signal independent of pooling).
    Xmax = pool_patches(D["patches_tr"], "max")
    Xmax_pk = pool_patches(D["patches_pk"], "max")
    sc_f, cf_f = fit_probe(Xmax, D["y_tr"])
    auroc_full = float(roc_auc_score(D["y_pk"], cf_f.decision_function(sc_f.transform(Xmax_pk))))
    sc_c, cf_c = fit_probe(Xmax[D["fit_rows"]], D["y_tr"][D["fit_rows"]])
    auroc_cal = float(roc_auc_score(D["y_pk"], cf_c.decision_function(sc_c.transform(Xmax_pk))))
    fit_pop = {"raw_max_pk_auroc_full_corpus": auroc_full,
               "raw_max_pk_auroc_cal_slice": auroc_cal,
               "p2_0_reported": 0.8535,
               "note": ("P2.0 fit on the seed=7 cal slice (n=2619); this battery matches that for "
                        "PK-eval checks. Full-corpus fit is a reference. A large cal-vs-full gap "
                        "means the 0.853 partly rode the small fit sample.")}
    print(f"fit-population sensitivity: cal-slice PK AUROC={auroc_cal:.3f} (P2.0=0.853) | "
          f"full-corpus PK AUROC={auroc_full:.3f}\n")

    print("===== CHECK 1: LODO-replay (leave-one-SOURCE-out, NO Pakistani) =====")
    c1 = check1_lodo_replay(D)
    for pool in ("max", "mean", "cls"):
        print(f"  {pool:5s} macro={c1[pool]['macro_mean_auroc']:.3f} "
              f"wmean={c1[pool]['patient_weighted_mean_auroc']:.3f}")
        for s in ORIG_SOURCES:
            f = c1[pool]["folds"][s]
            if f.get("skipped"):
                print(f"       {s:11s} SKIPPED ({f['reason']})")
            else:
                print(f"       {s:11s} AUROC={f['auroc']:.3f} {f['auroc_ci']} "
                      f"(n_pat={f['n_patients']}, pos={f['n_pos']})")
    print()

    print("===== CHECK 2: channel ablation (drop top-|coef| max channels) =====")
    c2 = check2_channel_ablation(D)
    for r in c2["curve"]:
        print(f"  drop_top={r['dropped_top']:2d} n_ch={r['n_channels']:3d} "
              f"PK_AUROC={r['pk_auroc']:.3f}")
    print(f"  drop5 retained frac={c2['drop5_retained_frac']:.3f}\n")

    print("===== CHECK 3: alternative pools (outlier sensitivity), PK =====")
    c3 = check3_alt_pools(D)
    for k in ("max", "p95", "top8mean", "lse", "wmax", "mean"):
        print(f"  {k:9s} PK_AUROC={c3[k]['auroc']:.3f} {c3[k]['auroc_ci']}")
    print()

    print("===== CHECK 4: token L2-norm before max, PK =====")
    c4 = check4_token_norm(D)
    print(f"  raw_max          PK_AUROC={c4['raw_max']['auroc']:.3f} {c4['raw_max']['auroc_ci']}")
    print(f"  l2norm_then_max  PK_AUROC={c4['l2norm_then_max']['auroc']:.3f} "
          f"{c4['l2norm_then_max']['auroc_ci']}\n")

    print("===== CHECK 5: negative control — predict SITE not TB =====")
    c5 = check5_site_prediction(D)
    print(f"  site 4-way CV acc={c5['site_4way']['cv_accuracy_mean']:.3f} "
          f"(chance={c5['site_4way']['majority_class_chance']:.3f})")
    print(f"  PK-vs-source AUROC={c5['pakistani_vs_source']['auroc']:.3f}\n")

    synthesis = synthesize(c1, c2, c3, c4, c5)
    print("=" * 70)
    print(f"VERDICT: {synthesis['verdict']}  (signals-for-real={synthesis['signals_for_real_count']}/4)")
    print("=" * 70)

    out = {
        "meta": {
            "deployed_pk_auroc": DEPLOYED_PK_AUROC,
            "probe": f"StandardScaler + LogisticRegression(L2, C={PROBE_C}, balanced)",
            "fit_only_on_source": True,
            "pk_eval_fit_population": "seed=7 cal slice (matches P2.0)",
            "n_source": int(len(D["y_tr"])),
            "n_fit_rows": int(len(D["fit_rows"])),
            "n_pk": int(len(D["y_pk"])),
            "fit_population_sensitivity": fit_pop,
        },
        "check1_lodo_replay": c1,
        "check2_channel_ablation": c2,
        "check3_alt_pools": c3,
        "check4_token_norm": c4,
        "check5_site_prediction": c5,
        "synthesis": synthesis,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
