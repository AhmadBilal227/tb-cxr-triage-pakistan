"""P1 ablation ladder under the FROZEN P0 locked protocol.

Runs the 4-rung additive ladder (one lever per rung), each evaluated under the
SAME frozen T + threshold on the SAME cached eval inputs:

  A. zonal-softor, mixstyle-p=0.0, sources=mont,shen,qatar,tbx11k          (deployed M22/M24 baseline)
  B. zonal-softor, mixstyle-p=0.0, +nih14_normals                          (NEGATIVE-DIVERSITY lever)
  C. soft-attn-pool, mixstyle-p=0.0, +nih14_normals                        (HEAD-SWAP lever)
  D. soft-attn-pool, mixstyle-p=0.5, +nih14_normals                        (full P1: + MixStyle)

For each rung we measure, on BOTH surfaces, under the locked protocol:
  - the HELD-OUT PAKISTANI cohort (data/features_mendeley_pk.npz) — the headline external number,
  - the LODO 80% eval slice (the 4-source OOF, complement of the seed=7 20% calibration split).
reporting AUROC (95% CI), sensitivity + specificity at the FROZEN threshold, and ECE.

DISCIPLINE (the whole point of P0):
  - T=1.5915, thr=0.6105, borderline_low=0.20 loaded from locked_protocol.load_locked_calibration()
    and APPLIED UNCHANGED. NEVER re-fit on Pakistani or LODO-eval. (A "+ local recalibration" Pakistani
    number is reported as a SECONDARY figure on a disjoint CAL_FRAC slice — the PRIMARY gate uses the
    frozen thr.)
  - All PRIOR-FIXED T2 knobs (g_min, zone_prior_init, r0, lambda_box, ...) stay at committed values.
  - The eval surfaces (Pakistani features + the 4-source OOF eval slice) are IDENTICAL across rungs, so
    the deltas are attributable to the single lever each rung adds. Paired bootstrap (same resampled
    indices) gives the delta CI for each rung vs rung A.

The Pakistani cohort enters ONLY as an EVAL surface (cached frozen-backbone features); it NEVER enters
training. NIH normals enter ONLY as TRAINING negatives; they are NOT in either eval surface.

  HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    training/.venv/bin/python training/p1_ablation.py --output data/p1_ablation.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from train_tb import (  # noqa: E402
    SEED, REMIX_SOURCES, _load_arrs, load_nih14_normals,
    train_head_t2, predict_logits_t2, ece,
)
from locked_protocol import (  # noqa: E402
    load_locked_calibration, make_calibration_split, _sigmoid,
)

DATA = REPO / "data"
FULL_LEVERS = frozenset({"fusion", "zonal", "box"})
ORIG_SOURCES = ("montgomery", "shenzhen", "qatar", "tbx11k")
CAL_FRAC_RECAL = 0.20  # disjoint Pakistani slice for the SECONDARY local-recalibration number
BOOTSTRAP_N = 2000

CONFIGS = [
    ("A", "zonal-softor", 0.0, False),
    ("B", "zonal-softor", 0.0, True),
    ("C", "soft-attn-pool", 0.0, True),
    ("D", "soft-attn-pool", 0.5, True),
]


def _load_training_corpus(with_nih: bool):
    """Return (arrs, y, src, groups) for the training corpus. with_nih appends NIH No_Finding
    negatives (fresh group ids past the existing max, so they can't collide with provenance clusters)."""
    d = np.load(DATA / "features.npz", allow_pickle=True)
    arrs = _load_arrs(d)
    y = d["y"].astype("int64")
    src = d["source"].astype(str)
    groups = d["group"].astype("int64") if "group" in d.files else None
    if with_nih:
        ex_arrs, ex_y, ex_src = load_nih14_normals()
        for k in arrs:
            arrs[k] = np.concatenate([arrs[k], ex_arrs[k]], axis=0)
        y = np.concatenate([y, ex_y])
        src = np.concatenate([src, ex_src])
        if groups is not None:
            new_groups = np.arange(len(ex_y)) + int(groups.max()) + 1
            groups = np.concatenate([groups, new_groups])
    return arrs, y, src, groups


def _load_pakistani():
    """Load cached held-out Pakistani features (frozen-backbone read; never in training)."""
    p = DATA / "features_mendeley_pk.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run: extract_features.py --mode external_holdout (P1 prerequisite)."
        )
    d = np.load(p, allow_pickle=True)
    arrs = {
        "cls": d["cls"].astype("float32"),
        "patches": d["patches"].astype("float32"),
        "txrv": d["txrv"].astype("float32"),
        "zones": d["zones"].astype("float32"),
    }
    y = d["y"].astype("int64")
    return arrs, y


def _gen_oof_logits(arrs, y, src, groups, head_kind, mixstyle_p):
    """LODO out-of-fold RAW image logits over the 4 ORIGINAL sources only (the fixed eval surface).
    When a non-held-out fold includes NIH negatives they stay in TRAIN; NIH is never a held-out fold
    and never an eval row. Returns (oof_logit[N_orig], y_orig, src_orig)."""
    orig_mask = np.isin(src, list(ORIG_SOURCES))
    oof = np.full(int(orig_mask.sum()), np.nan, dtype="float64")
    orig_positions = np.where(orig_mask)[0]
    pos_lookup = {int(g): i for i, g in enumerate(orig_positions)}
    for fold_index, ho in enumerate(ORIG_SOURCES):
        te = np.where(src == ho)[0]            # held-out site rows (always one of the 4 originals)
        tr_all = np.where(src != ho)[0]        # everything else INCLUDING NIH negatives
        if groups is not None:
            te_groups = set(int(x) for x in groups[te] if x >= 0)
            leak = np.array([g in te_groups and g >= 0 for g in groups[tr_all]])
            tr_all = tr_all[~leak]
        tr, va = train_test_split(tr_all, test_size=0.2, stratify=y[tr_all], random_state=SEED)
        model = train_head_t2(arrs, y, tr, va, FULL_LEVERS, seed=SEED + fold_index,
                              head_kind=head_kind, mixstyle_p=mixstyle_p)
        logits_te = predict_logits_t2(model, arrs, te)  # RAW logits (locked T applied at eval time)
        for j, abs_idx in enumerate(te):
            oof[pos_lookup[int(abs_idx)]] = logits_te[j]
        flag = "[LEAKAGE-PRONE re-mix]" if ho in REMIX_SOURCES else "[cleaner external]"
        print(f"   OOF holdout={ho:12s} n={len(te):5d} filled {flag}")
    if np.isnan(oof).any():
        raise RuntimeError("OOF logits incomplete — a 4-source fold was not covered")
    return oof, y[orig_mask], src[orig_mask]


def _train_full_and_score_pakistani(arrs, y, src, groups, pk_arrs, head_kind, mixstyle_p):
    """Train ONE full model on the ENTIRE training corpus, return RAW Pakistani logits + the model
    (model returned so config D can extract attention maps)."""
    allidx = np.arange(len(y))
    tr, va = train_test_split(allidx, test_size=0.15, stratify=y, random_state=SEED)
    model = train_head_t2(arrs, y, tr, va, FULL_LEVERS, seed=SEED,
                          head_kind=head_kind, mixstyle_p=mixstyle_p)
    pk_logits = predict_logits_t2(model, pk_arrs, np.arange(len(pk_arrs["cls"])))
    return pk_logits.astype("float64"), model


def _metrics_at_locked(logit, y, T, thr, borderline_low):
    """Apply the FROZEN locked protocol (T, thr, borderline band) to RAW logits. Returns a dict."""
    p = _sigmoid(logit / T)
    auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    preds = (p >= thr).astype(int)
    abstain = ((p >= borderline_low) & (p < thr)).astype(int)
    decided = abstain == 0
    tp = int(((y == 1) & (preds == 1) & decided).sum())
    fn = int(((y == 1) & (preds == 0) & decided).sum())
    fp = int(((y == 0) & (preds == 1) & decided).sum())
    tn = int(((y == 0) & (preds == 0) & decided).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "auroc": auroc,
        "sens": sens,
        "spec": spec,
        "ece": float(ece(y, p)),
        "abstain_rate": float(abstain.mean()),
        "tp_fn_fp_tn_abstain": [tp, fn, fp, tn, int(abstain.sum())],
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }


def _auroc_ci(logit, y, T, n=BOOTSTRAP_N, seed=1):
    p = _sigmoid(logit / T)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], p[b]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def _spec_at_thr(logit, y, T, thr, borderline_low, idx):
    """Specificity at the locked thr on a bootstrap index set (negatives only, abstain counts as not-flagged)."""
    p = _sigmoid(logit[idx] / T)
    yy = y[idx]
    neg = yy == 0
    if neg.sum() == 0:
        return float("nan")
    flagged = (p >= thr) & neg
    return float(1.0 - flagged.sum() / neg.sum())


def _paired_bootstrap_delta(logit_a, logit_x, y, T, thr, borderline_low, n=BOOTSTRAP_N, seed=2):
    """Paired bootstrap of (rung X - rung A) on the SAME resampled eval indices. Returns
    (spec_delta_lo, spec_delta_hi, auroc_delta_lo, auroc_delta_hi) 95% CIs."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    spec_d, auc_d = [], []
    pa = _sigmoid(logit_a / T)
    px = _sigmoid(logit_x / T)
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        yy = y[b]
        if len(np.unique(yy)) < 2:
            continue
        auc_d.append(roc_auc_score(yy, px[b]) - roc_auc_score(yy, pa[b]))
        neg = yy == 0
        if neg.sum() == 0:
            continue
        spec_a = 1.0 - ((pa[b] >= thr) & neg).sum() / neg.sum()
        spec_x = 1.0 - ((px[b] >= thr) & neg).sum() / neg.sum()
        spec_d.append(spec_x - spec_a)
    sd = (float(np.percentile(spec_d, 2.5)), float(np.percentile(spec_d, 97.5))) if spec_d else (float("nan"), float("nan"))
    ad = (float(np.percentile(auc_d, 2.5)), float(np.percentile(auc_d, 97.5))) if auc_d else (float("nan"), float("nan"))
    return sd[0], sd[1], ad[0], ad[1]


def _local_recal_spec(logit, y, T, target_sens=0.95):
    """SECONDARY number: re-fit ONLY the threshold on a disjoint CAL_FRAC Pakistani slice (T stays the
    locked T), eval spec on the rest at that local thr. Mirrors run_lodo's recal reporting. The PRIMARY
    gate still uses the frozen thr; this shows the headroom if a site could be locally recalibrated."""
    p = _sigmoid(logit / T)
    loc = np.arange(len(y))
    if int((y == 1).sum()) < 8 or int((y == 0).sum()) < 8:
        return None
    sens_vals, spec_vals = [], []
    for k in range(20):
        cal, ev = train_test_split(loc, test_size=1 - CAL_FRAC_RECAL, stratify=y, random_state=SEED + k)
        pos = np.sort(p[cal][y[cal] == 1])
        if len(pos) == 0:
            continue
        qi = max(0, int(np.floor((1 - target_sens) * len(pos))))
        thr_local = float(pos[qi])
        pred = (p[ev] >= thr_local).astype(int)
        ye = y[ev]
        sens_vals.append(int(((ye == 1) & (pred == 1)).sum()) / max(int((ye == 1).sum()), 1))
        spec_vals.append(int(((ye == 0) & (pred == 0)).sum()) / max(int((ye == 0).sum()), 1))
    if not spec_vals:
        return None
    return {"sens_median": float(np.median(sens_vals)), "spec_median": float(np.median(spec_vals))}


def _attention_diagnostic(model, pk_arrs, pk_y, locked, out_dir):
    """Config D only: dump SoftAttnPool attention over the held-out Pakistani TB+ cases the deployed
    model MISSED (low tb_prob) — the M24-class atypical-TB failure. Records the attention COM (centre of
    mass) row vs ZonalSoftOR's upper-zone prior, so we can say qualitatively whether the learned head
    attends mid/lower-lung where the consolidation lives. Returns a summary dict."""
    if model.soft_attn_pool is None:
        return {"note": "no soft_attn_pool on this model"}
    p_pk = _sigmoid(
        predict_logits_t2(model, pk_arrs, np.arange(len(pk_arrs["cls"]))).astype("float64") / locked.T
    )
    tb_idx = np.where(pk_y == 1)[0]
    missed = tb_idx[p_pk[tb_idx] < locked.thr_at_95sens]  # confident/near misses on TB+
    # run forward on the missed cases to capture attention (B,64)
    model.eval()
    attn_rows = []
    with torch.no_grad():
        for i in missed[:50]:
            cls = torch.tensor(pk_arrs["cls"][i:i + 1])
            pat = torch.tensor(pk_arrs["patches"][i:i + 1])
            txrv = torch.tensor(pk_arrs["txrv"][i:i + 1])
            from train_tb import DEVICE as _D
            model(cls.to(_D), pat.to(_D), txrv.to(_D))
            attn = model._last_attn[0].cpu().numpy().reshape(8, 8)  # row-major gy*8+gx
            com_row = float((attn.sum(axis=1) * np.arange(8)).sum())  # centre-of-mass row (0=top..7=bottom)
            attn_rows.append(com_row)
    return {
        "n_missed_tb": int(len(missed)),
        "attn_com_row_mean": float(np.mean(attn_rows)) if attn_rows else float("nan"),
        "attn_com_row_note": "0=apex .. 7=base; ZonalSoftOR carries a +0.4 upper-zone (apex) prior. "
                             "A higher COM row on missed atypical-TB = SoftAttnPool attends mid/lower lung.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DATA / "p1_ablation.json"))
    ap.add_argument("--only", default="", help="comma-separated config ids to run (default all A,B,C,D)")
    args = ap.parse_args()

    locked = load_locked_calibration()
    T, thr, blo = locked.T, locked.thr_at_95sens, locked.borderline_low
    print(f"LOCKED protocol (FROZEN, never re-fit): T={T:.4f} thr={thr:.4f} borderline_low={blo:.2f}")
    print(f"  git_sha={locked.git_sha}")

    pk_arrs, pk_y = _load_pakistani()
    print(f"Pakistani holdout: n={len(pk_y)} pos={int((pk_y==1).sum())} neg={int((pk_y==0).sum())}")

    only = set([c.strip() for c in args.only.split(",") if c.strip()]) if args.only else None

    results: dict = {
        "meta": {
            "locked_T": T, "locked_thr": thr, "locked_borderline_low": blo,
            "locked_git_sha": locked.git_sha,
            "frozen_baseline": {
                "pakistani": {"auroc": 0.781, "sens": 0.753, "spec": 0.675},
                "lodo_eval": {"sens": 0.846, "spec": 0.909},
            },
        },
        "configs": {},
    }
    raw = {}  # cid -> {"pk_logit","pk_y","lodo_logit","lodo_y"} for paired bootstrap

    for cid, head_kind, mixp, with_nih in CONFIGS:
        if only and cid not in only:
            continue
        print(f"\n===== CONFIG {cid}: head={head_kind} mixstyle_p={mixp} nih={with_nih} =====")
        arrs, y, src, groups = _load_training_corpus(with_nih)
        print(f"  training corpus: n={len(y)} pos={int((y==1).sum())} neg={int((y==0).sum())} "
              f"sources={ {s:int((src==s).sum()) for s in sorted(set(src.tolist()))} }")

        print("  -- LODO OOF over the 4 original sources (eval surface fixed across rungs) --")
        oof_logit, oof_y, oof_src = _gen_oof_logits(arrs, y, src, groups, head_kind, mixp)
        # the 20% calibration / 80% eval split is deterministic seed=7 on the 4-source OOF
        _, eval_idx = make_calibration_split(oof_logit, oof_y, oof_src)
        lodo_logit = oof_logit[eval_idx]
        lodo_y = oof_y[eval_idx]

        print("  -- full-corpus model -> score held-out Pakistani --")
        pk_logit, model = _train_full_and_score_pakistani(arrs, y, src, groups, pk_arrs, head_kind, mixp)

        lodo_m = _metrics_at_locked(lodo_logit, lodo_y, T, thr, blo)
        lodo_m["auroc_ci"] = list(_auroc_ci(lodo_logit, lodo_y, T))
        pk_m = _metrics_at_locked(pk_logit, pk_y, T, thr, blo)
        pk_m["auroc_ci"] = list(_auroc_ci(pk_logit, pk_y, T))
        pk_m["local_recal_secondary"] = _local_recal_spec(pk_logit, pk_y, T)

        cfg_out = {
            "head_kind": head_kind, "mixstyle_p": mixp, "with_nih": with_nih,
            "pakistani": pk_m, "lodo_eval": lodo_m,
        }
        if cid == "D":
            cfg_out["attention_diagnostic"] = _attention_diagnostic(model, pk_arrs, pk_y, locked, DATA)
        results["configs"][cid] = cfg_out
        raw[cid] = {"pk_logit": pk_logit, "pk_y": pk_y, "lodo_logit": lodo_logit, "lodo_y": lodo_y}

        print(f"  PAKISTANI: AUROC={pk_m['auroc']:.3f} {pk_m['auroc_ci']} sens={pk_m['sens']:.3f} "
              f"spec={pk_m['spec']:.3f} ece={pk_m['ece']:.3f} abstain={pk_m['abstain_rate']:.3f}")
        print(f"  LODO-EVAL: AUROC={lodo_m['auroc']:.3f} sens={lodo_m['sens']:.3f} spec={lodo_m['spec']:.3f} "
              f"ece={lodo_m['ece']:.3f} (n={lodo_m['n']})")

    # paired bootstrap deltas vs A (Pakistani spec + AUROC), only when A was run
    if "A" in raw:
        results["paired_delta_vs_A"] = {}
        for cid in ("B", "C", "D"):
            if cid not in raw:
                continue
            sd_lo, sd_hi, ad_lo, ad_hi = _paired_bootstrap_delta(
                raw["A"]["pk_logit"], raw[cid]["pk_logit"], pk_y, T, thr, blo)
            results["paired_delta_vs_A"][cid] = {
                "pakistani_spec_delta_ci": [sd_lo, sd_hi],
                "pakistani_auroc_delta_ci": [ad_lo, ad_hi],
            }
            print(f"  Δ vs A  {cid}: PK spec Δ95%CI [{sd_lo:+.3f},{sd_hi:+.3f}]  "
                  f"PK AUROC Δ95%CI [{ad_lo:+.3f},{ad_hi:+.3f}]")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
