"""50-image sensitivity + specificity check on unseen data.

TB+ side (25): cached LODO out-of-fold predictions — each prediction was made by
a model fold that did NOT see that image in training. Stratified across the four
training sources (montgomery/shenzhen/qatar/tbx11k). These are the honest
"unseen-by-predicting-fold" TB+ numbers; the only TB-positive labels we have
that satisfy a leave-one-out unseen-ness criterion.

TB- side (25): NIH ChestX-ray14, fully disjoint from all training sources (NIH
Clinical Center). Stratified across No_Finding (easy negatives) and the M18
worst-case mimic classes (Fibrosis, Nodule, Consolidation, Mass). The deployed
TriageEngine is run LIVE on each image — these are truly unseen.

Both halves apply the same deployed calibration: T=1.5915, thr=0.6105.

Run: training/.venv/bin/python training/test_50_sens_spec.py
"""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path

import numpy as np

from triage_core import TriageEngine  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NIH = DATA / "raw" / "nih14"
SEED = 43  # distinct from training SEED=0 and prior tests
BORDERLINE_LOW = 0.35


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def verdict_from(p: float, thr: float) -> str:
    if p >= thr:
        return "TB"
    if p >= BORDERLINE_LOW:
        return "ABSTAIN"
    return "NO_TB"


def sample_tb_positives(seed: int) -> list[dict]:
    """25 TB+ OOF predictions, stratified by source. Returns rows with
    {source, oof_logit, oof_tb_prob, verdict}."""
    oof = np.load(DATA / "image_oof_logits.npz", allow_pickle=True)
    logit = oof["image_logit"].astype("float64")
    label = oof["label"].astype("int64")
    source = oof["source"]

    cal = json.loads((DATA / "tb_threshold_t2.json").read_text())
    T = float(cal["temperature"])
    thr = float(cal["threshold"])

    rng = np.random.default_rng(seed)
    per_src = {"montgomery": 4, "shenzhen": 8, "qatar": 8, "tbx11k": 5}
    picks: list[dict] = []
    for src, n in per_src.items():
        idx_pool = np.where((source == src) & (label == 1))[0]
        chosen = rng.choice(idx_pool, size=n, replace=False)
        for j, idx in enumerate(chosen, 1):
            p = float(sigmoid(logit[idx] / T))
            picks.append({
                "kind": "TB+",
                "label": "TB",
                "id": f"{src}#{int(idx)}",
                "source": src,
                "findings": "(TB-positive)",
                "logit": float(logit[idx]),
                "tb_prob": p,
                "s_inactive": None,
                "predicted": verdict_from(p, thr),
            })
    return picks, T, thr


def sample_nih_negatives(seed: int) -> list[dict]:
    """25 NIH images stratified by finding type. All true label = NO_TB. Live engine run."""
    rows = list(csv.DictReader((NIH / "nih14_findings.csv").open()))
    rng = random.Random(seed)
    per_class = [
        ("No_Finding", 12),
        ("Fibrosis", 5),
        ("Nodule", 4),
        ("Consolidation", 3),
        ("Mass", 1),
    ]
    picks: list[dict] = []
    for col, n in per_class:
        pool = [r for r in rows if r.get(col) == "1"]
        for r in rng.sample(pool, n):
            picks.append({
                "kind": "TB-",
                "label": "NO_TB",
                "id": r["filename"],
                "source": "nih14",
                "findings": (r.get("findings") or "No Finding") or "No Finding",
            })
    return picks


def main() -> None:
    print("Loading validated model (Rad-DINO + TXRV + TBHeadT2 + sequelae)…")
    engine = TriageEngine()

    print("Building TB+ side (25 from LODO OOF cache — unseen by predicting fold)…")
    pos_rows, T, thr = sample_tb_positives(SEED)
    print(f"  T={T:.4f}  thr={thr:.4f}")

    print("Building TB- side (25 NIH images — running live engine)…")
    neg_rows = sample_nih_negatives(SEED)
    for r in neg_rows:
        img_path = NIH / r["id"]
        with img_path.open("rb") as f:
            result = engine.run(f.read())
        r["tb_prob"] = result.tb_prob
        r["s_inactive"] = result.s_inactive
        r["logit"] = result.tb_logit
        r["predicted"] = result.verdict.upper()

    rows = pos_rows + neg_rows

    print()
    print(f"{'#':<4}{'kind':<6}{'src':<14}{'true':<8}{'pred':<10}{'tb_prob':>10}{'s_inact':>10}  findings")
    print("-" * 110)
    correct = 0
    sens_tp = 0
    sens_fn = 0
    sens_abs = 0
    spec_tn = 0
    spec_fp = 0
    spec_abs = 0
    for i, r in enumerate(rows, 1):
        ok = r["predicted"] == r["label"]
        if ok:
            correct += 1
        marker = "✓" if ok else ("·" if r["predicted"] == "ABSTAIN" else "✗")
        si = f'{r["s_inactive"]:>10.4f}' if r["s_inactive"] is not None else f'{"n/a":>10}'
        print(f"{i:<4}{r['kind']:<6}{r['source']:<14}{r['label']:<8}{r['predicted']:<10}{r['tb_prob']:>10.4f}{si}  {r['findings'][:50]}  {marker}")
        if r["label"] == "TB":
            if r["predicted"] == "TB":
                sens_tp += 1
            elif r["predicted"] == "ABSTAIN":
                sens_abs += 1
            else:
                sens_fn += 1
        else:
            if r["predicted"] == "NO_TB":
                spec_tn += 1
            elif r["predicted"] == "ABSTAIN":
                spec_abs += 1
            else:
                spec_fp += 1

    print("-" * 110)
    print()
    total_decided = sens_tp + sens_fn + spec_tn + spec_fp
    print(f"OVERALL: {correct}/{len(rows)} correct  ({correct / len(rows):.0%})")
    print()
    print(f"  TB+ side (25 unseen-fold OOF predictions):")
    print(f"    caught (TP)        : {sens_tp:>3}/25")
    print(f"    missed (FN)        : {sens_fn:>3}/25  ← safety-critical")
    print(f"    abstained          : {sens_abs:>3}/25")
    decided_tb = sens_tp + sens_fn
    sens = sens_tp / max(decided_tb, 1)
    print(f"    sensitivity        : {sens_tp}/{decided_tb} = {sens:.3f}  (excluding abstains)")
    print()
    print(f"  TB- side (25 NIH images run live):")
    print(f"    cleared (TN)       : {spec_tn:>3}/25")
    print(f"    false-positive (FP): {spec_fp:>3}/25  ← spec cost")
    print(f"    abstained          : {spec_abs:>3}/25")
    decided_neg = spec_tn + spec_fp
    spec = spec_tn / max(decided_neg, 1)
    print(f"    specificity        : {spec_tn}/{decided_neg} = {spec:.3f}  (excluding abstains)")
    print()
    print(f"  Decided overall: {total_decided}/{len(rows)} ({(sens_abs + spec_abs)}/{len(rows)} abstained → 'requires re-read')")
    print()
    print("Endpoint: radiographic TB pattern. Per-site recalibration recommended for")
    print("any deployment site (per tb_threshold_t2.json).")


if __name__ == "__main__":
    main()
