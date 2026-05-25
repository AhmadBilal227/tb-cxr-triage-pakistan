"""How does the deployed T2 head perform on 100 unseen images?

Uses the cached LODO out-of-fold logits — each prediction was produced by a model
fold that did NOT include that image in training, so these are truly unseen-per-fold.
SEED=42 sampling (distinct from SEED=0 used for training splits) so the 100 are
independent of any training-time split.

Run: training/.venv/bin/python training/unseen_100.py
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42
N = 100


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    # 1) load cached OOF logits + deployed calibration
    oof = np.load(DATA / "image_oof_logits.npz", allow_pickle=True)
    logit = oof["image_logit"].astype("float64")
    label = oof["label"].astype("int64")
    source = oof["source"]
    cal = json.loads((DATA / "tb_threshold_t2.json").read_text())
    T = float(cal["temperature"])
    thr = float(cal["threshold"])
    print(f"Loaded {len(logit)} OOF preds | T={T:.4f} thr={thr:.4f} | sources: "
          f"{dict(zip(*np.unique(source, return_counts=True)))}\n")

    # 2) sample 100 stratified-ish (best-effort) by label so we get both classes
    rng = np.random.default_rng(SEED)
    pos_idx = np.where(label == 1)[0]
    neg_idx = np.where(label == 0)[0]
    n_pos = min(50, len(pos_idx))
    n_neg = N - n_pos
    pick = np.concatenate([
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=n_neg, replace=False),
    ])
    rng.shuffle(pick)

    # 3) apply deployed calibration -> verdicts
    p = sigmoid(logit[pick] / T)
    y = label[pick]
    src = source[pick]
    BORDERLINE_LOW = 0.35  # matches sequelaeEscalation.ts BORDERLINE_LOW
    verdict = np.where(p >= thr, "TB",
                       np.where((p >= BORDERLINE_LOW) & (p < thr), "ABSTAIN", "NO_TB"))

    # 4) confusion (ignoring ABSTAIN — those are referrals, not classified)
    decided = verdict != "ABSTAIN"
    pred_pos = (verdict == "TB") & decided
    pred_neg = (verdict == "NO_TB") & decided
    tp = int(((y == 1) & pred_pos).sum())
    fn = int(((y == 1) & pred_neg).sum())
    fp = int(((y == 0) & pred_pos).sum())
    tn = int(((y == 0) & pred_neg).sum())
    n_abstain = int((~decided).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    acc_decided = (tp + tn) / max(tp + tn + fp + fn, 1)

    print(f"=== 100 unseen images @ deployed operating point (thr={thr:.3f}, T={T:.3f}) ===\n")
    print(f"Verdict distribution:")
    for v, c in zip(*np.unique(verdict, return_counts=True)):
        print(f"  {v:10s} {int(c):3d}")
    print(f"\nConfusion (excluding {n_abstain} ABSTAIN/referrals):")
    print(f"            pred TB   pred NO_TB")
    print(f"  true TB     {tp:3d}        {fn:3d}")
    print(f"  true NO_TB  {fp:3d}        {tn:3d}")
    print(f"\nMetrics on the {tp + tn + fp + fn} DECIDED cases (the ABSTAIN-rule is *features*, not failures):")
    print(f"  sensitivity (catching TB)     = {tp}/{tp+fn} = {sens:.3f}")
    print(f"  specificity (clearing healthy) = {tn}/{tn+fp} = {spec:.3f}")
    print(f"  accuracy                       = {tp+tn}/{tp+tn+fp+fn} = {acc_decided:.3f}")
    print(f"  abstain rate                   = {n_abstain}/100 = {n_abstain/100:.2f}")

    # 5) by-source breakdown so the user sees which folds the unseen 100 came from
    print(f"\nBy source (n / pos / sens / spec on decided):")
    for s in sorted(set(src.tolist())):
        m = src == s
        ys = y[m]; vs = verdict[m]; dm = vs != "ABSTAIN"
        tps = int(((ys == 1) & (vs == "TB") & dm).sum())
        fns = int(((ys == 1) & (vs == "NO_TB") & dm).sum())
        fps = int(((ys == 0) & (vs == "TB") & dm).sum())
        tns = int(((ys == 0) & (vs == "NO_TB") & dm).sum())
        ns = int(m.sum()); pos = int((ys == 1).sum())
        ss = tps / max(tps + fns, 1) if (tps + fns) else float("nan")
        sp = tns / max(tns + fps, 1) if (tns + fps) else float("nan")
        print(f"  {s:12s} n={ns:3d} pos={pos:3d}  sens={ss:.2f}  spec={sp:.2f}")


if __name__ == "__main__":
    main()
