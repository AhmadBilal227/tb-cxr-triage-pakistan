"""Localization faithfulness of the T2 box-evidence map vs TBX11K active-TB boxes (Blueprint §2A).

This is the REAL anti-shortcut / localization metric — a head can get a high image AUROC off a
site/scanner cue while pointing nowhere near the lesion. We score the 8x8 evidence map against the
rasterized grid_label on the BOXED TBX11K images, held out LODO-style (the head is trained on the
NON-TBX11K sources, so the boxed images are a genuine holdout — the box loss never saw them):

  - POINTING GAME (Zhang et al.): hit = the argmax evidence cell lands on a positive grid cell.
  - mIoU: IoU between {evidence >= tau} and the positive grid cells, per image, averaged. tau is the
    per-image evidence threshold that the model itself would flag (we use Otsu-free: top-k where k =
    the image's positive-cell count, a label-free shape match — reported alongside a fixed-0.5 mIoU).
  - 1000-bootstrap 95% CIs over images.
  - RANDOM-ATTENTION FLOOR: the same metrics for a uniformly-random evidence map (same seed), so the
    numbers are read as "above a random floor", not as an absolute localization-accuracy claim.

HONEST CEILING (Blueprint §0.4, §4): the grid is 8x8 (pooled from Rad-DINO's 37x37) — coarse for small
TB foci. mIoU here is a RELATIVE tracking number vs the random floor, not an absolute localization claim.
The 37x37 native grid is the real fix.

    python training/localize.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from train_tb import (SEED, _load_arrs, evidence_maps_t2, fit_temperature, predict_logits_t2,
                      threshold_for_sensitivity, train_head_t2)

DATA = Path(__file__).resolve().parents[1] / "data"
HOLDOUT = "tbx11k"          # the only source with boxes -> held out so its boxed images are a true holdout
BOOTSTRAP_N = 1000
FULL_LEVERS = frozenset({"fusion", "zonal", "box"})


def _pointing_hit(ev: np.ndarray, gl: np.ndarray) -> bool:
    """Hit = the argmax evidence cell is a positive grid cell."""
    flat = ev.reshape(-1)
    j = int(flat.argmax())
    return bool(gl.reshape(-1)[j] > 0.5)


def _iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = float((pred_mask & gt_mask).sum())
    union = float((pred_mask | gt_mask).sum())
    return inter / union if union > 0 else float("nan")


def _miou_topk(ev: np.ndarray, gl: np.ndarray) -> float:
    """IoU at a LABEL-FREE threshold: flag the top-k evidence cells where k = the image's #positive
    cells. A shape match that doesn't peek at the label magnitude (only its count, a disclosed proxy)."""
    gt = gl.reshape(-1) > 0.5
    k = int(gt.sum())
    if k == 0:
        return float("nan")
    flat = ev.reshape(-1)
    order = np.argsort(-flat)[:k]
    pred = np.zeros_like(gt, dtype=bool)
    pred[order] = True
    return _iou(pred, gt)


def _miou_fixed(ev: np.ndarray, gl: np.ndarray, tau: float = 0.5) -> float:
    gt = gl.reshape(-1) > 0.5
    if gt.sum() == 0:
        return float("nan")
    return _iou(ev.reshape(-1) >= tau, gt)


def _bootstrap_ci(vals: np.ndarray, n: int = BOOTSTRAP_N) -> tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    means = [float(rng.choice(vals, len(vals), replace=True).mean()) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _scores(ev_maps: np.ndarray, gls: np.ndarray) -> dict:
    hits = np.array([_pointing_hit(ev_maps[i], gls[i]) for i in range(len(ev_maps))], dtype=float)
    miou_k = np.array([_miou_topk(ev_maps[i], gls[i]) for i in range(len(ev_maps))], dtype=float)
    miou_f = np.array([_miou_fixed(ev_maps[i], gls[i]) for i in range(len(ev_maps))], dtype=float)
    return {
        "hit": float(np.nanmean(hits)), "hit_ci": _bootstrap_ci(hits),
        "miou_topk": float(np.nanmean(miou_k)), "miou_topk_ci": _bootstrap_ci(miou_k),
        "miou_fixed": float(np.nanmean(miou_f)), "miou_fixed_ci": _bootstrap_ci(miou_f),
        "n": int(len(ev_maps)),
    }


def main() -> None:
    import torch
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    d = np.load(DATA / "features.npz", allow_pickle=True)
    arrs = _load_arrs(d)
    y = d["y"].astype("int64")
    src = d["source"].astype(str)
    groups = d["group"].astype("int64") if "group" in d.files else None

    te = np.where(src == HOLDOUT)[0]
    tr_all = np.where(src != HOLDOUT)[0]
    if groups is not None:
        te_groups = set(int(x) for x in groups[te] if x >= 0)
        leak = np.array([g in te_groups and g >= 0 for g in groups[tr_all]])
        tr_all = tr_all[~leak]
    tr, va = train_test_split(tr_all, test_size=0.2, stratify=y[tr_all], random_state=SEED)

    print(f"LOCALIZATION FAITHFULNESS — T2 box-evidence map vs {HOLDOUT} active-TB boxes")
    print(f"  head trained on NON-{HOLDOUT} sources (box loss never saw these images — true holdout)")
    # train the full T2 head on the non-TBX11K data; the box lever is supervised on NO boxes here
    # (montgomery/shenzhen/qatar have none) so the box scorer is driven only by the image label on
    # train — yet we still test whether its evidence map LOCALIZES on the held-out boxed images.
    print(f"  NOTE: the train sources have ZERO boxes, so the evidence map is learned WITHOUT direct")
    print(f"  cell supervision in this LODO fold — this is the hardest, most honest localization test.")
    model = train_head_t2(arrs, y, tr, va, FULL_LEVERS, seed=SEED)

    # held-out BOXED positives only (the localization target)
    hb = arrs["has_box"][te].astype(bool)
    box_idx = te[hb]
    print(f"  held-out boxed images: {len(box_idx)}")
    ev_maps = evidence_maps_t2(model, arrs, box_idx)
    gls = (arrs["grid_label"][box_idx] > 0.5).astype(np.float32)

    model_s = _scores(ev_maps, gls)
    # random-attention floor: same #cells, uniform random evidence
    rng = np.random.default_rng(SEED)
    rand_maps = rng.random(ev_maps.shape)
    rand_s = _scores(rand_maps, gls)

    def fmt(s: dict, label: str) -> None:
        print(f"\n  {label} (n={s['n']}):")
        print(f"    pointing-game hit-rate = {s['hit']:.3f}  [95% CI {s['hit_ci'][0]:.3f}-{s['hit_ci'][1]:.3f}]")
        print(f"    mIoU (top-k, label-free)= {s['miou_topk']:.3f}  "
              f"[95% CI {s['miou_topk_ci'][0]:.3f}-{s['miou_topk_ci'][1]:.3f}]")
        print(f"    mIoU (fixed tau=0.5)    = {s['miou_fixed']:.3f}  "
              f"[95% CI {s['miou_fixed_ci'][0]:.3f}-{s['miou_fixed_ci'][1]:.3f}]")

    fmt(model_s, "T2 evidence map")
    fmt(rand_s, "RANDOM-attention floor")
    print(f"\n  >>> hit-rate lift over random floor = {model_s['hit'] - rand_s['hit']:+.3f}")
    print(f"  >>> mIoU (top-k) lift over random floor = {model_s['miou_topk'] - rand_s['miou_topk']:+.3f}")
    print("  HONEST CAVEAT: 8x8 grid (pooled from 37x37) is coarse — mIoU is a RELATIVE tracking number")
    print("  vs the random floor, NOT an absolute localization-accuracy claim. 37x37 native grid is the fix.")


if __name__ == "__main__":
    main()
