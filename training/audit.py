"""Shortcut/site-bias audit. The site-leak canary: if a classifier can predict the DATASET
OF ORIGIN from the frozen features well above chance, the data is strongly site-separable and a
TB head can cheat on that signal instead of pathology. High leak + high LODO AUC = distrust.

Uses the SAME features the fusion head sees: Rad-DINO CLS (768) + mean-pooled patch tokens (768)
+ TorchXRayVision (1042). Reports balanced accuracy (robust to the large per-source imbalance).

    python training/audit.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def main() -> None:
    d = np.load(DATA / "features.npz", allow_pickle=True)
    cls = d["cls"].astype("float32")
    txrv = d["txrv"].astype("float32")
    patches = d["patches"].astype("float32")  # [N, 64, 768]
    X = np.concatenate([cls, patches.mean(axis=1), txrv], axis=1)  # what the fusion head sees
    src = d["source"].astype(str)
    codes, names = pd.factorize(src)
    n_sources = len(names)
    chance = 1.0 / n_sources

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=-1))
    acc = cross_val_score(clf, X, codes, cv=5).mean()
    bacc = cross_val_score(clf, X, codes, cv=5, scoring="balanced_accuracy").mean()
    print(f"sources: {list(names)}  (n={len(codes)})")
    print(f"SITE-LEAK canary: dataset-of-origin from features — "
          f"balanced-acc={bacc:.3f}, raw-acc={acc:.3f} (chance={chance:.3f})")
    if bacc > 0.9:
        print(">>> WARNING: features are highly site-separable. The head may be exploiting site "
              "cues (scanner/resolution), not pathology. Trust LODO less; strengthen lung-masking, "
              "resolution-harmonization, and dedup, and verify Grad-CAM is on-lung.")
    elif bacc > 0.7:
        print(">>> site separability MODERATE: some scanner/resolution signature survives "
              "preprocessing. LODO still informative but watch threshold transfer; verify Grad-CAM.")
    else:
        print(">>> site separability LOW: preprocessing harmonized the sources well.")


if __name__ == "__main__":
    main()
