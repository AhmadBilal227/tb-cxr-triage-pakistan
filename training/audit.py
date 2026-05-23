"""Shortcut/site-bias audit. The site-leak canary: if a classifier can predict the DATASET
OF ORIGIN from the frozen features near-perfectly, the data is strongly site-separable and any
TB head can cheat on that signal instead of pathology. High leak + high LODO AUC = distrust.

    python training/audit.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def main() -> None:
    d = np.load(DATA / "features.npz", allow_pickle=True)
    X = d["X"].astype("float32")
    src = d["source"].astype(str)
    codes, names = pd.factorize(src)
    n_sources = len(names)
    chance = 1.0 / n_sources

    acc = cross_val_score(
        LogisticRegression(max_iter=2000, n_jobs=-1), X, codes, cv=5
    ).mean()
    print(f"sources: {list(names)}")
    print(f"SITE-LEAK canary: dataset-of-origin predictable from features at {acc:.3f} "
          f"(chance={chance:.3f})")
    if acc > 0.9:
        print(">>> WARNING: features are highly site-separable. Lung-masking + stronger dedup "
              "recommended before trusting LODO numbers — the head may be exploiting site cues.")
    else:
        print(">>> site separability moderate; shortcut risk lower (still verify Grad-CAM on-lung).")


if __name__ == "__main__":
    main()
