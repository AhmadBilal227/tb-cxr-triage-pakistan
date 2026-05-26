"""P2.5 anti-shortcut check — is the Pakistani dose-response a TB/normal acquisition confound?

The P2.5 dose-response hit within-cohort AUROC 1.000 after adding ~600 in-domain
Pakistani films to training, which read like "the gap is data-closable." A 1.000
AUROC is exactly the signature this project distrusts, so this script runs the
decisive check: can each feature family, on its OWN, separate the Pakistani
`TB Chest X-rays` folder from the `Normal Chest X-rays` folder in within-cohort
patient-disjoint CV?

The CLS token carries NO pathology supervision (a self-supervised DINO embedding).
If it separates the two folders PERFECTLY, the folders differ by a global
ACQUISITION signature, not by pathology — the textbook single-site TB-dataset
confound (Shenzhen/Montgomery). That would make the dose-response a measurement of
shortcut-learnability, not data-closability: at a real deployment site TB+ and
normal share one acquisition pipeline, so the folder-shortcut does not exist.

Result (2026-05-26): CLS 1.0000, TXRV 1.0000, patch-mean 1.0000 — every fold.
Confound CONFIRMED. The P2.5 "~150 images closes the gap" headline is retracted;
the real finding is that the Mendeley Pakistani cohort is a booby-trapped benchmark.
"""
from __future__ import annotations

import statistics as st
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data"
PK = DATA / "features_mendeley_pk.npz"
# CLS being ~1.0 is the smoking gun: a pathology-blind global embedding should NOT
# perfectly separate TB+ from normal unless the two folders differ by acquisition.
CONFOUND_FLAG = 0.98


def _cv_auroc(X: np.ndarray, y: np.ndarray, seed: int = 7) -> tuple[float, list[float]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        s = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(float(roc_auc_score(y[te], s)))
    return st.mean(aucs), aucs


def main() -> None:
    d = np.load(PK, allow_pickle=True)
    y = d["y"].astype(int)
    print(f"Pakistani cohort: n={len(y)}  TB+={int((y == 1).sum())}  normal={int((y == 0).sum())}")
    branches = {
        "CLS (acquisition-heavy, pathology-blind)": d["cls"].astype("float64"),
        "TXRV (pathology logits)": d["txrv"].astype("float64"),
        "patch-mean": d["patches"].astype("float64").mean(axis=1),
    }
    worst_blind = 0.0
    for name, X in branches.items():
        mean_auc, folds = _cv_auroc(X, y)
        print(f"  within-PK 5-fold {name:42s} AUROC = {mean_auc:.4f}  folds={[round(a, 3) for a in folds]}")
        if name.startswith("CLS"):
            worst_blind = mean_auc
    confounded = worst_blind >= CONFOUND_FLAG
    print()
    print(
        f"VERDICT: {'CONFOUND CONFIRMED' if confounded else 'no acquisition confound detected'} "
        f"(pathology-blind CLS within-PK AUROC = {worst_blind:.4f}, flag>= {CONFOUND_FLAG})."
    )
    if confounded:
        print(
            "  The TB/normal folders are separable by a global acquisition signature, "
            "not pathology. The P2.5 dose-response measured shortcut-learnability, not "
            "data-closability. The cohort is valid ONLY for models that never train on it."
        )


if __name__ == "__main__":
    main()
