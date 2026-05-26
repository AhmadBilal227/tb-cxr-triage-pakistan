"""Confound audit — is TB-vs-normal in our datasets pathology, or acquisition?

Triggered by the P2.5 discovery (the Mendeley Pakistani dose-response hit 1.000
within-cohort, which a within-PK CLS probe revealed to be a TB/normal acquisition
confound). This generalizes the check to ALL cohorts and quantifies how much of the
LODO 0.92 is a shared cross-dataset acquisition shortcut versus genuine pathology.

Method: linear probe on the rad-dino CLS token (a self-supervised, pathology-LIGHT
global embedding) to separate TB+ from normal:
  - WITHIN each source (patient-disjoint k-fold): if ~1.0, that source's TB and
    normal films differ by a global acquisition signature, not pathology.
  - CROSS-source LODO (train on 3, test on the 4th): how much of the per-source
    confound is COMMON across datasets and therefore transfers.
  - CROSS-site (train on the 4 sources, test on never-seen Pakistani): genuine
    transfer to an independent site.

The diagnostic logic: if within-source >> cross-source >> cross-site, the high
within/cross-source numbers are largely a shared acquisition confound that does NOT
reach a truly new site — i.e. the LODO 0.92 is an optimistic upper bound and the
deployed external 0.78 is the honest field number.

Result (2026-05-26): within-source 0.97-1.00 (all 4 + PK), cross-source-LODO 0.908,
cross-site (PK) 0.604. Confound is pervasive and partly shared; LODO is inflated.
Confound-free TB data (single-pipeline TB+/normal, e.g. PadChest/VinDr) is the fix.
"""
from __future__ import annotations

import statistics as st
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data"


def _probe(Xtr, ytr, Xte) -> np.ndarray:
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def _within_source(X, y, g) -> float | None:
    if len(np.unique(y)) < 2 or (y == 1).sum() < 10 or (y == 0).sum() < 10:
        return None
    try:
        splitter = StratifiedGroupKFold(n_splits=5).split(X, y, g)
    except Exception:
        splitter = StratifiedKFold(5, shuffle=True, random_state=7).split(X, y)
    aucs = []
    for tr, te in splitter:
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(roc_auc_score(y[te], _probe(X[tr], y[tr], X[te])))
    return st.mean(aucs) if aucs else None


def main() -> None:
    d = np.load(DATA / "features.npz", allow_pickle=True)
    y = d["y"].astype(int)
    src = d["source"].astype(str)
    cls = d["cls"].astype("float64")
    grp = d["group"].astype(int) if "group" in d.files else np.arange(len(y))
    sources = sorted(set(src.tolist()))

    print("CLS-probe (pathology-light global embedding) TB+ vs normal:\n")
    print("WITHIN-SOURCE (patient-disjoint 5-fold):")
    for s in sources:
        m = src == s
        auc = _within_source(cls[m], y[m], grp[m])
        print(f"  {s:14s} n={int(m.sum()):5d} TB+={int(y[m].sum()):4d}  AUROC = "
              + (f"{auc:.4f}" if auc is not None else "n/a"))

    print("\nCROSS-SOURCE LODO (train 3 sources, test 4th):")
    lodo = []
    for s in sources:
        te = src == s
        tr = ~te
        if len(np.unique(y[te])) < 2:
            continue
        auc = roc_auc_score(y[te], _probe(cls[tr], y[tr], cls[te]))
        lodo.append(auc)
        print(f"  holdout {s:14s} AUROC = {auc:.4f}")
    print(f"  >>> mean cross-source CLS-LODO = {st.mean(lodo):.4f}")

    pk_path = DATA / "features_mendeley_pk.npz"
    if pk_path.exists():
        dp = np.load(pk_path, allow_pickle=True)
        s_pk = _probe(cls, y, dp["cls"].astype("float64"))
        auc_pk = roc_auc_score(dp["y"].astype(int), s_pk)
        print(f"\nCROSS-SITE (train 4 sources, test never-seen Pakistani): CLS AUROC = {auc_pk:.4f}")

    print("\nREAD: within-source ~1.0 >> cross-site ~0.60 => the TB/normal split is largely")
    print("a per-cohort ACQUISITION confound; the high cross-source LODO (~0.91) is a SHARED")
    print("confound across the assembled datasets that does NOT reach a new site. The LODO 0.92")
    print("is an optimistic upper bound; the deployed external ~0.78 is the honest field number.")
    print("Fix = confound-free data: single-pipeline TB+/normal (PadChest/VinDr), not more model surgery.")


if __name__ == "__main__":
    main()
