"""Anti-shortcut sanity checks (Adebayo et al., NeurIPS 2018, "Sanity Checks for Saliency Maps").

The load-bearing test for THIS project: LABEL RANDOMIZATION. Retrain the head on permuted TB
labels; the LODO AUROC must collapse to ~chance. If it stays high, the frozen features carry a
non-pathology signal correlated with the (site-structured) data — i.e. a site/scanner SHORTCUT.
This couples to the M12 site-leak canary: high site-separability + above-chance permuted-label
AUC = the head is cheating, regardless of how good the real-label LODO looks.

    python training/sanity.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from train_tb import TBHead, run_lodo, SEED

DATA = Path(__file__).resolve().parents[1] / "data"


def label_randomization_check(arrs: dict, y: np.ndarray, src: np.ndarray, groups,
                              n_perms: int = 3) -> dict:
    """Permute labels, retrain via the SAME LODO harness. Mean permuted-label AUROC must fall < ~0.60.

    A SINGLE permutation + a hard 0.60 cutoff is seed-dependent (one unlucky permutation can tip the
    verdict either way). Run N>=3 permutations with DISTINCT seeds and base PASS/FAIL on the MEAN,
    also reporting the range so a wide spread is visible. N=3 keeps it fast."""
    aucs: list[float] = []
    for k in range(n_perms):
        rng = np.random.default_rng(SEED + k)  # distinct seed per permutation
        y_perm = y.copy()
        rng.shuffle(y_perm)
        auc, _ = run_lodo(arrs, y_perm, src, groups, use_patches=True)
        aucs.append(float(auc))
    mean_auc = float(np.mean(aucs))
    lo, hi = float(np.min(aucs)), float(np.max(aucs))
    return {"n_perms": n_perms,
            "lodo_auc_permuted_labels_each": [round(a, 4) for a in aucs],
            "lodo_auc_permuted_labels_mean": round(mean_auc, 4),
            "lodo_auc_permuted_labels_range": [round(lo, 4), round(hi, 4)],
            "verdict": "PASS (mean <0.60 — no detectable shortcut)" if mean_auc < 0.60
                       else "FAIL (mean >=0.60 — features carry a site/scanner shortcut)"}


def model_randomization_check(model: TBHead, arrs: dict, n: int = 100) -> dict:
    """A faithful attention map must DEPEND on the trained weights; compare to a random head."""
    dev = next(model.parameters()).device
    rand = TBHead(arrs["patches"].shape[2], arrs["cls"].shape[1], arrs["txrv"].shape[1], True).to(dev)

    def attn(m: TBHead, i: int) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(arrs["patches"][i:i + 1]).to(dev)
            a = torch.tanh(m.att.V(x)) * torch.sigmoid(m.att.U(x))
            return torch.softmax(m.att.w(a), dim=1).squeeze().cpu().numpy()

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(arrs["cls"]), min(n, len(arrs["cls"])), replace=False)
    rhos = [spearmanr(attn(model, i), attn(rand, i)).correlation for i in idx]
    rho = float(np.nanmean(rhos))
    return {"attn_spearman_vs_random": rho,
            "verdict": "PASS (<0.3 — map depends on weights)" if abs(rho) < 0.3
                       else "FAIL (map ~invariant to weights — not faithful)"}


def main() -> None:
    d = np.load(DATA / "features.npz", allow_pickle=True)
    arrs = {"cls": d["cls"].astype("float32"), "patches": d["patches"].astype("float32"),
            "txrv": d["txrv"].astype("float32")}
    y = d["y"].astype("int64")
    src = d["source"].astype(str)
    groups = d["group"].astype("int64") if "group" in d.files else None
    print("=== ANTI-SHORTCUT: label randomization (permuted-label LODO AUROC must be < 0.60) ===")
    print(label_randomization_check(arrs, y, src, groups))


if __name__ == "__main__":
    main()
