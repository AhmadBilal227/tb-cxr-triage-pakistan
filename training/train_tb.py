"""Train the TB head on cached Rad-DINO features. Reports leave-one-dataset-out (LODO)
generalization (the honest score), then trains a final head on all data and calibrates a
threshold for >=92% sensitivity.

    python training/train_tb.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Head(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.3),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_head(X: np.ndarray, y: np.ndarray, epochs: int = 120, lr: float = 1e-3) -> Head:
    head = Head(X.shape[1]).to(DEVICE)
    Xt = torch.tensor(X, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    pos_weight = torch.tensor([n_neg / n_pos], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        loss_fn(head(Xt), yt).backward()
        opt.step()
    return head


def predict(head: Head, X: np.ndarray) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        return torch.sigmoid(head(torch.tensor(X, device=DEVICE))).cpu().numpy()


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.92):
    fpr, tpr, thr = roc_curve(y, p)
    idx = np.where(tpr >= target)[0]
    if len(idx) == 0:
        return 0.5, float("nan")
    i = idx[0]
    return float(thr[i]), float(1 - fpr[i])


def main() -> None:
    d = np.load(DATA / "features.npz", allow_pickle=True)
    X = d["X"].astype("float32")
    y = d["y"].astype("int64")
    src = d["source"].astype(str)

    sources = sorted(set(src.tolist()))
    print("sources:", {s: int((src == s).sum()) for s in sources})
    print(f"total: {len(y)}  pos={int((y==1).sum())}  neg={int((y==0).sum())}\n")

    aucs = []
    for ho in sources:
        te = src == ho
        tr = ~te
        if (y[te] == 1).sum() == 0 or (y[te] == 0).sum() == 0:
            print(f"LODO holdout={ho:14s} skipped (single-class test set)")
            continue
        head = train_head(X[tr], y[tr])
        p = predict(head, X[te])
        auc = roc_auc_score(y[te], p)
        thr, spec = threshold_for_sensitivity(y[te], p)
        aucs.append(auc)
        print(f"LODO holdout={ho:14s} n={int(te.sum()):5d}  AUC={auc:.3f}  thr@92%sens={thr:.3f}  spec={spec:.3f}")

    if aucs:
        print(f"\n>>> mean LODO AUC = {np.mean(aucs):.3f} (this is the honest generalization number)")

    # Final model on ALL data
    head = train_head(X, y)
    torch.save(head.state_dict(), DATA / "tb_head.pt")
    p = predict(head, X)
    thr, spec = threshold_for_sensitivity(y, p)
    json.dump({"threshold": thr, "target_sensitivity": 0.92}, open(DATA / "tb_threshold.json", "w"))
    print(f"\nfinal head -> data/tb_head.pt ; threshold@92%sens={thr:.3f} (in-sample spec={spec:.3f})")


if __name__ == "__main__":
    main()
