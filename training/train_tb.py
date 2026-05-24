"""Train the TB head on cached dual-backbone features.

Head = gated-attention (ABMIL) pooling over Rad-DINO patch tokens, fused with the CLS vector
and TorchXRayVision features, then an MLP. Loss = label-smoothed BCE with pos_weight.

Honest evaluation (panel-corrected):
  - Leave-one-dataset-out (LODO); within the training sources a TRAIN-ONLY validation split
    drives early stopping AND threshold selection (never the held-out fold).
  - TWO sensitivities per held-out source, because a fixed threshold does not transfer:
      (1) COLD-START — realized sensitivity at the FROZEN (train-derived) threshold, no local
          adaptation. The worst case: deploy at a new site with zero local labels.
      (2) + LOCAL RECALIBRATION — fit the threshold on a small labeled slice of the held-out
          site, evaluate on the DISJOINT rest. Simulates the designed deployment step (re-fit
          per site). Both with Clopper-Pearson 95% CIs. AUC (threshold-free) is reported once.
  - Ablation: fusion-only (CLS+TXRV) vs fusion+patch-attention, to substantiate the attention lift.

    python training/train_tb.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TARGET_SENS = 0.92
LABEL_SMOOTH = 0.05
BATCH = 256
CAL_FRAC = 0.3  # fraction of a held-out site used as the local calibration slice (rest = eval)


class GatedAttention(nn.Module):
    """Ilse et al. gated-attention MIL pooling over a token set [B,T,d] -> [B,d]."""

    def __init__(self, d: int = 768, h: int = 128):
        super().__init__()
        self.V = nn.Linear(d, h)
        self.U = nn.Linear(d, h)
        self.w = nn.Linear(h, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        a = torch.softmax(self.w(a), dim=1)  # [B,T,1]
        return (a * x).sum(dim=1)


class TBHead(nn.Module):
    def __init__(self, d_tok: int, d_cls: int, d_txrv: int, use_patches: bool = True):
        super().__init__()
        self.use_patches = use_patches
        self.att = GatedAttention(d_tok) if use_patches else None
        d = (d_tok if use_patches else 0) + d_cls + d_txrv
        self.mlp = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256, 1),
        )

    def forward(self, cls: torch.Tensor, patches: torch.Tensor, txrv: torch.Tensor) -> torch.Tensor:
        parts = [cls, txrv]
        if self.use_patches and self.att is not None:
            parts.insert(0, self.att(patches))
        return self.mlp(torch.cat(parts, dim=1)).squeeze(-1)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _batches(n: int, shuffle: bool):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, BATCH):
        yield idx[s : s + BATCH]


def _gather(arrs: dict, idx: np.ndarray) -> dict:
    return {k: torch.tensor(v[idx]).to(DEVICE) for k, v in arrs.items()}


def train_head(arrs: dict, y: np.ndarray, tr: np.ndarray, va: np.ndarray, use_patches: bool,
               max_epochs: int = 80, patience: int = 8) -> TBHead:
    model = TBHead(arrs["patches"].shape[2], arrs["cls"].shape[1], arrs["txrv"].shape[1], use_patches).to(DEVICE)
    n_pos = max(1, int((y[tr] == 1).sum()))
    n_neg = max(1, int((y[tr] == 0).sum()))
    pos_weight = torch.tensor([n_neg / n_pos], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    best_ap, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in _batches(len(tr), shuffle=True):
            ix = tr[b]
            g = _gather({k: arrs[k] for k in ("cls", "patches", "txrv")}, ix)
            yt = torch.tensor(y[ix], dtype=torch.float32, device=DEVICE)
            yt = yt * (1 - LABEL_SMOOTH) + (1 - yt) * LABEL_SMOOTH  # label smoothing
            opt.zero_grad()
            loss_fn(model(g["cls"], g["patches"], g["txrv"]), yt).backward()
            opt.step()
        ap = average_precision_score(y[va], predict(model, arrs, va))
        if ap > best_ap:
            best_ap, best_state, bad = ap, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: TBHead, arrs: dict, idx: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(len(idx), shuffle=False):
            ix = idx[b]
            g = _gather({k: arrs[k] for k in ("cls", "patches", "txrv")}, ix)
            out.append(torch.sigmoid(model(g["cls"], g["patches"], g["txrv"])).cpu().numpy())
    return np.concatenate(out)


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = TARGET_SENS):
    fpr, tpr, thr = roc_curve(y, p)
    idx = np.where(tpr >= target)[0]
    if len(idx) == 0:
        return 0.5
    return float(thr[idx[0]])


def _sens_spec(y: np.ndarray, p: np.ndarray, thr: float):
    """Sensitivity (+ Clopper-Pearson CI) and specificity at a given threshold."""
    pred = (p >= thr).astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / n_pos if n_pos else float("nan")
    spec = tn / n_neg if n_neg else float("nan")
    lo, hi = clopper_pearson(tp, n_pos)
    return sens, lo, hi, spec


def _recalibrated(yte: np.ndarray, pte: np.ndarray):
    """Fit the threshold on a CAL_FRAC labeled slice of the held-out site, evaluate on the
    disjoint rest. Non-leaky (cal/eval disjoint); models 'deploy with a small local labeled
    set'. Returns None if the site has too few positives to split meaningfully."""
    if int((yte == 1).sum()) < 8 or int((yte == 0).sum()) < 8:
        return None
    loc = np.arange(len(yte))
    cal, ev = train_test_split(loc, test_size=1 - CAL_FRAC, stratify=yte, random_state=0)
    if (yte[cal] == 1).sum() == 0 or (yte[ev] == 1).sum() == 0 or (yte[ev] == 0).sum() == 0:
        return None
    thr_local = threshold_for_sensitivity(yte[cal], pte[cal])  # fit on local cal slice
    sens, lo, hi, spec = _sens_spec(yte[ev], pte[ev], thr_local)
    return sens, lo, hi, spec, int((yte[ev] == 1).sum())


def run_lodo(arrs: dict, y: np.ndarray, src: np.ndarray, use_patches: bool) -> float:
    sources = sorted(set(src.tolist()))
    aucs = []
    tag = "fusion+attention" if use_patches else "fusion-only"
    print(f"\n--- LODO ({tag}) ---")
    for ho in sources:
        te = np.where(src == ho)[0]
        tr_all = np.where(src != ho)[0]
        if (y[te] == 1).sum() == 0 or (y[te] == 0).sum() == 0:
            print(f"  holdout={ho:14s} skipped (single-class test)")
            continue
        tr, va = train_test_split(tr_all, test_size=0.2, stratify=y[tr_all], random_state=0)
        model = train_head(arrs, y, tr, va, use_patches)
        thr = threshold_for_sensitivity(y[va], predict(model, arrs, va))  # FROZEN, train-derived
        pte = predict(model, arrs, te)
        yte = y[te]
        auc = roc_auc_score(yte, pte)
        aucs.append(auc)
        # (1) cold-start: frozen train-derived threshold, no local adaptation
        s_f, lo_f, hi_f, sp_f = _sens_spec(yte, pte, thr)
        # (2) with local recalibration: threshold fit on a labeled slice of the site, eval on rest
        rec = _recalibrated(yte, pte)
        print(f"  holdout={ho:14s} n={len(te):5d} AUC={auc:.3f}")
        print(f"      cold-start  (frozen thr)  sens={s_f:.3f} [95% CI {lo_f:.2f}-{hi_f:.2f}] spec={sp_f:.3f}")
        if rec is not None:
            s_r, lo_r, hi_r, sp_r, npos_e = rec
            print(f"      + local recalibration     sens={s_r:.3f} [95% CI {lo_r:.2f}-{hi_r:.2f}] spec={sp_r:.3f}"
                  f"  (eval n_pos={npos_e})")
        else:
            print(f"      + local recalibration     n/a (too few positives to split)")
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    print(f"  >>> mean LODO AUC ({tag}) = {mean_auc:.3f}")
    return mean_auc


def main() -> None:
    np.random.seed(0)   # reproducible head training (batch shuffles) + cal/eval splits
    torch.manual_seed(0)  # ~0.02 AUC run-to-run variance observed unseeded; pin it
    d = np.load(DATA / "features.npz", allow_pickle=True)
    arrs = {"cls": d["cls"].astype("float32"), "patches": d["patches"].astype("float32"), "txrv": d["txrv"].astype("float32")}
    y = d["y"].astype("int64")
    src = d["source"].astype(str)
    print("sources:", {s: int((src == s).sum()) for s in sorted(set(src.tolist()))})
    print(f"total {len(y)}  pos={int((y==1).sum())}  neg={int((y==0).sum())}")

    auc_fo = run_lodo(arrs, y, src, use_patches=False)
    auc_fa = run_lodo(arrs, y, src, use_patches=True)
    print(f"\nattention ablation: fusion-only {auc_fo:.3f} -> fusion+attention {auc_fa:.3f} "
          f"(delta {auc_fa-auc_fo:+.3f})")

    # final model on all data (with a small val split for early stop)
    allidx = np.arange(len(y))
    tr, va = train_test_split(allidx, test_size=0.15, stratify=y, random_state=0)
    model = train_head(arrs, y, tr, va, use_patches=True)
    torch.save(model.state_dict(), DATA / "tb_head.pt")
    thr = threshold_for_sensitivity(y[va], predict(model, arrs, va))
    json.dump({"threshold": thr, "target_sensitivity": TARGET_SENS}, open(DATA / "tb_threshold.json", "w"))
    # NOTE: threshold is in-sample/optimistic; the honest number is the frozen-threshold LODO
    # sensitivity above. Re-fit this threshold on a held-out deployment set in production.
    print(f"\nfinal head -> data/tb_head.pt ; threshold@{TARGET_SENS:.0%}sens={thr:.3f} (re-fit per site)")


if __name__ == "__main__":
    main()
