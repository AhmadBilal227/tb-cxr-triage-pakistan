"""Train the radiographic-TB-pattern head on cached dual-backbone features.

IMPORTANT (endpoint honesty — gpt-5.5 review + M9 panel): the open-dataset labels are RADIOGRAPHIC,
not microbiological. This head detects a *radiographic pattern associated with TB labels*, NOT
confirmed ACTIVE TB. Every metric below is "vs radiographic reference"; a WHO-TPP / active-TB claim
requires a bacteriologically-confirmed eval tier (TB Portals) — see docs/CASE_STUDY.md M12.

Head = gated-attention (ABMIL) pooling over Rad-DINO patch tokens, fused with CLS + TorchXRayVision,
then a small MLP. Loss = pos_weight BCE (NO label smoothing — it distorts the probability that gets
log-odds-fused). Probabilities are TEMPERATURE-SCALED on a validation split before thresholding.

Honest evaluation (audit-corrected):
  - Leave-one-dataset-out (LODO). Folds from RE-MIXED sources (Qatar/TBX11K aggregate the NLM sets)
    are flagged leakage-prone; a dup-cluster guard excludes train images that near-match a test image.
  - TWO sensitivities per fold: cold-start (frozen train-derived threshold) and + local recalibration.
  - AUROC with a bootstrap 95% CI; sensitivity with a Clopper-Pearson CI; calibration ECE.
  - PPV/NPV and confirmatory-tests-per-flagged-case at deployment prevalence (1%, 2%).
  - Attention ablation: fusion-only vs fusion+patch-attention.

    python training/train_tb.py
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TARGET_SENS = 0.92
BATCH = 256
CAL_FRAC = 0.3          # held-out-site slice used to fit the local-recalibration threshold
BOOTSTRAP_N = 2000      # resamples for the AUROC CI
PREVALENCES = (0.01, 0.02)  # community-screening prevalences for the PPV/NPV table
REMIX_SOURCES = {"qatar", "tbx11k"}  # aggregate NLM/India sets -> LODO fold is leakage-prone
SEED = 0


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
        # narrowed (128) — the small folds (montgomery n=138) don't support a wide head
        self.mlp = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 128), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(128, 1),
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


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single-parameter temperature scaling (Guo et al. 2017): minimize val NLL of sigmoid(z/T).
    NOTE: this calibrates to the validation distribution only — not transferable across sites/
    prevalence. Report ECE and re-fit per deployment site."""
    def nll(T: float) -> float:
        p = 1.0 / (1.0 + np.exp(-logits / max(T, 1e-3)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    r = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(r.x)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += abs(float(p[m].mean()) - float(y[m].mean())) * float(m.mean())
    return e


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n: int = BOOTSTRAP_N) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], p[b]))
    if not aucs:
        return (float("nan"), float("nan"))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def ppv_npv(sens: float, spec: float, prev: float) -> tuple[float, float, float]:
    """PPV, NPV, and confirmatory tests per flagged case at a given prevalence (radiographic endpoint)."""
    tp, fn = sens * prev, (1 - sens) * prev
    tn, fp = spec * (1 - prev), (1 - spec) * (1 - prev)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    tests_per_case = (tp + fp) / tp if tp > 0 else float("nan")
    return ppv, npv, tests_per_case


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
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # NO label smoothing (distorts probability)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    best_ap, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in _batches(len(tr), shuffle=True):
            ix = tr[b]
            g = _gather({k: arrs[k] for k in ("cls", "patches", "txrv")}, ix)
            yt = torch.tensor(y[ix], dtype=torch.float32, device=DEVICE)  # raw targets, no smoothing
            opt.zero_grad()
            loss_fn(model(g["cls"], g["patches"], g["txrv"]), yt).backward()
            opt.step()
        ap = average_precision_score(y[va], predict(model, arrs, va))  # AUPRC: right for imbalance
        if ap > best_ap:
            best_ap, best_state, bad = ap, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_logits(model: TBHead, arrs: dict, idx: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(len(idx), shuffle=False):
            ix = idx[b]
            g = _gather({k: arrs[k] for k in ("cls", "patches", "txrv")}, ix)
            out.append(model(g["cls"], g["patches"], g["txrv"]).cpu().numpy())
    return np.concatenate(out)


def predict(model: TBHead, arrs: dict, idx: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-predict_logits(model, arrs, idx) / temperature))


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = TARGET_SENS):
    fpr, tpr, thr = roc_curve(y, p)
    idx = np.where(tpr >= target)[0]
    if len(idx) == 0:
        return 0.5
    return float(thr[idx[0]])


def _sens_spec(y: np.ndarray, p: np.ndarray, thr: float):
    pred = (p >= thr).astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / n_pos if n_pos else float("nan")
    spec = tn / n_neg if n_neg else float("nan")
    lo, hi = clopper_pearson(tp, n_pos)
    return sens, lo, hi, spec


def _recalibrated(yte: np.ndarray, pte: np.ndarray):
    """Fit threshold on a CAL_FRAC labeled slice of the held-out site, eval on the disjoint rest."""
    if int((yte == 1).sum()) < 8 or int((yte == 0).sum()) < 8:
        return None
    loc = np.arange(len(yte))
    cal, ev = train_test_split(loc, test_size=1 - CAL_FRAC, stratify=yte, random_state=SEED)
    if (yte[cal] == 1).sum() == 0 or (yte[ev] == 1).sum() == 0 or (yte[ev] == 0).sum() == 0:
        return None
    thr_local = threshold_for_sensitivity(yte[cal], pte[cal])
    sens, lo, hi, spec = _sens_spec(yte[ev], pte[ev], thr_local)
    return sens, lo, hi, spec, int((yte[ev] == 1).sum()), spec


def run_lodo(arrs: dict, y: np.ndarray, src: np.ndarray, groups: np.ndarray | None,
             use_patches: bool) -> tuple[float, list[dict]]:
    sources = sorted(set(src.tolist()))
    aucs, ops = [], []
    tag = "fusion+attention" if use_patches else "fusion-only"
    print(f"\n--- LODO ({tag}) — radiographic-TB-pattern, vs radiographic reference ---")
    for ho in sources:
        te = np.where(src == ho)[0]
        tr_all = np.where(src != ho)[0]
        if (y[te] == 1).sum() == 0 or (y[te] == 0).sum() == 0:
            print(f"  holdout={ho:14s} skipped (single-class test)")
            continue
        if groups is not None:
            # dup-cluster guard: drop train images that near-match a held-out image (leak control)
            te_groups = set(int(x) for x in groups[te] if x >= 0)
            leak = np.array([g in te_groups and g >= 0 for g in groups[tr_all]])
            tr_all = tr_all[~leak]
        tr, va = train_test_split(tr_all, test_size=0.2, stratify=y[tr_all], random_state=SEED)
        model = train_head(arrs, y, tr, va, use_patches)
        T = fit_temperature(predict_logits(model, arrs, va), y[va])
        thr = threshold_for_sensitivity(y[va], predict(model, arrs, va, T))  # FROZEN, train-derived
        pte = predict(model, arrs, te, T)
        yte = y[te]
        auc = roc_auc_score(yte, pte)
        a_lo, a_hi = bootstrap_auc_ci(yte, pte)
        cal_ece = ece(yte, pte)
        aucs.append(auc)
        flag = "  [LEAKAGE-PRONE: re-mix source]" if ho in REMIX_SOURCES else "  [cleaner external]"
        s_f, lo_f, hi_f, sp_f = _sens_spec(yte, pte, thr)
        rec = _recalibrated(yte, pte)
        print(f"  holdout={ho:14s} n={len(te):5d} pos={int((yte==1).sum()):4d}  "
              f"AUC={auc:.3f} [95% CI {a_lo:.2f}-{a_hi:.2f}]  ECE={cal_ece:.3f}{flag}")
        print(f"      cold-start  (frozen thr)  sens={s_f:.3f} [95% CI {lo_f:.2f}-{hi_f:.2f}] spec={sp_f:.3f}")
        if rec is not None:
            s_r, lo_r, hi_r, sp_r, npos_e, spec_r = rec
            print(f"      + local recalibration     sens={s_r:.3f} [95% CI {lo_r:.2f}-{hi_r:.2f}] spec={sp_r:.3f}"
                  f"  (eval n_pos={npos_e})")
            ops.append({"source": ho, "sens": s_r, "spec": spec_r, "remix": ho in REMIX_SOURCES})
        else:
            print(f"      + local recalibration     n/a (too few positives to split)")
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    print(f"  >>> mean LODO AUC ({tag}) = {mean_auc:.3f}")
    return mean_auc, ops


def print_prevalence_table(ops: list[dict]) -> None:
    """PPV / confirmatory-tests-per-flagged-case at screening prevalence, using locally-recalibrated
    operating points on the CLEANER (non-re-mix) external folds. Radiographic endpoint."""
    clean = [o for o in ops if not o["remix"] and not np.isnan(o["sens"]) and not np.isnan(o["spec"])]
    if not clean:
        print("\n(no clean external operating points for a prevalence table)")
        return
    sens = float(np.mean([o["sens"] for o in clean]))
    spec = float(np.mean([o["spec"] for o in clean]))
    print(f"\n--- deployment utility (radiographic endpoint; mean of cleaner external folds: "
          f"sens={sens:.2f}, spec={spec:.2f}) ---")
    print(f"{'prevalence':>11s} {'PPV':>7s} {'NPV':>7s} {'tests/flagged case':>20s}")
    for prev in PREVALENCES:
        ppv, npv, tpc = ppv_npv(sens, spec, prev)
        print(f"{prev:>10.0%} {ppv:>7.1%} {npv:>7.3%} {tpc:>20.1f}")
    print("  NOTE: PPV/NPV are for the RADIOGRAPHIC-TB label, not bacteriologically-confirmed active TB.")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # NOTE: torch.use_deterministic_algorithms is intentionally NOT set — it destabilizes some
    # MPS ops (produced NaN logits here). Seeds give run-to-run reproducibility on this backend.
    d = np.load(DATA / "features.npz", allow_pickle=True)
    arrs = {"cls": d["cls"].astype("float32"), "patches": d["patches"].astype("float32"), "txrv": d["txrv"].astype("float32")}
    y = d["y"].astype("int64")
    src = d["source"].astype(str)
    groups = d["group"].astype("int64") if "group" in d.files else None
    print("ENDPOINT: radiographic-TB-pattern (NOT bacteriologically-confirmed active TB). Research preview.")
    print("sources:", {s: int((src == s).sum()) for s in sorted(set(src.tolist()))})
    print(f"total {len(y)}  pos={int((y==1).sum())}  neg={int((y==0).sum())}  "
          f"dup-cluster guard={'ON' if groups is not None else 'OFF (no group column)'}")

    auc_fo, _ = run_lodo(arrs, y, src, groups, use_patches=False)
    auc_fa, ops = run_lodo(arrs, y, src, groups, use_patches=True)
    print(f"\nattention ablation: fusion-only {auc_fo:.3f} -> fusion+attention {auc_fa:.3f} "
          f"(delta {auc_fa-auc_fo:+.3f})")
    print_prevalence_table(ops)

    # final model on all data (with a small val split for early stop + temperature)
    allidx = np.arange(len(y))
    tr, va = train_test_split(allidx, test_size=0.15, stratify=y, random_state=SEED)
    model = train_head(arrs, y, tr, va, use_patches=True)
    T = fit_temperature(predict_logits(model, arrs, va), y[va])
    torch.save(model.state_dict(), DATA / "tb_head.pt")
    thr = threshold_for_sensitivity(y[va], predict(model, arrs, va, T))
    json.dump({"threshold": thr, "temperature": T, "target_sensitivity": TARGET_SENS,
               "endpoint": "radiographic_tb_pattern", "note": "re-fit threshold+temperature per site"},
              open(DATA / "tb_threshold.json", "w"))
    print(f"\nfinal head -> data/tb_head.pt ; temperature={T:.3f} ; threshold@{TARGET_SENS:.0%}sens={thr:.3f} "
          f"(in-sample/optimistic — the honest numbers are the LODO folds above; re-fit per site)")


if __name__ == "__main__":
    main()
