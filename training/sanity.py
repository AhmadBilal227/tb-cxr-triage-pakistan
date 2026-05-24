"""Anti-shortcut sanity checks (Adebayo et al., NeurIPS 2018, "Sanity Checks for Saliency Maps").
Flags:
  --fast   Run ONLY the cheap CPU checks (conditional source-reliance + model randomization).
           Default runs the heavy label-randomization (3x LODO retrain) as well.


The load-bearing test for THIS project: LABEL RANDOMIZATION. Retrain the head on permuted TB
labels; the LODO AUROC must collapse to ~chance. If it stays high, the frozen features carry a
non-pathology signal correlated with the (site-structured) data — i.e. a site/scanner SHORTCUT.
This couples to the M12 site-leak canary: high site-separability + above-chance permuted-label
AUC = the head is cheating, regardless of how good the real-label LODO looks.

WHY THE NAIVE SITE-LEAK CANARY ASKS THE WRONG QUESTION (GPT strategy review). The old canary trains
a classifier to predict SOURCE from features. That ALWAYS succeeds (~0.95–0.98) because scanner,
preprocessing, and crop are intrinsically source-coded — so it does NOT tell us whether the TB
DECISION leans on source. We keep it (relabeled "source separability — expected high, NOT a gate")
as a diagnostic, and ADD the question that matters: does the TB decision use source BEYOND the
pathology it should be reading? See `conditional_source_reliance_check`.

    python training/sanity.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

import train_tb

# CPU-ONLY by contract (do not touch the GPU). Pin BEFORE importing predict helpers so _gather and
# any loaded head agree on device. The conditional-reliance test scores cached OOF logits (no train).
train_tb.DEVICE = "cpu"

from train_tb import TBHead, run_lodo, SEED  # noqa: E402  (after the DEVICE pin, intentionally)

DATA = Path(__file__).resolve().parents[1] / "data"
# TXRV named-finding (pathology) logits live at txrv[:, 1024:] — the 18-dim evidence block the TB
# decision is SUPPOSED to read. The conditional test residualizes the TB logit on these + the label.
N_TXRV_LOGITS = 18
OOF_CACHE = DATA / "image_oof_logits.npz"  # cached LODO out-of-fold image logits (honest, no leak)
RESIDUAL_GAP_GATE = 0.05  # residual source-separability that survives pathology+label must stay LOW;
# a large RAW->RESIDUAL gap that LEAVES high residual separability = the decision rides source.


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


def _scalar_source_separability(score: np.ndarray, src: np.ndarray) -> dict[str, float]:
    """One-vs-rest AUROC of a SCALAR predicting each source. A 1-feature logistic classifier's AUROC
    equals the rank-AUROC of the scalar, so this IS "predict source from the scalar". A scalar can be
    inverted for a class (AUROC<0.5), so separability = max(a, 1-a). Returns per-source separability
    plus the macro mean under key '_macro'."""
    out: dict[str, float] = {}
    for s in sorted(set(src.tolist())):
        z = (src == s).astype(int)
        if z.sum() == 0 or z.sum() == len(z):
            continue
        a = float(roc_auc_score(z, score))
        out[s] = max(a, 1.0 - a)  # direction-agnostic separability
    out["_macro"] = float(np.mean([v for k, v in out.items() if k != "_macro"])) if out else float("nan")
    return out


def conditional_source_reliance_check(logit: np.ndarray, y: np.ndarray, src: np.ndarray,
                                      path_logits: np.ndarray) -> dict:
    """Does the TB DECISION lean on SOURCE *beyond* the pathology it should be reading?

    The naive canary (predict source from features) always succeeds because source is coded into the
    pixels. This asks the conditional question: residualize the TB head's OUT-OF-FOLD (LODO) logit on
    the pathology evidence (the 18 TXRV pathology logits) + the binary label, then test how much of
    the logit's SOURCE-predictability SURVIVES in the residual.

      - RAW separability  = how well the raw TB logit predicts source (one-vs-rest AUROC, macro).
      - RESID separability = same, on the residual after linear-regressing the logit on [path, label].
      - gap = RAW - RESID  (how much source-predictability the pathology+label explained away).

    INTERPRETATION (the gap alone is not the verdict — the RESIDUAL LEVEL is):
      - If the residual separability is LOW (~chance), the decision's source-predictability was almost
        entirely mediated by pathology+label -> source is NOT an independent shortcut -> PASS.
      - If the residual STAYS HIGH, the decision carries source information the pathology does NOT
        explain -> a source shortcut beyond pathology -> FAIL.
    Also reports the within-label-class Pearson corr of the logit with each source indicator (a
    high |corr| inside one class is a direct, model-agnostic source-reliance read).

    Uses the cached LODO OOF logits (honest, leak-guarded, AUROC ~0.92) — no retraining, CPU-only."""
    raw = _scalar_source_separability(logit, src)
    # residualize: logit ~ [18 pathology logits, label]; take residuals (the part NOT explained)
    X = np.column_stack([path_logits.astype("float64"), y.astype("float64")])
    reg = LinearRegression().fit(X, logit.astype("float64"))
    residual = logit.astype("float64") - reg.predict(X)
    res = _scalar_source_separability(residual, src)
    gap = float(raw["_macro"] - res["_macro"])
    # within-label-class Pearson corr of the logit with each source indicator
    within: dict[str, dict[str, float]] = {}
    for cls in (0, 1):
        m = y == cls
        row: dict[str, float] = {}
        for s in sorted(set(src.tolist())):
            z = (src[m] == s).astype("float64")
            if z.std() == 0 or logit[m].std() == 0:
                continue
            r, _ = pearsonr(logit[m].astype("float64"), z)
            row[s] = float(r)
        within[f"label_{cls}"] = row
    max_abs_within = max((abs(v) for row in within.values() for v in row.values()), default=float("nan"))
    # Verdict on the RESIDUAL LEVEL (not the gap): how far above chance the residual still separates
    # source. RESIDUAL_GAP_GATE (0.05) is the excess-over-chance we tolerate; a separability scalar is
    # noisy on small folds so we PASS up to chance+0.20 but FLAG the chance+0.05..0.20 band as a watch.
    residual_excess = res["_macro"] - 0.5  # >0 = source still separable after removing pathology+label
    if residual_excess <= RESIDUAL_GAP_GATE:
        verdict = ("PASS (residual source-separability ~chance: the decision's source-predictability "
                   "is fully explained by pathology+label — no independent source shortcut)")
    elif residual_excess <= 0.20:
        verdict = (f"PASS-WITH-WATCH (residual separability {res['_macro']:.3f} sits "
                   f"{residual_excess:+.3f} above chance — a weak residual source signal survives "
                   f"pathology+label; monitor, do not gate-fail on a noisy scalar)")
    else:
        verdict = (f"FAIL (residual separability {res['_macro']:.3f} stays "
                   f"{residual_excess:+.3f} above chance — the decision uses site BEYOND pathology)")
    return {
        "raw_source_separability_per_src": {k: round(v, 4) for k, v in raw.items()},
        "residual_source_separability_per_src": {k: round(v, 4) for k, v in res.items()},
        "raw_macro": round(raw["_macro"], 4),
        "residual_macro": round(res["_macro"], 4),
        "gap_raw_minus_residual": round(gap, 4),
        "within_label_class_corr_logit_vs_source": {
            k: {s: round(v, 4) for s, v in row.items()} for k, row in within.items()},
        "max_abs_within_class_corr": round(max_abs_within, 4),
        "verdict": verdict,
    }


def main() -> None:
    fast = "--fast" in sys.argv  # skip the heavy label-randomization (3x LODO retrain)
    d = np.load(DATA / "features.npz", allow_pickle=True)
    arrs = {"cls": d["cls"].astype("float32"), "patches": d["patches"].astype("float32"),
            "txrv": d["txrv"].astype("float32")}
    y = d["y"].astype("int64")
    src = d["source"].astype(str)
    groups = d["group"].astype("int64") if "group" in d.files else None
    if not fast:
        print("=== ANTI-SHORTCUT: label randomization (permuted-label LODO AUROC must be < 0.60) ===")
        print(label_randomization_check(arrs, y, src, groups))
    else:
        print("(--fast: skipping label_randomization_check — re-run without --fast for the full suite)")

    print("\n=== ANTI-SHORTCUT: model randomization (trained attention must DIFFER from a random head) ===")
    head_path = DATA / "tb_head.pt"
    if not head_path.exists():
        print(f"SKIP model_randomization_check: {head_path} not found "
              f"(run training/train_tb.py first to write the trained head).")
    else:
        model = TBHead(arrs["patches"].shape[2], arrs["cls"].shape[1], arrs["txrv"].shape[1],
                       use_patches=True)
        model.load_state_dict(torch.load(head_path, map_location="cpu"))
        model.eval()
        print(model_randomization_check(model, arrs))

    # === CONDITIONAL SOURCE-RELIANCE (the question the naive site-leak canary cannot answer) ===
    # NOTE on the OLD canary: training/audit.py's "predict source from features" check is RELABELED
    # "source separability — expected high (~0.95), NOT a gate" (it is a diagnostic: source is coded
    # into pixels, so it always succeeds and says nothing about whether the DECISION uses source).
    print("\n=== ANTI-SHORTCUT: conditional source-reliance (does the TB DECISION lean on source "
          "BEYOND pathology?) ===")
    if not OOF_CACHE.exists():
        print(f"SKIP conditional_source_reliance_check: {OOF_CACHE.name} not found "
              f"(run training/multimodal.py to cache the LODO out-of-fold image logits first).")
    else:
        o = np.load(OOF_CACHE, allow_pickle=True)
        logit = o["image_logit"].astype("float64")
        # sanity: the OOF cache must be aligned to features.npz (same order, labels, source)
        if not (o["label"].astype("int64") == y).all() or not (o["source"].astype(str) == src).all():
            raise RuntimeError(f"{OOF_CACHE.name} is not aligned to features.npz (order/label/source "
                               f"mismatch) — re-cache via training/multimodal.py before trusting this.")
        path_logits = d["txrv"][:, 1024:].astype("float64")  # the 18 TXRV pathology logits
        if path_logits.shape[1] != N_TXRV_LOGITS:
            raise RuntimeError(f"expected {N_TXRV_LOGITS} TXRV pathology logits, "
                               f"got {path_logits.shape[1]}")
        res = conditional_source_reliance_check(logit, y, src, path_logits)
        print(f"  raw source separability (macro OvR AUROC):      {res['raw_macro']:.3f}")
        print(f"  residual source separability (after pathology+label): {res['residual_macro']:.3f}")
        print(f"  gap (raw - residual, = source explained by pathology+label): {res['gap_raw_minus_residual']:+.3f}")
        print(f"  per-source raw:      {res['raw_source_separability_per_src']}")
        print(f"  per-source residual: {res['residual_source_separability_per_src']}")
        print(f"  within-label-class corr(logit, source) [max |r| = {res['max_abs_within_class_corr']:.3f}]:")
        for cls, row in res["within_label_class_corr_logit_vs_source"].items():
            print(f"    {cls}: {row}")
        print(f"  VERDICT: {res['verdict']}")


if __name__ == "__main__":
    main()
