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
import sys
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
TARGET_SENS = 0.95  # WHO TPP OPTIMUM sensitivity for a TB triage test (floor is 0.90); paired with
# optimum spec 0.80. Screening prioritizes catching TB over false alarms (a missed case is worse than
# an extra confirmatory test). Realized specificity at this point is reported honestly per fold — at
# our current AUROC ~0.92 expect spec to fall to ~0.50-0.65 (the 95/80 ideal needs a higher AUROC).
BATCH = 256
CAL_FRAC = 0.3          # held-out-site slice used to fit the local-recalibration threshold
BOOTSTRAP_N = 2000      # resamples for the AUROC CI
PREVALENCES = (0.01, 0.02)  # community-screening prevalences for the PPV/NPV table
PRESUMPTIVE_PREVALENCES = (0.09, 0.12, 0.15)  # S-Asian presumptive/ACF-flagged band (qXR/CAD4TB cohorts)
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


# ---------------------------------------------------------------------------
# T2 "sharpening" head (feat/t2-heads). Blueprint §2 Phases B/C/D. Builds three
# evidence channels and fuses them with a learned log-odds blend:
#   1. ZONAL SOFT-OR (sensitivity lever): per-zone gated-attention pool over patches
#      weighted by the soft zone membership `zones[:,:,z]`, a FLOORED zone gate
#      (g_min=0.25, mild upper-zone init prior), then logsumexp soft-OR across the 7
#      zones so a single lower-zone lesion can fire the image logit alone.
#   2. BOX-EVIDENCE (interpretability): shared 1x1 scorer over the 64 patch tokens ->
#      8x8 evidence map -> BOUNDED-sharpness LSE-LBA pool -> a logit. Per-cell BCE on
#      grid_label, masked by has_box (box loss only on the 660 boxed TBX11K images);
#      per-cell pos_weight from the CELL-level imbalance; a small total-variation prior.
#   3. PATHOLOGY-GROUNDED FUSION (specificity): CLS + pooled-TXRV(1024) + a named-finding
#      embedding of the 18 TXRV logits -> MLP -> a logit. An aux head re-predicts the 18
#      logits (MSE, T=2) to anchor the representation in named findings, not site identity.
#
# PRIOR-FIXED hyper-parameters (NEVER tuned against the LODO test folds — selection bias):
#   g_min = 0.25 ; zone-gate upper-prior init = +0.4 logit on the 4 upper/hilar zones ;
#   r0 (LSE sharpness, softplus-parameterised, bounded) start ~5 capped at R_MAX = 8 ;
#   lambda_box = 0.2 ; lambda_distill = 0.2 ; lambda_tv = 0.01 ; distill T = 2.
N_ZONES = 7
# zone order from extract_features.ZONE_NAMES = (RUZ, RMZ, RLZ, LUZ, LMZ, LLZ, HILAR)
_UPPER_ZONE_IDX = (0, 3, 6)  # right-upper, left-upper, hilar/mediastinal — mild prior, NOT a hard anchor
G_MIN = 0.25
ZONE_PRIOR_INIT = 0.4
R0 = 5.0
R_MAX = 8.0
LAMBDA_BOX = 0.2
LAMBDA_DISTILL = 0.2
LAMBDA_TV = 0.01
DISTILL_T = 2.0
N_TXRV_LOGITS = 18  # TorchXRayVision named-finding logits live at txrv[:, 1024:]


class ZonalSoftOR(nn.Module):
    """Per-zone gated-attention pool over patches weighted by soft zone membership, then a
    FLOORED zone gate and a logsumexp soft-OR across zones. Returns (zonal_logit, zone_logits).

    Soft-OR (logsumexp) is the sensitivity lever: max-like, so a single zone with strong
    evidence drives the image logit even if every other zone is silent (the lower-zone-cavity
    miss). The gate is FLOORED at g_min=0.25 so no zone can be fully switched off."""

    def __init__(self, d: int = 768, h: int = 128):
        super().__init__()
        self.V = nn.Linear(d, h)
        self.U = nn.Linear(d, h)
        self.w = nn.Linear(h, 1)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        # learned-but-floored zone gate; init a MILD upper-zone prior (not a hard anchor)
        gate0 = torch.zeros(N_ZONES)
        for z in _UPPER_ZONE_IDX:
            gate0[z] = ZONE_PRIOR_INIT
        self.gate_logit = nn.Parameter(gate0)

    def forward(self, patches: torch.Tensor, zones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # patches [B,T,d]; zones [B,T,Z] soft membership (rows sum <=1; background patches ~0)
        a_pre = torch.tanh(self.V(patches)) * torch.sigmoid(self.U(patches))  # [B,T,h]
        raw = self.w(a_pre)  # [B,T,1] unnormalised attention scores
        zone_logits = []
        eps = 1e-8
        for z in range(N_ZONES):
            m = zones[:, :, z : z + 1]  # [B,T,1] membership weight for zone z
            # zone-restricted attention: softmax over patches, weighted by membership so out-of-zone
            # patches get ~0 mass. add log(m) so a zero-membership patch is excluded from the softmax.
            logits = raw + torch.log(m + eps)
            att = torch.softmax(logits, dim=1)  # [B,T,1]
            pooled = (att * patches).sum(dim=1)  # [B,d]
            zone_logits.append(self.scorer(pooled))  # [B,1]
        zl = torch.cat(zone_logits, dim=1)  # [B,Z]
        # floored zone gate in [g_min, 1]: g = g_min + (1-g_min)*sigmoid(gate_logit)
        gate = G_MIN + (1.0 - G_MIN) * torch.sigmoid(self.gate_logit)  # [Z]
        gated = zl * gate.unsqueeze(0)  # [B,Z]
        # logsumexp soft-OR across zones -> a single hot zone fires the image logit alone
        zonal_logit = torch.logsumexp(gated, dim=1)  # [B]
        return zonal_logit, zl


class BoxEvidence(nn.Module):
    """Shared 1x1 scorer over the 64 patch tokens -> 8x8 evidence map (logits) -> LSE-LBA pool with
    a BOUNDED learnable sharpness r (softplus, capped at R_MAX so it can't collapse to a hard max).
    Returns (box_logit, evidence_logits[B,64]). The evidence map is the localization target."""

    def __init__(self, d: int = 768):
        super().__init__()
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))  # shared 1x1 over tokens
        # sigmoid-parameterised sharpness BOUNDED in (0, R_MAX); init rho so r(rho0) ~= R0
        rho0 = float(np.log(R0 / (R_MAX - R0)))  # sigmoid(rho0)*R_MAX == R0
        self.rho = nn.Parameter(torch.tensor(rho0))

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ev = self.scorer(patches).squeeze(-1)  # [B,T] per-cell evidence logits
        r = R_MAX * torch.sigmoid(self.rho)  # bounded sharpness in (0, R_MAX)
        T = ev.shape[1]
        # LSE-LBA (Log-Sum-Exp Lower-Bounded Approximation, Li et al. 1803.07703): a smooth,
        # numerically stable max-approximation that stays calibrated. logit = (1/r)*logsumexp(r*ev).
        box_logit = (torch.logsumexp(r * ev, dim=1) - np.log(T)) / r  # [B]  (mean-normalised LSE)
        return box_logit, ev


class TBHeadT2(nn.Module):
    """T2 sharpening head: zonal soft-OR + box-evidence + pathology-grounded fusion, fused by a
    learned log-odds blend. `levers` selects which channels are active (for the ablation):
    a set drawn from {"zonal", "box", "fusion"}; "fusion" is always on (it carries CLS+TXRV)."""

    def __init__(self, d_tok: int, d_cls: int, d_txrv: int, levers: frozenset[str]):
        super().__init__()
        self.levers = levers
        self.use_zonal = "zonal" in levers
        self.use_box = "box" in levers
        d_pooled = d_txrv - N_TXRV_LOGITS  # 1024 pooled TXRV features (the 18 logits are split off)
        self.zonal = ZonalSoftOR(d_tok) if self.use_zonal else None
        self.box = BoxEvidence(d_tok) if self.use_box else None
        # named-finding embedding of the 18 TXRV logits (equal footing, not diluted in 1042-d)
        self.find_embed = nn.Sequential(nn.Linear(N_TXRV_LOGITS, 32), nn.GELU())
        d_fuse = d_cls + d_pooled + 32
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_fuse), nn.Dropout(0.3), nn.Linear(d_fuse, 128), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(128, 1),
        )
        # aux distillation head: re-predict the 18 TXRV logits from CLS (anchors rep in named findings)
        self.distill = nn.Sequential(nn.LayerNorm(d_cls), nn.Linear(d_cls, 64), nn.GELU(),
                                     nn.Linear(64, N_TXRV_LOGITS))
        # learned log-odds blend over the active channels (zonal, box, fusion). init equal-ish.
        self.blend = nn.Parameter(torch.zeros(3))

    def forward(self, cls: torch.Tensor, patches: torch.Tensor, txrv: torch.Tensor) -> dict:
        pooled = txrv[:, : txrv.shape[1] - N_TXRV_LOGITS]  # [B,1024]
        find_logits = txrv[:, txrv.shape[1] - N_TXRV_LOGITS :]  # [B,18] named-finding logits
        fuse_in = torch.cat([cls, pooled, self.find_embed(find_logits)], dim=1)
        fusion_logit = self.fusion(fuse_in).squeeze(-1)  # [B]
        distill_pred = self.distill(cls)  # [B,18] re-predicted TXRV logits
        # log-odds blend: softplus weights keep blend monotone-positive (channels can't cancel)
        w = torch.nn.functional.softplus(self.blend)  # [3]
        logit = w[2] * fusion_logit
        zone_logits = None
        evidence = None
        if self.zonal is not None:
            zonal_logit, zone_logits = self.zonal(patches, self._zones)
            logit = logit + w[0] * zonal_logit
        if self.box is not None:
            box_logit, evidence = self.box(patches)
            logit = logit + w[1] * box_logit
        return {"logit": logit, "evidence": evidence, "distill": distill_pred,
                "zone_logits": zone_logits, "find_target": find_logits}

    # zones are passed via an attribute set just before forward (keeps the predict() signature stable)
    _zones: torch.Tensor


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


def _batches(n: int, shuffle: bool, rng: np.random.Generator | None = None):
    # Reproducibility (P0): batch shuffling uses an EXPLICIT, fold-stable Generator threaded down
    # from train_head — never the global np.random state, whose position drifts across folds/runs.
    if shuffle:
        idx = rng.permutation(n) if rng is not None else np.random.permutation(n)
    else:
        idx = np.arange(n)
    for s in range(0, n, BATCH):
        yield idx[s : s + BATCH]


def _gather(arrs: dict, idx: np.ndarray) -> dict:
    return {k: torch.tensor(v[idx]).to(DEVICE) for k, v in arrs.items()}


def train_head(arrs: dict, y: np.ndarray, tr: np.ndarray, va: np.ndarray, use_patches: bool,
               max_epochs: int = 80, patience: int = 8, seed: int = SEED) -> TBHead:
    # `seed` is FOLD-STABLE (constructed per call as SEED + fold_index by the LODO caller) so each
    # fold's batch shuffle is reproducible and INDEPENDENT of how many folds ran before it. The old
    # code shared the global np.random state, so a fold's shuffle depended on prior folds' draws.
    rng = np.random.default_rng(seed)
    model = TBHead(arrs["patches"].shape[2], arrs["cls"].shape[1], arrs["txrv"].shape[1], use_patches).to(DEVICE)
    n_pos = max(1, int((y[tr] == 1).sum()))
    n_neg = max(1, int((y[tr] == 0).sum()))
    pos_weight = torch.tensor([n_neg / n_pos], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # NO label smoothing (distorts probability)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    best_ap, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in _batches(len(tr), shuffle=True, rng=rng):
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


# ---------------------------------------------------------------------------
# T2 head train / predict (multi-task: image BCE + masked box BCE + TV + distillation).
# Parallel to train_head/predict_logits so sanity.py's import of the baseline path is unaffected.

def _cell_pos_weight(arrs: dict, tr: np.ndarray) -> float:
    """Per-CELL pos_weight from the CELL-level imbalance among BOXED train images ONLY — NOT the
    image-level 1:6 ratio. (~9% of cells in boxed images are positive => pos_weight ~10.)"""
    hb = arrs["has_box"][tr].astype(bool)
    if hb.sum() == 0:
        return 1.0
    gl = (arrs["grid_label"][tr][hb] > 0.5).astype(np.float32)  # [n_box,8,8] binarised
    pos = float(gl.sum())
    neg = float(gl.size - pos)
    return max(1.0, neg / max(1.0, pos))


def _tv_prior(ev_map: torch.Tensor) -> torch.Tensor:
    """Total-variation prior on the 8x8 evidence map (encourages spatially coherent, not speckled,
    evidence). ev_map [B,8,8] sigmoid probabilities."""
    dh = (ev_map[:, 1:, :] - ev_map[:, :-1, :]).abs().mean()
    dw = (ev_map[:, :, 1:] - ev_map[:, :, :-1]).abs().mean()
    return dh + dw


def train_head_t2(arrs: dict, y: np.ndarray, tr: np.ndarray, va: np.ndarray, levers: frozenset[str],
                  max_epochs: int = 80, patience: int = 8, seed: int = SEED) -> TBHeadT2:
    rng = np.random.default_rng(seed)
    model = TBHeadT2(arrs["patches"].shape[2], arrs["cls"].shape[1], arrs["txrv"].shape[1], levers).to(DEVICE)
    n_pos = max(1, int((y[tr] == 1).sum()))
    n_neg = max(1, int((y[tr] == 0).sum()))
    img_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=DEVICE))  # no smoothing
    cell_pw = torch.tensor([_cell_pos_weight(arrs, tr)], device=DEVICE)
    cell_loss = nn.BCEWithLogitsLoss(pos_weight=cell_pw, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    use_box = "box" in levers

    best_ap, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in _batches(len(tr), shuffle=True, rng=rng):
            ix = tr[b]
            g = _gather({k: arrs[k] for k in ("cls", "patches", "txrv", "zones")}, ix)
            model._zones = g["zones"]
            yt = torch.tensor(y[ix], dtype=torch.float32, device=DEVICE)
            opt.zero_grad()
            out = model(g["cls"], g["patches"], g["txrv"])
            loss = img_loss(out["logit"], yt)
            # aux distillation: re-predict the 18 TXRV logits (MSE, T=2 soft target) from CLS
            tgt = out["find_target"] / DISTILL_T
            loss = loss + LAMBDA_DISTILL * nn.functional.mse_loss(out["distill"] / DISTILL_T, tgt)
            # masked per-cell box BCE — ONLY on the boxed images (never supervise negatives to all-zero)
            if use_box and out["evidence"] is not None:
                hb = torch.tensor(arrs["has_box"][ix].astype(np.float32), device=DEVICE)  # [B]
                if hb.sum() > 0:
                    gl = torch.tensor((arrs["grid_label"][ix] > 0.5).astype(np.float32),
                                      device=DEVICE).reshape(len(ix), -1)  # [B,64] row-major gy*8+gx
                    per = cell_loss(out["evidence"], gl).mean(dim=1)  # [B] per-image mean cell loss
                    box_bce = (per * hb).sum() / hb.sum()  # masked mean over boxed images only
                    ev_map = torch.sigmoid(out["evidence"]).reshape(len(ix), 8, 8)
                    loss = loss + LAMBDA_BOX * box_bce + LAMBDA_TV * _tv_prior(ev_map[hb.bool()])
            loss.backward()
            opt.step()
        ap = average_precision_score(y[va], predict_t2(model, arrs, va))
        if ap > best_ap:
            best_ap, best_state, bad = ap, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_logits_t2(model: TBHeadT2, arrs: dict, idx: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(len(idx), shuffle=False):
            ix = idx[b]
            g = _gather({k: arrs[k] for k in ("cls", "patches", "txrv", "zones")}, ix)
            model._zones = g["zones"]
            out.append(model(g["cls"], g["patches"], g["txrv"])["logit"].cpu().numpy())
    return np.concatenate(out)


def predict_t2(model: TBHeadT2, arrs: dict, idx: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-predict_logits_t2(model, arrs, idx) / temperature))


def evidence_maps_t2(model: TBHeadT2, arrs: dict, idx: np.ndarray) -> np.ndarray:
    """Return the 8x8 evidence-map probabilities [len(idx),8,8] for localization scoring."""
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(len(idx), shuffle=False):
            ix = idx[b]
            g = _gather({k: arrs[k] for k in ("cls", "patches", "txrv", "zones")}, ix)
            model._zones = g["zones"]
            ev = model(g["cls"], g["patches"], g["txrv"])["evidence"]
            if ev is None:
                raise RuntimeError("evidence_maps_t2 called on a model without the box lever")
            out.append(torch.sigmoid(ev).reshape(len(ix), 8, 8).cpu().numpy())
    return np.concatenate(out)


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


RECAL_SPLITS = 20  # random cal/eval splits — recalibrated sensitivity is reported as MEDIAN + IQR
                   # (a single split is a noisy draw; threshold-selection variance must be visible)


def _recalibrated(yte: np.ndarray, pte: np.ndarray):
    """Recalibrate the threshold on a CAL_FRAC labeled slice of the held-out site, eval on the disjoint
    rest. A SINGLE split is one noisy draw of a threshold-selection process, so we repeat over
    RECAL_SPLITS distinct-seed splits and report the MEDIAN sensitivity + IQR across splits. The
    Clopper-Pearson CI is kept on the MEDIAN-sensitivity split's eval slice (a representative single-eval
    finite-sample interval). Returns the per-split median operating point so the prevalence table and the
    acceptance gate see a stable number, not a coin flip.

    Returns: (sens_med, lo, hi, spec_med, npos_ev, sens_iqr_lo, sens_iqr_hi) or None when the floor
    (>=8 pos and >=8 neg) is not met or every split degenerates to a single-class slice."""
    if int((yte == 1).sum()) < 8 or int((yte == 0).sum()) < 8:
        return None
    loc = np.arange(len(yte))
    results = []  # (sens, lo, hi, spec, npos_ev) per valid split
    for k in range(RECAL_SPLITS):
        cal, ev = train_test_split(loc, test_size=1 - CAL_FRAC, stratify=yte, random_state=SEED + k)
        if (yte[cal] == 1).sum() == 0 or (yte[ev] == 1).sum() == 0 or (yte[ev] == 0).sum() == 0:
            continue
        thr_local = threshold_for_sensitivity(yte[cal], pte[cal])
        sens, lo, hi, spec = _sens_spec(yte[ev], pte[ev], thr_local)
        results.append((sens, lo, hi, spec, int((yte[ev] == 1).sum())))
    if not results:
        return None
    sens_vals = np.array([r[0] for r in results], dtype=float)
    spec_vals = np.array([r[3] for r in results], dtype=float)
    sens_med = float(np.median(sens_vals))
    spec_med = float(np.median(spec_vals))
    iqr_lo, iqr_hi = float(np.percentile(sens_vals, 25)), float(np.percentile(sens_vals, 75))
    # CI from the split whose sensitivity is closest to the median (representative single eval)
    med_split = min(results, key=lambda r: abs(r[0] - sens_med))
    _, lo, hi, _, npos_ev = med_split
    return sens_med, lo, hi, spec_med, npos_ev, iqr_lo, iqr_hi


def run_lodo(arrs: dict, y: np.ndarray, src: np.ndarray, groups: np.ndarray | None,
             use_patches: bool) -> tuple[float, list[dict]]:
    sources = sorted(set(src.tolist()))
    aucs, ops = [], []
    tag = "fusion+attention" if use_patches else "fusion-only"
    print(f"\n--- LODO ({tag}) — radiographic-TB-pattern, vs radiographic reference ---")
    for fold_index, ho in enumerate(sources):
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
        # FOLD-STABLE seed for batch shuffling: each fold gets SEED + its position in `sources`, so
        # the result is reproducible run-to-run and a fold's shuffle does not depend on prior folds.
        model = train_head(arrs, y, tr, va, use_patches, seed=SEED + fold_index)
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
            s_r, lo_r, hi_r, sp_r, npos_e, iqr_lo, iqr_hi = rec
            print(f"      + local recalibration     sens(median of {RECAL_SPLITS} splits)={s_r:.3f} "
                  f"[IQR {iqr_lo:.2f}-{iqr_hi:.2f}; median-split 95% CI {lo_r:.2f}-{hi_r:.2f}] "
                  f"spec={sp_r:.3f}  (eval n_pos={npos_e})")
            ops.append({"source": ho, "sens": s_r, "spec": sp_r, "remix": ho in REMIX_SOURCES,
                        "sens_ci": (lo_r, hi_r), "sens_iqr": (iqr_lo, iqr_hi)})
        else:
            print(f"      + local recalibration     n/a (too few positives to split)")
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    print(f"  >>> mean LODO AUC ({tag}) = {mean_auc:.3f}")
    # ACCEPTANCE GATE leads with the WORST-fold external (non-remix) recalibrated sensitivity, not the
    # mean — a screen is judged by its weakest site, and the re-mix folds are leakage-prone (excluded).
    ext = [o for o in ops if not o["remix"] and not np.isnan(o["sens"])]
    if ext:
        worst = min(ext, key=lambda o: o["sens"])
        wlo, whi = worst["sens_ci"]
        print(f"  >>> WORST-FOLD external (non-remix) recalibrated sens ({tag}) = {worst['sens']:.3f} "
              f"[95% CI {wlo:.2f}-{whi:.2f}]  holdout={worst['source']}  <-- acceptance gate")
    else:
        print(f"  >>> WORST-FOLD external (non-remix) recalibrated sens ({tag}) = n/a "
              f"(no clean external fold with a recalibration split)")
    return mean_auc, ops


def run_lodo_t2(arrs: dict, y: np.ndarray, src: np.ndarray, groups: np.ndarray | None,
                levers: frozenset[str], tag: str | None = None) -> tuple[float, list[dict]]:
    """LODO for the T2 head. `levers` (subset of {zonal, box, fusion}) selects active channels for the
    ablation. Mirrors run_lodo's honest reporting (cold-start + recal, bootstrap AUC CI, ECE, dup-cluster
    guard, worst-fold gate). Box supervision is 100% TBX11K — so per fold we flag whether the box lever
    is even active and the held-out site has no boxes (its AUROC effect on montgomery/shenzhen is the
    honesty check, reported in the ablation summary, not a leak)."""
    sources = sorted(set(src.tolist()))
    aucs, ops = [], []
    tag = tag or ("T2[" + "+".join(sorted(levers)) + "]")
    print(f"\n--- LODO ({tag}) — radiographic-TB-pattern, vs radiographic reference ---")
    for fold_index, ho in enumerate(sources):
        te = np.where(src == ho)[0]
        tr_all = np.where(src != ho)[0]
        if (y[te] == 1).sum() == 0 or (y[te] == 0).sum() == 0:
            print(f"  holdout={ho:14s} skipped (single-class test)")
            continue
        if groups is not None:
            te_groups = set(int(x) for x in groups[te] if x >= 0)
            leak = np.array([g in te_groups and g >= 0 for g in groups[tr_all]])
            tr_all = tr_all[~leak]
        tr, va = train_test_split(tr_all, test_size=0.2, stratify=y[tr_all], random_state=SEED)
        model = train_head_t2(arrs, y, tr, va, levers, seed=SEED + fold_index)
        T = fit_temperature(predict_logits_t2(model, arrs, va), y[va])
        thr = threshold_for_sensitivity(y[va], predict_t2(model, arrs, va, T))
        pte = predict_t2(model, arrs, te, T)
        yte = y[te]
        auc = roc_auc_score(yte, pte)
        a_lo, a_hi = bootstrap_auc_ci(yte, pte)
        cal_ece = ece(yte, pte)
        aucs.append(auc)
        n_box_te = int(arrs["has_box"][te].astype(bool).sum())
        flag = "  [LEAKAGE-PRONE: re-mix source]" if ho in REMIX_SOURCES else "  [cleaner external]"
        box_note = "" if "box" not in levers else (f"  box_te={n_box_te}" if n_box_te else "  box_te=0 (no boxes here)")
        s_f, lo_f, hi_f, sp_f = _sens_spec(yte, pte, thr)
        rec = _recalibrated(yte, pte)
        print(f"  holdout={ho:14s} n={len(te):5d} pos={int((yte==1).sum()):4d}  "
              f"AUC={auc:.3f} [95% CI {a_lo:.2f}-{a_hi:.2f}]  ECE={cal_ece:.3f}{flag}{box_note}")
        print(f"      cold-start  (frozen thr)  sens={s_f:.3f} [95% CI {lo_f:.2f}-{hi_f:.2f}] spec={sp_f:.3f}")
        if rec is not None:
            s_r, lo_r, hi_r, sp_r, npos_e, iqr_lo, iqr_hi = rec
            print(f"      + local recalibration     sens(median of {RECAL_SPLITS} splits)={s_r:.3f} "
                  f"[IQR {iqr_lo:.2f}-{iqr_hi:.2f}; median-split 95% CI {lo_r:.2f}-{hi_r:.2f}] "
                  f"spec={sp_r:.3f}  (eval n_pos={npos_e})")
            ops.append({"source": ho, "sens": s_r, "spec": sp_r, "remix": ho in REMIX_SOURCES,
                        "sens_ci": (lo_r, hi_r), "sens_iqr": (iqr_lo, iqr_hi), "auc": auc,
                        "box_te": n_box_te})
        else:
            print(f"      + local recalibration     n/a (too few positives to split)")
            ops.append({"source": ho, "sens": float("nan"), "spec": float("nan"),
                        "remix": ho in REMIX_SOURCES, "auc": auc, "box_te": n_box_te,
                        "sens_ci": (float("nan"), float("nan")), "sens_iqr": (float("nan"), float("nan"))})
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    print(f"  >>> mean LODO AUC ({tag}) = {mean_auc:.3f}")
    ext = [o for o in ops if not o["remix"] and not np.isnan(o["sens"])]
    if ext:
        worst = min(ext, key=lambda o: o["sens"])
        wlo, whi = worst["sens_ci"]
        print(f"  >>> WORST-FOLD external (non-remix) recalibrated sens ({tag}) = {worst['sens']:.3f} "
              f"[95% CI {wlo:.2f}-{whi:.2f}]  holdout={worst['source']}  <-- acceptance gate")
    else:
        print(f"  >>> WORST-FOLD external (non-remix) recalibrated sens ({tag}) = n/a")
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
    for prev in (*PREVALENCES, *PRESUMPTIVE_PREVALENCES):  # community (1-2%) + S-Asian presumptive (9-15%)
        ppv, npv, tpc = ppv_npv(sens, spec, prev)
        tag = "" if prev in PREVALENCES else "  <- presumptive/ACF band"
        print(f"{prev:>10.0%} {ppv:>7.1%} {npv:>7.3%} {tpc:>20.1f}{tag}")
    print("  NOTE: PPV/NPV are for the RADIOGRAPHIC-TB label, not bacteriologically-confirmed active TB.")


def _load_arrs(d) -> dict:
    return {
        "cls": d["cls"].astype("float32"),
        "patches": d["patches"].astype("float32"),
        "txrv": d["txrv"].astype("float32"),
        "zones": d["zones"].astype("float32"),
        "grid_label": d["grid_label"].astype("float32"),
        "has_box": d["has_box"].astype(bool),
    }


def _nonbox_auc(ops: list[dict]) -> float:
    """Mean LODO AUROC over the folds whose held-out site has NO boxes (montgomery+shenzhen+qatar) —
    the honest read of the box lever's effect where it CANNOT have leaked from box supervision."""
    v = [o["auc"] for o in ops if o.get("box_te", 0) == 0 and not np.isnan(o.get("auc", float("nan")))]
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # NOTE: torch.use_deterministic_algorithms is intentionally NOT set — it destabilizes some
    # MPS ops (produced NaN logits here). Seeds give run-to-run reproducibility on this backend.
    run_t2 = "--baseline-only" not in sys.argv
    d = np.load(DATA / "features.npz", allow_pickle=True)
    arrs = _load_arrs(d)
    y = d["y"].astype("int64")
    src = d["source"].astype(str)
    groups = d["group"].astype("int64") if "group" in d.files else None
    print("ENDPOINT: radiographic-TB-pattern (NOT bacteriologically-confirmed active TB). Research preview.")
    print("sources:", {s: int((src == s).sum()) for s in sorted(set(src.tolist()))})
    print(f"total {len(y)}  pos={int((y==1).sum())}  neg={int((y==0).sum())}  "
          f"dup-cluster guard={'ON' if groups is not None else 'OFF (no group column)'}")
    print(f"PRIOR-FIXED T2 knobs (NOT tuned on test folds): g_min={G_MIN} zone_prior_init={ZONE_PRIOR_INIT} "
          f"r0={R0} (cap {R_MAX}) lambda_box={LAMBDA_BOX} lambda_distill={LAMBDA_DISTILL} "
          f"lambda_tv={LAMBDA_TV} distill_T={DISTILL_T}")

    # ---- BASELINE (the drift-log "before": fusion-only vs fusion+attention) ----
    auc_fo, _ = run_lodo(arrs, y, src, groups, use_patches=False)
    auc_fa, ops = run_lodo(arrs, y, src, groups, use_patches=True)
    print(f"\nattention ablation: fusion-only {auc_fo:.3f} -> fusion+attention {auc_fa:.3f} "
          f"(delta {auc_fa-auc_fo:+.3f})")
    print_prevalence_table(ops)

    if run_t2:
        # ---- T2 ABLATION: each lever's marginal contribution, then the full head ----
        # fusion-only T2 (named-finding embed + distillation, no patches) isolates the grounding lever.
        ablation = [
            ("T2[fusion+distill]", frozenset({"fusion"})),
            ("T2[+zonal]", frozenset({"fusion", "zonal"})),
            ("T2[+box]", frozenset({"fusion", "box"})),
            ("T2[FULL]", frozenset({"fusion", "zonal", "box"})),
        ]
        t2_results: dict[str, tuple[float, list[dict]]] = {}
        for name, lev in ablation:
            t2_results[name] = run_lodo_t2(arrs, y, src, groups, lev, tag=name)
        full_auc, full_ops = t2_results["T2[FULL]"]
        print_prevalence_table(full_ops)

        print("\n=== ABLATION SUMMARY (mean LODO AUROC; baseline = current fusion+attention) ===")
        print(f"  baseline fusion-only        {auc_fo:.3f}")
        print(f"  baseline fusion+attention   {auc_fa:.3f}   <-- T2 'before'")
        for name, _ in ablation:
            a, o = t2_results[name]
            nb = _nonbox_auc(o)
            print(f"  {name:26s} {a:.3f}   (delta vs +att {a-auc_fa:+.3f}; non-box-fold AUROC {nb:.3f})")
        print("  HONESTY: box supervision is 100% TBX11K. 'non-box-fold AUROC' is the mean over folds")
        print("  whose held-out site has NO boxes (montgomery/shenzhen/qatar) — where the box lever")
        print("  cannot have leaked. Compare it across rows to see the box lever's HONEST effect.")
        drift = full_auc - auc_fa
        if drift > 0.05:
            print(f"  !! DRIFT WATCH: T2 FULL AUROC jumped {drift:+.3f} vs +att (> +0.05) — treat as SUSPECT")
            print("     (the heads should buy specificity/localization/trust, not big AUROC).")

        # save the full T2 head trained on all data
        allidx = np.arange(len(y))
        tr, va = train_test_split(allidx, test_size=0.15, stratify=y, random_state=SEED)
        model = train_head_t2(arrs, y, tr, va, frozenset({"fusion", "zonal", "box"}))
        T = fit_temperature(predict_logits_t2(model, arrs, va), y[va])
        torch.save(model.state_dict(), DATA / "tb_head_t2.pt")
        thr = threshold_for_sensitivity(y[va], predict_t2(model, arrs, va, T))
        json.dump({"threshold": thr, "temperature": T, "target_sensitivity": TARGET_SENS,
                   "endpoint": "radiographic_tb_pattern", "head": "t2",
                   "levers": ["fusion", "zonal", "box"],
                   "note": "re-fit threshold+temperature per site"},
                  open(DATA / "tb_threshold_t2.json", "w"))
        print(f"\nT2 head -> data/tb_head_t2.pt ; temperature={T:.3f} ; threshold@{TARGET_SENS:.0%}sens={thr:.3f} "
              f"(in-sample/optimistic — the honest numbers are the LODO folds above; re-fit per site)")
    else:
        # final baseline model on all data (with a small val split for early stop + temperature)
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
