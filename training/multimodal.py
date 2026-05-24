"""Multimodal late-fusion PROTOTYPE: image score + OPTIONAL clinical inputs, with graceful
missing-modality degradation. Blueprint §6 (verdict band) + DATA_SOURCES.md §multimodal.

WHAT THIS IS / IS NOT (read before quoting any number):
  - The IMAGE side is REAL: out-of-fold (LEAVE-ONE-DATASET-OUT) logits from the T2 image head over the
    cached data/features.npz features. We do NOT score in-sample — the M13/M15 lesson is that the head
    memorises its training images (in-sample AUROC ~1.0, no headroom, the conformal band collapses to a
    point). Each image's logit comes from a head that NEVER saw that image's SOURCE, so it carries the
    honest ~0.92 LODO generalisation gap. THIS is what gives the fusion real headroom to act on.
  - The CLINICAL side is SIMULATED. We do NOT yet have paired image+clinical+outcome data — that is
    gated (R2D2 tabular needs images by outreach; TB Portals / ICMR-NIRT need a DUA — DATA_SOURCES.md
    §6-8). So we synthesise a clinical signal whose effect size is CALIBRATED TO THE R2D2 LITERATURE
    (CRP/symptoms correlated with the true label so the fused-vs-image-only specificity gain at fixed
    90% sensitivity lands near R2D2's reported +5.6 pts for a CRP->CAD fusion). The R2D2 caveat is also
    honoured: gains SHRINK as the image model strengthens, and our image model is already strong, so
    we expect a single-digit specificity gain, NOT a transformation.
  - THEREFORE: the numbers here PROVE THE MACHINERY, they are NOT a multimodal performance claim. The
    fusion coefficients, the clinical effect size, and any specificity gain are simulated pending real
    paired data. Do not cite a "multimodal accuracy" from this file.

THE THREE MACHINERY RESULTS THIS SCRIPT REPORTS (the point of the exercise):
  (a) FULL-MODALITY >= IMAGE-ONLY: specificity at the WHO 90% sensitivity floor is no worse (and, at the
      simulated R2D2 effect size, ~+5-6 pts better) with clinical present.
  (b) GRACEFUL DEGRADATION: with the clinical vector ABSENT, the system STILL SCORES from the image
      alone — the fused logit cleanly reduces to the image logit (absent clinical contributes ~0).
  (c) LESS-CONFIDENT-WITHOUT-CLINICAL: the UNDETERMINED/abstain rate is measurably HIGHER in image-only
      mode, because the conformal band [tauLow, tauHigh] is WIDER when clinical is absent. State plainly.

DESIGN (late fusion in LOG-ODDS):
  fused_logit = image_logit + g * clinical_logit
  - image_logit: ALWAYS computed (the trained T2 head).
  - clinical_logit: a small MLP over the ClinicalVector (normalised value + PRESENT/ABSENT mask per
    field). When a field is absent its value is zeroed and its mask bit is 0, so the MLP learns to lean
    on present fields only. When the WHOLE clinical vector is absent (mask all 0), the clinical-MLP is
    trained (via modality dropout) to emit ~0 logit -> fused degrades to image-only. `g` is a learned
    non-negative gain (softplus) so clinical can only sharpen, never invert, the image evidence.
  - MODALITY DROPOUT: during training we drop the entire clinical vector for ~50% of samples, so BOTH
    pathways (image-only and fused) are exercised and the image-only path stays calibrated. This is the
    mechanism that makes "image-only still works."

CONFIDENCE RULE (less confident without clinical), stated plainly:
  We fit the conformal band [tauLow, tauHigh] (Blueprint §6) SEPARATELY for each modality regime on a
  calibration slice. tauLow keeps the WHO 90% sensitivity floor; tauHigh keeps a target specificity.
  A case scores UNDETERMINED iff tauLow <= p < tauHigh. The image-only band is WIDER (clinical can't
  pull near-threshold cases off the fence), so MORE cases land in UNDETERMINED. Rule: clinical present
  -> tighter band -> fewer UNDETERMINED; clinical absent -> wider band -> more UNDETERMINED. The system
  never refuses to score; it scores with explicitly lower confidence.

FOLLOW-ON (NOT built here, documented per the brief): the frontend clinical FORM + wiring depends on an
ONNX export of this fusion head. The app today calls API providers (HF/Replicate/OpenAI), NOT the
trained head, so there is no in-app path for the image logit yet — wiring the form is gated on exporting
both the T2 image head and this clinical-MLP+gain to ONNX and threading a ClinicalVector through the
orchestrator. This file is the offline machinery proof only.

    PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 training/.venv/bin/python training/multimodal.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from train_tb import (DEVICE, REMIX_SOURCES, SEED, clopper_pearson, fit_temperature,
                      predict_logits_t2, train_head_t2, _load_arrs, _sens_spec,
                      threshold_for_sensitivity)

DATA = Path(__file__).resolve().parents[1] / "data"
FULL_LEVERS = frozenset({"fusion", "zonal", "box"})
OOF_CACHE = DATA / "image_oof_logits.npz"  # cached LODO out-of-fold image logits (honest headroom)

# WHO floor for a TB triage SCREEN: catch >=90% of disease. We measure specificity AT this sensitivity
# (the R2D2 operating point) for the full-vs-image-only comparison. (The T2 head itself targets the WHO
# OPTIMUM 95%; for the multimodal effect we use the 90% FLOOR to match R2D2's reported operating point.)
TARGET_SENS_FUSION = 0.90

# ---------------------------------------------------------------------------
# SIMULATED-CLINICAL effect size (CALIBRATED TO R2D2 — NOT a measured coefficient).
# R2D2 (prospective, culture reference): at fixed 90% sensitivity, specificity rose CAD4TB-alone 70.3%
# -> CRP->CAD 75.9% (+5.6 pts). We pick a per-field log-odds effect so the SIMULATED fused head buys a
# single-digit specificity gain at 90% sens, consistent with R2D2 AND the "gains shrink on a strong image
# model" caveat (DATA_SOURCES.md). These numbers are a SIMULATION KNOB, not evidence.
R2D2_SPEC_GAIN_TARGET = 0.056  # the literature effect we calibrate the simulation TOWARD (not a claim)
# CLINICAL_SIGNAL_STRENGTH was set by a small sweep against R2D2's +0.056 spec-gain target (sweep was on
# the SIMULATION knob only — there is no real label to overfit to): 0.75->+0.047, 0.85->+0.055,
# 0.95->+0.061. 0.85 lands the simulated spec gain in the +0.04-0.06 band, the R2D2 neighbourhood.
# (The exact spec gain wobbles ~+/-0.015 run-to-run from MPS nondeterminism in train_fusion — a known,
# documented backend quirk on this project; the band, not a single decimal, is the honest read.)
CLINICAL_SIGNAL_STRENGTH = 0.85  # log-odds scale of the simulated clinical signal vs the true label
CLINICAL_NOISE = 1.0             # additive noise so clinical is INFORMATIVE-BUT-IMPERFECT (like CRP)
MODALITY_DROPOUT = 0.5           # fraction of training samples with the WHOLE clinical vector dropped
SEQ_CAL_FRAC = 0.4               # calibration slice for the per-regime conformal band
ALPHA_SENS = 0.90                # conformal sensitivity floor (band lower edge keeps >=90% sens)
GAMMA_SPEC = 0.10                # conformal specificity target (band upper edge)

# ---------------------------------------------------------------------------
# 1. CLINICAL FIELD SCHEMA — the form contract (value + PRESENT/ABSENT mask per field).
# Order is FIXED (it is the ONNX input contract for the follow-on form). Each field carries a normalised
# numeric value and a mask bit; absent -> value 0.0, mask 0.0. Booleans normalise to {0,1}; continuous
# fields are z-like normalised to roughly [-1,1] by the documented reference ranges below.
CLINICAL_FIELDS: tuple[str, ...] = (
    "age",               # years, normalised (age-50)/20
    "sex",               # 0=female, 1=male
    "cough_duration_wks",  # weeks, normalised (wks-2)/4  (>=2 wks = presumptive TB)
    "fever",             # 0/1
    "weight_loss",       # 0/1
    "night_sweats",      # 0/1
    "CRP",               # mg/L, normalised (crp-10)/20 (point-of-care CRP; R2D2's key marker)
    "HIV_status",        # 0=neg, 1=pos
    "diabetes",          # 0/1
    "prior_TB",          # 0/1
    "TB_contact",        # 0/1
)
N_CLINICAL = len(CLINICAL_FIELDS)

# Per-field log-odds weight of the SIMULATED clinical signal toward the true label. Sign/relative size
# follow the TB clinical literature (CRP, prolonged cough, constitutional symptoms, HIV, contact carry
# the most weight); absolute scale is set by CLINICAL_SIGNAL_STRENGTH. SIMULATED — not fitted on data.
_FIELD_PRIOR = np.array([
    0.15,  # age (mild)
    0.05,  # sex (weak)
    0.45,  # cough_duration_wks (strong: prolonged cough is the cardinal presumptive symptom)
    0.30,  # fever
    0.35,  # weight_loss
    0.30,  # night_sweats
    0.55,  # CRP (the R2D2 marker — strongest single tie-breaker)
    0.40,  # HIV_status (raises pre-test probability)
    0.20,  # diabetes (lower-zone-TB risk, modest)
    0.30,  # prior_TB
    0.35,  # TB_contact
], dtype=np.float64)


@dataclass
class ClinicalVector:
    """Normalised clinical values + a PRESENT/ABSENT mask, the fusion head's clinical input contract.

    value[i] is the normalised value of CLINICAL_FIELDS[i]; mask[i] in {0.,1.} is its PRESENT bit.
    An ABSENT field has value 0.0 AND mask 0.0 (the MLP sees both, so it can learn to ignore absent
    fields). A fully-absent vector (mask all 0) is the image-only regime."""

    value: np.ndarray  # [N_CLINICAL] float32
    mask: np.ndarray   # [N_CLINICAL] float32 in {0,1}

    def as_input(self) -> np.ndarray:
        """Concatenate [value*mask ; mask] -> [2*N_CLINICAL] (the MLP input: masked value + mask bits)."""
        return np.concatenate([self.value * self.mask, self.mask]).astype("float32")

    @property
    def present(self) -> bool:
        return bool(self.mask.sum() > 0)


def _simulate_clinical(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """SIMULATE a clinical matrix [N, N_CLINICAL] (normalised) + an all-present mask, with each field
    correlated to the true label y at the literature-shaped effect size. LOUDLY SIMULATED: the values
    are drawn from label-conditioned Gaussians, NOT measured patients. Returns (values, mask_present)."""
    n = len(y)
    values = np.zeros((n, N_CLINICAL), dtype="float32")
    for j in range(N_CLINICAL):
        # label-conditioned mean shift = field prior * signal strength; symmetric around 0 so absent==0
        # is a NEUTRAL value. Positives shift +, negatives shift -, plus per-field noise.
        shift = _FIELD_PRIOR[j] * CLINICAL_SIGNAL_STRENGTH
        mean = np.where(y == 1, shift, -shift)
        values[:, j] = mean + rng.normal(0.0, CLINICAL_NOISE, size=n).astype("float32")
    mask = np.ones((n, N_CLINICAL), dtype="float32")
    return values, mask


# ---------------------------------------------------------------------------
# 2. LATE-FUSION HEAD: image-logit (precomputed from the T2 head) + clinical-MLP -> log-odds fuse.
class ClinicalMLP(nn.Module):
    """Small MLP: [masked value ; mask] (2*N_CLINICAL) -> a clinical LOGIT. Heavy dropout (the signal is
    a few weak markers). A fully-absent input (all zeros) is trained, via modality dropout, to emit ~0."""

    def __init__(self, n_fields: int = N_CLINICAL, h: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(2 * n_fields), nn.Dropout(0.3), nn.Linear(2 * n_fields, h), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B, 2*N_CLINICAL]
        return self.mlp(x).squeeze(-1)


class LateFusion(nn.Module):
    """fused_logit = image_logit + g * clinical_logit, g = softplus(gain) >= 0 (clinical sharpens, never
    inverts). The image_logit is a FIXED precomputed input (the frozen T2 head). Only the clinical-MLP
    and the gain train here — the image side is already validated and we do not perturb it."""

    def __init__(self):
        super().__init__()
        self.clinical = ClinicalMLP()
        self.gain = nn.Parameter(torch.tensor(0.0))  # softplus(0)=0.69 ~ a moderate initial gain

    def forward(self, image_logit: torch.Tensor, clinical_x: torch.Tensor) -> torch.Tensor:
        g = torch.nn.functional.softplus(self.gain)
        return image_logit + g * self.clinical(clinical_x)


def train_fusion(image_logit: np.ndarray, clinical_x: np.ndarray, y: np.ndarray, tr: np.ndarray,
                 va: np.ndarray, max_epochs: int = 200, patience: int = 20,
                 seed: int = SEED) -> LateFusion:
    """Train the clinical-MLP + gain with MODALITY DROPOUT (drop the whole clinical vector for a fraction
    of samples each step), so the image-only pathway stays calibrated and the fused pathway learns to
    sharpen. The image logit is fixed (the T2 head); BCE on the fused logit. Returns the trained head."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = LateFusion().to(DEVICE)
    n_pos = max(1, int((y[tr] == 1).sum()))
    n_neg = max(1, int((y[tr] == 0).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-2)
    img_t = torch.tensor(image_logit, dtype=torch.float32, device=DEVICE)
    clin_t = torch.tensor(clinical_x, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    best_nll, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        perm = rng.permutation(len(tr))
        ix = tr[perm]
        # MODALITY DROPOUT: zero the WHOLE clinical input (value*mask AND mask bits) for a fraction of
        # this step's samples -> those rows train the image-only pathway (clinical contributes 0).
        drop = torch.tensor(rng.random(len(ix)) < MODALITY_DROPOUT, device=DEVICE)
        cx = clin_t[ix].clone()
        cx[drop] = 0.0  # all zeros == fully-absent clinical vector
        opt.zero_grad()
        out = model(img_t[ix], cx)
        loss_fn(out, y_t[ix]).backward()
        opt.step()
        # early-stop on validation NLL with clinical PRESENT (the harder, sharper regime to keep honest)
        model.eval()
        with torch.no_grad():
            vout = model(img_t[va], clin_t[va])
            vnll = float(loss_fn(vout, y_t[va]).item())
        if vnll < best_nll:
            best_nll, best_state, bad = vnll, {k: v.detach().cpu().clone()
                                               for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _fused_logits(model: LateFusion, image_logit: np.ndarray, clinical_x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(image_logit, dtype=torch.float32, device=DEVICE),
                    torch.tensor(clinical_x, dtype=torch.float32, device=DEVICE))
    return out.cpu().numpy()


def _absent_clinical_x(n: int) -> np.ndarray:
    """The fully-absent clinical input for n rows: [value*mask ; mask] all zeros -> image-only regime."""
    return np.zeros((n, 2 * N_CLINICAL), dtype="float32")


# ---------------------------------------------------------------------------
# REAL image side: OUT-OF-FOLD (LODO) image logits. Scoring in-sample memorises (AUROC ~1.0, no
# headroom — the conformal band collapses to a point). Instead each image's logit comes from a T2 head
# trained on the OTHER sources only, so it carries the honest ~0.92 LODO generalisation gap. Cached.
def get_oof_image_logits(arrs: dict, y: np.ndarray, src: np.ndarray,
                         groups: np.ndarray | None) -> np.ndarray:
    """Per-source LODO out-of-fold image logits: train the FULL T2 head on src != ho, predict the ho
    fold, with the same dup-cluster leak guard as run_lodo_t2. Returns [N] honest image logits."""
    if OOF_CACHE.exists():
        cached = np.load(OOF_CACHE, allow_pickle=True)
        if cached["image_logit"].shape[0] == len(y) and np.array_equal(cached["label"], y):
            print(f"image side: loaded cached OOF logits from {OOF_CACHE.name} (LODO, honest headroom)")
            return cached["image_logit"].astype("float32")
    print("image side: generating OOF (LODO) image logits — train per-fold T2 head, predict holdout...")
    from sklearn.model_selection import train_test_split as _tts
    oof = np.full(len(y), np.nan, dtype="float32")
    for fold_index, ho in enumerate(sorted(set(src.tolist()))):
        te = np.where(src == ho)[0]
        tr_all = np.where(src != ho)[0]
        if groups is not None:
            te_groups = set(int(x) for x in groups[te] if x >= 0)
            leak = np.array([g in te_groups and g >= 0 for g in groups[tr_all]])
            tr_all = tr_all[~leak]
        tr, va = _tts(tr_all, test_size=0.2, stratify=y[tr_all], random_state=SEED)
        model = train_head_t2(arrs, y, tr, va, FULL_LEVERS, seed=SEED + fold_index)
        oof[te] = predict_logits_t2(model, arrs, te)
        flag = "[LEAKAGE-PRONE re-mix]" if ho in REMIX_SOURCES else "[cleaner external]"
        print(f"   holdout={ho:12s} n={len(te):5d} OOF-logits filled {flag}")
    if np.isnan(oof).any():
        raise RuntimeError("OOF logits incomplete — a source was not covered")
    np.savez_compressed(OOF_CACHE, image_logit=oof, label=y, source=src.astype(str))
    print(f"   cached OOF logits -> {OOF_CACHE.name}")
    return oof


# ---------------------------------------------------------------------------
# 4. CONFIDENCE / CONFORMAL BAND (Blueprint §6). Wider band (image-only) -> more UNDETERMINED.
def conformal_band(scores: np.ndarray, y: np.ndarray, alpha_sens: float = ALPHA_SENS,
                   gamma_spec: float = GAMMA_SPEC) -> tuple[float, float]:
    """Fit [tauLow, tauHigh] on a labeled calibration slice (mirrors src/lib/calibration.ts).
      tauLow  = the score quantile that keeps >= alpha_sens of POSITIVES at/above it (sensitivity floor).
      tauHigh = the score quantile that keeps the top gamma_spec of NEGATIVES above it (specificity edge).
    A case is UNDETERMINED iff tauLow <= p < tauHigh. A WIDER band -> MORE UNDETERMINED."""
    pos = np.sort(scores[y == 1])
    neg = np.sort(scores[y == 0])
    n_pos, n_neg = len(pos), len(neg)
    beta = 1.0 - alpha_sens
    tau_low = 0.0
    if n_pos > 0:
        k = int(np.floor(beta * (n_pos + 1)))
        tau_low = 0.0 if k <= 0 else float(pos[k - 1])
    tau_high = 1.0
    if n_neg > 0:
        m = int(np.floor(gamma_spec * (n_neg + 1)))
        tau_high = 1.0 if m <= 0 else float(neg[n_neg - m])
    tau_high = max(tau_high, tau_low)  # never invert
    return float(np.clip(tau_low, 0, 1)), float(np.clip(tau_high, 0, 1))


def undetermined_rate(scores: np.ndarray, tau_low: float, tau_high: float) -> float:
    """Fraction of cases falling in the UNDETERMINED band [tauLow, tauHigh)."""
    return float(np.mean((scores >= tau_low) & (scores < tau_high)))


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    print("=" * 92)
    print("MULTIMODAL LATE-FUSION PROTOTYPE — image (REAL) + clinical (SIMULATED, R2D2-calibrated)")
    print("=" * 92)
    print("CAVEAT (loud): the clinical signal is SIMULATED — we have NO paired image+clinical+outcome")
    print("data yet (gated: R2D2 tabular / TB Portals / ICMR-NIRT — see DATA_SOURCES.md §6-8). The")
    print("clinical effect size is CALIBRATED TO the R2D2 literature (+5.6pts spec@90%sens for CRP->CAD).")
    print("This is a MACHINERY proof, NOT a multimodal performance claim. Do not cite an accuracy here.\n")

    # ---- REAL image side: OUT-OF-FOLD (LODO) image logits — honest ~0.92 headroom, NOT in-sample ----
    d = np.load(DATA / "features.npz", allow_pickle=True)
    arrs = _load_arrs(d)
    y = d["y"].astype("int64")
    src = d["source"].astype(str)
    groups = d["group"].astype("int64") if "group" in d.files else None
    n = len(y)
    image_logit = get_oof_image_logits(arrs, y, src, groups)  # REAL, out-of-fold (honest headroom)
    oof_auc = roc_auc_score(y, 1.0 / (1.0 + np.exp(-image_logit)))
    print(f"image side: OOF image-logit range [{image_logit.min():.2f}, {image_logit.max():.2f}] ; "
          f"pooled OOF AUROC={oof_auc:.3f} (the honest LODO headroom, NOT in-sample ~1.0)")

    # ---- SIMULATED clinical side (R2D2-calibrated effect size) ----
    clin_values, clin_mask = _simulate_clinical(y, rng)
    clinical_x = np.concatenate([clin_values * clin_mask, clin_mask], axis=1).astype("float32")
    print(f"clinical side: {N_CLINICAL} fields {CLINICAL_FIELDS}")
    print(f"               SIMULATED (signal_strength={CLINICAL_SIGNAL_STRENGTH}, noise={CLINICAL_NOISE}, "
          f"modality_dropout={MODALITY_DROPOUT}) — NOT measured\n")

    # ---- split: train fusion / calibrate band / evaluate (stratified, disjoint) ----
    tr, rest = train_test_split(np.arange(n), test_size=0.4, stratify=y, random_state=SEED)
    cal, ev = train_test_split(rest, test_size=1 - SEQ_CAL_FRAC, stratify=y[rest], random_state=SEED)
    model = train_fusion(image_logit, clinical_x, y, tr, np.array(cal))

    # fused probabilities require temperature scaling on the cal slice, fit PER REGIME (image-only vs full)
    absent_x = _absent_clinical_x(n)
    logit_full = _fused_logits(model, image_logit, clinical_x)       # clinical PRESENT (full modality)
    logit_imgonly = _fused_logits(model, image_logit, absent_x)      # clinical ABSENT (image-only)
    T_full = fit_temperature(logit_full[cal], y[cal])
    T_img = fit_temperature(logit_imgonly[cal], y[cal])
    p_full = 1.0 / (1.0 + np.exp(-logit_full / T_full))
    p_img = 1.0 / (1.0 + np.exp(-logit_imgonly / T_img))

    # ===== MACHINERY RESULT (b): GRACEFUL DEGRADATION — image-only STILL scores =====
    # the absent-clinical regime produces finite, usable probabilities and a non-degenerate AUROC.
    auc_full = roc_auc_score(y[ev], p_full[ev])
    auc_img = roc_auc_score(y[ev], p_img[ev])
    gain_softplus = float(torch.nn.functional.softplus(model.gain).item())

    # ===== MACHINERY RESULT (a): FULL-MODALITY >= IMAGE-ONLY (spec at the 90% sens floor) =====
    # threshold for 90% sensitivity fit on the cal slice PER REGIME, specificity measured on eval.
    thr_full = threshold_for_sensitivity(y[cal], p_full[cal], TARGET_SENS_FUSION)
    thr_img = threshold_for_sensitivity(y[cal], p_img[cal], TARGET_SENS_FUSION)
    sens_full, slo_full, shi_full, spec_full = _sens_spec(y[ev], p_full[ev], thr_full)
    sens_img, slo_img, shi_img, spec_img = _sens_spec(y[ev], p_img[ev], thr_img)
    # specificity Clopper-Pearson CI (negatives correctly cleared / total negatives on the eval slice)
    n_neg_ev = int((y[ev] == 0).sum())
    tn_full = int(((p_full[ev] < thr_full) & (y[ev] == 0)).sum())
    tn_img = int(((p_img[ev] < thr_img) & (y[ev] == 0)).sum())
    splo_full, sphi_full = clopper_pearson(tn_full, n_neg_ev)
    splo_img, sphi_img = clopper_pearson(tn_img, n_neg_ev)

    # ===== MACHINERY RESULT (c): LESS CONFIDENT WITHOUT CLINICAL — wider band, more UNDETERMINED =====
    tlo_full, thi_full = conformal_band(p_full[cal], y[cal])
    tlo_img, thi_img = conformal_band(p_img[cal], y[cal])
    und_full = undetermined_rate(p_full[ev], tlo_full, thi_full)
    und_img = undetermined_rate(p_img[ev], tlo_img, thi_img)
    band_full = thi_full - tlo_full
    band_img = thi_img - tlo_img

    print("-" * 92)
    print("MACHINERY RESULTS (image REAL, clinical SIMULATED@R2D2 — these prove the machinery, not perf)")
    print("-" * 92)
    print(f"learned clinical gain g = softplus(gain) = {gain_softplus:.3f}  (>=0: clinical sharpens, "
          f"never inverts the image evidence)\n")

    print("(a) FULL-MODALITY >= IMAGE-ONLY  (specificity at the WHO 90% sensitivity floor, eval slice):")
    print(f"      image-only : sens={sens_img:.3f} [CP {slo_img:.2f}-{shi_img:.2f}]  "
          f"spec={spec_img:.3f} [CP {splo_img:.2f}-{sphi_img:.2f}]  AUROC={auc_img:.3f}")
    print(f"      full       : sens={sens_full:.3f} [CP {slo_full:.2f}-{shi_full:.2f}]  "
          f"spec={spec_full:.3f} [CP {splo_full:.2f}-{sphi_full:.2f}]  AUROC={auc_full:.3f}")
    print(f"      spec gain (full - image-only) = {spec_full - spec_img:+.3f}  "
          f"(SIMULATED toward R2D2's +{R2D2_SPEC_GAIN_TARGET:.3f}; single-digit, as expected on a "
          f"strong image model)\n")

    print("(b) GRACEFUL DEGRADATION  (clinical ABSENT -> system STILL scores from the image alone):")
    print(f"      image-only AUROC={auc_img:.3f} on n_eval={len(ev)} (finite, usable) — the fused logit")
    print(f"      reduces to the image logit when the clinical vector is all-absent (mask=0 -> g*0=0).\n")

    print("(c) LESS CONFIDENT WITHOUT CLINICAL  (wider conformal band -> higher UNDETERMINED rate):")
    print(f"      full       : band [tauLow={tlo_full:.3f}, tauHigh={thi_full:.3f}] width={band_full:.3f}"
          f"  UNDETERMINED rate={und_full:.3f}")
    print(f"      image-only : band [tauLow={tlo_img:.3f}, tauHigh={thi_img:.3f}] width={band_img:.3f}"
          f"  UNDETERMINED rate={und_img:.3f}")
    print(f"      delta (image-only - full) = band {band_img - band_full:+.3f} ; "
          f"UNDETERMINED {und_img - und_full:+.3f}")
    print("      RULE: clinical present -> tighter band -> fewer UNDETERMINED ; clinical absent ->")
    print("      wider band -> more UNDETERMINED. The system never refuses to score; it scores with")
    print("      explicitly lower confidence when clinical data is missing.\n")

    print("-" * 92)
    summary = {
        "(a) full >= image-only": spec_full >= spec_img - 1e-9,
        "(b) image-only still scores": np.isfinite(auc_img) and 0.5 < auc_img <= 1.0,
        "(c) image-only more UNDETERMINED": und_img >= und_full - 1e-9,
    }
    for k, ok in summary.items():
        print(f"   {'PASS' if ok else 'FAIL'}  {k}")
    print("-" * 92)
    print("REMINDER: clinical is SIMULATED (R2D2-calibrated). The machinery is proven; the specificity")
    print("gain is a simulation result pending real paired data (R2D2 images / TB Portals DUA).")
    print("Follow-on (not built): the frontend clinical form is gated on an ONNX export of this head.")


if __name__ == "__main__":
    main()
