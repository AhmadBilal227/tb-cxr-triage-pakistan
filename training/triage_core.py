"""TriageEngine — local-mode full pipeline (Milestone 22).

The user's M4 ships with the full validated stack (Rad-DINO + TorchXRayVision + the
deployed TBHeadT2 + InactiveSequelaeHead under their calibrated temperatures). M21
flipped gpt-5.5 vision to PRIMARY because the M19 ONNX heads cannot execute in the
browser today (no in-browser pathway produces Rad-DINO patch tokens). M22 closes the
loop on local hardware: this module is the SINGLE-SOURCE-OF-TRUTH engine called from
the CLI, the FastAPI server, and (transitively) the frontend's local-mode pathway.

DESIGN INVARIANTS (load-bearing — do not paraphrase the preprocessing):
  - All image preprocessing IMPORTS the exact functions from extract_features.py.
    `_harmonize` (percentile-clip + min-max + antialiased resize to WORKING_RES=1024)
    and the lung-mask + dilate + 18% margin + letterbox-to-square crop are the
    canonical surfaces; any reimplementation here would silently drift the model off
    its training distribution (M12 audit).
  - Rad-DINO (microsoft/rad-dino) is loaded from the HF cache (HF_HUB_OFFLINE=1).
    Its preprocessor does shortest_edge=518 -> center_crop=518x518; our letterbox
    keeps that exact uniform-resize property (no chopping on non-square crops).
  - TorchXRayVision densenet121-res224-all runs on CPU with `op_threshs=None` so
    `dm(x)` returns raw logits (consistent with extract_features.py audit P0-6).
  - Features are stacked exactly as in extract_features.py: cls [768] + patches
    [64, 768] pooled 37x37 -> 8x8 on CPU + txrv [1042] = pooled[1024] + 18 logits.
  - Calibration constants are READ FROM JSON (tb_threshold_t2.json + tb_inactive_meta.json),
    never pasted into source. Threshold 0.6104530692100525 IS our deployed thr@95sens —
    legitimate to reuse here because the head IS the deployed head.
  - The verdict rule is a Python translation of sequelaeEscalation.ts so the local
    pathway emits THE SAME verdict structure the frontend already knows.

AUDIT FIELDS (TriageResult.audit): model_sha (sha256 of tb_head_t2.pt), git_sha,
calibration_version, ISO 8601 timestamp, per-stage latency dict.

Single-engine load: TriageEngine.__init__ does the heavy lifting (Rad-DINO + TXRV +
heads + segmenter). Every subsequent .run() reuses warm models. The FastAPI server
imports a module-level singleton; the CLI constructs one fresh per invocation.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
sys.path.insert(0, str(REPO / "training"))

# Canonical preprocessing — imported, NOT reimplemented (M12 audit). Any change to
# the harmonize/letterbox math must happen in extract_features.py so train + serve
# share one preprocessor. PATCH_GRID lives here so we get the 37x37 -> 8x8 reshape
# the deployed head was trained on.
from extract_features import (  # noqa: E402
    PATCH_GRID,
    WORKING_RES,
    _detect_inversion,
    _harmonize,
    _read_gray,
    letterbox_to_square,
    zone_matrix_from_masks,
)
from preprocess import _get_seg  # noqa: E402  same segmenter used by extract_features
from train_tb import InactiveSequelaeHead, TBHeadT2  # noqa: E402

RAD_ID = "microsoft/rad-dino"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Calibration source: thresholds + temperatures are READ from JSON on every load
# (NEVER hardcoded — the file is the source of truth and a re-fit must propagate).
_CAL_T2_PATH = DATA / "tb_threshold_t2.json"
_CAL_SEQ_PATH = DATA / "tb_inactive_meta.json"
_HEAD_T2_PATH = DATA / "tb_head_t2.pt"
_HEAD_SEQ_PATH = DATA / "tb_head_inactive.pt"

# Verdict thresholds: these are deliberately the SAME constants the frontend
# sequelaeEscalation.ts uses — the rule was designed for THIS head, so the
# constants are legitimate reuse (not "tuned-on-test" leakage). Keep them in lock-
# step with sequelaeEscalation.ts. If you change one, change the other.
BORDERLINE_LOW = 0.35
DEFAULT_BORDERLINE_HIGH = 0.6105  # mirrored from tb_threshold_t2.json @ 0.95 sens
S_INACTIVE_ESCALATE_THRESHOLD = 0.7126  # q30 of confirmed-scar s_inactive under T_seq
SCAR_ABSTAIN_REASON = "scar-shape pattern flagged for re-read"


# ---------------------------------------------------------------------------
# Result dataclass — the wire shape the CLI, server, and frontend all consume.
# Keep field names exactly aligned with localTriage.ts (TypeScript validation
# treats them as canonical).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TriageAudit:
    model_id: str
    model_sha: str
    calibration: dict[str, float]
    git_sha: str
    version: int
    timestamp: str


@dataclass(frozen=True)
class ImageQuality:
    warnings: list[str]


# ---------------------------------------------------------------------------
# M24 enrichment fields — INTERMEDIATES the validated model already computes
# and that this engine USED to discard. None of these change the verdict; they
# are surfaced for UI evidence panels and (downstream) the gpt-as-interpreter
# narrative. ALL OPTIONAL on the wire — older servers omit them; the TS side
# parses with `?:` so the orchestrator degrades gracefully.
#
# Honest-mapping note: `zonal_scores` has 7 keys (upper/mid/lower L+R + hilar)
# because the trained ZonalSoftOR module uses ZONE_NAMES = ("RUZ", "RMZ",
# "RLZ", "LUZ", "LMZ", "LLZ", "HILAR") and the hilar/mediastinal channel is a
# single combined zone. We do NOT invent a L/R split where the model does not
# compute one — see CASE_STUDY M24.
# ---------------------------------------------------------------------------
ZONE_KEYS: tuple[str, ...] = (
    "upper_r", "mid_r", "lower_r", "upper_l", "mid_l", "lower_l", "hilar",
)
# 18 TorchXRayVision label names in the canonical order the DenseNet emits. The
# trained head consumes these as the last 18 entries of the txrv feature block
# (see N_TXRV_LOGITS in train_tb.py). These names come VERBATIM from
# `xrv.models.DenseNet(weights="densenet121-res224-all").pathologies` and the
# ORDER matters — column i of the head's `find_target` is the i-th label below.
# (The DenseNet returns "Lung Lesion" / "Lung Opacity" / "Enlarged
# Cardiomediastinum" with spaces; we keep them as-is to preserve the wire
# contract with the source of truth.)
TXRV_LABELS: tuple[str, ...] = (
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture",
    "Lung Opacity", "Enlarged Cardiomediastinum",
)


@dataclass(frozen=True)
class TriageResult:
    tb_prob: float
    tb_logit: float
    s_inactive: float
    verdict: str  # "tb" | "no_tb" | "abstain"
    decided_at_threshold: float
    safety_net_applied: str | None
    image_quality: ImageQuality
    latency_ms: dict[str, int]
    audit: TriageAudit
    # ---- M24 enrichment intermediates (None when a lever was off or the model lacks the channel) ----
    box_evidence_grid: list[list[float]] | None = None
    """8x8 per-cell SIGMOID probabilities from the box-evidence head, BEFORE the LSE-LBA pool.
    Captured from TBHeadT2's `evidence` return key; the same map `train_tb.evidence_maps_t2`
    produces for localization scoring. `None` when the box lever is off."""

    zonal_scores: dict[str, float] | None = None
    """Per-zone calibrated TB probability `sigmoid(zone_logit / T)` for the 7 trained zones
    (`upper_r, mid_r, lower_r, upper_l, mid_l, lower_l, hilar`). Captured from
    ZonalSoftOR's `zone_logits` return. `None` when the zonal lever is off OR a zero
    zone-membership matrix was supplied (the current deployment passes zeros — see the
    `_zones` assignment in `run()` — so this surface honestly tells the UI 'zone evidence
    is not available on this run' rather than fabricating a distribution)."""

    txrv_pathologies: dict[str, float] | None = None
    """Sigmoid-calibrated probabilities for the 18 TorchXRayVision named-finding logits the
    pipeline already computes (txrv[:, 1024:]). These ARE the features fed into TBHeadT2's
    fusion lever; surfacing them tells the UI which other findings the perception backbone
    sees alongside TB. NOT independent diagnoses — feature scores."""

    crop_box: dict[str, int] | None = None
    """Letterbox seg-crop in ORIGINAL image pixel coordinates: {x, y, w, h}. The UI uses
    this to align the 8x8 box-evidence overlay with the source CXR."""

    inversion_detected: bool | None = None
    """`_detect_inversion(g)`'s heuristic result (suspected MONOCHROME1 polarity). Already
    drives the image_quality warning; surfacing the bool lets the UI render a chip."""

    tta_passes: list[float] | None = None
    """When `use_tta=True`, the K_PASSES per-augmentation calibrated TB probabilities whose
    mean IS `tb_prob`. `None` on the single-pass (M22 deployed) path so the default wire
    shape is unchanged. CXR-safe augs only (identity + hflip + brighten + darken + contrast)."""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tb_prob": self.tb_prob,
            "tb_logit": self.tb_logit,
            "s_inactive": self.s_inactive,
            "verdict": self.verdict,
            "decided_at_threshold": self.decided_at_threshold,
            "safety_net_applied": self.safety_net_applied,
            "image_quality": {"warnings": list(self.image_quality.warnings)},
            "latency_ms": dict(self.latency_ms),
            "audit": asdict(self.audit),
        }
        # Optional enrichment fields — only emit when present so the wire stays
        # backwards-compatible (older clients ignore unknown keys; new clients
        # use `?:` for these).
        if self.box_evidence_grid is not None:
            out["box_evidence_grid"] = [list(row) for row in self.box_evidence_grid]
        if self.zonal_scores is not None:
            out["zonal_scores"] = dict(self.zonal_scores)
        if self.txrv_pathologies is not None:
            out["txrv_pathologies"] = dict(self.txrv_pathologies)
        if self.crop_box is not None:
            out["crop_box"] = dict(self.crop_box)
        if self.inversion_detected is not None:
            out["inversion_detected"] = bool(self.inversion_detected)
        if self.tta_passes is not None:
            out["tta_passes"] = [float(p) for p in self.tta_passes]
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _platt_sigmoid(logit: float, temperature: float) -> float:
    """Calibrated probability under temperature scaling (Guo et al.): p = sigmoid(logit / T).
    The training pipeline fits T on a validation slice and stores it in tb_threshold_t2.json."""
    z = float(logit) / max(float(temperature), 1e-3)
    if z >= 0:
        e = float(np.exp(-z))
        return 1.0 / (1.0 + e)
    e = float(np.exp(z))
    return e / (1.0 + e)


def _verdict_from(p_tb: float, s_inactive: float, borderline_high: float) -> tuple[str, str | None]:
    """Python translation of src/lib/pipeline/sequelaeEscalation.ts.

    The frontend rule is the same shape: deterministic safety net wraps the head;
    a borderline tb_prob with a high s_inactive escalates NO_TB to ABSTAIN; a high
    s_inactive can NEVER clear a TB call. We compute the base verdict here from
    p_tb vs the calibrated 0.95-sensitivity threshold (`borderline_high`), then
    apply the escalation rule the same way the orchestrator will after the local
    call returns. Returns (verdict, safety_net_applied_reason_or_None).
    """
    if p_tb >= borderline_high:
        return "tb", None
    base_verdict = "no_tb"
    # Escalation: borderline band [BORDERLINE_LOW, borderline_high) + high s_inactive -> abstain.
    in_band = (p_tb >= BORDERLINE_LOW) and (p_tb < borderline_high)
    if in_band and s_inactive >= S_INACTIVE_ESCALATE_THRESHOLD:
        return "abstain", SCAR_ABSTAIN_REASON
    return base_verdict, None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class TriageEngine:
    """Warm-loaded local triage. Construct ONCE per process; call .run() many times.

    Loads on construct: the Rad-DINO ViT (microsoft/rad-dino) onto MPS, the TXRV
    DenseNet on CPU with op_threshs=None, the TBHeadT2 weights, the InactiveSequelaeHead
    weights, and the lung segmenter. Calibration constants are read from JSON. Per-run
    latency is sub-second on warm M4."""

    def __init__(
        self,
        hf_offline: bool = True,
        *,
        use_tta: bool = False,
        use_locked_protocol: bool = False,
        head_kind: str = "zonal-softor",
    ) -> None:
        # Force the HF cache to be the sole source — no silent network fetches at runtime.
        # Set BEFORE the transformers import path triggers any cache lookup downstream.
        if hf_offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Calibration constants — READ, never hardcoded.
        with open(_CAL_T2_PATH) as f:
            cal_t2 = json.load(f)
        with open(_CAL_SEQ_PATH) as f:
            cal_seq = json.load(f)
        self.T: float = float(cal_t2["temperature"])
        self.thr_at_95sens: float = float(cal_t2["threshold"])
        self.T_sequelae: float = float(cal_seq["temperature"])

        # --- P0 opt-in levers (default OFF → M22 deployed behavior byte-for-byte) ---
        # use_tta: K=5 CXR-safe test-time augmentation, probabilities averaged.
        # use_locked_protocol: T + decision threshold come from the pre-registered
        #   data/p0_locked_calibration.json instead of the deployed tb_threshold_t2.json.
        #   This is the structural defense against per-config calibration leakage: every
        #   P1/P2/P3 evaluation loads the SAME locked T+thr and never re-fits them.
        self.use_tta: bool = use_tta
        if use_locked_protocol:
            from locked_protocol import load_locked_calibration

            locked = load_locked_calibration()
            self.T = float(locked.T)
            self.thr_at_95sens = float(locked.thr_at_95sens)
            # `thr` is the canonical name the locked-protocol tests + measurement read.
            self.thr: float = float(locked.thr_at_95sens)
            self._calibration_source = "p0_locked"
        else:
            self.thr = self.thr_at_95sens
            self._calibration_source = "tb_threshold_t2_json"

        # Mirror the borderline_high to the calibrated threshold so the rule and the
        # decided_at_threshold are one consistent number.
        self.borderline_high: float = self.thr_at_95sens

        # Audit pins (load-time invariants — re-read per .run() for the timestamp only).
        self.model_sha: str = _sha256_file(_HEAD_T2_PATH)
        self.git_sha: str = _git_sha()
        self.calibration_version: int = 1  # bump on contract changes

        # --- Rad-DINO (heavy ViT -> MPS) ---
        self.proc = AutoImageProcessor.from_pretrained(RAD_ID)
        self.rad = AutoModel.from_pretrained(RAD_ID).to(DEVICE).eval()
        for p in self.rad.parameters():
            p.requires_grad_(False)

        # --- TorchXRayVision (light DenseNet on CPU; raw logits) ---
        import torchxrayvision as xrv

        self.dm = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
        self.dm.op_threshs = None  # raw logits, matches extract_features.py audit P0-6
        for p in self.dm.parameters():
            p.requires_grad_(False)
        self._xcrop = xrv.datasets.XRayCenterCrop()
        self._xresize = xrv.datasets.XRayResizer(224)
        self._xrv = xrv

        # --- Lung segmenter (same one extract_features uses) ---
        seg, lung_idx = _get_seg()
        if seg is not None:
            seg = seg.to(DEVICE)
        self._seg = seg
        self._lung_idx = lung_idx
        # M24: SEPARATE channel indices for the lung-only and hilus/mediastinum channels
        # so we can build a real zone-membership matrix on the fly (matches
        # extract_features.py main() which reads both indices off `seg.targets`). When the
        # segmenter is unavailable, both lists are empty — `zonal_scores` then reports None.
        self._lung_only_idx: list[int] = []
        self._hilus_med_idx: list[int] = []
        if seg is not None:
            targets = [str(t).lower() for t in seg.targets]
            self._lung_only_idx = [i for i, t in enumerate(targets) if "lung" in t]
            self._hilus_med_idx = [i for i, t in enumerate(targets) if "hilus" in t or "mediastinum" in t]

        # --- Heads ---
        # Architecture (d_tok=768, d_cls=768, d_txrv=1042) and lever set MUST match how
        # the .pt was saved by train_tb.py — i.e. the full {fusion, zonal, box} blend.
        d_tok, d_cls, d_txrv = 768, 768, 1042
        # head_kind defaults to 'zonal-softor' (the deployed M22/M24 head) for backward compat;
        # 'soft-attn-pool' loads a P1 SoftAttnPool artifact instead (same lever set, ONE lever swap).
        self.head_kind = head_kind
        self.head_t2 = TBHeadT2(d_tok, d_cls, d_txrv, frozenset({"fusion", "zonal", "box"}),
                                head_kind=head_kind).to(DEVICE)
        self.head_t2.load_state_dict(torch.load(_HEAD_T2_PATH, map_location=DEVICE))
        self.head_t2.eval()
        self.head_seq = InactiveSequelaeHead(d_tok, d_cls).to(DEVICE)
        self.head_seq.load_state_dict(torch.load(_HEAD_SEQ_PATH, map_location=DEVICE))
        self.head_seq.eval()

    # ----------------------------------------------------------------------
    # Public run() — bytes in, TriageResult out. Five timed stages.
    # ----------------------------------------------------------------------
    def run(self, image_bytes: bytes) -> TriageResult:
        latency: dict[str, int] = {}
        warnings: list[str] = []
        t_total0 = time.perf_counter()

        # --- 1. harmonize (read + percentile-clip + min-max + antialiased resize) ---
        t0 = time.perf_counter()
        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        g = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if g is None:
            # PIL fallback for formats opencv can't decode (e.g. palette PNG, DICOM-as-RGBA).
            try:
                g = np.array(Image.open(io.BytesIO(image_bytes)).convert("L"))
            except Exception as e:
                raise ValueError(f"could not decode image bytes: {e!r}") from e
        inversion = bool(_detect_inversion(g))
        if inversion:
            warnings.append("suspected MONOCHROME1 polarity (heuristic only; DICOM tag needed for medical-grade)")
        orig_h, orig_w = int(g.shape[0]), int(g.shape[1])
        harm = _harmonize(g)  # uint8, shorter-side == WORKING_RES, antialiased
        latency["harmonize"] = int((time.perf_counter() - t0) * 1000)

        # --- 2. seg + crop + letterbox to square (also exposes lung/hilus masks for zones) ---
        t0 = time.perf_counter()
        crop_img_pil, crop_box_tuple, harm_wh, lung_crop, hilus_crop = self._seg_crop_single(harm, warnings)
        latency["seg"] = int((time.perf_counter() - t0) * 1000)

        # --- 3. Rad-DINO (MPS) -> cls[768] + patches[64,768] ---
        t0 = time.perf_counter()
        cls_arr, patches_arr = self._rad_forward(crop_img_pil)
        latency["rad_dino"] = int((time.perf_counter() - t0) * 1000)

        # --- 4. TXRV (CPU) -> txrv[1042] ---
        t0 = time.perf_counter()
        txrv_arr = self._txrv_forward(harm)
        latency["txrv"] = int((time.perf_counter() - t0) * 1000)

        # --- 5. Heads -> calibrated p_tb + calibrated s_inactive ---
        t0 = time.perf_counter()
        cls_t = torch.tensor(cls_arr[None, ...], dtype=torch.float32, device=DEVICE)
        patches_t = torch.tensor(patches_arr[None, ...], dtype=torch.float32, device=DEVICE)
        txrv_t = torch.tensor(txrv_arr[None, ...], dtype=torch.float32, device=DEVICE)
        # zones: the deployed head trained with zone supervision; at inference we pass an
        # all-zero soft zone matrix when we are not computing zones on the fly. The fitted
        # zonal lever's gate has g_min=0.25 so it still contributes via the floor. This mirrors
        # the M19 ONNX export path (training/export_onnx.py uses the same convention).
        #
        # M24 invariant — DO NOT touch this forward. The 0.9999769 headline on tb-sample-1.jpg
        # is conditioned on the zero-zones convention; switching to real zones here would
        # change tb_prob and bust the M22 EXPERIMENT_LOG tripwire. Real zones are used in a
        # SEPARATE, post-hoc forward (see below) solely to populate `zonal_scores` for the
        # UI; the verdict pin stays.
        zones_zero = torch.zeros(1, PATCH_GRID * PATCH_GRID, 7, dtype=torch.float32, device=DEVICE)
        self.head_t2._zones = zones_zero
        with torch.no_grad():
            out_t2 = self.head_t2(cls_t, patches_t, txrv_t)
            logit_t2 = float(out_t2["logit"].cpu().numpy()[0])
            # Box-evidence: TBHeadT2 already emits the 8x8 per-cell logits we used to discard.
            # `evidence` is [B, 64] of LOGITS; sigmoid + reshape to 8x8 matches `train_tb.evidence_maps_t2`.
            box_evidence_grid: list[list[float]] | None = None
            ev_t = out_t2.get("evidence")
            if ev_t is not None:
                ev_probs = torch.sigmoid(ev_t).reshape(1, PATCH_GRID, PATCH_GRID).cpu().numpy()[0]
                box_evidence_grid = [[float(ev_probs[r, c]) for c in range(PATCH_GRID)] for r in range(PATCH_GRID)]
            logit_seq = float(self.head_seq(cls_t, patches_t, txrv_t).cpu().numpy()[0])
        tb_prob = _platt_sigmoid(logit_t2, self.T)
        s_inactive = _platt_sigmoid(logit_seq, self.T_sequelae)
        latency["heads"] = int((time.perf_counter() - t0) * 1000)

        # --- 5a. P0 test-time augmentation (opt-in) ---
        # The single-pass `tb_prob` above IS the identity pass. When use_tta is on we
        # apply the K-1 remaining CXR-safe augmentations to the SEG-CROPPED, normalized
        # crop tensor (not raw PIL), re-run Rad-DINO + the t2 head per variant, and average
        # the calibrated probabilities. The verdict logit / enrichment fields stay pinned to
        # the identity forward (M24 invariant); only `tb_prob` becomes the averaged value.
        tta_probs: list[float] | None = None
        if self.use_tta:
            from tta import tta_passes, tta_average_probs

            base_chw = self._pil_to_chw01(crop_img_pil)
            tta_probs = []
            for i, variant in enumerate(tta_passes(base_chw)):
                if i == 0:
                    # identity pass — reuse the already-computed verdict probability
                    tta_probs.append(float(tb_prob))
                    continue
                aug_pil = self._chw01_to_pil(variant)
                tta_probs.append(self._t2_prob_for_crop(aug_pil, txrv_t))
            tb_prob = tta_average_probs(tta_probs)

        # --- 5b. M24 enrichment forward (zonal_scores only) — SEPARATE from the verdict forward.
        # If a real zone matrix can be built from the lung+hilus masks, run TBHeadT2 again on the
        # SAME features but with real zones, and capture `zone_logits` to report per-zone
        # calibrated TB probabilities under T. The verdict logit stays from the zero-zones forward.
        zonal_scores: dict[str, float] | None = None
        if lung_crop is not None and hilus_crop is not None:
            zm = zone_matrix_from_masks(lung_crop, hilus_crop, G=PATCH_GRID)  # [64, 7]
            if zm.sum() > 0:  # at least one non-background patch
                zones_t = torch.tensor(zm[None, ...], dtype=torch.float32, device=DEVICE)
                # Snapshot + restore: leave the engine state identical to the verdict forward.
                self.head_t2._zones = zones_t
                with torch.no_grad():
                    out_real = self.head_t2(cls_t, patches_t, txrv_t)
                self.head_t2._zones = zones_zero  # restore
                zl_t = out_real.get("zone_logits")
                if zl_t is not None:
                    zl = zl_t.cpu().numpy()[0]  # [7]
                    # Per-zone calibrated TB probability under the SAME temperature the image
                    # logit is calibrated with. The per-zone logits are scored by the SAME
                    # 1-dim scorer + zone-floored gate that produces the image-level zonal logit
                    # (which the trained head log-odds-blends into the final tb_logit), so applying
                    # T is the natural calibration choice; same alphabet as `tb_prob`.
                    zonal_scores = {key: _platt_sigmoid(float(zl[i]), self.T) for i, key in enumerate(ZONE_KEYS)}

        # TXRV pathologies: the last 18 entries of txrv_arr are RAW logits; sigmoid -> probs.
        # These ARE the features the trained head's fusion lever consumes — surfacing them shows
        # the UI what OTHER findings the perception backbone sees. NOT independent diagnoses;
        # uncalibrated per-class (TXRV densenet121-res224-all probabilities, not site-recalibrated).
        find_logits = txrv_arr[len(txrv_arr) - 18 :]
        txrv_pathologies: dict[str, float] = {
            label: float(1.0 / (1.0 + np.exp(-float(find_logits[i]))))
            for i, label in enumerate(TXRV_LABELS)
        }

        # crop_box in the HARMONIZED frame; map back to ORIGINAL pixel coords. _harmonize
        # antialias-resizes the SHORTER side to WORKING_RES while preserving aspect ratio,
        # so the harmonized-to-original scale factor is uniform = orig_short / WORKING_RES.
        # (We can't recover sub-pixel translation if _harmonize ever added padding — it does
        # not; harmonize is a pure resize — so this is a faithful pixel-space mapping.)
        y0h, y1h, x0h, x1h = crop_box_tuple
        harm_w, harm_h = harm_wh  # harmonized width, height
        scale_x = orig_w / float(harm_w) if harm_w > 0 else 1.0
        scale_y = orig_h / float(harm_h) if harm_h > 0 else 1.0
        crop_box = {
            "x": int(round(x0h * scale_x)),
            "y": int(round(y0h * scale_y)),
            "w": int(round((x1h - x0h) * scale_x)),
            "h": int(round((y1h - y0h) * scale_y)),
        }

        latency["total"] = int((time.perf_counter() - t_total0) * 1000)

        verdict, safety_net = _verdict_from(tb_prob, s_inactive, self.borderline_high)

        audit = TriageAudit(
            model_id="tb_head_t2",
            model_sha=self.model_sha,
            calibration={
                "T": self.T,
                "thr_at_95sens": self.thr_at_95sens,
                "T_sequelae": self.T_sequelae,
            },
            git_sha=self.git_sha,
            version=self.calibration_version,
            timestamp=_iso_now(),
        )
        return TriageResult(
            tb_prob=float(tb_prob),
            tb_logit=float(logit_t2),
            s_inactive=float(s_inactive),
            verdict=verdict,
            decided_at_threshold=self.thr_at_95sens,
            safety_net_applied=safety_net,
            image_quality=ImageQuality(warnings=warnings),
            latency_ms=latency,
            audit=audit,
            box_evidence_grid=box_evidence_grid,
            zonal_scores=zonal_scores,
            txrv_pathologies=txrv_pathologies,
            crop_box=crop_box,
            inversion_detected=inversion,
            tta_passes=tta_probs,
        )

    # ----------------------------------------------------------------------
    # Stage internals — same math as extract_features.py, batch size 1.
    # ----------------------------------------------------------------------
    def _seg_crop_single(
        self, harm: np.ndarray, warnings: list[str]
    ) -> tuple[
        Image.Image,
        tuple[int, int, int, int],
        tuple[int, int],
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Lung-mask + dilate + 18% margin crop + LETTERBOX to square. Mirrors extract_features.seg_crop
        for batch size 1. Falls back to the whole harmonized frame (letterboxed) when seg is unavailable
        or returns an empty mask; both fallbacks record an image_quality warning.

        M24 extension — ALSO returns the LETTERBOXED lung-only and hilus/mediastinum SOFT masks (or
        None when the segmenter is unavailable / channels missing) so the engine can build the same
        zone-membership matrix `extract_features.zone_matrix_from_masks` expects. The masks register
        pixel-for-pixel with the Rad-DINO patch tokens (extract_features.py letterbox invariant).
        """
        H, W = harm.shape
        if self._seg is None or not self._lung_idx:
            warnings.append("no lung segmenter available (fallback to whole frame)")
            cb = (0, H, 0, W)
            lb = letterbox_to_square(harm, cb, pad_value=0.0)
            return Image.fromarray(cv2.cvtColor(lb, cv2.COLOR_GRAY2RGB)), cb, (W, H), None, None

        norm = self._xrv.datasets.normalize(cv2.resize(harm, (512, 512)).astype("float32"), 255)
        t = torch.from_numpy(norm)[None, None, ...].to(DEVICE)
        with torch.no_grad():
            seg_prob = torch.sigmoid(self._seg(t))  # [1, C, 512, 512]
        m_small = seg_prob[:, self._lung_idx].amax(dim=1)[0].cpu().numpy()  # [512,512]
        mask = (cv2.resize(m_small, (W, H)) > 0.5).astype("uint8")

        # M24: also fetch the SEPARATE lung-only and hilus/mediastinum channels (the combined
        # `_lung_idx` is the crop driver; zones need them split — extract_features.py main()
        # makes the same split).
        lung_s_full: np.ndarray | None = None
        hil_s_full: np.ndarray | None = None
        if self._lung_only_idx:
            lung_s_small = seg_prob[:, self._lung_only_idx].amax(dim=1)[0].cpu().numpy()
            lung_s_full = cv2.resize(lung_s_small, (W, H))
        if self._hilus_med_idx:
            hil_s_small = seg_prob[:, self._hilus_med_idx].amax(dim=1)[0].cpu().numpy()
            hil_s_full = cv2.resize(hil_s_small, (W, H))

        if mask.sum() == 0:
            warnings.append("no clear lung mask (fallback to whole frame)")
            cb = (0, H, 0, W)
            soft = harm
            lung_crop_raw = lung_s_full
            hilus_crop_raw = hil_s_full
        else:
            k = max(5, int(0.04 * max(H, W)))
            md = cv2.dilate(mask, np.ones((k, k), np.uint8))
            ys, xs = np.where(md > 0)
            y0, x0 = int(ys.min()), int(xs.min())
            y1, x1 = min(H, int(ys.max()) + 1), min(W, int(xs.max()) + 1)
            my, mx = int(0.18 * (y1 - y0)), int(0.18 * (x1 - x0))
            y0, y1 = max(0, y0 - my), min(H, y1 + my)
            x0, x1 = max(0, x0 - mx), min(W, x1 + mx)
            soft = (harm.astype("float32") * (0.3 + 0.7 * md)).clip(0, 255).astype("uint8")[y0:y1, x0:x1]
            cb = (y0, y1, x0, x1)
            lung_crop_raw = lung_s_full[y0:y1, x0:x1] if lung_s_full is not None else None
            hilus_crop_raw = hil_s_full[y0:y1, x0:x1] if hil_s_full is not None else None

        # LETTERBOX: pad cropped frame to square so Rad-DINO's shortest_edge=518 + center_crop=518x518
        # becomes a uniform resize (no chopping of apical/costophrenic content). This is the M12 fix.
        lb = letterbox_to_square(soft, cb, pad_value=0.0)
        S = lb.shape[0]
        if S != lb.shape[1]:  # defensive — letterbox_to_square produces SxS
            warnings.append("non-square crop after letterbox (unexpected; fed as-is to Rad-DINO)")
        # Letterbox the masks IDENTICALLY (pad_value=0 → background / zero lung), so they register
        # with the Rad-DINO patch tokens (same letterbox invariant as extract_features.py).
        lung_crop_lb = (
            letterbox_to_square(lung_crop_raw, cb, pad_value=0.0) if lung_crop_raw is not None else None
        )
        hilus_crop_lb = (
            letterbox_to_square(hilus_crop_raw, cb, pad_value=0.0) if hilus_crop_raw is not None else None
        )
        return (
            Image.fromarray(cv2.cvtColor(lb, cv2.COLOR_GRAY2RGB)),
            cb,
            (W, H),
            lung_crop_lb,
            hilus_crop_lb,
        )

    # ----------------------------------------------------------------------
    # P0 TTA internals — augmentations applied to the NORMALIZED seg-cropped
    # tensor (not raw PIL): harmonize+seg run ONCE, then the K CXR-safe variants
    # diverge only at the Rad-DINO input. The identity pass is bit-identical to
    # the single-pass path, so `tta_passes[0]` equals the non-TTA tb_prob.
    # ----------------------------------------------------------------------
    @staticmethod
    def _pil_to_chw01(img: Image.Image) -> torch.Tensor:
        """RGB PIL → CHW float tensor in [0, 1] (the alphabet tta.tta_passes expects)."""
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW

    @staticmethod
    def _chw01_to_pil(t: torch.Tensor) -> Image.Image:
        """CHW float tensor in [0, 1] → RGB PIL uint8."""
        arr = (t.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
        return Image.fromarray(arr)

    def _t2_prob_for_crop(self, crop_img_pil: Image.Image, txrv_t: torch.Tensor) -> float:
        """Calibrated TB probability for ONE crop variant (zero-zones convention).

        Mirrors the verdict forward exactly: Rad-DINO → TBHeadT2 with all-zero zone
        matrix → sigmoid(logit / T). Used per TTA pass. txrv_t is shared across passes
        (TXRV runs on the harmonized frame, which TTA does not alter)."""
        cls_arr, patches_arr = self._rad_forward(crop_img_pil)
        cls_t = torch.tensor(cls_arr[None, ...], dtype=torch.float32, device=DEVICE)
        patches_t = torch.tensor(patches_arr[None, ...], dtype=torch.float32, device=DEVICE)
        zones_zero = torch.zeros(1, PATCH_GRID * PATCH_GRID, 7, dtype=torch.float32, device=DEVICE)
        self.head_t2._zones = zones_zero
        with torch.no_grad():
            out_t2 = self.head_t2(cls_t, patches_t, txrv_t)
            logit_t2 = float(out_t2["logit"].cpu().numpy()[0])
        return _platt_sigmoid(logit_t2, self.T)

    def _rad_forward(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Rad-DINO pooler_output (CLS, [768]) + patch tokens pooled 37x37 -> 8x8 ([64, 768] fp32).
        Pooling done on CPU because adaptive_avg_pool2d on non-divisible sizes is unsupported on MPS
        (same workaround as extract_features.py)."""
        rin = self.proc(images=[img], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.rad(**rin)
            cls = out.pooler_output.float().cpu().numpy()[0]  # [768]
            tok = out.last_hidden_state[:, 1:, :]  # drop CLS; no register tokens
            g = int(round(tok.shape[1] ** 0.5))
            if tok.shape[1] != g * g:
                raise RuntimeError(f"unexpected Rad-DINO token count {tok.shape[1]} (registers?)")
            grid = tok.transpose(1, 2).reshape(tok.shape[0], tok.shape[2], g, g).float().cpu()
            patches = F.adaptive_avg_pool2d(grid, (PATCH_GRID, PATCH_GRID))
            patches = patches.reshape(grid.shape[0], grid.shape[1], PATCH_GRID ** 2).transpose(1, 2)
            patches_np = patches.numpy().astype("float32")[0]  # [64, 768]
        return cls.astype("float32"), patches_np

    def _txrv_forward(self, harm: np.ndarray) -> np.ndarray:
        """TXRV pooled [1024] + raw 18 logits = [1042]. Operates on the HARMONIZED frame (not the
        cropped/letterboxed one) — same as extract_features.py."""
        norm = self._xrv.datasets.normalize(harm.astype("float32"), 255)[None, ...]  # [1,H,W]
        norm = self._xresize(self._xcrop(norm)).astype("float32")  # [1,224,224]
        with torch.no_grad():
            tx = torch.from_numpy(norm[None, ...])  # [1,1,224,224]
            pooled = self.dm.features2(tx)  # [1, 1024]
            logits = self.dm.classifier(pooled)  # [1, 18]
            arr = torch.cat([pooled, logits], dim=1).numpy()[0]  # [1042]
        return arr.astype("float32")


# Module-level lazy singleton — the FastAPI server hits this on first request,
# the CLI builds one per invocation. Cheap fast-path that avoids re-loading
# Rad-DINO + TXRV + the heads on every server request.
_ENGINE: TriageEngine | None = None


def get_engine() -> TriageEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TriageEngine()
    return _ENGINE
