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

    def to_dict(self) -> dict[str, Any]:
        return {
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

    def __init__(self, hf_offline: bool = True) -> None:
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

        # --- Heads ---
        # Architecture (d_tok=768, d_cls=768, d_txrv=1042) and lever set MUST match how
        # the .pt was saved by train_tb.py — i.e. the full {fusion, zonal, box} blend.
        d_tok, d_cls, d_txrv = 768, 768, 1042
        self.head_t2 = TBHeadT2(d_tok, d_cls, d_txrv, frozenset({"fusion", "zonal", "box"})).to(DEVICE)
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
        if _detect_inversion(g):
            warnings.append("suspected MONOCHROME1 polarity (heuristic only; DICOM tag needed for medical-grade)")
        harm = _harmonize(g)  # uint8, shorter-side == WORKING_RES, antialiased
        latency["harmonize"] = int((time.perf_counter() - t0) * 1000)

        # --- 2. seg + crop + letterbox to square ---
        t0 = time.perf_counter()
        crop_img_pil, crop_box, harm_wh = self._seg_crop_single(harm, warnings)
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
        zones_t = torch.zeros(1, PATCH_GRID * PATCH_GRID, 7, dtype=torch.float32, device=DEVICE)
        self.head_t2._zones = zones_t
        with torch.no_grad():
            out_t2 = self.head_t2(cls_t, patches_t, txrv_t)
            logit_t2 = float(out_t2["logit"].cpu().numpy()[0])
            logit_seq = float(self.head_seq(cls_t, patches_t, txrv_t).cpu().numpy()[0])
        tb_prob = _platt_sigmoid(logit_t2, self.T)
        s_inactive = _platt_sigmoid(logit_seq, self.T_sequelae)
        latency["heads"] = int((time.perf_counter() - t0) * 1000)

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
        )

    # ----------------------------------------------------------------------
    # Stage internals — same math as extract_features.py, batch size 1.
    # ----------------------------------------------------------------------
    def _seg_crop_single(
        self, harm: np.ndarray, warnings: list[str]
    ) -> tuple[Image.Image, tuple[int, int, int, int], tuple[int, int]]:
        """Lung-mask + dilate + 18% margin crop + LETTERBOX to square. Mirrors extract_features.seg_crop
        for batch size 1. Falls back to the whole harmonized frame (letterboxed) when seg is unavailable
        or returns an empty mask; both fallbacks record an image_quality warning."""
        H, W = harm.shape
        if self._seg is None or not self._lung_idx:
            warnings.append("no lung segmenter available (fallback to whole frame)")
            cb = (0, H, 0, W)
            lb = letterbox_to_square(harm, cb, pad_value=0.0)
            return Image.fromarray(cv2.cvtColor(lb, cv2.COLOR_GRAY2RGB)), cb, (W, H)

        norm = self._xrv.datasets.normalize(cv2.resize(harm, (512, 512)).astype("float32"), 255)
        t = torch.from_numpy(norm)[None, None, ...].to(DEVICE)
        with torch.no_grad():
            seg_prob = torch.sigmoid(self._seg(t))  # [1, C, 512, 512]
        m_small = seg_prob[:, self._lung_idx].amax(dim=1)[0].cpu().numpy()  # [512,512]
        mask = (cv2.resize(m_small, (W, H)) > 0.5).astype("uint8")

        if mask.sum() == 0:
            warnings.append("no clear lung mask (fallback to whole frame)")
            cb = (0, H, 0, W)
            soft = harm
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

        # LETTERBOX: pad cropped frame to square so Rad-DINO's shortest_edge=518 + center_crop=518x518
        # becomes a uniform resize (no chopping of apical/costophrenic content). This is the M12 fix.
        lb = letterbox_to_square(soft, cb, pad_value=0.0)
        S = lb.shape[0]
        if S != lb.shape[1]:  # defensive — letterbox_to_square produces SxS
            warnings.append("non-square crop after letterbox (unexpected; fed as-is to Rad-DINO)")
        return Image.fromarray(cv2.cvtColor(lb, cv2.COLOR_GRAY2RGB)), cb, (W, H)

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
