"""CXR preprocessing: CLAHE + lung SOFT-mask (panel-corrected — no hard crop).

Feeds the Rad-DINO path. CLAHE normalizes scanner contrast; a pretrained TorchXRayVision
lung segmenter (PSPNet) provides a mask that we DILATE + soft-attenuate (not zero) outside
the lungs and crop to a margined bbox — retaining hila, costophrenic angles, and apices
(all TB-relevant), while removing borders/markers/collimation that drive site shortcuts.

Segmentation is best-effort: if the seg model is unavailable or fails on an image, we fall
back to CLAHE-only (still functional). TorchXRayVision features use their OWN preprocessing,
NOT this function (see extract_features.py).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_SEG = None
_LUNG_IDX: list[int] | None = None
_SEG_TRIED = False


def _get_seg():
    """Lazy-load the pretrained lung segmenter; identify lung target channels by name."""
    global _SEG, _LUNG_IDX, _SEG_TRIED
    if _SEG_TRIED:
        return _SEG, _LUNG_IDX
    _SEG_TRIED = True
    try:
        import torchxrayvision as xrv

        m = xrv.baseline_models.chestx_det.PSPNet().eval()
        targets = [str(t).lower() for t in m.targets]
        idx = [i for i, t in enumerate(targets) if "lung" in t]
        _SEG, _LUNG_IDX = (m, idx) if idx else (None, None)
    except Exception:
        _SEG, _LUNG_IDX = None, None
    return _SEG, _LUNG_IDX


def _lung_mask(gray_u8: np.ndarray) -> np.ndarray | None:
    seg, idx = _get_seg()
    if seg is None or not idx:
        return None
    try:
        import torchxrayvision as xrv

        norm = xrv.datasets.normalize(gray_u8.astype("float32"), 255)  # -> [-1024, 1024]
        t = torch.from_numpy(norm)[None, None, ...]
        t = F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = torch.sigmoid(seg(t))  # [1, C, 512, 512]
        lung = out[0, idx].amax(0).cpu().numpy()
        lung = cv2.resize(lung, (gray_u8.shape[1], gray_u8.shape[0]))
        return (lung > 0.5).astype("uint8")
    except Exception:
        return None


def preprocess_image(path: str) -> Image.Image:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(path).convert("L"))
    img = _CLAHE.apply(img)

    mask = _lung_mask(img)
    if mask is not None and int(mask.sum()) > 0:
        k = max(5, int(0.04 * max(img.shape)))
        mask_d = cv2.dilate(mask, np.ones((k, k), np.uint8))
        ys, xs = np.where(mask_d > 0)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        my, mx = int(0.10 * (y1 - y0 + 1)), int(0.10 * (x1 - x0 + 1))
        y0, y1 = max(0, y0 - my), min(img.shape[0], y1 + my)
        x0, x1 = max(0, x0 - mx), min(img.shape[1], x1 + mx)
        # soft-attenuate outside the lungs to 30% (NOT zero — keeps foundation features sane)
        soft = (img.astype("float32") * (0.3 + 0.7 * mask_d)).clip(0, 255).astype("uint8")
        img = soft[y0:y1, x0:x1]

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
