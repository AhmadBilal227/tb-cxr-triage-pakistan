"""Dual-backbone frozen feature extraction — BATCHED + threaded for M4 utilization.

Per image, cache: Rad-DINO CLS (768) + patch tokens pooled to 8x8=64 tokens (fp16) +
TorchXRayVision DenseNet pooled (1024) + 18 pathology LOGITS = 1042.

RELIABILITY (2026-05-24 audit — see CASE_STUDY.md M12): both backbones now see the SAME
harmonized intensity + resolution so the model cannot cheat on scanner/site cues:
  - monotonic percentile-clip + min-max normalization (NOT CLAHE — CLAHE is non-monotonic and
    pushes a frozen encoder off its pretraining distribution; Rad-DINO was trained on min-max
    [0,255] images, no CLAHE — model card / Pérez-García et al. 2024).
  - resolution harmonization: every source resized to a common shorter-side then lightly blurred,
    so a 4892px source (downsampled) and a 512px source (upsampled) share a band-limit and stop
    being trivially separable (the site-leak canary was balanced-acc 1.000 before this).
  - MONOCHROME1 inversion guard (heuristic; per-source rate logged).
TorchXRayVision then applies its OWN [-1024,1024] normalization on top of the harmonized image;
`op_threshs` is disabled so dm(x) returns true logits (not op-norm'd scores).

    PYTORCH_ENABLE_MPS_FALLBACK=1 python training/extract_features.py
"""
from __future__ import annotations
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
from preprocess import _get_seg  # noqa: E402  (reuse the lung segmenter)

DATA = REPO / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RAD_ID = "microsoft/rad-dino"
BATCH = 16
WORKERS = 6                 # CPU loader threads (capped to leave the machine usable, <90% CPU)
CPU_TORCH_THREADS = 4       # threads the CPU DenseNet may use
# PATCH_GRID is defined in the GEOMETRY section below (kept next to the rasterizer that consumes it).
WORKING_RES = 1024          # common shorter-side w/ antialiased downsample (locked, consistent kernel)
PCLIP = (1.0, 99.0)         # percentile clip for monotonic intensity normalization
INVERT_MARGIN = 25.0        # threshold for DETECTING (not auto-applying) suspected MONOCHROME1
# NOTE (gpt-5.5 steelman + 4-agent audit): we deliberately do NOT add a Gaussian blur to "harmonize"
# resolution — it suppresses the high-frequency signal TB sensitivity depends on (miliary nodules,
# subtle apical infiltrates). Antialiased downsampling (INTER_AREA) is the principled step. The
# site-leak canary is a DIAGNOSTIC we track, not an objective to optimize preprocessing against.


def _read_gray(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        g = np.array(Image.open(path).convert("L"))  # palette/RGB -> luminance, deterministic
    return g


def _detect_inversion(g: np.ndarray) -> bool:
    """DETECT (do not auto-apply) suspected MONOCHROME1 polarity. A correctly-oriented CXR has a
    DARK unexposed border; a border much brighter than the body center suggests inversion. This is
    a brittle heuristic, so we only LOG the per-source rate — medical-grade ingest must read the
    DICOM PhotometricInterpretation tag, not guess from pixels (gpt-5.5 steelman). For these open
    PNG sets (almost certainly MONOCHROME2) the rate should be ~0; a high rate flags a real problem."""
    h, w = g.shape
    bh, bw = max(1, h // 20), max(1, w // 20)
    border = np.concatenate([g[:bh].ravel(), g[-bh:].ravel(), g[:, :bw].ravel(), g[:, -bw:].ravel()])
    center = g[h // 3:2 * h // 3, w // 3:2 * w // 3]
    return float(border.mean()) > float(center.mean()) + INVERT_MARGIN


def _harmonize(g: np.ndarray) -> np.ndarray:
    """Monotonic intensity normalization + resolution harmonization -> uint8.
    Keeps the frozen encoders near their pretraining distribution and removes the per-source
    contrast/DC offset and resolution/sharpness signature that the site-leak canary exposed."""
    lo, hi = np.percentile(g, PCLIP)
    hi = hi if hi > lo else lo + 1.0
    f = np.clip((g.astype("float32") - lo) / (hi - lo), 0.0, 1.0)
    h, w = f.shape
    scale = WORKING_RES / min(h, w)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC  # antialiased downsample, fixed kernel
    f = cv2.resize(f, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)
    return (f * 255.0).clip(0, 255).astype("uint8")


# ============================================================================================
# GEOMETRY (pure numpy — unit-tested in training/test_extract_geom.py)
#
# COORDINATE FRAMES. Four frames matter:
#   (1) ORIGINAL  — the source image as on disk; TBX11K boxes in data.csv are xywh in this frame
#                   (TBX11K originals are 512x512, so original==512x512 for boxed images).
#   (2) HARMONIZED — after `_harmonize` resizes the shorter side to WORKING_RES (uniform scale).
#   (3) CROPPED   — after `seg_crop` crops the harmonized image to the dilated-lung bbox + margin.
#   (4) LETTERBOXED — the CROPPED frame symmetrically PADDED to a SQUARE (side S = max(cw, ch)) before
#                   the Rad-DINO processor. THIS is the frame the patch grid lives in (see below).
#
# WHY LETTERBOX (2026-05-24 fix — center-crop misregistration, found by independent review).
#   The Rad-DINO AutoImageProcessor (preprocessor_config.json) does do_resize size.shortest_edge=518
#   THEN do_center_crop crop_size 518x518. For a NON-SQUARE crop the resize only makes the SHORTER
#   side 518 (longer side stays >518), and the center-crop then CHOPS the long-axis edge strips. So
#   the 37x37 patch tokens span ONLY THE CENTER SQUARE of the cropped rectangle — NOT the full rect.
#   Mapping a box/mask in [0,1] over the FULL cropped rectangle to the 8x8 grid is therefore wrong on
#   the long axis (misregistered by up to ~0.4-0.9 grid cells; apical/costophrenic content in the
#   chopped strips would be assigned to rows Rad-DINO never sees). The 9 original square-crop tests
#   missed it because center-crop is a NO-OP when cropped width == height.
#   FIX: symmetrically pad the cropped frame to a square (letterbox) BEFORE the processor. Then
#   shortest_edge=518 + center_crop(518) is a TRUE uniform resize with no chopping, and [0,1] over the
#   LETTERBOXED square maps linearly to the grid. The IDENTICAL letterbox is applied to the box coords
#   (rasterize_boxes_to_grid) and to the lung/hilus masks (zone_matrix_from_masks) so grid_label and
#   zones live in the SAME letterboxed-square frame as the tokens. Padding bars map to empty cells
#   (no box, zero lung membership), preserving ALL lung content incl. the apices.
#
# The 8x8 patch grid is over the LETTERBOXED frame. So a TBX11K box is carried original -> harmonized
# (scale by WORKING_RES/min(H0,W0); done by the caller, which knows the source) -> cropped (subtract
# the crop offset) -> letterboxed (ADD the symmetric pad offset) -> [0,1] over the square side S ->
# 8x8 cells. `rasterize_boxes_to_grid` takes boxes ALREADY in the HARMONIZED frame plus the crop box,
# and finishes the crop->letterbox->[0,1]->grid mapping (the letterbox is computed from crop_box).
#
# TOKEN ORDER. The main loop pools Rad-DINO tokens via
#   grid = tok.transpose(1,2).reshape(B, D, g, g);  patches = adaptive_avg_pool2d(grid,(8,8))
#   patches = patches.reshape(B, D, 64).transpose(1,2)
# so patch index i == gy*8 + gx (row-major, gy=row/vertical, gx=col/horizontal). grid_label is stored
# [8,8] indexed [gy, gx]; flattening row-major (.reshape(64)) matches the token order. Asserted below.
# ============================================================================================
PATCH_GRID = 8
ZONE_NAMES = ("RUZ", "RMZ", "RLZ", "LUZ", "LMZ", "LLZ", "HILAR")  # 6 lung zones + hilar/mediastinal
N_ZONES = len(ZONE_NAMES)


def parse_bbox_field(raw: object) -> list[tuple[float, float, float, float]]:
    """Parse the index `bbox` column -> list of (x,y,w,h) boxes in the ORIGINAL image frame.
    build_index writes a JSON list of {'xmin','ymin','width','height'} dicts ("[]" when none); we also
    tolerate the old single-dict / "none" forms defensively. Returns [] on anything unparseable."""
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s.lower() in ("none", "nan", "[]"):
        return []
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        try:
            import ast
            obj = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    if isinstance(obj, dict):
        obj = [obj]
    boxes: list[tuple[float, float, float, float]] = []
    if isinstance(obj, list):
        for d in obj:
            if isinstance(d, dict) and {"xmin", "ymin", "width", "height"} <= set(d):
                try:  # one malformed numeric value must skip THAT box, not crash the whole parse
                    boxes.append((float(d["xmin"]), float(d["ymin"]), float(d["width"]), float(d["height"])))
                except (ValueError, TypeError):
                    print(f"parse_bbox_field: skipping malformed box {d!r}")
                    continue
    return boxes


def scale_boxes_orig_to_harmonized(
    boxes_xywh: list[tuple[float, float, float, float]],
    orig_hw: tuple[int, int],
    working_res: int = WORKING_RES,
) -> list[tuple[float, float, float, float]]:
    """`_harmonize` resizes the shorter side to `working_res` with a UNIFORM scale
    s = working_res / min(H0, W0). Apply the same s to every box coord so boxes land in the
    harmonized frame. (For TBX11K originals are 512x512 -> s = working_res/512.)"""
    H0, W0 = orig_hw
    s = float(working_res) / float(max(1, min(H0, W0)))
    return [(x * s, y * s, w * s, h * s) for (x, y, w, h) in boxes_xywh]


def letterbox_square_params(crop_box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Given a crop_box (y0, y1, x0, x1) -> (S, pad_x, pad_y) for the SQUARE letterbox of the cropped
    frame. S = max(cw, ch) is the square side; pad_x/pad_y are the SYMMETRIC top-left pad offsets that
    center the cropped content in the SxS square (pad_before = (S - extent) // 2, mirroring how the
    processor's center_crop floors its offset — so resize-to-shortest-edge + center_crop is a true
    uniform resize with no chopping). The cropped content occupies [pad_x, pad_x+cw) x [pad_y, pad_y+ch)
    in the letterboxed frame; everything else is padding (maps to empty grid cells / zero lung)."""
    y0, y1, x0, x1 = crop_box
    cw = max(1, x1 - x0)
    ch = max(1, y1 - y0)
    S = max(cw, ch)
    pad_x = (S - cw) // 2
    pad_y = (S - ch) // 2
    return S, pad_x, pad_y


def letterbox_to_square(arr: np.ndarray, crop_box: tuple[int, int, int, int], pad_value: float = 0.0) -> np.ndarray:
    """Symmetrically pad a cropped-frame array (image [H,W] / [H,W,C] or a mask [H,W]) to the SxS square
    defined by `letterbox_square_params(crop_box)`. The cropped content is centered; the padding bars
    use `pad_value` (image background = 0 / zero lung membership). Returns the SxS (or SxSxC) array.

    `arr`'s height/width MUST equal the crop_box extent (ch x cw); this keeps the image, the box
    coordinate frame, and the masks padded IDENTICALLY so all three live in the same letterboxed frame.
    """
    S, pad_x, pad_y = letterbox_square_params(crop_box)
    ch, cw = arr.shape[0], arr.shape[1]
    pad = [(pad_y, S - ch - pad_y), (pad_x, S - cw - pad_x)]
    if arr.ndim == 3:
        pad.append((0, 0))
    return np.pad(arr, pad, mode="constant", constant_values=pad_value)


def rasterize_boxes_to_grid(
    boxes_xywh: list[tuple[float, float, float, float]],
    crop_box: tuple[int, int, int, int],
    harmonized_wh: tuple[int, int],
    G: int = PATCH_GRID,
    mode: str = "soft",
) -> np.ndarray:
    """Rasterize active-TB boxes (in the HARMONIZED frame) onto a GxG patch grid over the LETTERBOXED frame.

    Args:
        boxes_xywh:    list of (x, y, w, h) boxes in HARMONIZED-frame pixels (top-left origin).
        crop_box:      (y0, y1, x0, x1) crop of the harmonized image -> cropped frame (x1/y1 exclusive).
        harmonized_wh: (W_harm, H_harm) — full harmonized size (used only to clamp boxes sanely).
        G:             grid side (8).
        mode:          'soft' = fractional area coverage of each cell by the box (in [0,1]); a cell
                       partly covered gets a fractional value. 'hard' = 1.0 if any overlap, else 0.

    Returns [G, G] float in [0,1], indexed [gy, gx], = max over boxes of per-cell coverage. A box fully
    outside the crop contributes nothing (all-zero). The crop transform (subtract crop offset), the
    SYMMETRIC LETTERBOX pad (add pad_x/pad_y), and normalization by the SQUARE side S are applied here
    so the grid registers EXACTLY with the Rad-DINO patch tokens (which see the letterboxed square).
    A box that falls entirely in the padding bars contributes nothing.
    """
    W_harm, H_harm = harmonized_wh
    y0, y1, x0, x1 = crop_box
    ch = max(1, y1 - y0)   # cropped-frame height
    cw = max(1, x1 - x0)   # cropped-frame width
    S, pad_x, pad_y = letterbox_square_params(crop_box)  # square side + symmetric pad offsets
    grid = np.zeros((G, G), dtype=np.float32)
    if not boxes_xywh:
        return grid
    for (bx, by, bw, bh) in boxes_xywh:
        # box corners in harmonized frame, clamped to the harmonized image
        hx0 = float(np.clip(bx, 0.0, W_harm))
        hy0 = float(np.clip(by, 0.0, H_harm))
        hx1 = float(np.clip(bx + bw, 0.0, W_harm))
        hy1 = float(np.clip(by + bh, 0.0, H_harm))
        # -> cropped frame, then intersect with the crop window [0,cw]x[0,ch]
        cx0 = max(0.0, hx0 - x0)
        cy0 = max(0.0, hy0 - y0)
        cx1 = min(float(cw), hx1 - x0)
        cy1 = min(float(ch), hy1 - y0)
        if cx1 <= cx0 or cy1 <= cy0:
            continue  # box does not intersect the crop -> contributes nothing
        # -> LETTERBOXED frame (add symmetric pad offset), then [0,1] over the SQUARE side S -> grid cells
        gx0, gx1 = (cx0 + pad_x) / S * G, (cx1 + pad_x) / S * G
        gy0, gy1 = (cy0 + pad_y) / S * G, (cy1 + pad_y) / S * G
        for gy in range(int(np.floor(gy0)), min(G, int(np.ceil(gy1)))):
            cell_oy = min(gy + 1.0, gy1) - max(float(gy), gy0)  # vertical overlap of cell [gy,gy+1]
            if cell_oy <= 0:
                continue
            for gx in range(int(np.floor(gx0)), min(G, int(np.ceil(gx1)))):
                cell_ox = min(gx + 1.0, gx1) - max(float(gx), gx0)  # horizontal overlap
                if cell_ox <= 0:
                    continue
                if mode == "hard":
                    val = 1.0
                else:
                    val = float(cell_oy * cell_ox)  # fractional area of this unit cell covered (<=1)
                if val > grid[gy, gx]:
                    grid[gy, gx] = val  # max over boxes
    return np.clip(grid, 0.0, 1.0)


def _downsample_area(mask: np.ndarray, G: int) -> np.ndarray:
    """Downsample a [H,W] float mask to [G,G] by AVERAGE pooling (= fractional area coverage per cell)."""
    h, w = mask.shape
    t = torch.from_numpy(mask.astype("float32"))[None, None]
    pooled = F.adaptive_avg_pool2d(t, (G, G))[0, 0].numpy()
    return pooled


def zone_matrix_from_masks(
    lung_mask: np.ndarray,
    hilus_med_mask: np.ndarray,
    G: int = PATCH_GRID,
    min_lung_frac: float = 0.05,
) -> np.ndarray:
    """Build a [G*G, N_ZONES] soft zone-membership matrix from masks in the LETTERBOXED (square) frame.

    The masks are expected already padded to the SxS letterbox (see `seg_crop`/`letterbox_to_square`),
    so they register with the Rad-DINO patch tokens. The padding bars are zero, fall below
    `min_lung_frac`, and become all-zero (background) rows automatically — no special handling needed.

    Zones: 6 lung zones (RUZ,RMZ,RLZ,LUZ,LMZ,LLZ) split L/R by the per-row lung centroid, then within
    EACH hemithorax cut vertically at 1/3 and 2/3 of THAT hemithorax's lung vertical extent. 7th zone
    = hilus+mediastinum. NOTE 'R'/'L' are the PATIENT's right/left (image left = patient right), so the
    image's left half is RUZ/RMZ/RLZ and the right half is LUZ/LMZ/LLZ.

    Steps: rasterize each of the 7 zone masks at full cropped-frame resolution, area-downsample each to
    GxG, row-normalize across the 7 channels, and ZERO any patch whose lung coverage < min_lung_frac
    (so background patches get an all-zero row). Returns [G*G, N_ZONES] float, rows sum to <= 1.
    Row r flattens row-major: r = gy*G + gx (matches the Rad-DINO token order).
    """
    h, w = lung_mask.shape
    lung = (lung_mask > 0.5).astype("float32")
    hil = (hilus_med_mask > 0.5).astype("float32")
    zmask = np.zeros((h, w, N_ZONES), dtype="float32")

    ys_all, xs_all = np.where(lung > 0)
    if ys_all.size > 0:
        # per-row lung centroid (column) -> L/R split that follows a tilted/asymmetric thorax
        col_idx = np.arange(w, dtype="float32")
        row_has_lung = lung.sum(axis=1) > 0
        # default split = global lung centroid column; refine per-row where the row has lung pixels
        split_col = np.full(h, float(xs_all.mean()), dtype="float32")
        row_sum = lung.sum(axis=1)
        row_cx = (lung * col_idx[None, :]).sum(axis=1)
        split_col[row_has_lung] = row_cx[row_has_lung] / row_sum[row_has_lung]

        right_half = lung.copy()  # patient-right = image-left (cols < split)
        left_half = lung.copy()   # patient-left  = image-right (cols >= split)
        col_grid = np.tile(col_idx[None, :], (h, 1))
        is_left_img = col_grid >= split_col[:, None]
        right_half[is_left_img] = 0.0
        left_half[~is_left_img] = 0.0

        for half, (zu, zm, zl) in ((right_half, (0, 1, 2)), (left_half, (3, 4, 5))):
            hy, _ = np.where(half > 0)
            if hy.size == 0:
                continue
            top, bot = float(hy.min()), float(hy.max())
            ext = max(1.0, bot - top + 1.0)
            c1 = top + ext / 3.0   # upper|middle cut
            c2 = top + 2.0 * ext / 3.0  # middle|lower cut
            row_band = np.arange(h, dtype="float32")[:, None]
            upper = (row_band < c1)
            middle = (row_band >= c1) & (row_band < c2)
            lower = (row_band >= c2)
            zmask[:, :, zu] = half * upper
            zmask[:, :, zm] = half * middle
            zmask[:, :, zl] = half * lower
    zmask[:, :, 6] = hil  # hilar/mediastinal channel

    # downsample each channel to GxG by area, then assemble [G*G, N_ZONES]
    pooled = np.stack([_downsample_area(zmask[:, :, k], G) for k in range(N_ZONES)], axis=-1)  # [G,G,7]
    flat = pooled.reshape(G * G, N_ZONES)  # row-major gy*G+gx — matches token order
    # patch lung coverage = sum of the 6 lung-zone channels (the hilar channel is excluded from the gate)
    lung_cover = flat[:, :6].sum(axis=1)
    flat[lung_cover < min_lung_frac] = 0.0
    # row-normalize across the 7 zones so each patch's memberships sum to <= 1
    rs = flat.sum(axis=1, keepdims=True)
    nz = rs[:, 0] > 0
    flat[nz] = flat[nz] / rs[nz]
    return flat.astype("float32")


def main(mode: str = "main") -> None:
    df, out_path = _load_index(mode)
    idx_path = out_path  # for error messages
    print(f"mode={mode}  device={DEVICE}  images={len(df)}  batch={BATCH}  workers={WORKERS}  "
          f"working_res={WORKING_RES}  -> {out_path.name}")

    import torchxrayvision as xrv

    proc = AutoImageProcessor.from_pretrained(RAD_ID)
    rad = AutoModel.from_pretrained(RAD_ID).to(DEVICE).eval()  # heavy ViT -> GPU (MPS)
    dm = xrv.models.DenseNet(weights="densenet121-res224-all").eval()  # light DenseNet -> CPU
    dm.op_threshs = None  # return RAW logits from dm(x), not op-norm'd scores (audit P0-6)
    for m in (rad, dm):
        for p in m.parameters():
            p.requires_grad_(False)
    torch.set_num_threads(CPU_TORCH_THREADS)  # cap CPU DenseNet so the machine stays usable (<90%)
    xcrop, xresize = xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)
    seg, lung_idx = _get_seg()  # lung_idx = lung+hilus+mediastinum (drives the existing soft-crop bbox)
    # For the zone matrix we need lung-only vs hilus/mediastinum SEPARATELY (the 7th zone is hilar/med).
    lung_only_idx: list[int] = []
    hilus_med_idx: list[int] = []
    if seg is not None:
        targets = [str(t).lower() for t in seg.targets]
        lung_only_idx = [i for i, t in enumerate(targets) if "lung" in t]
        hilus_med_idx = [i for i, t in enumerate(targets) if "hilus" in t or "mediastinum" in t]
        seg = seg.to(DEVICE)

    # FAIL-VISIBLE (P1): count every image we silently dropped (unreadable / harmonize failure) per
    # source. A non-zero count means features.npz has FEWER images than the index — surfaced in the
    # final summary so a degraded extraction is never hidden. Counter is touched from worker threads.
    skip_by_src: dict[str, int] = {}
    skip_lock = threading.Lock()

    def load_prep(arg: tuple[str, str]):
        """Thread-pool CPU work: returns (harmonized_gray_u8, txrv_tensor[1,224,224], inv_suspected,
        orig_hw). orig_hw is the ORIGINAL (pre-harmonize) (H0,W0) so TBX11K boxes (in original coords)
        can be scaled into the harmonized frame downstream. On any failure, increments the per-source
        skip counter and returns None (the row is dropped from features.npz)."""
        path, source = arg
        try:
            g = _read_gray(path)
            inv = _detect_inversion(g)  # detect-and-log only; do NOT silently transform pixels
            orig_hw = (int(g.shape[0]), int(g.shape[1]))
            h = _harmonize(g)  # common-res, percentile-normed uint8 — fed to BOTH backbones
            norm = xrv.datasets.normalize(h.astype("float32"), 255)[None, ...]  # -> [-1024,1024]
            norm = xresize(xcrop(norm)).astype("float32")  # [1,224,224]
            return h, norm, inv, orig_hw
        except Exception as e:
            print("skip", path, repr(e)[:100])
            with skip_lock:
                skip_by_src[source] = skip_by_src.get(source, 0) + 1
            return None

    @torch.no_grad()
    def seg_crop(harm_list: list[np.ndarray]):
        """Soft-mask + crop each harmonized image to the dilated-lung bbox + 18% margin.

        The cropped frame is LETTERBOXED (symmetrically padded) to a SQUARE before the Rad-DINO
        processor so its shortest_edge resize + center_crop is a true uniform resize (no chopping) —
        see the COORDINATE FRAMES note. 'img' and the masks are returned in this LETTERBOXED frame; the
        box rasterizer recomputes the same letterbox from 'crop_box', so all three stay registered.

        Returns a list of per-image dicts:
          {'img': RGB PIL of the LETTERBOXED-square cropped frame (fed to Rad-DINO),
           'crop_box': (y0,y1,x0,x1) crop in the HARMONIZED frame (y1/x1 exclusive),
           'harm_wh': (W_harm, H_harm) harmonized image size,
           'lung_crop': lung-only soft mask [S,S] in the LETTERBOXED frame (for zones),
           'hilus_crop': hilus+mediastinum soft mask [S,S] in the LETTERBOXED frame (for the 7th zone)}.
        When seg is unavailable, crop_box is the whole image and masks are None (zone matrix -> zeros)."""
        if seg is None or not lung_idx:
            res = []
            for c in harm_list:
                cb = (0, c.shape[0], 0, c.shape[1])
                lb = letterbox_to_square(c, cb, pad_value=0.0)  # pad whole harmonized frame to square
                res.append({
                    "img": Image.fromarray(cv2.cvtColor(lb, cv2.COLOR_GRAY2RGB)),
                    "crop_box": cb,
                    "harm_wh": (c.shape[1], c.shape[0]),
                    "lung_crop": None, "hilus_crop": None,
                })
            return res
        stack = np.stack([xrv.datasets.normalize(cv2.resize(c, (512, 512)).astype("float32"), 255) for c in harm_list])
        t = torch.from_numpy(stack)[:, None, ...].to(DEVICE)
        seg_prob = torch.sigmoid(seg(t))                           # [b,C,512,512] on-device
        # Select only the needed channels ON-DEVICE (max over the relevant channel group), then copy
        # JUST those reductions to CPU — avoids hauling all C segmentation channels across the bus.
        masks = seg_prob[:, lung_idx].amax(dim=1).cpu().numpy()    # [b,512,512] crop driver (lung+hilus+med)
        lung_s = seg_prob[:, lung_only_idx].amax(dim=1).cpu().numpy() if lung_only_idx else None
        hil_s = seg_prob[:, hilus_med_idx].amax(dim=1).cpu().numpy() if hilus_med_idx else None
        out = []
        for bi, (c, m) in enumerate(zip(harm_list, masks)):
            H, W = c.shape
            mask = (cv2.resize(m, (W, H)) > 0.5).astype("uint8")
            if mask.sum() > 0:
                k = max(5, int(0.04 * max(c.shape)))
                md = cv2.dilate(mask, np.ones((k, k), np.uint8))
                ys, xs = np.where(md > 0)
                # INCLUSIVE mask bbox -> EXCLUSIVE crop bounds: +1 on the max so the bottom/right
                # edge row/col is kept (Python slicing and rasterize_boxes_to_grid both treat y1/x1
                # as exclusive). Clamp to image size.
                y0, x0 = int(ys.min()), int(xs.min())
                y1, x1 = min(H, int(ys.max()) + 1), min(W, int(xs.max()) + 1)
                # generous 18% margin so apical / costophrenic / pleural TB signs are not cropped out
                my, mx = int(0.18 * (y1 - y0)), int(0.18 * (x1 - x0))
                y0, y1 = max(0, y0 - my), min(H, y1 + my)
                x0, x1 = max(0, x0 - mx), min(W, x1 + mx)
                soft = (c.astype("float32") * (0.3 + 0.7 * md)).clip(0, 255).astype("uint8")[y0:y1, x0:x1]
            else:
                y0, y1, x0, x1 = 0, H, 0, W
                soft = c
            # soft masks resized to the harmonized frame, then cropped with the SAME window so they
            # align pixel-for-pixel with the cropped image.
            lung_crop = cv2.resize(lung_s[bi], (W, H))[y0:y1, x0:x1] if lung_s is not None else None
            hilus_crop = cv2.resize(hil_s[bi], (W, H))[y0:y1, x0:x1] if hil_s is not None else None
            # LETTERBOX: pad the cropped image AND the masks IDENTICALLY to the SxS square (pad=0 =
            # background / zero lung) so they register pixel-for-pixel with the Rad-DINO patch tokens,
            # which see the letterboxed square (not the raw non-square crop). Box coords get the same
            # letterbox inside rasterize_boxes_to_grid (recomputed from crop_box).
            cb = (y0, y1, x0, x1)
            soft = letterbox_to_square(soft, cb, pad_value=0.0)
            lung_crop = letterbox_to_square(lung_crop, cb, pad_value=0.0) if lung_crop is not None else None
            hilus_crop = letterbox_to_square(hilus_crop, cb, pad_value=0.0) if hilus_crop is not None else None
            out.append({
                "img": Image.fromarray(cv2.cvtColor(soft, cv2.COLOR_GRAY2RGB)),
                "crop_box": cb,
                "harm_wh": (W, H),
                "lung_crop": lung_crop, "hilus_crop": hilus_crop,
            })
        return out

    @torch.no_grad()
    def txrv_cpu(txrv_list: list[np.ndarray]) -> np.ndarray:
        """Runs on CPU (worker thread) concurrently with the GPU Rad-DINO/seg. ONE backbone pass."""
        tx = torch.from_numpy(np.stack(txrv_list))   # [b,1,224,224] on CPU
        pooled = dm.features2(tx)                     # [b,1024] (relu + adaptive-avg-pool inside)
        logits = dm.classifier(pooled)               # [b,18] raw logits (op_threshs disabled)
        return torch.cat([pooled, logits], dim=1).numpy()  # [b,1042]

    cls_l: list[np.ndarray] = []
    tok_l: list[np.ndarray] = []
    txv_l: list[np.ndarray] = []
    grid_l: list[np.ndarray] = []
    hasbox_l: list[bool] = []
    zones_l: list[np.ndarray] = []
    keep: list[int] = []
    inv_by_src: dict[str, int] = {}
    paths = [str(p) if str(p).startswith("/") else str(REPO / p) for p in df["path"]]
    srcs = df["source"].astype(str).tolist()
    # per-row active-TB boxes in ORIGINAL coords (only tbx11k active-TB rows are non-empty)
    bbox_col = df["bbox"].tolist() if "bbox" in df.columns else [None] * len(df)
    boxes_orig = [parse_bbox_field(b) for b in bbox_col]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for s in tqdm(range(0, len(df), BATCH)):
            rows = list(range(s, min(s + BATCH, len(df))))
            prepped = list(ex.map(load_prep, [(paths[i], srcs[i]) for i in rows]))
            harm_list, txrv_list, ok_rows, orig_hws = [], [], [], []
            for i, pr in zip(rows, prepped):
                if pr is not None:
                    harm_list.append(pr[0])
                    txrv_list.append(pr[1])
                    ok_rows.append(i)
                    orig_hws.append(pr[3])
                    if pr[2]:
                        inv_by_src[srcs[i]] = inv_by_src.get(srcs[i], 0) + 1
            if not ok_rows:
                continue
            # CPU+GPU overlap: TorchXRayVision (CPU) runs concurrently with Rad-DINO+seg (GPU/MPS).
            fut_txrv = ex.submit(txrv_cpu, list(txrv_list))
            with torch.no_grad():
                crops = seg_crop(harm_list)
                rgbs = [cr["img"] for cr in crops]
                rin = proc(images=rgbs, return_tensors="pt").to(DEVICE)
                out = rad(**rin)
                cls = out.pooler_output.float().cpu().numpy()  # [b,768]
                tok = out.last_hidden_state[:, 1:, :]          # Rad-DINO: no register tokens (verified)
                g = int(round(tok.shape[1] ** 0.5))
                assert tok.shape[1] == g * g, f"unexpected Rad-DINO token count {tok.shape[1]} (registers?)"
                # adaptive_avg_pool2d with non-divisible sizes (37->8) is unsupported on MPS; do it on CPU.
                grid = tok.transpose(1, 2).reshape(tok.shape[0], tok.shape[2], g, g).float().cpu()
                patches = F.adaptive_avg_pool2d(grid, (PATCH_GRID, PATCH_GRID))
                patches = patches.reshape(grid.shape[0], grid.shape[1], PATCH_GRID ** 2).transpose(1, 2)
                patches = patches.numpy().astype("float16")  # [b,64,768]
            txv = fut_txrv.result()  # [b,1042] — computed on CPU in parallel with the GPU work
            for j, i in enumerate(ok_rows):
                cls_l.append(cls[j])
                tok_l.append(patches[j])
                txv_l.append(txv[j])
                # ---- localization grid_label + has_box (active-TB boxes only) ----
                cr = crops[j]
                bxs_h = scale_boxes_orig_to_harmonized(boxes_orig[i], orig_hws[j])
                gl = rasterize_boxes_to_grid(bxs_h, cr["crop_box"], cr["harm_wh"], G=PATCH_GRID, mode="soft")
                # has_box reflects whether a box SURVIVED the crop/rasterization, not the pre-crop list:
                # an image whose lesion was cropped out (or had no active box) gets an all-zero grid_label
                # and MUST be has_box=False so the per-cell box loss is not supervised on a false all-zero map.
                has_box = bool(gl.sum() > 0)
                grid_l.append(gl.astype("float16"))
                hasbox_l.append(has_box)
                # ---- per-patch soft zone membership [64,7] ----
                # Lung-zone supervision (the 6 lung zones) only needs the lung mask. If the hilar
                # channel is unavailable, still build the 6 lung zones and just zero the 7th (hilar)
                # channel — don't throw away all zone supervision for the missing 7th mask.
                lung_c, hil_c = cr["lung_crop"], cr["hilus_crop"]
                if lung_c is not None:
                    hil_in = hil_c if hil_c is not None else np.zeros_like(lung_c)
                    zm = zone_matrix_from_masks(lung_c, hil_in, G=PATCH_GRID)
                else:
                    zm = np.zeros((PATCH_GRID * PATCH_GRID, N_ZONES), dtype="float32")
                zones_l.append(zm.astype("float16"))
                keep.append(i)
            done = min(s + BATCH, len(df))
            el = time.time() - t0
            rate = el / max(1, done)
            with open(DATA / "extract_progress.txt", "w") as pf:  # live, greppable progress
                pf.write(f"{done}/{len(df)} ({100 * done // len(df)}%) | elapsed {int(el)}s | "
                         f"{rate:.3f} s/img | eta {int((len(df) - done) * rate)}s\n")

    if not cls_l:
        raise SystemExit(f"no features extracted — check {idx_path} paths")
    if inv_by_src:
        print(f"MONOCHROME1 SUSPECTED (detect-only, not transformed) per source: {inv_by_src}  "
              f"— investigate if a source has a non-trivial rate; medical-grade needs DICOM tags")
    n_skipped = int(sum(skip_by_src.values()))
    if n_skipped:
        print(f"SKIPPED (unreadable / harmonize failure, dropped from features.npz): {n_skipped}/{len(df)} "
              f"images per source: {skip_by_src}  — investigate; features.npz is short by this many rows")
    else:
        print(f"skipped images: 0/{len(df)} (every indexed image extracted cleanly)")
    sub = df.iloc[keep]
    grid_arr = np.stack(grid_l).astype("float16")          # [N,8,8] indexed [gy,gx]
    zones_arr = np.stack(zones_l).astype("float16")        # [N,64,7] row-major gy*8+gx
    hasbox_arr = np.asarray(hasbox_l, dtype=bool)          # [N]
    # ROW-MAJOR CONSISTENCY: grid_label[gy,gx].reshape(64) must put a hot cell at gy*8+gx so it lines
    # up with the Rad-DINO patch tokens produced by `tok.transpose(1,2).reshape(B,D,37,37)`.
    # NOTE: the probe below ONLY proves numpy.reshape is C-order (a tautology). It does NOT prove the
    # MODEL emits tokens row-major. The REAL empirical proof — feed Rad-DINO a top-white/bottom-black
    # image, run the production reshape, confirm gy increases downward — lives in
    # training/test_extract_geom.py::test_radino_token_order_is_row_major (slow; needs the model).
    _probe = np.zeros((PATCH_GRID, PATCH_GRID), dtype="float32"); _probe[2, 5] = 1.0
    assert int(_probe.reshape(-1).argmax()) == 2 * PATCH_GRID + 5, "grid_label flatten is not row-major gy*8+gx"
    assert zones_arr.shape[1] == PATCH_GRID * PATCH_GRID and zones_arr.shape[2] == N_ZONES
    save_kwargs = dict(
        cls=np.stack(cls_l).astype("float32"),
        patches=np.stack(tok_l).astype("float16"),
        txrv=np.stack(txv_l).astype("float32"),
        y=sub["label"].to_numpy().astype("int64"),
        source=sub["source"].to_numpy().astype(str),
        patient_id=sub["patient_id"].astype(str).to_numpy(),
        grid_label=grid_arr,
        has_box=hasbox_arr,
        zones=zones_arr,
    )
    if "group" in df.columns:  # provenance cluster id from dedup.py -> train_tb leak-guard
        save_kwargs["group"] = sub["group"].to_numpy().astype("int64")
    np.savez_compressed(out_path, **save_kwargs)
    print(f"saved {out_path}  cls={np.stack(cls_l).shape}  patches={np.stack(tok_l).shape}  "
          f"txrv={np.stack(txv_l).shape}  grid_label={grid_arr.shape}  zones={zones_arr.shape}  "
          f"has_box={int(hasbox_arr.sum())}/{len(hasbox_arr)}  "
          f"pos={int(sub['label'].sum())}  neg={int((sub['label']==0).sum())}")


def _load_index(mode: str) -> tuple[pd.DataFrame, Path]:
    """Return (index_df, output_npz_path) for the requested extraction mode.

    'main'             -> data/index_dedup.csv             -> data/features.npz
    'sequelae'         -> data/index_tbx_latent.csv DEDUPED to unique image paths -> data/features_sequelae.npz
                          (the inactive-sequelae specificity probe: 239 rows but 169 UNIQUE images; all
                           label 1, source 'tbx11k_sequelae', no boxes — kept OUT of features.npz).
    'external_holdout' -> data/index_external_holdout.csv  -> data/features_mendeley_pk.npz
                          (P1 prerequisite: the held-out Pakistani cohort, label 1 = TB+ / 0 = normal,
                           source 'mendeley_pk'. Extracting features here is a FROZEN-BACKBONE READ — the
                           cohort STILL never enters training; it is the standing external eval set. The
                           identical pipeline (rad-dino-base + TXRV + zones, same harmonize/letterbox/crop)
                           guarantees a PAIRED comparison against the training-set features.)"""
    if mode == "sequelae":
        p = DATA / "index_tbx_latent.csv"
        if not p.exists():
            raise SystemExit(f"missing {p} — run build_index.py first")
        df = pd.read_csv(p)
        df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)  # 239 rows -> 169 unique images
        df["label"] = 1
        df["source"] = "tbx11k_sequelae"
        df["bbox"] = "[]"  # sequelae boxes are NOT activity supervision (ethos) -> no grid_label
        return df, DATA / "features_sequelae.npz"
    if mode == "external_holdout":
        p = DATA / "index_external_holdout.csv"
        if not p.exists():
            raise SystemExit(f"missing {p} — run build_index.py first (registers mendeley_pk holdout)")
        df = pd.read_csv(p).reset_index(drop=True)
        if "bbox" not in df.columns:  # Pakistani normals/TB carry no localization boxes -> no grid_label
            df["bbox"] = "[]"
        return df, DATA / "features_mendeley_pk.npz"
    p = DATA / "index_dedup.csv"
    if not p.exists():
        raise SystemExit(f"missing {p} — run build_index.py + dedup.py first")
    return pd.read_csv(p).reset_index(drop=True), DATA / "features.npz"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dual-backbone frozen feature extraction.")
    ap.add_argument("--mode", choices=["main", "sequelae", "external_holdout"], default="main",
                    help="'main' = the LODO binary set -> features.npz; "
                         "'sequelae' = the 169-image inactive-sequelae probe -> features_sequelae.npz; "
                         "'external_holdout' = the held-out Pakistani cohort -> features_mendeley_pk.npz")
    args = ap.parse_args()
    main(args.mode)
