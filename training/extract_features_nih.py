"""NIH ChestX-ray14 feature extraction — IDENTICAL preprocessing to extract_features.py.

WHY THIS EXISTS (per-finding FPR stress test). The deployed T2 head over-calls 84% of TBX11K
healed scars and the scar-subgroup Brier (0.806) / ECE (0.846) suggest a catastrophic decision-
layer mis-calibration. The open question is whether that mis-calibration is SCAR-SPECIFIC (the
sequelae head is the targeted fix) or whether it generalizes to all radiographic TB MIMICS
(fibrosis, nodule, consolidation, mass, pleural thickening, infiltration, atelectasis) — in which
case the decision layer needs a broader fix. NIH ChestX-ray14 gives us labeled, sample-rich
per-finding subgroups to settle that question.

CRITICAL SAFETY RULE. This script REUSES the exact functions from `training/extract_features.py`
(`_read_gray`, `_harmonize` percentile-clip [1,99] + min-max + antialiased resize at WORKING_RES,
`seg_crop` lung-mask + letterbox-to-square, Rad-DINO 518px, TXRV 224 with op_threshs=None for raw
logits, `_detect_inversion` log-only, the same constants WORKING_RES=1024 PATCH_GRID=8 BATCH=16
WORKERS=6). NO new preprocessing variants. The point of this run is to score NIH through the
SAME front-end the deployed head was trained with — anything else would silently change the
distribution we measure.

WHAT WE OMIT FROM THE OUTPUT NPZ (and why):
  - grid_label / has_box: NIH ChestX-ray14 has no per-image active-TB box annotations (the boxed
    subset is a different ~1k file we do not use here). These are SUPERVISION labels, not features,
    and are not needed for inference.
  - We DO compute `zones` (soft lung-zone membership per patch) from the lung segmenter — it is a
    structural input to the deployed T2 head's ZonalSoftOR forward pass (NOT a label). Omitting it
    would force us to feed all-zero zones at inference, which silently degrades the zonal lever
    and changes the model we are measuring. The user's instruction grouped zones with grid_label/
    has_box; we computed zones anyway because not doing so would violate "deployed-style" scoring.
    Documented here and in the commit message.

OUTPUTS schema (data/features_nih14.npz):
  cls(N,768) float32, patches(N,64,768) fp16, txrv(N,1042) float32, zones(N,64,7) fp16,
  y(N,) int64 (all 0 — non-TB by NIH labels), filename(N,) str, n_findings(N,) int64,
  view_position(N,) str, plus the 15 ChestX-ray14 per-finding columns as separate int64 keys
  (No_Finding, Atelectasis, ... Hernia).

RUNTIME: ~30-55 min on M4/MPS for 10k images. Live progress to data/extract_nih_progress.txt.

    PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \\
        training/.venv/bin/python training/extract_features_nih.py
"""
from __future__ import annotations

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

# Reuse the EXACT preprocessing + geometry from the main extractor (zero variants).
from extract_features import (  # noqa: E402
    BATCH,
    CPU_TORCH_THREADS,
    N_ZONES,
    PATCH_GRID,
    RAD_ID,
    WORKERS,
    WORKING_RES,
    _detect_inversion,
    _harmonize,
    _read_gray,
    letterbox_to_square,
    zone_matrix_from_masks,
)
from preprocess import _get_seg  # noqa: E402

DATA = REPO / "data"
NIH_DIR = DATA / "raw" / "nih14"
SIDECAR = NIH_DIR / "nih14_findings.csv"
OUT_NPZ = DATA / "features_nih14.npz"
PROGRESS_TXT = DATA / "extract_nih_progress.txt"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 15 ChestX-ray14 finding columns (locked order; matches the sidecar).
FINDING_COLS: tuple[str, ...] = (
    "No_Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
)


def _load_sidecar() -> pd.DataFrame:
    if not SIDECAR.exists():
        raise SystemExit(f"missing sidecar {SIDECAR}; run scripts/fetch_nih14.py first")
    df = pd.read_csv(SIDECAR)
    # Sanity-check every PNG referenced exists (locked corpus).
    missing = [f for f in df["filename"].tolist() if not (NIH_DIR / f).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} sidecar rows reference missing PNGs (e.g. {missing[:3]})")
    return df.reset_index(drop=True)


def main() -> None:
    df = _load_sidecar()
    paths = [str(NIH_DIR / f) for f in df["filename"].tolist()]
    n = len(df)
    print(f"NIH extraction  device={DEVICE}  images={n}  batch={BATCH}  workers={WORKERS}  "
          f"working_res={WORKING_RES}  -> {OUT_NPZ.name}")

    import torchxrayvision as xrv

    proc = AutoImageProcessor.from_pretrained(RAD_ID)
    rad = AutoModel.from_pretrained(RAD_ID).to(DEVICE).eval()
    dm = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    dm.op_threshs = None  # raw logits (audit P0-6)
    for m in (rad, dm):
        for p in m.parameters():
            p.requires_grad_(False)
    torch.set_num_threads(CPU_TORCH_THREADS)
    xcrop, xresize = xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)

    seg, lung_idx = _get_seg()
    lung_only_idx: list[int] = []
    hilus_med_idx: list[int] = []
    if seg is not None:
        targets = [str(t).lower() for t in seg.targets]
        lung_only_idx = [i for i, t in enumerate(targets) if "lung" in t]
        hilus_med_idx = [i for i, t in enumerate(targets) if "hilus" in t or "mediastinum" in t]
        seg = seg.to(DEVICE)

    skip_count = 0
    skip_lock = threading.Lock()
    inv_count = 0

    def load_prep(arg: tuple[str, int]) -> tuple[np.ndarray, np.ndarray, bool] | None:
        path, _row_i = arg
        nonlocal skip_count
        try:
            g = _read_gray(path)
            inv = _detect_inversion(g)
            h = _harmonize(g)
            norm = xrv.datasets.normalize(h.astype("float32"), 255)[None, ...]
            norm = xresize(xcrop(norm)).astype("float32")
            return h, norm, inv
        except Exception as e:
            print("skip", path, repr(e)[:100])
            with skip_lock:
                skip_count += 1
            return None

    @torch.no_grad()
    def seg_crop(harm_list: list[np.ndarray]) -> list[dict]:
        """Same as extract_features.seg_crop, minus the harmonized_wh/crop_box bookkeeping the box
        rasterizer needs (NIH has no boxes). Returns img (PIL RGB of letterboxed square), lung_crop,
        hilus_crop. Re-implemented here (rather than imported) because the original returns extra
        fields and we do not want to pull in box-rasterization code paths we never exercise."""
        if seg is None or not lung_idx:
            res = []
            for c in harm_list:
                cb = (0, c.shape[0], 0, c.shape[1])
                lb = letterbox_to_square(c, cb, pad_value=0.0)
                res.append({
                    "img": Image.fromarray(cv2.cvtColor(lb, cv2.COLOR_GRAY2RGB)),
                    "lung_crop": None,
                    "hilus_crop": None,
                })
            return res
        stack = np.stack([
            xrv.datasets.normalize(cv2.resize(c, (512, 512)).astype("float32"), 255)
            for c in harm_list
        ])
        t = torch.from_numpy(stack)[:, None, ...].to(DEVICE)
        seg_prob = torch.sigmoid(seg(t))
        masks = seg_prob[:, lung_idx].amax(dim=1).cpu().numpy()
        lung_s = seg_prob[:, lung_only_idx].amax(dim=1).cpu().numpy() if lung_only_idx else None
        hil_s = seg_prob[:, hilus_med_idx].amax(dim=1).cpu().numpy() if hilus_med_idx else None
        out: list[dict] = []
        for bi, (c, m) in enumerate(zip(harm_list, masks)):
            H, W = c.shape
            mask = (cv2.resize(m, (W, H)) > 0.5).astype("uint8")
            if mask.sum() > 0:
                k = max(5, int(0.04 * max(c.shape)))
                md = cv2.dilate(mask, np.ones((k, k), np.uint8))
                ys, xs = np.where(md > 0)
                y0, x0 = int(ys.min()), int(xs.min())
                y1, x1 = min(H, int(ys.max()) + 1), min(W, int(xs.max()) + 1)
                my, mx = int(0.18 * (y1 - y0)), int(0.18 * (x1 - x0))
                y0, y1 = max(0, y0 - my), min(H, y1 + my)
                x0, x1 = max(0, x0 - mx), min(W, x1 + mx)
                soft = (c.astype("float32") * (0.3 + 0.7 * md)).clip(0, 255).astype("uint8")[y0:y1, x0:x1]
            else:
                y0, y1, x0, x1 = 0, H, 0, W
                soft = c
            lung_crop = cv2.resize(lung_s[bi], (W, H))[y0:y1, x0:x1] if lung_s is not None else None
            hilus_crop = cv2.resize(hil_s[bi], (W, H))[y0:y1, x0:x1] if hil_s is not None else None
            cb = (y0, y1, x0, x1)
            soft = letterbox_to_square(soft, cb, pad_value=0.0)
            lung_crop = letterbox_to_square(lung_crop, cb, pad_value=0.0) if lung_crop is not None else None
            hilus_crop = letterbox_to_square(hilus_crop, cb, pad_value=0.0) if hilus_crop is not None else None
            out.append({
                "img": Image.fromarray(cv2.cvtColor(soft, cv2.COLOR_GRAY2RGB)),
                "lung_crop": lung_crop,
                "hilus_crop": hilus_crop,
            })
        return out

    @torch.no_grad()
    def txrv_cpu(txrv_list: list[np.ndarray]) -> np.ndarray:
        tx = torch.from_numpy(np.stack(txrv_list))
        pooled = dm.features2(tx)
        logits = dm.classifier(pooled)
        return torch.cat([pooled, logits], dim=1).numpy()

    cls_l: list[np.ndarray] = []
    tok_l: list[np.ndarray] = []
    txv_l: list[np.ndarray] = []
    zones_l: list[np.ndarray] = []
    keep: list[int] = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for s in tqdm(range(0, n, BATCH)):
            rows = list(range(s, min(s + BATCH, n)))
            prepped = list(ex.map(load_prep, [(paths[i], i) for i in rows]))
            harm_list: list[np.ndarray] = []
            txrv_list: list[np.ndarray] = []
            ok_rows: list[int] = []
            for i, pr in zip(rows, prepped):
                if pr is not None:
                    harm_list.append(pr[0])
                    txrv_list.append(pr[1])
                    ok_rows.append(i)
                    if pr[2]:
                        inv_count += 1
            if not ok_rows:
                continue
            fut_txrv = ex.submit(txrv_cpu, list(txrv_list))
            with torch.no_grad():
                crops = seg_crop(harm_list)
                rgbs = [cr["img"] for cr in crops]
                rin = proc(images=rgbs, return_tensors="pt").to(DEVICE)
                out = rad(**rin)
                cls = out.pooler_output.float().cpu().numpy()
                tok = out.last_hidden_state[:, 1:, :]
                g = int(round(tok.shape[1] ** 0.5))
                assert tok.shape[1] == g * g, f"unexpected Rad-DINO token count {tok.shape[1]}"
                grid = tok.transpose(1, 2).reshape(tok.shape[0], tok.shape[2], g, g).float().cpu()
                patches = F.adaptive_avg_pool2d(grid, (PATCH_GRID, PATCH_GRID))
                patches = patches.reshape(grid.shape[0], grid.shape[1], PATCH_GRID ** 2).transpose(1, 2)
                patches = patches.numpy().astype("float16")
            txv = fut_txrv.result()
            for j, i in enumerate(ok_rows):
                cls_l.append(cls[j])
                tok_l.append(patches[j])
                txv_l.append(txv[j])
                cr = crops[j]
                lung_c, hil_c = cr["lung_crop"], cr["hilus_crop"]
                if lung_c is not None:
                    hil_in = hil_c if hil_c is not None else np.zeros_like(lung_c)
                    zm = zone_matrix_from_masks(lung_c, hil_in, G=PATCH_GRID)
                else:
                    zm = np.zeros((PATCH_GRID * PATCH_GRID, N_ZONES), dtype="float32")
                zones_l.append(zm.astype("float16"))
                keep.append(i)
            done = min(s + BATCH, n)
            el = time.time() - t0
            rate = el / max(1, done)
            with open(PROGRESS_TXT, "w") as pf:
                pf.write(f"{done}/{n} ({100 * done // n}%) | elapsed {int(el)}s | "
                         f"{rate:.3f} s/img | eta {int((n - done) * rate)}s\n")

    if not cls_l:
        raise SystemExit("no features extracted")
    if inv_count:
        print(f"MONOCHROME1 SUSPECTED (detect-only, not transformed): {inv_count}/{n} images")
    if skip_count:
        print(f"SKIPPED (unreadable / harmonize failure): {skip_count}/{n}")
    else:
        print(f"skipped images: 0/{n}")

    sub = df.iloc[keep].reset_index(drop=True)
    cls_arr = np.stack(cls_l).astype("float32")
    tok_arr = np.stack(tok_l).astype("float16")
    txv_arr = np.stack(txv_l).astype("float32")
    zones_arr = np.stack(zones_l).astype("float16")
    save_kwargs: dict[str, np.ndarray] = dict(
        cls=cls_arr,
        patches=tok_arr,
        txrv=txv_arr,
        zones=zones_arr,
        y=sub["y_nontb"].to_numpy().astype("int64"),
        filename=sub["filename"].astype(str).to_numpy(),
        n_findings=sub["n_findings"].to_numpy().astype("int64"),
        view_position=sub["view_position"].astype(str).to_numpy(),
    )
    for col in FINDING_COLS:
        save_kwargs[col] = sub[col].to_numpy().astype("int64")
    np.savez_compressed(OUT_NPZ, **save_kwargs)
    print(f"saved {OUT_NPZ}  cls={cls_arr.shape}  patches={tok_arr.shape}  "
          f"txrv={txv_arr.shape}  zones={zones_arr.shape}  rows={len(sub)}")


if __name__ == "__main__":
    main()
