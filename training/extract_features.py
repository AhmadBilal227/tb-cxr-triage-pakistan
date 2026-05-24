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
import sys
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
PATCH_GRID = 8
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


def main() -> None:
    idx_path = DATA / "index_dedup.csv"
    if not idx_path.exists():
        raise SystemExit(f"missing {idx_path} — run build_index.py + dedup.py first")
    df = pd.read_csv(idx_path).reset_index(drop=True)
    print(f"device={DEVICE}  images={len(df)}  batch={BATCH}  workers={WORKERS}  working_res={WORKING_RES}")

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
    seg, lung_idx = _get_seg()
    if seg is not None:
        seg = seg.to(DEVICE)

    def load_prep(path: str):
        """Thread-pool CPU work: returns (harmonized_gray_u8, txrv_tensor[1,224,224], inv_suspected)."""
        try:
            g = _read_gray(path)
            inv = _detect_inversion(g)  # detect-and-log only; do NOT silently transform pixels
            h = _harmonize(g)  # common-res, percentile-normed uint8 — fed to BOTH backbones
            norm = xrv.datasets.normalize(h.astype("float32"), 255)[None, ...]  # -> [-1024,1024]
            norm = xresize(xcrop(norm)).astype("float32")  # [1,224,224]
            return h, norm, inv
        except Exception as e:
            print("skip", path, repr(e)[:100])
            return None

    @torch.no_grad()
    def seg_crop(harm_list: list[np.ndarray]) -> list[Image.Image]:
        if seg is None or not lung_idx:
            return [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)) for c in harm_list]
        stack = np.stack([xrv.datasets.normalize(cv2.resize(c, (512, 512)).astype("float32"), 255) for c in harm_list])
        t = torch.from_numpy(stack)[:, None, ...].to(DEVICE)
        masks = torch.sigmoid(seg(t))[:, lung_idx].amax(1).cpu().numpy()  # [b,512,512]
        out = []
        for c, m in zip(harm_list, masks):
            mask = (cv2.resize(m, (c.shape[1], c.shape[0])) > 0.5).astype("uint8")
            if mask.sum() > 0:
                k = max(5, int(0.04 * max(c.shape)))
                md = cv2.dilate(mask, np.ones((k, k), np.uint8))
                ys, xs = np.where(md > 0)
                y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
                # generous 18% margin so apical / costophrenic / pleural TB signs are not cropped out
                my, mx = int(0.18 * (y1 - y0 + 1)), int(0.18 * (x1 - x0 + 1))
                y0, y1 = max(0, y0 - my), min(c.shape[0], y1 + my)
                x0, x1 = max(0, x0 - mx), min(c.shape[1], x1 + mx)
                soft = (c.astype("float32") * (0.3 + 0.7 * md)).clip(0, 255).astype("uint8")[y0:y1, x0:x1]
            else:
                soft = c
            out.append(Image.fromarray(cv2.cvtColor(soft, cv2.COLOR_GRAY2RGB)))
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
    keep: list[int] = []
    inv_by_src: dict[str, int] = {}
    paths = [str(p) if str(p).startswith("/") else str(REPO / p) for p in df["path"]]
    srcs = df["source"].astype(str).tolist()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for s in tqdm(range(0, len(df), BATCH)):
            rows = list(range(s, min(s + BATCH, len(df))))
            prepped = list(ex.map(load_prep, [paths[i] for i in rows]))
            harm_list, txrv_list, ok_rows = [], [], []
            for i, pr in zip(rows, prepped):
                if pr is not None:
                    harm_list.append(pr[0])
                    txrv_list.append(pr[1])
                    ok_rows.append(i)
                    if pr[2]:
                        inv_by_src[srcs[i]] = inv_by_src.get(srcs[i], 0) + 1
            if not ok_rows:
                continue
            # CPU+GPU overlap: TorchXRayVision (CPU) runs concurrently with Rad-DINO+seg (GPU/MPS).
            fut_txrv = ex.submit(txrv_cpu, list(txrv_list))
            with torch.no_grad():
                rgbs = seg_crop(harm_list)
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
                keep.append(i)

    if not cls_l:
        raise SystemExit("no features extracted — check data/index_dedup.csv paths")
    if inv_by_src:
        print(f"MONOCHROME1 SUSPECTED (detect-only, not transformed) per source: {inv_by_src}  "
              f"— investigate if a source has a non-trivial rate; medical-grade needs DICOM tags")
    sub = df.iloc[keep]
    save_kwargs = dict(
        cls=np.stack(cls_l).astype("float32"),
        patches=np.stack(tok_l).astype("float16"),
        txrv=np.stack(txv_l).astype("float32"),
        y=sub["label"].to_numpy().astype("int64"),
        source=sub["source"].to_numpy().astype(str),
        patient_id=sub["patient_id"].astype(str).to_numpy(),
    )
    if "group" in df.columns:  # provenance cluster id from dedup.py -> train_tb leak-guard
        save_kwargs["group"] = sub["group"].to_numpy().astype("int64")
    np.savez_compressed(DATA / "features.npz", **save_kwargs)
    print(f"saved data/features.npz  cls={np.stack(cls_l).shape}  patches={np.stack(tok_l).shape}  "
          f"txrv={np.stack(txv_l).shape}  pos={int(sub['label'].sum())}  neg={int((sub['label']==0).sum())}")


if __name__ == "__main__":
    main()
