"""Dual-backbone frozen feature extraction — BATCHED + threaded for M4 utilization.

Per image, cache: Rad-DINO CLS (768) + patch tokens pooled to 8x8=64 tokens (fp16) +
TorchXRayVision DenseNet pooled (1024) + 18 logits = 1042. Rad-DINO sees CLAHE+lung-soft-mask;
TorchXRayVision uses its own normalization on the raw image.

Speed: image decode + CLAHE + TXRV-normalize run in a THREAD POOL (parallel CPU); the lung-seg,
Rad-DINO, and TorchXRayVision forwards run in BATCHES on MPS (instead of one image at a time),
so both CPU and GPU stay busy.

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
from preprocess import _CLAHE, _get_seg  # noqa: E402  (reuse the same CLAHE + lung-seg)

DATA = REPO / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RAD_ID = "microsoft/rad-dino"
BATCH = 16
WORKERS = 6           # CPU loader threads (capped to leave the machine usable, <90% CPU)
CPU_TORCH_THREADS = 4  # threads the CPU DenseNet may use
PATCH_GRID = 8


def main() -> None:
    idx_path = DATA / "index_dedup.csv"
    if not idx_path.exists():
        raise SystemExit(f"missing {idx_path} — run build_index.py + dedup.py first")
    df = pd.read_csv(idx_path).reset_index(drop=True)
    print(f"device={DEVICE}  images={len(df)}  batch={BATCH}  workers={WORKERS}")

    import torchxrayvision as xrv

    proc = AutoImageProcessor.from_pretrained(RAD_ID)
    rad = AutoModel.from_pretrained(RAD_ID).to(DEVICE).eval()  # heavy ViT -> GPU (MPS)
    dm = xrv.models.DenseNet(weights="densenet121-res224-all").eval()  # light DenseNet -> CPU (concurrent)
    for m in (rad, dm):
        for p in m.parameters():
            p.requires_grad_(False)
    torch.set_num_threads(CPU_TORCH_THREADS)  # cap CPU DenseNet so the machine stays usable (<90%)
    xcrop, xresize = xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)
    seg, lung_idx = _get_seg()
    if seg is not None:
        seg = seg.to(DEVICE)

    def load_prep(path: str):
        """Thread-pool CPU work: returns (clahe_gray_u8, txrv_tensor[1,224,224]) or None."""
        try:
            g8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if g8 is None:
                g8 = np.array(Image.open(path).convert("L"))
            clahe = _CLAHE.apply(g8)
            norm = xrv.datasets.normalize(g8.astype("float32"), 255)[None, ...]
            norm = xresize(xcrop(norm)).astype("float32")  # [1,224,224]
            return clahe, norm
        except Exception as e:
            print("skip", path, repr(e)[:100])
            return None

    @torch.no_grad()
    def seg_crop(clahe_list: list[np.ndarray]) -> list[Image.Image]:
        if seg is None or not lung_idx:
            return [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)) for c in clahe_list]
        stack = np.stack([xrv.datasets.normalize(cv2.resize(c, (512, 512)).astype("float32"), 255) for c in clahe_list])
        t = torch.from_numpy(stack)[:, None, ...].to(DEVICE)
        masks = torch.sigmoid(seg(t))[:, lung_idx].amax(1).cpu().numpy()  # [b,512,512]
        out = []
        for c, m in zip(clahe_list, masks):
            mask = (cv2.resize(m, (c.shape[1], c.shape[0])) > 0.5).astype("uint8")
            if mask.sum() > 0:
                k = max(5, int(0.04 * max(c.shape)))
                md = cv2.dilate(mask, np.ones((k, k), np.uint8))
                ys, xs = np.where(md > 0)
                y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
                my, mx = int(0.10 * (y1 - y0 + 1)), int(0.10 * (x1 - x0 + 1))
                y0, y1 = max(0, y0 - my), min(c.shape[0], y1 + my)
                x0, x1 = max(0, x0 - mx), min(c.shape[1], x1 + mx)
                soft = (c.astype("float32") * (0.3 + 0.7 * md)).clip(0, 255).astype("uint8")[y0:y1, x0:x1]
            else:
                soft = c
            out.append(Image.fromarray(cv2.cvtColor(soft, cv2.COLOR_GRAY2RGB)))
        return out

    @torch.no_grad()
    def txrv_cpu(txrv_list: list[np.ndarray]) -> np.ndarray:
        """Runs on CPU (in a worker thread) concurrently with the GPU Rad-DINO/seg."""
        tx = torch.from_numpy(np.stack(txrv_list))  # [b,1,224,224] on CPU
        feats = F.relu(dm.features(tx))
        pooled = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        return torch.cat([pooled, dm(tx)], dim=1).numpy()  # [b,1042]

    cls_l: list[np.ndarray] = []
    tok_l: list[np.ndarray] = []
    txv_l: list[np.ndarray] = []
    keep: list[int] = []
    paths = [str(p) if str(p).startswith("/") else str(REPO / p) for p in df["path"]]

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for s in tqdm(range(0, len(df), BATCH)):
            rows = list(range(s, min(s + BATCH, len(df))))
            prepped = list(ex.map(load_prep, [paths[i] for i in rows]))
            clahe_list, txrv_list, ok_rows = [], [], []
            for i, pr in zip(rows, prepped):
                if pr is not None:
                    clahe_list.append(pr[0])
                    txrv_list.append(pr[1])
                    ok_rows.append(i)
            if not ok_rows:
                continue
            # CPU+GPU overlap: TorchXRayVision (light DenseNet) runs on CPU in a worker thread
            # concurrently with the heavy Rad-DINO + lung-seg on the GPU (torch releases the GIL).
            fut_txrv = ex.submit(txrv_cpu, list(txrv_list))
            with torch.no_grad():
                rgbs = seg_crop(clahe_list)
                rin = proc(images=rgbs, return_tensors="pt").to(DEVICE)
                out = rad(**rin)
                cls = out.pooler_output.float().cpu().numpy()  # [b,768]
                tok = out.last_hidden_state[:, 1:, :]
                g = int(round(tok.shape[1] ** 0.5))
                # adaptive_avg_pool2d with non-divisible sizes (37->8) is unsupported on MPS;
                # do this pool on CPU (we copy to CPU anyway).
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
    sub = df.iloc[keep]
    np.savez_compressed(
        DATA / "features.npz",
        cls=np.stack(cls_l).astype("float32"),
        patches=np.stack(tok_l).astype("float16"),
        txrv=np.stack(txv_l).astype("float32"),
        y=sub["label"].to_numpy().astype("int64"),
        source=sub["source"].to_numpy().astype(str),
        patient_id=sub["patient_id"].astype(str).to_numpy(),
    )
    print(f"saved data/features.npz  cls={np.stack(cls_l).shape}  patches={np.stack(tok_l).shape}  "
          f"txrv={np.stack(txv_l).shape}  pos={int(sub['label'].sum())}  neg={int((sub['label']==0).sum())}")


if __name__ == "__main__":
    main()
