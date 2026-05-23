"""Dual-backbone frozen feature extraction (Apple MPS), cached to data/features.npz.

Per image, cache:
  - Rad-DINO CLS pooler_output (768)               [feeds the global head]
  - Rad-DINO patch tokens, 37x37 grid adaptive-pooled to 8x8 = 64 tokens x 768, fp16
                                                    [feeds the ABMIL attention head]
  - TorchXRayVision DenseNet pooled features (1024) + 18 pathology logits = 1042
                                                    [supervised, pathology-grounded, anti-shortcut]
Rad-DINO sees the CLAHE+soft-mask preprocessing (preprocess.py); TorchXRayVision uses its OWN
normalization on the raw image (two preprocessing paths). Both backbones frozen.

    PYTORCH_ENABLE_MPS_FALLBACK=1 python training/extract_features.py
"""
from __future__ import annotations
import sys
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
from preprocess import preprocess_image  # noqa: E402

DATA = REPO / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RAD_ID = "microsoft/rad-dino"
PATCH_GRID = 8  # adaptive-pool the 37x37 token grid to 8x8 = 64 tokens


def main() -> None:
    idx_path = DATA / "index_dedup.csv"
    if not idx_path.exists():
        raise SystemExit(f"missing {idx_path} — run build_index.py + dedup.py first")
    df = pd.read_csv(idx_path).reset_index(drop=True)
    print(f"device={DEVICE}  images={len(df)}")

    import torchxrayvision as xrv

    proc = AutoImageProcessor.from_pretrained(RAD_ID)
    rad = AutoModel.from_pretrained(RAD_ID).to(DEVICE).eval()
    for p in rad.parameters():
        p.requires_grad_(False)
    dm = xrv.models.DenseNet(weights="densenet121-res224-all").eval()  # CPU; small
    for p in dm.parameters():
        p.requires_grad_(False)
    xcrop, xresize = xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)

    cls_l: list[np.ndarray] = []
    tok_l: list[np.ndarray] = []
    txv_l: list[np.ndarray] = []
    keep: list[int] = []

    @torch.no_grad()
    def rad_features(path: str) -> tuple[np.ndarray, np.ndarray]:
        pil = preprocess_image(path)
        inp = proc(images=pil, return_tensors="pt").to(DEVICE)
        out = rad(**inp)
        cls = out.pooler_output[0].float().cpu().numpy()
        tok = out.last_hidden_state[0, 1:, :]  # [num_patches, 768]
        g = int(round(tok.shape[0] ** 0.5))
        grid = tok.transpose(0, 1).reshape(1, tok.shape[1], g, g)
        pooled = F.adaptive_avg_pool2d(grid, (PATCH_GRID, PATCH_GRID))
        toks = pooled.reshape(tok.shape[1], PATCH_GRID * PATCH_GRID).transpose(0, 1)
        return cls, toks.float().cpu().numpy().astype("float16")

    @torch.no_grad()
    def txrv_features(path: str) -> np.ndarray:
        g8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g8 is None:
            g8 = np.array(Image.open(path).convert("L"))
        norm = xrv.datasets.normalize(g8.astype("float32"), 255)[None, ...]  # [1,H,W]
        norm = xresize(xcrop(norm))  # [1,224,224]
        t = torch.from_numpy(norm)[None, ...].float()  # [1,1,224,224]
        feats = F.relu(dm.features(t))
        pooled = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)[0]
        logits = dm(t)[0]
        return torch.cat([pooled, logits]).cpu().numpy()

    for i, row in tqdm(df.iterrows(), total=len(df)):
        path = str(row["path"]) if str(row["path"]).startswith("/") else str(REPO / row["path"])
        try:
            cls, tok = rad_features(path)
            txv = txrv_features(path)
            cls_l.append(cls)
            tok_l.append(tok)
            txv_l.append(txv)
            keep.append(i)
        except Exception as e:
            print("skip", path, repr(e)[:120])

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
    print(
        f"saved data/features.npz  cls={np.stack(cls_l).shape}  patches={np.stack(tok_l).shape}  "
        f"txrv={np.stack(txv_l).shape}  pos={int(sub['label'].sum())}  neg={int((sub['label']==0).sum())}"
    )


if __name__ == "__main__":
    main()
