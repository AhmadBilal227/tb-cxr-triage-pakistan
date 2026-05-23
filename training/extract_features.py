"""Frozen Rad-DINO feature extraction (Apple MPS), cached to data/features.npz.

One-time pass: reads data/index_dedup.csv, applies CLAHE preprocessing, runs the frozen
Rad-DINO ViT-B, and caches the 768-d CLS vector per image. Head training then iterates on
the cache instantly. Run from the repo root with the training venv active:

    PYTORCH_ENABLE_MPS_FALLBACK=1 python training/extract_features.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
from preprocess import preprocess_image  # noqa: E402

DATA = REPO / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 16
MODEL_ID = "microsoft/rad-dino"


def main() -> None:
    idx_path = DATA / "index_dedup.csv"
    if not idx_path.exists():
        raise SystemExit(f"missing {idx_path} — run build_index.py + dedup.py first")
    df = pd.read_csv(idx_path).reset_index(drop=True)
    print(f"device={DEVICE}  images={len(df)}")

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    feats: list[np.ndarray] = []
    kept_rows: list[int] = []
    buf_imgs: list = []
    buf_idx: list[int] = []

    @torch.no_grad()
    def flush() -> None:
        if not buf_imgs:
            return
        inp = proc(images=buf_imgs, return_tensors="pt").to(DEVICE)
        out = model(**inp).pooler_output.float().cpu().numpy()
        for j, ridx in enumerate(buf_idx):
            feats.append(out[j])
            kept_rows.append(ridx)
        buf_imgs.clear()
        buf_idx.clear()

    for ridx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            buf_imgs.append(preprocess_image(str(row["path"])))
            buf_idx.append(ridx)
            if len(buf_imgs) >= BATCH:
                flush()
        except Exception as e:  # corrupt image etc. — skip, don't abort the run
            # preprocess_image is the arg to append, so on failure nothing was buffered.
            print("skip", row["path"], repr(e))
    flush()

    if not feats:
        raise SystemExit("no features extracted — check data/index_dedup.csv paths")
    X = np.stack(feats)
    sub = df.iloc[kept_rows]
    np.savez_compressed(
        DATA / "features.npz",
        X=X.astype("float32"),
        y=sub["label"].to_numpy().astype("int64"),
        source=sub["source"].to_numpy().astype(str),
        patient_id=sub["patient_id"].astype(str).to_numpy(),
    )
    print(f"saved data/features.npz  X={X.shape}  pos={int(sub['label'].sum())}  neg={int((sub['label']==0).sum())}")


if __name__ == "__main__":
    main()
