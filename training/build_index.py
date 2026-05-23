"""Build data/index.csv (path,label,source,patient_id) from data/raw/<source>/.

Generic per-source heuristics:
  - label 1 (TB): path contains 'tubercul', or filename ends '_1', or a '/tb/' folder.
  - label 0 (NORMAL): path contains 'normal'/'health', or filename ends '_0'.
  - mask images (path contains 'mask') and the 'lungseg' source are excluded (not TB train/test).
After running, eyeball the printed per-source counts; patch any source that mislabels.
"""
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
IMG = re.compile(r"\.(png|jpe?g|bmp)$", re.I)
TB_SUF = re.compile(r"_1\.(png|jpe?g|bmp)$", re.I)
NEG_SUF = re.compile(r"_0\.(png|jpe?g|bmp)$", re.I)


def label_from_path(rel: Path) -> int | None:
    """Classify using the path RELATIVE to the source dir to avoid ancestor dir pollution."""
    s = str(rel).lower()
    parts = [pt.lower() for pt in rel.parts]
    if "mask" in s:
        return None
    if "tubercul" in s or TB_SUF.search(s) or "tb" in parts:
        return 1
    if any(pt in ("normal", "healthy") for pt in parts) or NEG_SUF.search(s):
        return 0
    return None


def patient_id(name: str) -> str:
    return re.sub(r"_[01]$", "", Path(name).stem)


def main() -> None:
    rows = []
    for src_dir in sorted(d for d in RAW.iterdir() if d.is_dir()):
        src = src_dir.name
        if src == "lungseg":
            continue
        n = 0
        for p in src_dir.rglob("*"):
            if not (p.is_file() and IMG.search(p.name)):
                continue
            lab = label_from_path(p.relative_to(src_dir))
            if lab is None:
                continue
            rows.append(
                {
                    "path": str(p.relative_to(REPO)),
                    "label": lab,
                    "source": src,
                    "patient_id": f"{src}_{patient_id(p.name)}",
                }
            )
            n += 1
        print(f"{src}: {n} labeled images")
    df = pd.DataFrame(rows)
    out = REPO / "data" / "index.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  total={len(df)}  pos={int((df.label==1).sum())}  neg={int((df.label==0).sum())}")
    if len(df):
        print(df.groupby(["source", "label"]).size())


if __name__ == "__main__":
    main()
