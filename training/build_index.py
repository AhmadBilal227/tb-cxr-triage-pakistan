"""Build data/index.csv (path,label,source,patient_id) from data/raw/<source>/.

Generic per-source heuristics:
  - label 1 (TB): path contains 'tubercul', or filename ends '_1', or a '/tb/' folder.
  - label 0 (NORMAL): path contains 'normal'/'health', or filename ends '_0'.
  - mask images (path contains 'mask') and the 'lungseg' source are excluded (not TB train/test).

TBX11K is special: its labels live in data.csv, not the path. Binary target = active_tb (1) vs
no_tb (0, healthy + sick-but-no-TB hard negatives); latent_tb (sequelae) is held OUT of the binary
task into data/index_tbx_latent.csv as a probe set; the unlabeled competition test/ images are
excluded. (Decision: 2026-05-24 — screen for ACTIVE TB; latent = old/healed, radiographically
distinct, measured separately.)

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


def build_tbx11k(src_dir: Path) -> tuple[list[dict], list[dict]]:
    """TBX11K-simplified: labels come from data.csv. Returns (binary_rows, latent_probe_rows).
    Binary: active_tb -> 1, no_tb (healthy + sick-but-no-TB) -> 0. latent_tb -> probe (held out).
    Unlabeled test/ images are not in the CSV, so they are naturally excluded."""
    csvs = list(src_dir.rglob("data.csv"))
    if not csvs:
        print("tbx11k: data.csv not found — skipping")
        return [], []
    df = pd.read_csv(csvs[0])
    base = csvs[0].parent
    by_name = {p.name: p for p in base.rglob("*") if p.is_file() and IMG.search(p.name)}
    binary: list[dict] = []
    latent: list[dict] = []
    miss = 0
    for _, r in df.iterrows():
        fp = by_name.get(str(r["fname"]))
        if fp is None:
            miss += 1
            continue
        if "test" in (pt.lower() for pt in fp.relative_to(base).parts):
            continue  # unlabeled competition split
        common = {"path": str(fp.relative_to(REPO)), "patient_id": f"tbx11k_{fp.stem}"}
        tb_type, target = str(r["tb_type"]), str(r["target"])
        if tb_type == "active_tb":
            binary.append({**common, "label": 1, "source": "tbx11k"})
        elif target == "no_tb":
            binary.append({**common, "label": 0, "source": "tbx11k"})
        elif tb_type == "latent_tb":
            latent.append({**common, "label": 1, "source": "tbx11k_latent"})
    pos = sum(b["label"] for b in binary)
    if miss:
        print(f"tbx11k: {miss} csv rows had no matching image file (skipped)")
    print(f"tbx11k: {len(binary)} binary ({pos} active-TB pos, {len(binary)-pos} neg) "
          f"+ {len(latent)} latent-TB held out as a probe")
    return binary, latent


def main() -> None:
    rows: list[dict] = []
    latent_rows: list[dict] = []
    for src_dir in sorted(d for d in RAW.iterdir() if d.is_dir()):
        src = src_dir.name
        if src == "lungseg":
            continue
        if src == "tbx11k":
            b, lat = build_tbx11k(src_dir)
            rows.extend(b)
            latent_rows.extend(lat)
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
    if latent_rows:
        lat_out = REPO / "data" / "index_tbx_latent.csv"
        pd.DataFrame(latent_rows).to_csv(lat_out, index=False)
        print(f"\nwrote {lat_out}  latent-TB probe={len(latent_rows)} (held out of the binary LODO)")


if __name__ == "__main__":
    main()
