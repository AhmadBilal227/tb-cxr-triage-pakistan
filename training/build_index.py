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

NIH14 (data/raw/nih14/) is DELIBERATELY EXCLUDED from this index. It is a LOCKED,
never-train, stratified-by-finding FPR stress test produced by scripts/fetch_nih14.py; its
per-finding labels live in the sidecar data/raw/nih14/nih14_findings.csv and the stratified
mimic-FPR probe in training/stress_metrics.py reads that sidecar directly. Registering NIH14
as label 0 here would risk it leaking into the TB training set, so the raw and parquet-cache
dirs are added to the loop's skip list. (Decision: 2026-05-24, revised same day after a
strategy review: NIH14 stays out of the training manifest by design.)

After running, eyeball the printed per-source counts; patch any source that mislabels.
"""
from __future__ import annotations
import ast
import json
import re
from pathlib import Path

import pandas as pd


def _parse_bbox(raw: str) -> dict | None:
    """data.csv `bbox` is ONE box per row: a python-dict string
    "{'xmin':..,'ymin':..,'width':..,'height':..}" (COCO-ish xywh in 512x512) or the literal "none".
    Returns the dict (with float xmin/ymin/width/height) or None for "none"/unparseable."""
    s = str(raw).strip()
    if not s or s.lower() == "none" or s.lower() == "nan":
        return None
    try:
        d = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(d, dict) or not {"xmin", "ymin", "width", "height"} <= set(d):
        return None
    try:  # a malformed numeric value must skip THIS box (return None), not crash index-building
        return {k: float(d[k]) for k in ("xmin", "ymin", "width", "height")}
    except (ValueError, TypeError):
        print(f"_parse_bbox: skipping box with non-numeric field {d!r}")
        return None

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
    Unlabeled test/ images are not in the CSV, so they are naturally excluded.

    bbox handling: data.csv has ONE box per row, so an image with K active-TB boxes spans K rows.
    We GROUP by fname and emit exactly ONE row per image, aggregating only the ACTIVE-TB boxes into
    a JSON list string in the `bbox` column (COCO-ish [{'xmin','ymin','width','height'}, ...] in the
    image's original 512x512 frame; "[]" when there are none). Latent/sequelae boxes are deliberately
    EXCLUDED from this list (ethos: only active TB supervises localization). 30 images carry BOTH
    active and latent boxes — for those we keep the active boxes and drop the latent ones."""
    csvs = list(src_dir.rglob("data.csv"))
    if not csvs:
        print("tbx11k: data.csv not found — skipping")
        return [], []
    df = pd.read_csv(csvs[0])
    base = csvs[0].parent
    by_name = {p.name: p for p in base.rglob("*") if p.is_file() and IMG.search(p.name)}
    # group csv rows by fname: collect tb_types seen, the (single) target, and ACTIVE-TB boxes only
    agg: dict[str, dict] = {}
    miss_names: set[str] = set()
    for _, r in df.iterrows():
        fname = str(r["fname"])
        fp = by_name.get(fname)
        if fp is None:
            miss_names.add(fname)
            continue
        if "test" in (pt.lower() for pt in fp.relative_to(base).parts):
            continue  # unlabeled competition split
        a = agg.setdefault(fname, {"fp": fp, "tb_types": set(), "target": str(r["target"]), "boxes": []})
        tb_type = str(r["tb_type"])
        a["tb_types"].add(tb_type)
        if tb_type == "active_tb":
            box = _parse_bbox(r["bbox"])
            if box is not None:
                a["boxes"].append(box)

    binary: list[dict] = []
    latent: list[dict] = []
    for fname, a in agg.items():
        fp = a["fp"]
        common = {"path": str(fp.relative_to(REPO)), "patient_id": f"tbx11k_{fp.stem}"}
        if "active_tb" in a["tb_types"]:
            # active TB present -> positive; carry ONLY its active boxes (latent boxes excluded above)
            binary.append({**common, "label": 1, "source": "tbx11k", "bbox": json.dumps(a["boxes"])})
        elif a["target"] == "no_tb":
            binary.append({**common, "label": 0, "source": "tbx11k", "bbox": "[]"})
        elif "latent_tb" in a["tb_types"]:
            latent.append({**common, "label": 1, "source": "tbx11k_latent", "bbox": "[]"})
    pos = sum(b["label"] for b in binary)
    n_boxed = sum(1 for b in binary if b["label"] == 1 and b["bbox"] != "[]")
    if miss_names:
        print(f"tbx11k: {len(miss_names)} csv fnames had no matching image file (skipped)")
    print(f"tbx11k: {len(binary)} binary ({pos} active-TB pos [{n_boxed} with boxes], "
          f"{len(binary)-pos} neg) + {len(latent)} latent-TB held out as a probe")
    return binary, latent


def main() -> None:
    rows: list[dict] = []
    latent_rows: list[dict] = []
    for src_dir in sorted(d for d in RAW.iterdir() if d.is_dir()):
        src = src_dir.name
        if src in ("lungseg", "nih14", "nih14_parquet"):
            # lungseg = masks (not TB train/test).
            # nih14 = NIH ChestX-ray14: locked, never-train per-finding FPR stress test — read by
            #         training/stress_metrics.py from its sidecar, NOT registered as TB training data.
            # nih14_parquet = the raw parquet cache for nih14.
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
                    "bbox": "[]",  # non-TBX11K sources have no boxes; keep the column consistent
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
