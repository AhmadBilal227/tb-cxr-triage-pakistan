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


def build_mendeley_pk() -> list[dict]:
    """Mendeley Kiran/Jabeen Pakistani TB CXR dataset (May 2024, CC-BY-4.0, v2).

    EXTERNAL VALIDATION HOLDOUT — NOT training data. (Plan correction 2026-05-25.)
    The external blind eval on this cohort (3,008 images, AUROC 0.781 external vs
    0.922 LODO, specificity 0.675) established it as our only well-powered external
    TB+ set (2,494 positives). It is the standing external eval set that P1/P2/P3
    measure generalization against — a site the model NEVER trained on. So every
    row here is tagged `split='external_holdout'` and the aggregator routes these
    rows to data/index_external_holdout.csv, NOT into the training data/index.csv.

    Layout on disk: data/raw/Kiran:Jabeen/{TB Chest X-rays, Normal Chest X-rays}/*.png|jpg
    (the directory name carries the colon as Mendeley shipped it). Label heuristic:
    a directory part containing 'tb' (case-insensitive token) -> TB+; a directory
    part containing 'normal' -> TB-. Logs a warning if no rows are produced so the
    user knows to inspect the raw dir.
    """
    root = RAW / "Kiran:Jabeen"
    if not root.exists():
        # Tolerate the canonical-id directory too, in case a future fetch lands there.
        alt = RAW / "mendeley_pk"
        if alt.exists():
            root = alt
        else:
            return []
    rows: list[dict] = []
    for p in root.rglob("*"):
        if not (p.is_file() and IMG.search(p.name)):
            continue
        rel = p.relative_to(root)
        parts = [pt.lower() for pt in rel.parts]
        relstr = str(rel).lower()
        if "mask" in relstr:
            continue
        if any("tb" in pt or "tubercul" in pt for pt in parts):
            y = 1
        elif any("normal" in pt or "healthy" in pt for pt in parts):
            y = 0
        else:
            continue
        rows.append(
            {
                "path": str(p.relative_to(REPO)),
                "label": y,
                "source": "mendeley_pk",
                "patient_id": f"mendeley_pk_{p.stem}",
                "bbox": "[]",  # no bbox available
                "split": "external_holdout",
            }
        )
    pos = sum(r["label"] for r in rows)
    if rows:
        print(f"mendeley_pk (EXTERNAL HOLDOUT): {len(rows)} rows ({pos} TB+, {len(rows) - pos} TB-)")
    else:
        print("mendeley_pk: 0 rows — inspect data/raw/Kiran:Jabeen/ (expected ~3008 images)")
    return rows


def build_padchest_tb() -> list[dict]:
    """PadChest TB-7-label union subset (BIMCV, Spain, ~152-176+ TB+ studies).

    The canonical TB-positive filter is the union of seven PadChest labels:
      tuberculosis, sequelae tuberculosis, cavitation, calcified adenopathy,
      granuloma, calcified granuloma, apical pleural thickening.
    See PMC11843218 for the published harvest protocol.

    Expects data/raw/padchest_tb/{tb,normal}/*.png after manual BIMCV download
    + filtering by the 7-label union. If the directory is empty (PadChest DUA
    not yet through), returns an empty list silently. This is a NO-OP at runtime
    today; the builder is registered so the data drops in cleanly later.

    Unlike mendeley_pk, PadChest is intended as TRAINING data (atypical-TB
    richness directly addresses the M24 weak spot), so its rows are NOT tagged
    external_holdout.
    """
    root = RAW / "padchest_tb"
    if not root.exists():
        return []
    rows: list[dict] = []
    for p in root.rglob("*"):
        if not (p.is_file() and IMG.search(p.name)):
            continue
        rel = p.relative_to(root).as_posix().lower()
        if rel.startswith("tb/"):
            y = 1
        elif rel.startswith("normal/"):
            y = 0
        else:
            continue
        rows.append(
            {
                "path": str(p.relative_to(REPO)),
                "label": y,
                "source": "padchest_tb",
                "patient_id": f"padchest_{p.stem}",
                "bbox": "[]",
            }
        )
    if rows:
        pos = sum(r["label"] for r in rows)
        print(f"padchest_tb: {len(rows)} rows ({pos} TB+, {len(rows) - pos} TB-)")
    return rows


def main() -> None:
    rows: list[dict] = []
    latent_rows: list[dict] = []
    holdout_rows: list[dict] = []
    for src_dir in sorted(d for d in RAW.iterdir() if d.is_dir()):
        src = src_dir.name
        if src in ("lungseg", "nih14", "nih14_parquet", "Kiran:Jabeen", "mendeley_pk", "padchest_tb"):
            # lungseg = masks (not TB train/test).
            # nih14 = NIH ChestX-ray14: locked, never-train per-finding FPR stress test — read by
            #         training/stress_metrics.py from its sidecar, NOT registered as TB training data.
            # nih14_parquet = the raw parquet cache for nih14.
            # Kiran:Jabeen / mendeley_pk = the EXTERNAL VALIDATION HOLDOUT (built explicitly below
            #         via build_mendeley_pk() and written to index_external_holdout.csv, NOT here).
            # padchest_tb = future training source, built explicitly below via build_padchest_tb().
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

    # PadChest TB-7-union is a TRAINING source (no-op until the BIMCV DUA lands).
    rows.extend(build_padchest_tb())

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

    # EXTERNAL VALIDATION HOLDOUT — written to a SEPARATE manifest so it can NEVER
    # enter training. (Plan correction 2026-05-25: Mendeley PK = holdout, not train.)
    holdout_rows.extend(build_mendeley_pk())
    if holdout_rows:
        ho_df = pd.DataFrame(holdout_rows)
        ho_out = REPO / "data" / "index_external_holdout.csv"
        ho_df.to_csv(ho_out, index=False)
        ho_pos = int((ho_df.label == 1).sum())
        print(f"\nwrote {ho_out}  EXTERNAL HOLDOUT total={len(ho_df)}  pos={ho_pos}  "
              f"neg={len(ho_df) - ho_pos}  (EXCLUDED from training index.csv)")
        print(ho_df.groupby(["source", "label"]).size())


if __name__ == "__main__":
    main()
