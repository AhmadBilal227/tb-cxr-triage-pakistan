"""Acquire NIH ChestX-ray14 as a LOCKED, never-train, stratified-by-finding FPR stress test.

Source: HF dataset `g-ronimo/NIH-Chest-X-ray-dataset_10k` — a 10k Parquet subset (300px) of the
*official* NIH Clinical Center ChestX-ray14 (Wang et al. 2017, CVPR), traced from
`alkzar90/NIH-Chest-X-ray-dataset`. Data-only Parquet (NO trust_remote_code, NO registration/DUA).
Two shards only: data/train-00000-of-00001.parquet (7,500) + data/test-00000-of-00001.parquet (2,500).

ROLE (revised 2026-05-24 after a strategy review): NIH14 is NOT a blind "all label 0" training
pool. Our real failure mode is false positives on TB MIMICS (fibrosis/scarring, nodules,
consolidation), not easy normals — so NIH14 is a held-out FPR stress test we measure PER FINDING:
FPR(No Finding), FPR(Fibrosis), FPR(Nodule), FPR(Consolidation), ... separately. The whole point is
the per-finding breakdown, not a single binary non-TB FPR (we keep y_nontb=0 too, but stratified is
the deliverable). NIH14 must NEVER enter the TB training set — it is NOT registered in
data/index.csv; the sidecar below is its standalone manifest.

WHY THIS SOURCE (provenance + value):
  - Provenance-INDEPENDENT external fold: NIH Clinical Center (US) cohort, disjoint from our
    NLM/India/China TB sets (Montgomery, Shenzhen, Qatar, TBX11K) — no patient/file overlap.
  - NIH14 has NO tuberculosis label, so every image is a true non-TB case carrying its real
    radiographic findings — exactly the labeled non-TB abnormals our specificity probe lacked.

This script DOWNLOADS (resumable curl/hub transfer, NOT fsspec streaming — that stalls on flaky
links) + DECODES images to data/raw/nih14/<name>.png and writes the sidecar
data/raw/nih14/nih14_findings.csv with ONE ROW PER IMAGE and columns:
  filename, y_nontb (always 0), n_findings, findings (pipe-joined), view_position,
  + one 0/1 column per ChestX-ray14 class: No_Finding, Atelectasis, Cardiomegaly, Effusion,
    Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema,
    Fibrosis, Pleural_Thickening, Hernia.
It does NOT touch the GPU or the TB training index. The sidecar plugs into
training/stress_metrics.py's build_nih_subgroups() (the deployment-gates infrastructure).

VIEW POSITION: the full NIH `Data_Entry_2017.csv` has AP/PA per image, but THIS 10k subset exposes
only `image`+`labels` (no projection field), so view_position is written as "unknown". To recover
AP/portable subgroup FPR you must join on the original NIH metadata (filenames were also dropped by
this subset — see the synthetic-name note). Recorded as a known limitation.

g-ronimo labels are NUMERIC indices (its README is empty). We read the ClassLabel names from the
parquet's own Arrow schema when present, else fall back to the canonical ChestX-ray14 order
(verified against alkzar90's card).

Sampling: by default keep ALL rows (no balance cap) so the per-finding stress test is as large as
possible; --normal-frac<1 optionally caps "No Finding" if you want a leaner abnormal-heavy slice.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "raw" / "nih14"
PARQUET_DIR = REPO / "data" / "raw" / "nih14_parquet"  # local parquet cache (curl-downloaded)
HF_REPO = "g-ronimo/NIH-Chest-X-ray-dataset_10k"
SHARDS = ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]
# Resolve order matters: test (~153MB) before train (~457MB) so a slow/flaky link still yields a
# usable artifact. On flaky networks prefer pre-downloading the parquet with curl -C - --retry-all-errors
#   curl -L -C - -o data/raw/nih14_parquet/test.parquet  <repo>/resolve/main/data/test-00000-of-00001.parquet
# (fsspec streaming + the hub client both stalled here; plain resumable curl was the only reliable path).
LOCAL_SHARDS = [PARQUET_DIR / "test.parquet", PARQUET_DIR / "train.parquet"]

# Canonical NIH ChestX-ray14 class order (alkzar90 dataset card). Index 0 == "No Finding".
NIH_CLASSES = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def _col(name: str) -> str:
    """CSV-safe column name for a finding ('No Finding' -> 'No_Finding')."""
    return name.replace(" ", "_")


def _ensure_shards() -> list[Path]:
    """Resolve parquet shard paths.

    Prefer locally pre-downloaded shards under data/raw/nih14_parquet/ (curl is the reliable path
    on flaky links — see header). Fall back to the resumable hub transfer for any shard not present
    locally. Test split is listed first so a partial pull still yields a usable subset."""
    local = [p for p in LOCAL_SHARDS if p.exists() and p.stat().st_size > 0]
    if local:
        return local

    from huggingface_hub import hf_hub_download

    paths: list[Path] = []
    for fn in SHARDS:
        p = hf_hub_download(HF_REPO, fn, repo_type="dataset", resume_download=True)
        paths.append(Path(p))
    return paths


def _class_names_from_schema(table_schema) -> list[str]:
    """If the parquet embeds HF ClassLabel names (datasets stores them in field metadata as JSON),
    use them. Otherwise fall back to the canonical NIH order."""
    try:
        meta = table_schema.metadata or {}
        hf = meta.get(b"huggingface")
        if hf:
            feats = json.loads(hf.decode())["info"]["features"]
            lab = feats.get("labels")
            # Sequence(ClassLabel) -> {'feature': {'names': [...]}} ; ClassLabel -> {'names': [...]}
            node = lab.get("feature", lab) if isinstance(lab, dict) else {}
            names = node.get("names")
            if names:
                return list(names)
    except Exception as e:  # noqa: BLE001 - schema is best-effort; canonical fallback is safe
        print(f"schema label-name read failed ({e!r}); using canonical NIH order")
    return NIH_CLASSES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10000, help="max images to save (across shards)")
    ap.add_argument("--normal-frac", type=float, default=1.0,
                    help="max fraction of subset that may be 'No Finding' (1.0 = keep all; "
                         "<1.0 = abnormal-heavy slice). The FPR stress test prefers ALL rows.")
    ap.add_argument("--shard", choices=["both", "train", "test"], default="both",
                    help="limit to one shard (useful when a slow link only got one down)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    max_normal = int(args.target * args.normal_frac)
    max_abnormal = args.target  # abnormals are the point; only "No Finding" is ever capped

    shard_paths = _ensure_shards()
    if args.shard == "train":
        shard_paths = [p for p in shard_paths if "train" in p.name]
    elif args.shard == "test":
        shard_paths = [p for p in shard_paths if "test" in p.name]

    n_normal = n_abnormal = saved = 0
    finding_cols = [_col(c) for c in NIH_CLASSES]  # No_Finding, Atelectasis, ... Hernia
    sidecar = OUT_DIR / "nih14_findings.csv"
    with sidecar.open("w", newline="") as fh:
        w = csv.writer(fh)
        # one row per image; binary y_nontb kept, but the per-finding 0/1 columns are the deliverable
        w.writerow(["filename", "y_nontb", "n_findings", "findings", "view_position"] + finding_cols)
        for sp in shard_paths:
            pf = pq.ParquetFile(sp)
            names = _class_names_from_schema(pf.schema_arrow)
            print(f"{sp.name}: {pf.metadata.num_rows} rows; class[0]={names[0]!r}", flush=True)
            for batch in pf.iter_batches(batch_size=256):
                d = batch.to_pydict()
                img_col = d.get("image")
                lab_col = d.get("labels")
                fn_col = d.get("filename") or d.get("image_file_path")
                view_col = d.get("view_position") or d.get("View Position")
                for i in range(len(lab_col)):
                    idxs = lab_col[i] if isinstance(lab_col[i], (list, tuple)) else [lab_col[i]]
                    finding_names = [names[j] if 0 <= j < len(names) else str(j) for j in idxs]
                    findings = "|".join(finding_names)
                    is_normal = finding_names == ["No Finding"]
                    if is_normal and n_normal >= max_normal:
                        continue
                    if not is_normal and n_abnormal >= max_abnormal:
                        continue

                    # filename: prefer provided; else synthesize a stable name (subset dropped names)
                    raw_fn = fn_col[i] if fn_col is not None else None
                    fname = Path(str(raw_fn)).name if raw_fn else f"nih14_{saved:06d}.png"
                    if not fname.lower().endswith(".png"):
                        fname = f"{Path(fname).stem}.png"
                    out_path = OUT_DIR / fname
                    if out_path.exists():
                        continue

                    cell = img_col[i]
                    raw = cell["bytes"] if isinstance(cell, dict) else cell
                    img = Image.open(io.BytesIO(raw)) if isinstance(raw, (bytes, bytearray)) else cell
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(out_path, format="PNG")

                    present = set(finding_names)
                    onehot = [1 if c in present else 0 for c in NIH_CLASSES]
                    view = str(view_col[i]) if view_col is not None and view_col[i] else "unknown"
                    w.writerow([fname, 0, len(finding_names), findings, view] + onehot)

                    if is_normal:
                        n_normal += 1
                    else:
                        n_abnormal += 1
                    saved += 1
                    if saved % 1000 == 0:
                        print(f"saved {saved} (No Finding={n_normal} abnormal={n_abnormal})", flush=True)
                    if n_normal >= max_normal and n_abnormal >= max_abnormal:
                        break
                if n_normal >= max_normal and n_abnormal >= max_abnormal:
                    break
            if n_normal >= max_normal and n_abnormal >= max_abnormal:
                break

    print(f"\nDONE: saved {saved} non-TB images to {OUT_DIR}")
    print(f"  No Finding (independent normals): {n_normal}")
    print(f"  >=1 non-TB abnormal finding:      {n_abnormal}")
    print(f"  per-finding stratified sidecar -> {sidecar}")
    print("  view_position: unavailable in this subset (written 'unknown') — known limitation")


if __name__ == "__main__":
    main()
