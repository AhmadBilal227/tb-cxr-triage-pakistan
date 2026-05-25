"""Audit cross-source overlap between the EXTERNAL HOLDOUT and the TRAINING index.

The Mendeley Pakistani cohort is the external validation holdout (P0.5). If a
training image leaks into the holdout (or vice versa) the external eval is
invalidated — a model that "generalizes" to an image it trained on tells us
nothing. This script protects the holdout's validity.

It loads data/index.csv (training) + data/index_external_holdout.csv (holdout),
computes md5 (exact) + pHash-256 (near) for every image, and reports ONLY
cross-set matches (a holdout image near-identical to a training image) to
data/dedup_audit.log. It does NOT modify data/index_dedup.csv — the holdout
must never enter the training dedup graph that train_tb.py reads.

Reuses the exact match policy from training/dedup.py:
  PHASH_SIZE=16 (256-bit), PHASH_THRESHOLD=6 (~2.3%) — only near-identical match.

Run: training/.venv/bin/python training/audit_holdout_overlap.py
"""
from __future__ import annotations
import hashlib
from datetime import datetime
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PHASH_SIZE = 16          # 256-bit perceptual hash (matches dedup.py)
PHASH_THRESHOLD = 6      # out of 256 bits (~2.3%) — only near-identical images match
LOG = DATA / "dedup_audit.log"


def _hash_index(df: pd.DataFrame, tag: str) -> tuple[list[dict], np.ndarray, list[str]]:
    """Return (kept_rows, phash_bits[n,256], md5s) for the readable images in df."""
    rows: list[dict] = []
    bits: list[np.ndarray] = []
    md5s: list[str] = []
    skipped = 0
    for _, r in df.iterrows():
        full = r["path"] if str(r["path"]).startswith("/") else str(REPO / r["path"])
        try:
            raw = Path(full).read_bytes()
            ph = imagehash.phash(Image.open(full).convert("L"), hash_size=PHASH_SIZE)
        except Exception:
            skipped += 1
            continue
        rec = dict(r)
        rec["_set"] = tag
        rows.append(rec)
        bits.append(ph.hash.flatten().astype(np.uint8))
        md5s.append(hashlib.md5(raw).hexdigest())
    if skipped:
        print(f"  {tag}: skipped {skipped} unreadable images")
    H = np.stack(bits) if bits else np.zeros((0, PHASH_SIZE * PHASH_SIZE), dtype=np.uint8)
    return rows, H, md5s


def main() -> None:
    train_csv = DATA / "index.csv"
    holdout_csv = DATA / "index_external_holdout.csv"
    if not holdout_csv.exists():
        raise SystemExit(f"missing {holdout_csv}; run training/build_index.py first")
    train_df = pd.read_csv(train_csv).reset_index(drop=True)
    holdout_df = pd.read_csv(holdout_csv).reset_index(drop=True)

    print(f"hashing TRAINING index ({len(train_df)} rows)...")
    tr_rows, tr_H, tr_md5 = _hash_index(train_df, "train")
    print(f"hashing EXTERNAL HOLDOUT index ({len(holdout_df)} rows)...")
    ho_rows, ho_H, ho_md5 = _hash_index(holdout_df, "holdout")

    matches: list[dict] = []

    # exact md5 cross-set matches
    tr_md5_to_idx: dict[str, int] = {}
    for i, m in enumerate(tr_md5):
        tr_md5_to_idx.setdefault(m, i)
    for j, m in enumerate(ho_md5):
        if m in tr_md5_to_idx:
            i = tr_md5_to_idx[m]
            matches.append({
                "kind": "md5-exact",
                "hamming": 0,
                "holdout": ho_rows[j]["path"],
                "train": tr_rows[i]["path"],
                "holdout_label": int(ho_rows[j]["label"]),
                "train_label": int(tr_rows[i]["label"]),
            })

    # near-duplicate pHash cross-set matches (holdout row vs every training row)
    if len(tr_H) and len(ho_H):
        for j in range(len(ho_H)):
            d = (tr_H != ho_H[j]).sum(axis=1)
            for i in np.where(d <= PHASH_THRESHOLD)[0]:
                matches.append({
                    "kind": "phash-near",
                    "hamming": int(d[i]),
                    "holdout": ho_rows[j]["path"],
                    "train": tr_rows[i]["path"],
                    "holdout_label": int(ho_rows[j]["label"]),
                    "train_label": int(tr_rows[int(i)]["label"]),
                })

    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    header = (
        f"\n===== HOLDOUT-vs-TRAINING OVERLAP AUDIT {ts} =====\n"
        f"train rows hashed: {len(tr_rows)}  |  holdout rows hashed: {len(ho_rows)}\n"
        f"policy: md5-exact + pHash-256 hamming<={PHASH_THRESHOLD}\n"
        f"cross-set matches found: {len(matches)}\n"
    )
    lines = [header]
    if matches:
        for m in matches:
            lines.append(
                f"  MATCH [{m['kind']} h={m['hamming']}] "
                f"holdout({m['holdout_label']})={m['holdout']}  <==>  "
                f"train({m['train_label']})={m['train']}\n"
            )
        lines.append(
            "  >>> ALARM: holdout image(s) overlap the training index. The external eval "
            "is invalid for these images — remove them from the holdout or the training set.\n"
        )
    else:
        lines.append("  CLEAN: zero overlap. Holdout validity preserved.\n")

    with LOG.open("a") as f:
        f.writelines(lines)
    print("".join(lines))
    print(f"appended audit to {LOG}")


if __name__ == "__main__":
    main()
