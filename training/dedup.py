"""Cross-source dedup -> data/index_dedup.csv.

Qatar aggregates NLM (Montgomery/Shenzhen) + NIAID, so cross-source COPIES leak LODO unless
removed. Two dup types:
  - exact: md5 of the RAW file bytes (catches literal copies).
  - near-dup: a discriminative perceptual hash (pHash, hash_size=16 -> 256-bit).

Chest X-rays are globally similar (all frontal chests), so coarse average_hash collapses
DISTINCT patients (it flagged TB and Normal images as "duplicates"). So we (a) use pHash at a
tight threshold, and (b) only drop a near-dup when LABELS AGREE — a TB/Normal near-match is a
contradiction (distinct patients), so we KEEP both and just report the count.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PHASH_SIZE = 16          # 256-bit perceptual hash (far more discriminative than 64-bit aHash)
PHASH_THRESHOLD = 6      # out of 256 bits (~2.3%) — only near-identical images match


def main() -> None:
    df = pd.read_csv(REPO / "data" / "index.csv")
    seen_md5: set[str] = set()
    kept_ph: list[tuple[imagehash.ImageHash, int]] = []  # (phash, label)
    keep: list = []
    dropped_exact = dropped_near = conflicts = 0
    for _, r in df.iterrows():
        label = int(r["label"])
        full = r["path"] if str(r["path"]).startswith("/") else str(REPO / r["path"])
        try:
            raw = Path(full).read_bytes()
        except Exception:
            continue
        md5 = hashlib.md5(raw).hexdigest()
        if md5 in seen_md5:
            dropped_exact += 1
            continue
        try:
            ph = imagehash.phash(Image.open(full).convert("L"), hash_size=PHASH_SIZE)
        except Exception:
            continue
        near = any((ph - kp) <= PHASH_THRESHOLD and kl == label for kp, kl in kept_ph)
        if near:
            dropped_near += 1
            continue
        # different-label near-match => distinct patients, keep both, just count
        if any((ph - kp) <= PHASH_THRESHOLD and kl != label for kp, kl in kept_ph):
            conflicts += 1
        seen_md5.add(md5)
        kept_ph.append((ph, label))
        keep.append(r)

    out = pd.DataFrame(keep)
    out.to_csv(REPO / "data" / "index_dedup.csv", index=False)
    print(
        f"dedup: kept {len(out)} / {len(df)} "
        f"(exact {dropped_exact}, same-label near {dropped_near} dropped; "
        f"{conflicts} cross-label near-matches KEPT as distinct)"
    )
    if len(out):
        print(out.groupby(["source", "label"]).size())


if __name__ == "__main__":
    main()
