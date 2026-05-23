"""Cross-source dedup -> data/index_dedup.csv. These TB sets overlap (Qatar aggregates
NLM Montgomery/Shenzhen + NIAID), so training on one and 'externally' testing on another
leaks unless duplicates are removed first. Drops exact (md5) and near-duplicate (average-hash)
images, keeping the first occurrence.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]

# Hamming-distance threshold for near-duplicate detection (average hash, 64-bit).
# Distance <= 5 treats images as duplicates; tuned for 256x256 greyscale CXRs.
AHASH_THRESHOLD = 5


def main() -> None:
    df = pd.read_csv(REPO / "data" / "index.csv")
    seen_md5: set[str] = set()
    # Keep imagehash objects (not strings) so we can compute Hamming distance.
    # Near-dup detection is a one-time offline O(n·k) cost where k = kept images so far.
    kept_ahashes: list[imagehash.ImageHash] = []
    keep: list = []
    dropped = 0
    for _, r in df.iterrows():
        path = r["path"]
        full = path if str(path).startswith("/") else str(REPO / path)
        try:
            im = Image.open(full).convert("L").resize((256, 256))
        except Exception:
            continue
        md5 = hashlib.md5(im.tobytes()).hexdigest()
        ah = imagehash.average_hash(im)
        # Drop if exact md5 match OR Hamming distance <= threshold vs any kept hash.
        if md5 in seen_md5 or (
            kept_ahashes and min(abs(ah - kh) for kh in kept_ahashes) <= AHASH_THRESHOLD
        ):
            dropped += 1
            continue
        seen_md5.add(md5)
        kept_ahashes.append(ah)
        keep.append(r)
    out = pd.DataFrame(keep)
    out.to_csv(REPO / "data" / "index_dedup.csv", index=False)
    print(f"dedup: kept {len(out)} / {len(df)} ({dropped} duplicates dropped)")
    if len(out):
        print(out.groupby(["source", "label"]).size())


if __name__ == "__main__":
    main()
