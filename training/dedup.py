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


def main() -> None:
    df = pd.read_csv(REPO / "data" / "index.csv")
    seen_md5: set[str] = set()
    seen_ahash: set[str] = set()
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
        ah = str(imagehash.average_hash(im))
        if md5 in seen_md5 or ah in seen_ahash:
            dropped += 1
            continue
        seen_md5.add(md5)
        seen_ahash.add(ah)
        keep.append(r)
    out = pd.DataFrame(keep)
    out.to_csv(REPO / "data" / "index_dedup.csv", index=False)
    print(f"dedup: kept {len(out)} / {len(df)} ({dropped} duplicates dropped)")
    if len(out):
        print(out.groupby(["source", "label"]).size())


if __name__ == "__main__":
    main()
