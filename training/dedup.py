"""Cross-source dedup + provenance grouping -> data/index_dedup.csv (adds a `group` column).

Qatar aggregates NLM (Montgomery/Shenzhen) + NIAID; TBX11K aggregates India-DA/DB + Montgomery +
Shenzhen. So the "sources" overlap and naive per-source LODO is NOT external. We therefore:
  1. md5 (exact) + pHash-256 (near) match EVERY image against every other ACROSS sources;
  2. union matches into connected components = provenance CLUSTERS (a `group` id per image);
  3. keep ONE representative per (group, label) so redundant copies don't inflate counts;
  4. CROSS-LABEL clusters (same film labelled TB in one wrapper, Normal in another) are a
     data-quality ALARM — reported loudly, and kept in the SAME group so LODO's dup-guard never
     splits them across the train/test boundary (the leak path gpt-5.5 flagged).
train_tb.py uses `group` to exclude train images that near-match a held-out test image.

(Embedding-NN dedup on Rad-DINO CLS is a documented follow-up — it needs features.npz, which is
built after this step.)
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PHASH_SIZE = 16          # 256-bit perceptual hash
PHASH_THRESHOLD = 6      # out of 256 bits (~2.3%) — only near-identical images match


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main() -> None:
    df = pd.read_csv(REPO / "data" / "index.csv").reset_index(drop=True)
    rows, bits, md5s, labels = [], [], [], []
    skip_by_src: dict[str, int] = {}  # FAIL-VISIBLE (P1): count unreadable images dropped, per source
    for _, r in df.iterrows():
        full = r["path"] if str(r["path"]).startswith("/") else str(REPO / r["path"])
        try:
            raw = Path(full).read_bytes()
            ph = imagehash.phash(Image.open(full).convert("L"), hash_size=PHASH_SIZE)
        except Exception:
            src = str(r["source"]) if "source" in r else "unknown"
            skip_by_src[src] = skip_by_src.get(src, 0) + 1
            continue
        rows.append(r)
        bits.append(ph.hash.flatten().astype(np.uint8))
        md5s.append(hashlib.md5(raw).hexdigest())
        labels.append(int(r["label"]))
    n = len(rows)
    if n == 0:
        raise SystemExit("no readable images")
    H = np.stack(bits)               # [n, 256] uint8
    labels = np.asarray(labels)
    uf = _UF(n)

    # union exact (md5) duplicates
    by_md5: dict[str, int] = {}
    for i, m in enumerate(md5s):
        if m in by_md5:
            uf.union(i, by_md5[m])
        else:
            by_md5[m] = i
    # union near-duplicates (vectorized Hamming, upper triangle)
    for i in range(n - 1):
        d = (H[i + 1:] != H[i]).sum(axis=1)
        for j in np.where(d <= PHASH_THRESHOLD)[0]:
            uf.union(i, i + 1 + int(j))

    comp = np.array([uf.find(i) for i in range(n)])
    _, group = np.unique(comp, return_inverse=True)  # remap component roots -> 0..G-1

    # cross-label clusters = data-quality alarm
    conflict_groups = {int(g) for g in np.unique(group) if len(set(labels[group == g])) > 1}

    # keep one representative per (group, label)
    seen: set[tuple[int, int]] = set()
    keep_idx, keep_groups = [], []
    for i in range(n):
        key = (int(group[i]), int(labels[i]))
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)
        keep_groups.append(int(group[i]))

    out = pd.DataFrame([rows[i] for i in keep_idx]).reset_index(drop=True)
    out["group"] = keep_groups
    out.to_csv(REPO / "data" / "index_dedup.csv", index=False)
    n_clusters = int(len(np.unique(group)))
    print(f"dedup: kept {len(out)} / {n} readable (collapsed {n - len(out)} redundant copies into "
          f"{n_clusters} provenance clusters)")
    n_skipped = int(sum(skip_by_src.values()))
    if n_skipped:
        print(f"SKIPPED (unreadable, dropped before dedup): {n_skipped}/{len(df)} images per source: "
              f"{skip_by_src}  — investigate; index_dedup.csv is short by these source images")
    else:
        print(f"skipped images: 0/{len(df)} (every indexed image was readable)")
    if conflict_groups:
        print(f">>> DATA-QUALITY ALARM: {len(conflict_groups)} cross-label clusters (same image labelled "
              f"BOTH TB and Normal across wrappers). Kept in one group (no train/test leak), but the "
              f"labels disagree — adjudicate before any clinical claim.")
    else:
        print("no cross-label conflict clusters detected.")
    print(out.groupby(["source", "label"]).size())


if __name__ == "__main__":
    main()
