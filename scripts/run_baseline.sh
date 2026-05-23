#!/usr/bin/env bash
# Baseline run on already-downloaded complete sources (qatar/montgomery/shenzhen).
# Validates the full pipeline end-to-end and records LODO numbers to compare against the
# full run later. Balanced subsample (cap 700 per source-label) for speed; TBX11K excluded
# (not yet unzipped). NOT the final number — a recorded baseline.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1
mkdir -p docs/baselines
REPORT="docs/baselines/2026-05-24-baseline.txt"

echo "=== dedup (cross-source) ===" | tee "$REPORT"
training/.venv/bin/python training/dedup.py 2>&1 | tee -a "$REPORT"

echo "=== balanced subsample (cap 700/source-label; exclude tbx11k) ===" | tee -a "$REPORT"
training/.venv/bin/python - <<'PY' 2>&1 | tee -a "$REPORT"
import pandas as pd
df = pd.read_csv("data/index_dedup.csv")
df = df[df["source"] != "tbx11k"]
parts = [g.sample(min(len(g), 700), random_state=0) for _, g in df.groupby(["source", "label"])]
out = pd.concat(parts).reset_index(drop=True)
out.to_csv("data/index_dedup.csv", index=False)
print("baseline set:", len(out))
print(out.groupby(["source", "label"]).size())
PY

echo "=== extract_features (dual-backbone, cached) ===" | tee -a "$REPORT"
training/.venv/bin/python training/extract_features.py 2>&1 | grep -vE "skip|Warning" | tail -3 | tee -a "$REPORT"

echo "=== train_tb (LODO + fusion-only-vs-attention ablation + frozen-threshold sensitivity) ===" | tee -a "$REPORT"
training/.venv/bin/python training/train_tb.py 2>&1 | tee -a "$REPORT"

echo "=== BASELINE DONE -> $REPORT ===" | tee -a "$REPORT"
