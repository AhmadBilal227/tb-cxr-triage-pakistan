#!/usr/bin/env bash
# FULL validation run on all downloaded sources (qatar/montgomery/shenzhen/tbx11k).
# No subsample — the honest number on everything we have. Pipeline:
#   build_index -> cross-source dedup -> dual-backbone extract (CPU+GPU) -> LODO + dual report.
# CPU stays capped (~60-70%) so the machine remains usable. ~2h, dominated by extraction.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1
mkdir -p docs/baselines
REPORT="docs/baselines/2026-05-24-full-audit-fixed.txt"

echo "=== build_index (all downloaded sources) ===" | tee "$REPORT"
training/.venv/bin/python training/build_index.py 2>&1 | tee -a "$REPORT"

echo "=== dedup (cross-source: remove qatar/tbx11k re-mix copies) ===" | tee -a "$REPORT"
training/.venv/bin/python training/dedup.py 2>&1 | tee -a "$REPORT"

echo "=== extract_features (dual-backbone, CPU+GPU, capped, harmonized) ===" | tee -a "$REPORT"
training/.venv/bin/python training/extract_features.py 2>&1 | grep -vE "skip|Warning" | tail -6 | tee -a "$REPORT"

echo "=== site-leak audit (AFTER harmonization — compare to 1.000 before) ===" | tee -a "$REPORT"
training/.venv/bin/python training/audit.py 2>&1 | grep -avE "Warning|warn" | tee -a "$REPORT"

echo "=== train_tb (LODO + dual report + bootstrap CIs + ECE + PPV-at-prevalence) ===" | tee -a "$REPORT"
training/.venv/bin/python training/train_tb.py 2>&1 | tee -a "$REPORT"

echo "=== FULL RUN DONE -> $REPORT ===" | tee -a "$REPORT"
