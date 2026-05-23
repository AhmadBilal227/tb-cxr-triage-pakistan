#!/usr/bin/env bash
# Download TB chest-X-ray datasets into data/raw/. Token-free: reads KAGGLE_API_TOKEN
# from the environment (never hard-code the token here — this file is committed).
#
#   export KAGGLE_API_TOKEN=KGAT_xxx
#   bash scripts/fetch_tb_data.sh
#
# Disk-aware: stops downloading if free space drops below MIN_FREE_GB. Distinct sources
# (qatar / montgomery / shenzhen / tbx11k) are required for leave-one-dataset-out validation;
# lungseg provides masks for the lung-field cropping preprocessing.
set -uo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW"
export PATH="$HOME/.local/bin:$PATH"
MIN_FREE_GB=5

free_gb() { df -g "$RAW" | tail -1 | awk '{print $4}'; }

dl() {
  local slug="$1" name="$2"
  local free; free=$(free_gb)
  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    echo "STOP: only ${free}GB free (< ${MIN_FREE_GB}GB) — skipping $slug and the rest."
    return 1
  fi
  if [ -d "$RAW/$name" ] && [ -n "$(ls -A "$RAW/$name" 2>/dev/null)" ]; then
    echo "=== $name already present, skipping $slug ==="
    return 0
  fi
  echo "=== downloading $slug -> $RAW/$name (free ${free}GB) ==="
  mkdir -p "$RAW/$name"
  kaggle datasets download -d "$slug" -p "$RAW/$name" --unzip 2>&1 | tail -4
  rm -f "$RAW/$name"/*.zip 2>/dev/null || true
  echo "--- done $slug; $(free_gb)GB free now ---"
}

# Priority order: small distinct sources first, then the large volume set, then masks.
dl tawsifurrahman/tuberculosis-tb-chest-xray-dataset qatar      || true
dl raddar/tuberculosis-chest-xrays-montgomery        montgomery || true
dl raddar/tuberculosis-chest-xrays-shenzhen          shenzhen   || true
dl vbookshelf/tbx11k-simplified                      tbx11k     || true
dl iamtapendu/chest-x-ray-lungs-segmentation         lungseg    || true

echo "=== ALL DOWNLOADS DONE ==="
df -h "$RAW" | tail -1
du -sh "$RAW"/* 2>/dev/null || true
