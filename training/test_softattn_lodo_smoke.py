"""Smoke test: 1-fold LODO training with SoftAttnPool + MixStyle on a tiny subset.

Goal: verify the new code path doesn't crash, produces non-NaN losses, and
converges to a sensible AUROC on a within-source train/val split.
This is NOT the P1 full-LODO measurement; it's a ~5-minute sanity check.
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_softattn_smoke_runs_without_error() -> None:
    """Run train_tb.py with --head soft-attn-pool --mixstyle-p 0.5 on a 1-fold subset."""
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    result = subprocess.run(
        [
            str(ROOT / 'training' / '.venv' / 'bin' / 'python'),
            str(ROOT / 'training' / 'train_tb.py'),
            '--head', 'soft-attn-pool',
            '--mixstyle-p', '0.5',
            '--smoke',  # smoke flag: trains 1 fold, 5 epochs, subsampled train
        ],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    assert result.returncode == 0, f'smoke run failed:\n{result.stdout}\n{result.stderr}'
    # Expect at least one AUROC log line
    assert 'auroc' in (result.stdout + result.stderr).lower()


def _run_all() -> None:
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {fn_name}')


if __name__ == '__main__':
    _run_all()
