"""Locked-protocol OOD calibration + threshold fitting.

After P0 lands, NO subsequent evaluation re-tunes T, threshold, or abstain
thresholds on the evaluation set. The calibration split is pre-registered and
deterministic for SEED=7 (P0_CALIBRATION_SEED, defined here).

This module is the structural defense against the steelman objection:
'per-config calibration tuning IS test-set leakage.' Every phase after P0
calls `load_locked_calibration()` and applies it as-is.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

P0_CALIBRATION_SEED = 7
P0_CAL_FRAC = 0.20  # 20% of LODO OOF goes into the calibration slice
LOCKED_JSON = Path(__file__).parent.parent / 'data' / 'p0_locked_calibration.json'


@dataclass(frozen=True)
class LockedCalibration:
    """Locked T, threshold, and abstain thresholds. Once written, never re-fit."""
    T: float                     # temperature
    thr_at_95sens: float         # decision threshold at 95% sens
    borderline_low: float        # lower edge of borderline band (escalates to VLM verifier)
    s_inactive_escalate: float   # sequelae escalation threshold (M19)
    asymmetric_evidence_thr: float  # M26 asymmetric-evidence threshold
    seed: int
    cal_frac: float
    n_cal: int
    n_eval: int
    git_sha: str
    timestamp: str


def make_calibration_split(
    logit: np.ndarray,
    label: np.ndarray,
    source: np.ndarray,
    seed: int = P0_CALIBRATION_SEED,
    cal_frac: float = P0_CAL_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratify by (label, source) and split deterministically.

    Returns (cal_indices, eval_indices) as numpy int arrays into the input arrays.
    """
    n = len(logit)
    assert len(label) == n and len(source) == n
    rng = np.random.default_rng(seed)
    cal_mask = np.zeros(n, dtype=bool)
    sources = np.unique(source)
    labels = np.unique(label)
    for src in sources:
        for lab in labels:
            idx = np.where((source == src) & (label == lab))[0]
            if len(idx) == 0:
                continue
            n_cal = max(1, int(round(len(idx) * cal_frac))) if len(idx) > 1 else 0
            picked = rng.choice(idx, size=n_cal, replace=False)
            cal_mask[picked] = True
    cal_idx = np.where(cal_mask)[0]
    eval_idx = np.where(~cal_mask)[0]
    return cal_idx, eval_idx


if __name__ == '__main__':
    print(f'P0_CALIBRATION_SEED={P0_CALIBRATION_SEED}, P0_CAL_FRAC={P0_CAL_FRAC}')
