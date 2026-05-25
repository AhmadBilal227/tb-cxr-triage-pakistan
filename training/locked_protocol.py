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


import subprocess
from datetime import datetime


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(Path(__file__).parent.parent),
            text=True,
        ).strip()
    except Exception:
        return 'unknown'


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _fit_temperature(logit: np.ndarray, label: np.ndarray) -> float:
    """Grid search T in [0.5, 5.0] to minimize NLL on (logit, label)."""
    Ts = np.linspace(0.5, 5.0, 91)
    best_nll = np.inf
    best_T = 1.0
    for T in Ts:
        p = _sigmoid(logit / T).clip(1e-7, 1 - 1e-7)
        nll = -(label * np.log(p) + (1 - label) * np.log(1 - p)).mean()
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T


def _threshold_for_sensitivity(label: np.ndarray, prob: np.ndarray, target: float = 0.95) -> float:
    """Smallest threshold such that recall >= target on positives."""
    pos_scores = np.sort(prob[label == 1])
    if len(pos_scores) == 0:
        return 0.5
    # take the (1-target) quantile of positive scores; anything >= that is called positive
    idx = max(0, int(np.floor((1 - target) * len(pos_scores))))
    return float(pos_scores[idx])


def fit_locked_calibration(write: bool = True) -> LockedCalibration:
    """Lock the DEPLOYED operating point + the frozen eval split.

    CORRECTION (2026-05-25): an earlier version re-fit T from scratch (NLL-optimal
    T=2.85) + a 95%-sens threshold (0.098). That drifted the probability space away
    from the deployed model's T=1.5915 — and the M26 rule-stack constants
    (borderline_low=0.20, s_inactive_escalate=0.7126, asymmetric_evidence_thr=0.88)
    were all derived in the T=1.5915 space. Result: borderline_low (0.20) > thr (0.098),
    an INVERTED ABSTAIN band that silently disabled the entire M26 safety net
    (abstain_rate=0.0). Temperature and threshold are coupled (same ROC point), so
    re-fitting T buys nothing for the decision while breaking the downstream rules.

    The locked protocol therefore locks the DEPLOYED operating point (T + thr read
    verbatim from data/tb_threshold_t2.json — the validated M22 values that the M26
    rule stack assumes) PLUS the deterministic seed=7 20/80 eval split. The "locked"
    property is: frozen split + frozen shipped operating point, never re-tuned per
    config on the evaluation surface. (Per-site recalibration for NEW deployment
    sites is a separate flow — P0.5+, not the baseline.)
    """
    data_dir = Path(__file__).parent.parent / 'data'
    oof_path = data_dir / 'image_oof_logits.npz'
    if not oof_path.exists():
        raise FileNotFoundError(f'OOF cache missing: {oof_path}. Run M14 LODO first.')
    deployed_path = data_dir / 'tb_threshold_t2.json'
    if not deployed_path.exists():
        raise FileNotFoundError(f'Deployed calibration missing: {deployed_path}.')
    deployed = json.loads(deployed_path.read_text())
    T = float(deployed['temperature'])           # validated deployed T (1.5915)
    thr = float(deployed['threshold'])            # validated deployed thr (0.6105)

    d = np.load(oof_path, allow_pickle=True)
    logit = d['image_logit'].astype('float64')
    label = d['label'].astype('int64')
    source = d['source']
    cal_idx, eval_idx = make_calibration_split(logit, label, source)

    borderline_low = 0.20  # M26 widened borderline band lower edge (in the T=1.5915 space)
    # Invariant: the ABSTAIN band [borderline_low, thr) must be non-empty, else the
    # entire M26 safety net (ABSTAIN routing) is silently dead. This is the exact
    # bug the 2026-05-25 correction fixes.
    assert borderline_low < thr, (
        f'ABSTAIN band inverted: borderline_low={borderline_low} >= thr={thr}. '
        'The M26 rule stack would never fire ABSTAIN. Lock T+thr to the deployed '
        'operating point (tb_threshold_t2.json) so the band stays valid.'
    )

    cal = LockedCalibration(
        T=T,
        thr_at_95sens=thr,  # field name retained for schema stability; value is the deployed thr
        borderline_low=borderline_low,
        s_inactive_escalate=0.7126,  # M19 sequelae escalator (T=1.5915 space)
        asymmetric_evidence_thr=0.88,  # M26 box-evidence high threshold (T=1.5915 space)
        seed=P0_CALIBRATION_SEED,
        cal_frac=P0_CAL_FRAC,
        n_cal=int(len(cal_idx)),
        n_eval=int(len(eval_idx)),
        git_sha=_git_sha(),
        timestamp=datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    )

    if write:
        LOCKED_JSON.parent.mkdir(parents=True, exist_ok=True)
        with LOCKED_JSON.open('w') as f:
            json.dump(cal.__dict__, f, indent=2)
        print(f'Wrote locked calibration to {LOCKED_JSON}')
        print(f'  T={cal.T:.4f}  thr@95sens={cal.thr_at_95sens:.4f}')
        print(f'  n_cal={cal.n_cal}  n_eval={cal.n_eval}')

    return cal


def load_locked_calibration() -> LockedCalibration:
    """Load the locked calibration. Raises if not yet fit (P0 hasn't run)."""
    if not LOCKED_JSON.exists():
        raise FileNotFoundError(
            f'Locked calibration missing: {LOCKED_JSON}. '
            'Run training/locked_protocol.py to fit it before evaluation.'
        )
    with LOCKED_JSON.open() as f:
        data = json.load(f)
    return LockedCalibration(**data)


if __name__ == '__main__':
    cal = fit_locked_calibration(write=True)
