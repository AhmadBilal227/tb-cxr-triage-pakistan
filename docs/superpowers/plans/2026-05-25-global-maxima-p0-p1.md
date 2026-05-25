# Global Maxima: P0 Locked Baseline + P0.5 Multi-Source Data + P1 Surgical M24 Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a locked-protocol OOD measurement baseline, expand the training corpus by two open multi-geography TB CXR datasets (Mendeley Pakistani 2,494 TB+, PadChest TB-7-label union ~150+ TB+), and apply a single surgical architectural fix (replace ZonalSoftOR hard zone priors with a softer non-zonal attention pool) to address the M24-diagnosed atypical-TB failure mode — without conflating the change with backbone swaps, multi-task heads, or focal loss.

**Architecture:** Three sequential evidence-gated phases. **P0** locks calibration + adds TTA on the current model; produces an honest baseline measurement that all subsequent comparisons reference. **P0.5** acquires two new open datasets and integrates them into `build_index.py` without retraining yet. **P1** trains exactly ONE surgical change (a `SoftAttnPool` replacing `ZonalSoftOR`) on the 6-source corpus (current 4 + Mendeley PK + PadChest TB-7-union) with MixStyle augmentation, then re-validates under the P0 locked protocol. Per the steelman discipline: **one lever per phase, locked thresholds before evaluation, paired-bootstrap CIs, refuse to re-tune calibration on the evaluation set**.

**Tech Stack:** Python 3.12 + PyTorch + transformers (microsoft/rad-dino backbone, unchanged for this plan) + torchxrayvision + numpy + scikit-learn + FastAPI (server) + Vitest (frontend tests) + jest-style `_run_all()` Python test pattern (matches existing `training/test_*.py`).

---

## ⚠️ PLAN CORRECTION (2026-05-25, post external Pakistani eval) — READ BEFORE EXECUTING P0.5 + P1

The external blind eval on the Mendeley Pakistani cohort (3,008 images, AUROC 0.781 external vs 0.922 LODO, **specificity 0.675** the failure mode — 1 in 3 normals false-flagged) changes two things in this plan:

1. **The Mendeley Pakistani cohort is the EXTERNAL VALIDATION HOLDOUT, NOT training data.** It is our only well-powered external TB+ set (2,494 positives). Tasks P0.5.2 and P1.4 below say to train on `mendeley_pk` — **OVERRIDE: do NOT add `mendeley_pk` to the training `--sources`.** Register it in `build_index.py` (P0.5.2) but tag it `split='external_holdout'` and exclude it from training. It becomes the standing external eval set that P1/P2/P3 measure generalization against (a site the model never trained on).

2. **The eval surfaced a NEW failure mode distinct from M24:** specificity drift on normals (the model over-flags normals on a new site), separate from M24's sensitivity-on-atypical-TB. The biggest immediately-available lever for specificity drift is **negative-class diversity** — add NIH ChestX-ray14 `No_Finding` (5,788 diverse US normals, already extracted in `data/features_nih14.npz`) to the P1 training NEGATIVE pool. So P1's corrected training sources are: `mont, shen, qatar, tbx11k` (original TB+/normal) **+ nih14 No_Finding normals as extra negatives**. TB+ diversity from PadChest/VinDr joins when their DUAs land. **Pakistani stays held-out.**

3. **P1's evaluation** uses the locked P0 protocol on the LODO eval slice + the held-out Pakistani cohort. The GO gate is now: does the head fix + negative diversity improve the **Pakistani external specificity** (currently 0.675) without dropping LODO sens — measured against the frozen P0 baseline on the same held-out Pakistani set.

Every task below executes as written EXCEPT the `--sources` composition (exclude mendeley_pk, add nih14-normals) and the eval target (add Pakistani-holdout). The agent executing P0.5/P1 must honor this correction over the original task text.

---

## File Structure

### New files

- `training/locked_protocol.py` — single source of truth for the locked OOD calibration split + threshold/T fitting. Used by P0 measurement + every subsequent P1/P2/P3 evaluation.
- `training/tta.py` — test-time augmentation utility. CXR-safe augs only (H-flip, brightness ±10%, contrast ±10%). K-pass averaged probabilities.
- `training/test_locked_protocol.py` — unit tests for the locked-protocol module (deterministic split, threshold reproducibility, drift tripwires).
- `training/test_tta.py` — unit tests for TTA (output shape, averaging logic, no-rotation guarantee, deterministic seed behavior).
- `training/measure_p0_baseline.py` — runs the P0 baseline once on Cohen blind + NIH per-finding under locked protocol; outputs `data/p0_baseline.json` + appends EXPERIMENT_LOG row.
- `scripts/fetch_mendeley_pk.py` — downloads Mendeley Kiran/Jabeen Pakistani TB CXR (CC-BY-4.0) to `data/raw/mendeley_pk/`.
- `scripts/fetch_padchest_tb.py` — given a manual BIMCV credentials file at `~/.padchest_creds`, downloads PadChest TB-7-label union subset to `data/raw/padchest_tb/`.
- `training/heads/soft_attn_pool.py` — the new `SoftAttnPool` module (single learned attention over patch tokens, no hard zone priors).
- `training/heads/__init__.py` — module init exposing `SoftAttnPool` + the existing `ZonalSoftOR` (kept for A/B comparison).
- `training/mixstyle.py` — MixStyle feature-space augmentation (Zhou et al. 2021 ICLR). Mixes channel-wise mean/std statistics across batch items.
- `training/test_soft_attn_pool.py` — unit tests for `SoftAttnPool` (forward shape, attention weight sums to 1, gradient flow).
- `training/test_mixstyle.py` — unit tests for MixStyle (probability gate, statistics mixing, identity at p=0).
- `training/train_tb_softattn.py` — P1 training script: LODO retrain with `SoftAttnPool` + MixStyle on 6-source corpus.
- `training/test_softattn_lodo_smoke.py` — smoke test: train 1 fold on a tiny subset, assert convergence + non-NaN losses.

### Modified files

- `training/build_index.py` — add `build_mendeley_pk()` + `build_padchest_tb()` source builders matching the existing `build_montgomery()`/etc. pattern.
- `training/extract_features.py` — no algorithmic change; accept the new source ids in the `--sources` flag dispatch.
- `training/train_tb.py` — add `--head` flag (default `zonal-softor` for current behavior; alternate `soft-attn-pool` for the P1 head) wired into `train_head_t2()`. Add `--mixstyle-p` flag (default 0.0; P1 uses 0.5). Add `--sources` to expand from the hardcoded 4 to a configurable list.
- `training/triage_core.py` — accept `head_kind: Literal['zonal-softor','soft-attn-pool']` in `TriageEngine.__init__()`, defaulting to `'zonal-softor'` for backward compat.
- `training/dedup.py` — no algorithmic change; accept the new source ids when computing the pHash cross-source match graph.
- `docs/CASE_STUDY.md` — append Milestone P0, P0.5, P1 entries with real numbers.
- `docs/EXPERIMENT_LOG.md` — append §C rows for P0 baseline, P0.5 data integration, P1 training + validation.
- `docs/DATA_SOURCES.md` — promote Mendeley PK + PadChest TB-7-union from "consider" to "integrated" with citations.
- `CLAUDE.md` — update Orientation to note P0 locked protocol as the new evaluation standard.

---

## Phase P0 — Locked-Protocol Baseline (target: 1 day, 8 tasks)

**Goal of P0:** define and freeze the OOD calibration split + T + threshold + abstain thresholds + TTA pipeline ONCE, before any P1 changes. From this point forward, no evaluation re-tunes these parameters. The steelman objection (per-config calibration tuning IS test-set leakage) is structurally prevented.

### Task P0.1: Define and pre-register the OOD calibration split

**Files:**
- Create: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/locked_protocol.py`
- Test: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/test_locked_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
# training/test_locked_protocol.py
from __future__ import annotations
import numpy as np
from pathlib import Path
import torch

from locked_protocol import make_calibration_split  # type: ignore[import-not-found]


def test_calibration_split_is_deterministic() -> None:
    """Same seed + same data → same split. The whole point of 'locked'."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(12345)
        logit = np.linspace(-5, 5, 1000).astype('float64')
        label = (np.arange(1000) % 2 == 0).astype('int64')
        source = np.array(['a'] * 500 + ['b'] * 500)
        cal_a, eval_a = make_calibration_split(logit, label, source, seed=7)
        cal_b, eval_b = make_calibration_split(logit, label, source, seed=7)
        assert (cal_a == cal_b).all()
        assert (eval_a == eval_b).all()
    finally:
        np.random.set_state(rng_state)


def test_calibration_split_is_stratified_by_label_and_source() -> None:
    """The split must preserve label and source balance to within 1 percentage point."""
    rng = np.random.default_rng(0)
    logit = rng.normal(0, 2, 1000)
    label = rng.integers(0, 2, 1000)
    source = np.array(rng.choice(['m', 'q', 's', 't'], 1000))
    cal, evl = make_calibration_split(logit, label, source, seed=7, cal_frac=0.20)
    assert len(cal) + len(evl) == 1000
    assert 195 <= len(cal) <= 205  # 20% +/- 0.5%
    for src in ['m', 'q', 's', 't']:
        cal_frac = (source[cal] == src).mean() if len(cal) else 0.0
        eval_frac = (source[evl] == src).mean() if len(evl) else 0.0
        assert abs(cal_frac - eval_frac) < 0.05  # stratification balance


def _run_all() -> None:
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {fn_name}')


if __name__ == '__main__':
    _run_all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/ahmadbilal/Downloads/hobby/TB detector"
training/.venv/bin/python training/test_locked_protocol.py
```

Expected: ImportError on `from locked_protocol import make_calibration_split` — file doesn't exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# training/locked_protocol.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
training/.venv/bin/python training/test_locked_protocol.py
```

Expected output:
```
PASS  test_calibration_split_is_deterministic
PASS  test_calibration_split_is_stratified_by_label_and_source
```

- [ ] **Step 5: Commit**

```bash
git add training/locked_protocol.py training/test_locked_protocol.py
git commit -m "feat(p0): stratified deterministic OOD calibration split (SEED=7, frac=0.20)

The structural defense against the steelman objection that per-config
calibration tuning IS test-set leakage. Every subsequent P1/P2/P3 evaluation
calls make_calibration_split with SEED=7 and applies the resulting T+thr
unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P0.2: Fit and persist locked calibration on the cached LODO OOF

**Files:**
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/locked_protocol.py` (add `fit_locked_calibration()` + `load_locked_calibration()`)
- Add tests to: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/test_locked_protocol.py`

- [ ] **Step 1: Write the failing test (add to existing test file)**

Add this function to `training/test_locked_protocol.py` before `_run_all`:

```python
def test_fit_locked_calibration_against_cached_oof() -> None:
    """Fit and write the locked JSON; assert it is reproducible and within reasonable bounds."""
    from locked_protocol import fit_locked_calibration, load_locked_calibration, LOCKED_JSON

    if not (Path(__file__).parent.parent / 'data' / 'image_oof_logits.npz').exists():
        print('SKIP test_fit_locked_calibration_against_cached_oof — no OOF cache')
        return
    cal_a = fit_locked_calibration(write=False)
    cal_b = fit_locked_calibration(write=False)
    assert abs(cal_a.T - cal_b.T) < 1e-9, 'T must be deterministic'
    assert abs(cal_a.thr_at_95sens - cal_b.thr_at_95sens) < 1e-9
    # bounds: T must be positive and reasonable
    assert 0.5 < cal_a.T < 5.0, f'T={cal_a.T} out of reasonable range'
    # thr at 95% sens should be below 1.0 and above the M22 deployed 0.61 minus 0.20
    assert 0.30 < cal_a.thr_at_95sens < 0.95, f'thr={cal_a.thr_at_95sens} out of range'
    # write once and reload
    cal_a = fit_locked_calibration(write=True)
    cal_loaded = load_locked_calibration()
    assert abs(cal_loaded.T - cal_a.T) < 1e-9
    assert abs(cal_loaded.thr_at_95sens - cal_a.thr_at_95sens) < 1e-9
    # do NOT clean up the file; this is the locked artifact
```

- [ ] **Step 2: Run test, verify failure**

```bash
training/.venv/bin/python training/test_locked_protocol.py
```

Expected: `ImportError: cannot import name 'fit_locked_calibration'` — function doesn't exist yet.

- [ ] **Step 3: Implement `fit_locked_calibration` and `load_locked_calibration`**

Append to `training/locked_protocol.py`:

```python
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
    """Fit T + thr@95sens on the calibration slice of the cached LODO OOF.

    The calibration slice is the deterministic 20% stratified split from
    make_calibration_split(seed=P0_CALIBRATION_SEED). The remaining 80% is
    the evaluation surface that all subsequent measurement uses unchanged.
    """
    data_dir = Path(__file__).parent.parent / 'data'
    oof_path = data_dir / 'image_oof_logits.npz'
    if not oof_path.exists():
        raise FileNotFoundError(f'OOF cache missing: {oof_path}. Run M14 LODO first.')
    d = np.load(oof_path, allow_pickle=True)
    logit = d['image_logit'].astype('float64')
    label = d['label'].astype('int64')
    source = d['source']
    cal_idx, eval_idx = make_calibration_split(logit, label, source)
    T = _fit_temperature(logit[cal_idx], label[cal_idx])
    prob_cal = _sigmoid(logit[cal_idx] / T)
    thr = _threshold_for_sensitivity(label[cal_idx], prob_cal, target=0.95)

    cal = LockedCalibration(
        T=T,
        thr_at_95sens=thr,
        borderline_low=0.20,  # M26 widening, locked here
        s_inactive_escalate=0.7126,  # M19 sequelae escalator, locked here
        asymmetric_evidence_thr=0.88,  # M26 box-evidence high threshold
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
```

- [ ] **Step 4: Run test, verify passes**

```bash
training/.venv/bin/python training/test_locked_protocol.py
```

Expected:
```
PASS  test_calibration_split_is_deterministic
PASS  test_calibration_split_is_stratified_by_label_and_source
PASS  test_fit_locked_calibration_against_cached_oof
```

- [ ] **Step 5: Run the fit command itself to write the locked JSON**

```bash
training/.venv/bin/python training/locked_protocol.py
```

Expected output (numbers will vary, but these bounds must hold):
```
Wrote locked calibration to /Users/ahmadbilal/Downloads/hobby/TB detector/data/p0_locked_calibration.json
  T=<value 0.5-5.0>  thr@95sens=<value 0.3-0.95>
  n_cal=<value>  n_eval=<value>
```

- [ ] **Step 6: Commit**

```bash
git add training/locked_protocol.py training/test_locked_protocol.py data/p0_locked_calibration.json
git commit -m "feat(p0): fit and persist locked calibration on 20% stratified LODO OOF slice

T + thr@95sens fit ONCE on the deterministic 20% stratified calibration slice
(seed=7). The remaining 80% is the locked evaluation surface for all P1/P2/P3
measurement. No subsequent phase re-fits these values on the evaluation set.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P0.3: Test-time augmentation utility (CXR-safe only)

**Files:**
- Create: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/tta.py`
- Create: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/test_tta.py`

- [ ] **Step 1: Write the failing test**

```python
# training/test_tta.py
from __future__ import annotations
import numpy as np
import torch

from tta import tta_passes, K_PASSES, AUG_NAMES  # type: ignore[import-not-found]


def test_tta_produces_K_passes_per_image() -> None:
    """tta_passes returns K image variants for a given single CHW input."""
    img = torch.rand(3, 224, 224)
    variants = list(tta_passes(img))
    assert len(variants) == K_PASSES
    for v in variants:
        assert v.shape == img.shape


def test_tta_first_pass_is_identity() -> None:
    """First pass MUST be the unmodified image so single-pass eval is recoverable."""
    img = torch.rand(3, 100, 100)
    variants = list(tta_passes(img))
    assert torch.allclose(variants[0], img), 'first variant must be identity (no augmentation)'


def test_tta_includes_hflip_not_rotation() -> None:
    """CXR-safe: H-flip is OK, rotation is NOT (left-right anatomy distinction matters)."""
    assert 'hflip' in AUG_NAMES
    assert 'rotate' not in AUG_NAMES
    assert 'rotation' not in AUG_NAMES


def test_tta_brightness_change_is_bounded() -> None:
    """Brightness shift must be small (±10% per Wave 1.4 CXR-safe constraint)."""
    img = torch.full((3, 50, 50), 0.5)
    variants = list(tta_passes(img))
    for v in variants:
        # max +/- 0.10 change from baseline; tolerate 0.0001 numerical
        assert v.min() >= 0.5 - 0.10 - 1e-3
        assert v.max() <= 0.5 + 0.10 + 1e-3


def _run_all() -> None:
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {fn_name}')


if __name__ == '__main__':
    _run_all()
```

- [ ] **Step 2: Run test, verify it fails**

```bash
training/.venv/bin/python training/test_tta.py
```

Expected: `ImportError` on `from tta import ...`.

- [ ] **Step 3: Implement `tta.py`**

```python
# training/tta.py
"""Test-time augmentation, CXR-safe.

CXR-safe constraint (per Wave 1.4 literature survey, ranked #3 by ROI):
  - H-flip OK (handled by anatomic-symmetry assumption + sequelae head's left/right structure)
  - Brightness +/- 10% OK
  - Contrast +/- 10% OK
  - NO rotation (left-right diagnostic anatomy)
  - NO random crop (apices + costophrenic angles are diagnostic landmarks)
  - NO elastic deformation
  - NO color jitter (CXR is grayscale)

Inference cost: K_PASSES x baseline. With K=5 the wall-time becomes ~1.5x
of single-pass (the seg-crop preprocessing dominates, not the backbone).
"""
from __future__ import annotations
from typing import Iterator
import torch

K_PASSES = 5
AUG_NAMES = ('identity', 'hflip', 'brighten', 'darken', 'contrast_up')


def tta_passes(img: torch.Tensor) -> Iterator[torch.Tensor]:
    """Yield K_PASSES augmented variants of a single CHW image tensor.

    First yield is identity (so K=1 collapses to no-TTA cleanly).
    """
    assert img.dim() == 3, f'expected CHW, got {img.shape}'
    yield img  # 0: identity
    yield torch.flip(img, dims=(-1,))  # 1: H-flip
    yield (img + 0.10).clamp(0.0, 1.0)  # 2: brighten
    yield (img - 0.10).clamp(0.0, 1.0)  # 3: darken
    yield ((img - 0.5) * 1.10 + 0.5).clamp(0.0, 1.0)  # 4: contrast up


def tta_average_probs(probs_per_pass: list[float]) -> float:
    """Average a list of TTA probabilities into a single calibrated probability."""
    assert len(probs_per_pass) == K_PASSES, f'expected {K_PASSES} passes, got {len(probs_per_pass)}'
    return float(sum(probs_per_pass) / len(probs_per_pass))
```

- [ ] **Step 4: Run test, verify passes**

```bash
training/.venv/bin/python training/test_tta.py
```

Expected:
```
PASS  test_tta_brightness_change_is_bounded
PASS  test_tta_first_pass_is_identity
PASS  test_tta_includes_hflip_not_rotation
PASS  test_tta_produces_K_passes_per_image
```

- [ ] **Step 5: Commit**

```bash
git add training/tta.py training/test_tta.py
git commit -m "feat(p0): CXR-safe test-time augmentation (K=5, hflip + brightness +/- 0.1 + contrast)

CXR-safe per Wave 1.4: no rotation, no random crop, no elastic. First pass is
identity so single-pass eval is recoverable. Used by P0 baseline + P1
evaluation under the locked protocol.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P0.4: Wire TTA + locked calibration into `triage_core.py`

**Files:**
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/triage_core.py`
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/test_triage_engine.py`

- [ ] **Step 1: Write the failing test (append to existing test_triage_engine.py)**

Add the following to `training/test_triage_engine.py` before `_run_all`:

```python
def test_engine_supports_tta_mode() -> None:
    """When use_tta=True the verdict is computed by averaging K_PASSES probs."""
    from triage_core import TriageEngine
    from tta import K_PASSES

    eng = TriageEngine(use_tta=True)
    sample_path = Path(__file__).parent.parent / 'public' / 'samples' / 'tb-sample-1.jpg'
    if not sample_path.exists():
        print('SKIP test_engine_supports_tta_mode — sample missing')
        return
    img_bytes = sample_path.read_bytes()
    result = eng.run(img_bytes)
    # tta_passes attribute should be populated to K_PASSES probabilities
    assert hasattr(result, 'tta_passes')
    assert len(result.tta_passes) == K_PASSES
    # baseline tb_prob should be within +/- 0.10 of the TTA-averaged tb_prob
    # (TTA shouldn't massively swing the calibrated probability on a clear case)
    assert abs(result.tb_prob - sum(result.tta_passes) / K_PASSES) < 1e-6


def test_engine_uses_locked_calibration_when_available() -> None:
    """If data/p0_locked_calibration.json exists, the engine uses its T and thr."""
    import json
    from pathlib import Path
    from triage_core import TriageEngine

    locked_path = Path(__file__).parent.parent / 'data' / 'p0_locked_calibration.json'
    if not locked_path.exists():
        print('SKIP test_engine_uses_locked_calibration_when_available — locked JSON missing')
        return
    with locked_path.open() as f:
        locked = json.load(f)
    eng = TriageEngine(use_locked_protocol=True)
    assert abs(eng.T - locked['T']) < 1e-9
    assert abs(eng.thr - locked['thr_at_95sens']) < 1e-9
```

- [ ] **Step 2: Run test, verify it fails**

```bash
training/.venv/bin/python training/test_triage_engine.py
```

Expected: `TypeError: TriageEngine.__init__() got an unexpected keyword argument 'use_tta'` (or similar).

- [ ] **Step 3: Modify `triage_core.py` — add `use_tta` + `use_locked_protocol` kwargs**

Locate the `TriageEngine.__init__` method in `training/triage_core.py`. Add two new keyword-only parameters and the TTA branch. The existing constructor signature should be extended (do NOT remove existing params); add at end:

```python
# Add to TriageEngine.__init__ signature (keyword-only):
#     *,
#     use_tta: bool = False,
#     use_locked_protocol: bool = False,

# Inside __init__, after current T/thr loading:

self.use_tta = use_tta
if use_locked_protocol:
    from locked_protocol import load_locked_calibration
    locked = load_locked_calibration()
    self.T = locked.T
    self.thr = locked.thr_at_95sens
    self._calibration_source = 'p0_locked'
else:
    self._calibration_source = 'tb_threshold_t2_json'
```

And modify `TriageResult` dataclass (or equivalent in triage_core) to add an optional `tta_passes: list[float] | None = None` field.

In the `run()` method, add the TTA branch ABOVE the existing inference logic:

```python
def run(self, image_bytes: bytes) -> 'TriageResult':
    if self.use_tta:
        from tta import tta_passes, K_PASSES, tta_average_probs
        # Decode once
        from PIL import Image
        import io
        img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        passes_probs: list[float] = []
        # ... run the existing per-image pipeline K_PASSES times with the augmentations applied
        # ... (implementation detail: do harmonize+seg ONCE, apply augs at the tensor stage)
        # store passes_probs into result.tta_passes; set result.tb_prob = tta_average_probs(passes_probs)
        # FALL THROUGH to existing logic for non-TTA path
        # ...
    # ... existing single-pass code unchanged ...
```

The implementer should consult the existing `run()` method to identify the right interception point (after seg_crop, before backbone forward) so the augmentations apply at the normalized tensor stage, not raw PIL.

- [ ] **Step 4: Run test, verify passes**

```bash
training/.venv/bin/python training/test_triage_engine.py
training/.venv/bin/python training/test_locked_protocol.py
training/.venv/bin/python training/test_tta.py
```

Expected: all three test files PASS.

- [ ] **Step 5: Commit**

```bash
git add training/triage_core.py training/test_triage_engine.py
git commit -m "feat(p0): wire TTA + locked calibration into TriageEngine

Adds two opt-in keyword-only kwargs: use_tta (K=5 CXR-safe averaging) and
use_locked_protocol (T+thr from data/p0_locked_calibration.json). Default
behavior unchanged; defaults preserve the M22 deployed pipeline byte-for-byte.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P0.5: Implement `measure_p0_baseline.py` — runs the locked-protocol baseline once

**Files:**
- Create: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/measure_p0_baseline.py`

- [ ] **Step 1: Write the script**

```python
# training/measure_p0_baseline.py
"""P0 baseline measurement under locked protocol.

Runs the CURRENT model (rad-dino-base + TBHeadT2) with use_locked_protocol=True
and use_tta=True on:
  - The 23-image Cohen blind set (data/blind_test/)
  - The full NIH per-finding stress set (data/features_nih14.npz)
  - The 80% LODO evaluation slice (the complement of the calibration split)

Outputs data/p0_baseline.json with:
  - per-image probabilities and verdicts
  - aggregate sens / spec / abstain at locked threshold
  - per-source / per-finding subgroup breakdowns
  - paired-bootstrap 95% CIs for sens and spec

This is the FROZEN reference any P1 evaluation compares against.

Run: training/.venv/bin/python training/measure_p0_baseline.py
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from triage_core import TriageEngine  # type: ignore[import-not-found]
from locked_protocol import load_locked_calibration, make_calibration_split, _sigmoid


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
BLIND_TB = ROOT / 'data' / 'blind_test' / 'tb_positive'
BLIND_NO = ROOT / 'data' / 'blind_test' / 'no_tb'
OUT = DATA / 'p0_baseline.json'


def _bootstrap_ci(label: np.ndarray, pred: np.ndarray, n_bootstrap: int = 2000, seed: int = 1) -> tuple[float, float]:
    """Clopper-Pearson-style 95% CI via bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(label)
    rates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        ll = label[idx]
        pp = pred[idx]
        if ll.sum() == 0:
            continue
        rates.append((ll & pp).sum() / max(ll.sum(), 1))
    rates = np.array(rates)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def _verdict_from(p: float, thr: float, borderline_low: float = 0.20) -> str:
    if p >= thr:
        return 'TB'
    if p >= borderline_low:
        return 'ABSTAIN'
    return 'NO_TB'


def main() -> None:
    print('Loading locked calibration…')
    locked = load_locked_calibration()
    print(f'  T={locked.T:.4f}  thr@95sens={locked.thr_at_95sens:.4f}  seed={locked.seed}  n_eval={locked.n_eval}')

    print('Loading TriageEngine with locked protocol + TTA…')
    eng = TriageEngine(use_tta=True, use_locked_protocol=True)

    results: dict = {
        'meta': {
            'git_sha': locked.git_sha,
            'locked_T': locked.T,
            'locked_thr_at_95sens': locked.thr_at_95sens,
            'locked_borderline_low': locked.borderline_low,
            'measured_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        },
        'cohen_blind': {'images': []},
        'lodo_eval_slice': {},
        'nih_per_finding': {},
    }

    # === Cohen blind set ===
    print('\nMeasuring Cohen blind set under locked protocol + TTA…')
    cohen_labels: list[int] = []
    cohen_preds: list[str] = []
    cohen_probs: list[float] = []
    for label_name, dir_ in (('TB', BLIND_TB), ('NO_TB', BLIND_NO)):
        for img_path in sorted(dir_.iterdir()):
            if img_path.is_dir():
                continue
            res = eng.run(img_path.read_bytes())
            v = _verdict_from(res.tb_prob, locked.thr_at_95sens, locked.borderline_low)
            cohen_labels.append(1 if label_name == 'TB' else 0)
            cohen_preds.append(v)
            cohen_probs.append(res.tb_prob)
            results['cohen_blind']['images'].append({
                'filename': img_path.name,
                'true_label': label_name,
                'tb_prob': res.tb_prob,
                'verdict': v,
            })

    labels = np.array(cohen_labels)
    preds_tb = np.array([p == 'TB' for p in cohen_preds]).astype(int)
    preds_abstain = np.array([p == 'ABSTAIN' for p in cohen_preds]).astype(int)
    decided = ~preds_abstain.astype(bool)
    tp = int(((labels == 1) & preds_tb & decided).sum())
    fn = int(((labels == 1) & ~preds_tb & decided).sum())
    fp = int(((labels == 0) & preds_tb & decided).sum())
    tn = int(((labels == 0) & ~preds_tb & decided).sum())
    abs_pos = int(((labels == 1) & preds_abstain.astype(bool)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    sens_lo, sens_hi = _bootstrap_ci(labels, preds_tb, n_bootstrap=2000, seed=1)
    results['cohen_blind']['strict_sens'] = sens
    results['cohen_blind']['strict_sens_ci'] = [sens_lo, sens_hi]
    results['cohen_blind']['strict_spec'] = spec
    results['cohen_blind']['effective_sens_with_abstain'] = (tp + abs_pos) / max(tp + fn + abs_pos, 1)
    results['cohen_blind']['tp_fn_fp_tn_abstain'] = [tp, fn, fp, tn, int(preds_abstain.sum())]

    # === LODO 80% eval slice ===
    print('\nMeasuring LODO eval slice (80%) under locked thr…')
    oof = np.load(DATA / 'image_oof_logits.npz', allow_pickle=True)
    logit = oof['image_logit'].astype('float64')
    label = oof['label'].astype('int64')
    source = oof['source']
    _, eval_idx = make_calibration_split(logit, label, source)
    p_eval = _sigmoid(logit[eval_idx] / locked.T)
    y_eval = label[eval_idx]
    preds = (p_eval >= locked.thr_at_95sens).astype(int)
    abstain = ((p_eval >= locked.borderline_low) & (p_eval < locked.thr_at_95sens)).astype(int)
    tp = int(((y_eval == 1) & (preds == 1) & (abstain == 0)).sum())
    fn = int(((y_eval == 1) & (preds == 0) & (abstain == 0)).sum())
    fp = int(((y_eval == 0) & (preds == 1) & (abstain == 0)).sum())
    tn = int(((y_eval == 0) & (preds == 0) & (abstain == 0)).sum())
    results['lodo_eval_slice'] = {
        'n_eval': int(len(eval_idx)),
        'strict_sens': tp / max(tp + fn, 1),
        'strict_spec': tn / max(tn + fp, 1),
        'abstain_rate': float(abstain.mean()),
        'tp_fn_fp_tn': [tp, fn, fp, tn],
    }

    OUT.write_text(json.dumps(results, indent=2))
    print(f'\nWrote P0 baseline to {OUT}')
    print(f'  Cohen strict sens: {results["cohen_blind"]["strict_sens"]:.3f} [{sens_lo:.2f}-{sens_hi:.2f}]')
    print(f'  Cohen effective sens (with abstain): {results["cohen_blind"]["effective_sens_with_abstain"]:.3f}')
    print(f'  LODO eval-slice sens: {results["lodo_eval_slice"]["strict_sens"]:.3f}, spec: {results["lodo_eval_slice"]["strict_spec"]:.3f}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the baseline script**

```bash
training/.venv/bin/python training/measure_p0_baseline.py
```

Expected: prints the Cohen + LODO numbers and writes `data/p0_baseline.json`. Takes ~5-15 minutes (23 Cohen images × 5 TTA passes + LODO slice analysis).

- [ ] **Step 3: Sanity-check the output**

```bash
cat data/p0_baseline.json | python3 -m json.tool | head -40
```

Verify:
- `meta.locked_T` matches `data/p0_locked_calibration.json`'s T
- `cohen_blind.strict_sens` is reported with a CI
- `lodo_eval_slice.strict_sens` >= 0.75 (the baseline isn't broken by the locked thr)

- [ ] **Step 4: Commit**

```bash
git add training/measure_p0_baseline.py data/p0_baseline.json
git commit -m "feat(p0): measure baseline under locked protocol + TTA (Cohen + LODO slice)

The frozen reference any P1 evaluation compares against. Cohen blind strict
sens + CI, LODO 80%-eval-slice sens/spec at the locked thr. data/p0_baseline.json
is the audit trail.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P0.6: Append P0 entry to CASE_STUDY + EXPERIMENT_LOG

**Files:**
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/docs/CASE_STUDY.md`
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/docs/EXPERIMENT_LOG.md`

- [ ] **Step 1: Append CASE_STUDY P0 entry**

After the existing M27a section in `docs/CASE_STUDY.md`, before the "Maintenance log" footer list, insert a new section:

```markdown
## Milestone P0 — Locked-Protocol OOD Baseline (2026-05-25)

A GPT steelman review of the proposed M28 Option C (CheXFound + GLoRI head + multi-task TXRV + focal loss + 2×2 ablation) caught a load-bearing methodological flaw I had built in: per-config calibration re-fitting on the evaluation surface IS test-set leakage. I'd built the apparent rigor (paired bootstrap, factorial ablation) without locking the calibration step. The fix has to be structural, not procedural — so I made it structural.

`training/locked_protocol.py` defines `make_calibration_split(seed=7, cal_frac=0.20)` — a deterministic stratified split of the 13k LODO OOF cache into a 20% calibration slice and an 80% evaluation slice. T + thr@95sens are fit ONCE on the calibration slice and persisted to `data/p0_locked_calibration.json`. Every subsequent measurement (P1, P2, P3) calls `load_locked_calibration()` and applies the values unchanged. The eval-set 80% is never seen by calibration.

I added `training/tta.py` with CXR-safe augmentations: H-flip + brightness ±0.10 + contrast 1.10 — exactly the set Wave 1.4 ranked as CXR-safe per the 2024-2026 literature. No rotation (left-right anatomy matters for diagnostics — a flipped diagnosis is a misdiagnosis). No random crop (apical disease + costophrenic angles are diagnostic landmarks the model needs to see). K=5 passes averaged. The first pass is identity so K=1 collapses cleanly.

I wired both into `TriageEngine` as opt-in keyword-only kwargs `use_tta` and `use_locked_protocol`. Defaults preserve M22 deployed behavior byte-for-byte. The locked-protocol path becomes the new evaluation standard going forward.

**Measured baseline numbers** (the frozen reference all P1+ work measures against):

| Metric | Cohen blind (n=23) | LODO 80% eval slice |
|--------|---|---|
| Strict sensitivity | [to be filled by measure_p0_baseline.py] | [same] |
| Strict specificity | [same] | [same] |
| Effective sens (with ABSTAIN as rescue) | [same] | n/a |
| Abstain rate | [same] | [same] |

The 80% eval slice gives a paired-bootstrap-ready evaluation surface that is structurally insulated from calibration leakage. Any future architecture change that "improves" Cohen blind without improving the LODO eval slice is suspect — and any change that improves both ONLY via threshold movement is suspect because the threshold is locked at the P0 fit.

**What I learned.** The steelman taught me that the appearance of rigor (factorial table, paired bootstrap) is not rigor. Rigor is locked thresholds + named eval slices + deterministic seeds. Three lines of test code (`test_calibration_split_is_deterministic`, `test_engine_uses_locked_calibration_when_available`, `test_engine_supports_tta_mode`) now make this structural property check-able by any future implementer. The hidden coupling between "let's just refit T per config" and "publishable artifact" is the most subtle methodology error I've made in this project and the most consequential — caught only because I asked GPT to argue against my own plan.

```

- [ ] **Step 2: Append the EXPERIMENT_LOG §C row**

After the M27a row in `docs/EXPERIMENT_LOG.md` §C table, append:

```markdown
| 05-25 | **P0 LOCKED-PROTOCOL BASELINE** — `training/locked_protocol.py` + `training/tta.py` + `training/measure_p0_baseline.py`. Stratified deterministic 20% calibration slice (seed=7) on the 13k LODO OOF; T + thr@95sens locked into `data/p0_locked_calibration.json`. CXR-safe K=5 TTA (H-flip + brightness +/- 0.10 + contrast 1.10; no rotation/crop). `TriageEngine` gains opt-in `use_tta` + `use_locked_protocol`. Baseline measured ONCE on Cohen blind + LODO 80% eval slice. | Locked T in [0.5, 5.0]; locked thr@95sens in [0.3, 0.95]. P0 baseline Cohen sens >= M26 effective sens 0.818 +/- 0.05 (TTA should give a small bump). LODO 80% eval-slice sens >= 0.75. | [to be filled after measure_p0_baseline.py runs] | No drift (P0 is methodology, not architecture) | ACCEPT as new evaluation standard. All P1/P2/P3 measurement runs with use_tta=True + use_locked_protocol=True. Per-config calibration tuning on the evaluation surface is now structurally prevented. |
```

- [ ] **Step 3: Commit**

```bash
git add docs/CASE_STUDY.md docs/EXPERIMENT_LOG.md
git commit -m "docs(p0): CASE_STUDY P0 entry + EXPERIMENT_LOG row

Records the locked-protocol structural defense against per-config calibration
leakage (the steelman objection that triggered the M28 Option C rejection).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase P0.5 — Multi-Source Data Acquisition (target: 1 day active + waits)

**Goal of P0.5:** Add two open multi-geography TB CXR sources to the build index without retraining yet. The new sources are integrated into the same preprocessing + dedup pipeline as the current 4 sources. Training on them is P1's job.

### Task P0.5.1: Download Mendeley Pakistani TB CXR dataset

**Files:**
- Create: `/Users/ahmadbilal/Downloads/hobby/TB detector/scripts/fetch_mendeley_pk.py`

- [ ] **Step 1: Write the fetch script**

```python
# scripts/fetch_mendeley_pk.py
"""Download the Kiran/Jabeen Pakistani TB CXR dataset (CC-BY-4.0, May 2024).

DOI: 10.17632/8j2g3csprk.2
Contents: 2,494 TB+ + 514 normal CXRs from a Pakistani hospital.
Output: data/raw/mendeley_pk/{tb,normal}/*.{jpg,png}
"""
from __future__ import annotations
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'raw' / 'mendeley_pk'

# Mendeley exposes per-file download URLs at:
#   https://data.mendeley.com/public-api/datasets/8j2g3csprk/files
# The v2 dataset bundle URL is documented at the DOI landing page; this script
# downloads the published ZIP and unpacks it.
MENDELEY_ZIP_URL = 'https://data.mendeley.com/public-files/datasets/8j2g3csprk/files/version2.zip'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'Downloading Mendeley PK TB dataset to {OUT} ...')
    print(f'  URL: {MENDELEY_ZIP_URL}')
    try:
        req = urllib.request.Request(MENDELEY_ZIP_URL, headers={'User-Agent': 'tb-triage-research/1.0'})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
    except Exception as e:
        print(f'ERROR: download failed: {e}', file=sys.stderr)
        print('  -> Falling back to manual download instructions:')
        print('     1. Open https://data.mendeley.com/datasets/8j2g3csprk/2')
        print('     2. Click "Download All"')
        print(f'     3. Unzip to {OUT}')
        sys.exit(1)

    print(f'  downloaded {len(data)/1e6:.1f} MB; sha256={hashlib.sha256(data).hexdigest()[:16]}…')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(OUT)
    print(f'  extracted to {OUT}')

    # Count what landed
    n_tb = sum(1 for _ in (OUT.rglob('*.jpg')) if 'tb' in str(_).lower())
    n_norm = sum(1 for _ in (OUT.rglob('*.jpg')) if 'normal' in str(_).lower())
    print(f'  approx {n_tb} TB images, {n_norm} normal images')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the fetch**

```bash
training/.venv/bin/python scripts/fetch_mendeley_pk.py
```

Expected: downloads + extracts. If the Mendeley URL has changed (this happens), the script falls through to the manual instructions; follow them.

- [ ] **Step 3: Verify file counts**

```bash
find data/raw/mendeley_pk -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l
```

Expected: approximately 3,000 files (2,494 TB + 514 normal).

- [ ] **Step 4: Commit the script (NOT the data; `data/` is gitignored)**

```bash
git add scripts/fetch_mendeley_pk.py
git commit -m "feat(p0.5): fetch script for Mendeley Pakistani TB CXR (Kiran/Jabeen 2024, CC-BY-4.0)

2,494 TB+ + 514 normal CXRs from a Pakistani hospital. Single-site,
preprocessed-and-resized. Use as multi-source augmentation, not standalone
validation. DOI 10.17632/8j2g3csprk.2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P0.5.2: Extend `build_index.py` to register `mendeley_pk` as a source

**Files:**
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/build_index.py`

- [ ] **Step 1: Locate the existing source builders**

```bash
grep -n "def build_" training/build_index.py
```

Expected output should list `build_montgomery`, `build_shenzhen`, `build_qatar`, `build_tbx11k`, `build_nih14` (the M27a addition).

- [ ] **Step 2: Add `build_mendeley_pk()` function**

After the last existing builder function in `training/build_index.py`, add:

```python
def build_mendeley_pk() -> list[dict]:
    """Mendeley Kiran/Jabeen Pakistani TB CXR dataset (May 2024, CC-BY-4.0, v2).

    Filename heuristic: any file under a directory whose name contains 'tb'
    (case-insensitive) is labeled TB+; any under a directory with 'normal' is
    TB-. The dataset's exact directory layout depends on what Mendeley shipped
    in version 2; this function logs warnings if no rows are produced so the
    user knows to inspect data/raw/mendeley_pk/.
    """
    root = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'mendeley_pk'
    if not root.exists():
        return []
    rows: list[dict] = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        relpath = p.relative_to(root).as_posix().lower()
        if 'tb' in relpath:
            y = 1
        elif 'normal' in relpath:
            y = 0
        else:
            continue
        rows.append({
            'filename': str(p),
            'source': 'mendeley_pk',
            'active_tb': y,
            'no_tb': 1 - y,
            'latent_tb': 0,
            'patient_id': f'mendeley_pk_{p.stem}',
            'bbox_xyxy': '',  # no bbox available
        })
    print(f'  mendeley_pk: {len(rows)} rows ({sum(r["active_tb"] for r in rows)} TB+, {sum(r["no_tb"] for r in rows)} TB-)')
    return rows
```

Then locate the function in build_index.py that aggregates source builders (search for `build_montgomery() + build_shenzhen() + ...`) and add `+ build_mendeley_pk()` to the aggregate.

- [ ] **Step 3: Run build_index to verify the new source is picked up**

```bash
training/.venv/bin/python training/build_index.py
```

Expected: stdout includes a line like `mendeley_pk: 3008 rows (2494 TB+, 514 TB-)` (numbers approximate).

- [ ] **Step 4: Commit**

```bash
git add training/build_index.py
git commit -m "feat(p0.5): register mendeley_pk source in build_index

2,494 TB+ + 514 normal Pakistani CXRs added to the training index. Source id
'mendeley_pk'. Will be deduped + extracted as part of the multi-source P1
training corpus.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P0.5.3: Document PadChest manual access + write a placeholder builder

**Files:**
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/build_index.py`
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/docs/DATA_SOURCES.md`

PadChest requires manual BIMCV form-request (days, not immediate). We register the builder + filter logic now so when the data arrives, the implementer just drops it into `data/raw/padchest_tb/` and re-runs build_index.

- [ ] **Step 1: Add `build_padchest_tb()` builder**

Append to `training/build_index.py`:

```python
def build_padchest_tb() -> list[dict]:
    """PadChest TB-7-label union subset (BIMCV, Spain, ~152-176+ TB+ studies).

    The canonical TB-positive filter is the union of seven PadChest labels:
      tuberculosis, sequelae tuberculosis, cavitation, calcified adenopathy,
      granuloma, calcified granuloma, apical pleural thickening.
    See PMC11843218 for the published harvest protocol.

    Expects data/raw/padchest_tb/{tb,normal}/*.png after manual BIMCV download
    + filtering by the 7-label union. If the directory is empty (PadChest
    DUA not yet through), returns an empty list silently.

    Single-site Spain — atypical-TB richness (sequelae/granuloma/calcified
    adenopathy) directly addresses M24 weak spot.
    """
    root = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'padchest_tb'
    if not root.exists():
        return []
    rows: list[dict] = []
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        relpath = p.relative_to(root).as_posix().lower()
        if relpath.startswith('tb/'):
            y = 1
        elif relpath.startswith('normal/'):
            y = 0
        else:
            continue
        rows.append({
            'filename': str(p),
            'source': 'padchest_tb',
            'active_tb': y,
            'no_tb': 1 - y,
            'latent_tb': 0,
            'patient_id': f'padchest_{p.stem}',
            'bbox_xyxy': '',
        })
    if rows:
        print(f'  padchest_tb: {len(rows)} rows ({sum(r["active_tb"] for r in rows)} TB+, {sum(r["no_tb"] for r in rows)} TB-)')
    return rows
```

Add `+ build_padchest_tb()` to the aggregator alongside `+ build_mendeley_pk()`.

- [ ] **Step 2: Update DATA_SOURCES.md**

Add a section to `docs/DATA_SOURCES.md` under "Do this week (open / quick)":

```markdown
### 0. Mendeley Pakistani TB CXR (Kiran/Jabeen 2024, CC-BY-4.0) — INTEGRATED (P0.5)
2,494 TB+ + 514 normal from a Pakistani hospital. Single-site, preprocessed.
- DOI: https://doi.org/10.17632/8j2g3csprk.2
- Direct download: see `scripts/fetch_mendeley_pk.py`
- Source id in build_index: `mendeley_pk`
- Use case: multi-source MixStyle augmentation; deployment geography (Pakistan = the CAD4TB cohort context).

### PadChest TB-7-label union (BIMCV, Spain, ~150+ TB+) — REQUESTED via BIMCV form, BUILDER REGISTERED (P0.5)
TB-positive filter is the union of: tuberculosis + sequelae tuberculosis + cavitation + calcified adenopathy + granuloma + calcified granuloma + apical pleural thickening (see PMC11843218 for the published harvest protocol).
- Access: BIMCV form request at https://bimcv.cipf.es/bimcv-projects/padchest/
- Source id in build_index: `padchest_tb`
- When the credentialed download lands: place PNGs under `data/raw/padchest_tb/{tb,normal}/` and re-run `build_index.py`
- Use case: atypical-TB richness (sequelae/granuloma/calcified adenopathy) directly addresses M24 weak spot.
- Status: DUA form submitted on [DATE — fill in when submitted]. Expected delivery: days.
```

- [ ] **Step 3: Submit the BIMCV form (manual step — out-of-band)**

```
USER ACTION (cannot be automated): visit https://bimcv.cipf.es/bimcv-projects/padchest/
fill the form and wait for credentials. When credentials arrive, save them to
~/.padchest_creds with the format documented in the BIMCV email.
```

- [ ] **Step 4: Commit (builder + docs only; no data yet)**

```bash
git add training/build_index.py docs/DATA_SOURCES.md
git commit -m "feat(p0.5): register padchest_tb source + document BIMCV manual access

Builder is in place; populates from data/raw/padchest_tb/{tb,normal}/*.png
when the manual BIMCV DUA download lands. TB-positive filter is the published
7-label union (PMC11843218).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P0.5.4: Re-run dedup + extract on the expanded index

**Files:**
- No source code changes — pure pipeline run.

- [ ] **Step 1: Re-run dedup on the expanded index**

```bash
training/.venv/bin/python training/dedup.py
```

Expected: stdout shows `mendeley_pk` rows joining the dedup graph. Any cross-source TB/normal label conflicts are flagged. Total dedup count should rise by ~3,000 (Mendeley PK) minus any cross-source matches.

- [ ] **Step 2: Inspect any cross-source conflicts**

```bash
grep -E "WARN|conflict" data/dedup_audit.log | head -30
```

Expected: 0-5 cross-source matches (most likely from Mendeley PK images that happen to be re-publications of older Pakistani studies).

- [ ] **Step 3: Re-run feature extraction including the new source**

```bash
HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
training/.venv/bin/python training/extract_features.py --sources mendeley_pk --output data/features_mendeley_pk.npz
```

Expected: ~30-50 min M4 inference. Produces `data/features_mendeley_pk.npz` with `cls`, `patches`, `txrv`, `y`, `source`, `patient_id`.

- [ ] **Step 4: Verify the new feature file shapes**

```bash
training/.venv/bin/python -c "
import numpy as np
d = np.load('data/features_mendeley_pk.npz', allow_pickle=True)
print('keys:', list(d.keys()))
print('cls:', d['cls'].shape, 'patches:', d['patches'].shape, 'txrv:', d['txrv'].shape)
print('y dist:', dict(zip(*np.unique(d['y'], return_counts=True))))
print('source dist:', dict(zip(*np.unique(d['source'], return_counts=True))))
"
```

Expected: ~3000 rows; `y` has two classes; source is uniform `mendeley_pk`.

- [ ] **Step 5: Commit the dedup audit log (the features are gitignored)**

```bash
git add data/dedup_audit.log
git commit -m "chore(p0.5): re-dedup with mendeley_pk added; cross-source conflict audit logged

Pure pipeline run; no code change. features_mendeley_pk.npz produced
separately (gitignored).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P0.5.5: Append P0.5 entry to CASE_STUDY + EXPERIMENT_LOG

**Files:**
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/docs/CASE_STUDY.md`
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/docs/EXPERIMENT_LOG.md`

- [ ] **Step 1: Append CASE_STUDY P0.5 section**

After the P0 section in CASE_STUDY.md, before the maintenance log, insert:

```markdown
## Milestone P0.5 — Multi-Source Data Acquisition (2026-05-25)

I expanded the training corpus from 4 sources to 6 by adding:

1. **Mendeley Kiran/Jabeen Pakistani TB CXR** (2,494 TB+ + 514 normal, CC-BY-4.0, May 2024) — single-site Pakistani hospital, the exact deployment geography the project is framed toward (CAD4TB cohort context). Single-site is the caveat; preprocessing was already done by the dataset authors so original-resolution provenance is lost. Use as multi-source diversity, not standalone validation. Source id `mendeley_pk`.

2. **PadChest TB-7-label union** (BIMCV, Spain, ~150+ TB+ estimated) — DUA-gated through BIMCV form, builder + docs registered, awaiting credentials. The 7-label union (tuberculosis + sequelae tuberculosis + cavitation + calcified adenopathy + granuloma + calcified granuloma + apical pleural thickening) per PMC11843218 directly addresses the M24 atypical-TB failure mode — sequelae, granuloma, and calcified adenopathy are exactly the radiographic categories my model under-recognized. Source id `padchest_tb`.

**Why these two specifically (per Wave 1.5 dataset survey).** Wave 1.5 surfaced that:
- Most published TB CXR work uses only Montgomery + Shenzhen + Qatar + TBX11K (East Asia + US/IN). The Mendeley PK set is the first openly-licensed Pakistani TB CXR corpus at meaningful scale.
- BRAX, CheXpert, MIMIC-CXR have NO TB labels in their official taxonomies — adding them as positive-source candidates is wrong.
- TB Portals (NIAID) is still the #1 ask but DUA-gated for institutional ARs (weeks-to-months).
- VinDr-CXR has ~479 TB+ but PhysioNet credentialed (CITI training, days) — queued for P0.5+ but not blocking P1.

The dedup pipeline ran on the expanded index. Cross-source pHash matches are logged in `data/dedup_audit.log` — these are the only places a Mendeley PK image could overlap with our existing 4 sources, and the audit catches it before P1 training so we don't accidentally evaluate on a training image.

**What I learned.** Open multi-geography TB CXR data exists and is closer at hand than DATA_SOURCES.md previously suggested. The right Wave-1 question wasn't "is there gated bacteriological data" (TB Portals, well-known) but "is there OPEN multi-geography data I missed" — and the answer was yes, a 3,000-image Pakistani set released in 2024. The literature gap pattern is consistent: dataset announcements outpace data-source compilations by ~12 months.
```

- [ ] **Step 2: Append EXPERIMENT_LOG §C row for P0.5**

```markdown
| 05-25 | **P0.5 MULTI-SOURCE DATA ACQUISITION** — Added `mendeley_pk` (Mendeley Kiran/Jabeen, 2,494 TB+ + 514 normal, CC-BY-4.0, Pakistani single-site) to the build index. Registered `padchest_tb` builder (BIMCV DUA pending; populates from `data/raw/padchest_tb/{tb,normal}/` when credentials land — TB-7-label union per PMC11843218). Re-ran dedup; logged cross-source matches to `data/dedup_audit.log`. Extracted features for `mendeley_pk` into `data/features_mendeley_pk.npz`. | mendeley_pk rows >= 2500; padchest_tb returns 0 rows (DUA pending); cross-source dedup matches < 10. | [to be filled after extract_features.py runs] | No drift (data integration, not architecture) | ACCEPT. Training corpus now 6 sources for P1 (Mont, Shen, Qatar, TBX11K, mendeley_pk, padchest_tb-when-arrived). |
```

- [ ] **Step 3: Commit**

```bash
git add docs/CASE_STUDY.md docs/EXPERIMENT_LOG.md
git commit -m "docs(p0.5): CASE_STUDY P0.5 entry + EXPERIMENT_LOG row

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase P1 — Surgical M24 Fix (`SoftAttnPool` replaces `ZonalSoftOR`) (target: 3-5 days)

**Goal of P1:** ONE lever change: replace the `ZonalSoftOR` head's hard-zone-prior pooling with a learned non-zonal soft-attention pooling. Train under MixStyle augmentation on the 6-source corpus. Evaluate under the P0 locked protocol. Steelman discipline: backbone unchanged (still rad-dino-base), no GLoRI multi-query, no focal loss, no multi-task TXRV head — those bundles are deferred to future plans so this milestone's gain is attributable to a single named lever.

### Task P1.1: Define `SoftAttnPool` head module

**Files:**
- Create: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/heads/__init__.py`
- Create: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/heads/soft_attn_pool.py`
- Create: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/test_soft_attn_pool.py`

- [ ] **Step 1: Write the failing test**

```python
# training/test_soft_attn_pool.py
from __future__ import annotations
import torch

from heads.soft_attn_pool import SoftAttnPool  # type: ignore[import-not-found]


def test_soft_attn_pool_forward_shape() -> None:
    """SoftAttnPool(d_in=768) takes (B, T, 768) patches and emits (B, 768)."""
    mod = SoftAttnPool(d_in=768, d_hidden=128)
    x = torch.randn(4, 64, 768)
    out, attn = mod(x)
    assert out.shape == (4, 768), out.shape
    assert attn.shape == (4, 64), attn.shape


def test_soft_attn_pool_attention_sums_to_one() -> None:
    """Attention weights must form a probability distribution over patch tokens."""
    torch.manual_seed(0)
    mod = SoftAttnPool(d_in=768, d_hidden=128)
    x = torch.randn(2, 64, 768)
    _, attn = mod(x)
    sums = attn.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), sums


def test_soft_attn_pool_no_hard_zone_priors() -> None:
    """SoftAttnPool has no fixed-zone partition — only a learnable attention layer."""
    mod = SoftAttnPool(d_in=768, d_hidden=128)
    # All parameters must be learnable; no buffer of zone indices like ZonalSoftOR has
    n_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    n_buffers = sum(b.numel() for b in mod.buffers())
    assert n_params > 0
    assert n_buffers == 0, f'SoftAttnPool must have no zone-index buffers, got {n_buffers}'


def test_soft_attn_pool_gradient_flows() -> None:
    """Backward pass through SoftAttnPool produces non-zero gradients on params."""
    mod = SoftAttnPool(d_in=768, d_hidden=128)
    x = torch.randn(2, 64, 768)
    out, _ = mod(x)
    out.sum().backward()
    for name, p in mod.named_parameters():
        assert p.grad is not None, f'no grad on {name}'
        assert p.grad.abs().sum().item() > 0, f'zero grad on {name}'


def _run_all() -> None:
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {fn_name}')


if __name__ == '__main__':
    _run_all()
```

- [ ] **Step 2: Run test, verify it fails**

```bash
training/.venv/bin/python training/test_soft_attn_pool.py
```

Expected: ImportError on `from heads.soft_attn_pool import SoftAttnPool`.

- [ ] **Step 3: Implement `SoftAttnPool`**

```python
# training/heads/__init__.py
from .soft_attn_pool import SoftAttnPool

__all__ = ['SoftAttnPool']
```

```python
# training/heads/soft_attn_pool.py
"""SoftAttnPool — non-zonal learned attention pooling.

Drop-in replacement for ZonalSoftOR. Designed to fix the M24 atypical-TB
failure mode by removing the hard zone-prior partition that caused mid-lung
consolidative TB (M24 cases #6, #7, #9, #10) to be attenuated when the
backbone clearly encoded patch-level evidence.

Architecture:
  - Input: (B, T, d_in) patch tokens, T = patch_grid^2 (typically 64 for 8x8)
  - Single learnable attention layer:
      a_t = softmax( w^T tanh( W x_t + b ) )    over t = 1..T
      out = sum_t a_t * x_t                     -> (B, d_in)
  - Returns (out, attn_weights) so downstream code can visualize attention
    similar to the M24 BoxEvidence heatmap (the attention map IS interpretable).

No fixed zones, no SoftOR, no LogSumExp pooling. The same fusion+box+distill
heads of the original TBHeadT2 consume the pooled (B, d_in) vector unchanged.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttnPool(nn.Module):
    """Learned non-zonal soft-attention pooling over patch tokens.

    Total params: d_in * d_hidden + d_hidden (W,b) + d_hidden (w) = O(d_in * d_hidden).
    With d_in=768 and d_hidden=128 that's ~98k params — same order as the existing
    ZonalSoftOR (which is bigger because of zone-conditional gating).
    """

    def __init__(self, d_in: int, d_hidden: int = 128) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(d_in, d_hidden)
        self.attn_weight = nn.Parameter(torch.empty(d_hidden))
        nn.init.normal_(self.attn_weight, mean=0.0, std=0.02)

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """patches: (B, T, d_in). Returns (pooled (B, d_in), attn (B, T))."""
        # (B, T, d_hidden) -> (B, T) via attn_weight dot product
        h = torch.tanh(self.attn_proj(patches))
        scores = (h * self.attn_weight).sum(dim=-1)  # (B, T)
        attn = F.softmax(scores, dim=-1)             # (B, T)
        pooled = (attn.unsqueeze(-1) * patches).sum(dim=1)  # (B, d_in)
        return pooled, attn
```

- [ ] **Step 4: Run test, verify passes**

```bash
training/.venv/bin/python training/test_soft_attn_pool.py
```

Expected:
```
PASS  test_soft_attn_pool_attention_sums_to_one
PASS  test_soft_attn_pool_forward_shape
PASS  test_soft_attn_pool_gradient_flows
PASS  test_soft_attn_pool_no_hard_zone_priors
```

- [ ] **Step 5: Commit**

```bash
git add training/heads/ training/test_soft_attn_pool.py
git commit -m "feat(p1): SoftAttnPool — non-zonal attention pooling head module

Surgical replacement for ZonalSoftOR's hard-zone-prior pooling. Single
learnable attention over patch tokens, no fixed zones. Designed to fix the
M24 atypical-TB failure mode (mid-lung consolidative TB attenuated by
zone-prior averaging) per Wave 1.3 architectural diagnosis.

Tests cover: (1) forward shape, (2) attention sums to 1, (3) no hard-zone
buffers, (4) gradient flow. Backbone unchanged (rad-dino-base); ONE lever.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P1.2: Implement MixStyle augmentation

**Files:**
- Create: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/mixstyle.py`
- Create: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/test_mixstyle.py`

- [ ] **Step 1: Write the failing test**

```python
# training/test_mixstyle.py
from __future__ import annotations
import torch
from mixstyle import MixStyle  # type: ignore[import-not-found]


def test_mixstyle_identity_when_disabled() -> None:
    """MixStyle(p=0) must be identity — for inference."""
    mod = MixStyle(p=0.0)
    mod.train()
    x = torch.randn(4, 64, 768)
    out = mod(x)
    assert torch.allclose(out, x)


def test_mixstyle_identity_in_eval_mode() -> None:
    """MixStyle never fires during eval."""
    mod = MixStyle(p=1.0)
    mod.eval()
    x = torch.randn(4, 64, 768)
    out = mod(x)
    assert torch.allclose(out, x)


def test_mixstyle_changes_stats_when_active() -> None:
    """When training and p=1.0, MixStyle mixes channel mean/std between batch items."""
    mod = MixStyle(p=1.0, alpha=0.3)
    mod.train()
    torch.manual_seed(0)
    x = torch.randn(8, 64, 768) * 2.0 + 5.0  # nontrivial mean/std
    out = mod(x)
    # Output and input cannot be byte-identical
    assert not torch.allclose(out, x)
    # But shape must match
    assert out.shape == x.shape


def test_mixstyle_preserves_dtype_and_device() -> None:
    mod = MixStyle(p=1.0)
    mod.train()
    x = torch.randn(4, 64, 768).float()
    out = mod(x)
    assert out.dtype == x.dtype
    assert out.device == x.device


def _run_all() -> None:
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {fn_name}')


if __name__ == '__main__':
    _run_all()
```

- [ ] **Step 2: Run test, verify it fails**

```bash
training/.venv/bin/python training/test_mixstyle.py
```

Expected: ImportError.

- [ ] **Step 3: Implement MixStyle**

```python
# training/mixstyle.py
"""MixStyle feature-space augmentation for domain generalization.

Reference: Zhou et al., ICLR 2021, "Domain Generalization with MixStyle."
Adapted from the official PyTorch implementation for feature-space tokens
(rather than CNN spatial features). Per Wave 1.4 ranking #2 (TB-targeted
Hwang 2025), MixStyle on the feature side is orthogonal to architectural
changes and to LoRA-style adapter training.

How it works:
  - Each forward in training mode, with probability p, mixes channel-wise
    mean/std between random pairs of batch items, using a Beta(alpha, alpha)
    interpolation weight. This introduces feature-style variability that
    simulates cross-source shifts.
  - Identity in eval mode.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """Feature-side MixStyle augmentation.

    Applied to (B, T, d) patch-token features. Mixes channel-wise stats
    along the (B, T) axes for each channel d.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        assert 0.0 <= p <= 1.0
        assert alpha > 0
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self._beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) or (B, d). Returns same shape with mixed stats in training."""
        if not self.training or self.p == 0.0:
            return x
        if torch.rand(1).item() > self.p:
            return x

        if x.dim() == 2:
            x_3d = x.unsqueeze(1)  # (B, 1, d)
            squeeze = True
        else:
            x_3d = x
            squeeze = False

        B, T, d = x_3d.shape
        # channel-wise mean/std over T
        mu = x_3d.mean(dim=1, keepdim=True)            # (B, 1, d)
        var = x_3d.var(dim=1, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_norm = (x_3d - mu) / sig

        # random permutation of batch
        perm = torch.randperm(B, device=x.device)
        mu_perm = mu[perm]
        sig_perm = sig[perm]

        # interpolation weight per item
        lam = self._beta.sample((B,)).to(x.device).view(B, 1, 1)
        mu_mix = lam * mu + (1 - lam) * mu_perm
        sig_mix = lam * sig + (1 - lam) * sig_perm

        out = x_norm * sig_mix + mu_mix
        if squeeze:
            out = out.squeeze(1)
        return out
```

- [ ] **Step 4: Run test, verify passes**

```bash
training/.venv/bin/python training/test_mixstyle.py
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/mixstyle.py training/test_mixstyle.py
git commit -m "feat(p1): MixStyle feature-space augmentation for domain generalization

Zhou et al. ICLR 2021. Mixes channel-wise mean/std between random batch
pairs in training mode. Identity in eval. p=0.5 by default; configurable via
train_tb.py --mixstyle-p flag (P1 will use 0.5 on the 6-source corpus per
Wave 1.4 #2 ranking).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P1.3: Wire `SoftAttnPool` + MixStyle into the training script

**Files:**
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/training/train_tb.py`

(Note: this task involves a moderate refactor of an existing complex file. The implementer should READ `training/train_tb.py` from end to end first — specifically the `TBHeadT2` class definition and `train_head_t2()` function — before editing.)

- [ ] **Step 1: Add CLI flag plumbing in `train_tb.py`**

Locate the argument parser in `train_tb.py` (search for `argparse` or the function that builds the CLI). Add three new arguments:

```python
parser.add_argument('--head', type=str, default='zonal-softor', choices=('zonal-softor', 'soft-attn-pool'),
                    help='Head architecture for patch pooling. Default keeps existing M22 behavior.')
parser.add_argument('--mixstyle-p', type=float, default=0.0,
                    help='Probability of MixStyle feature-space augmentation per forward. 0.0 = no MixStyle (backward-compatible).')
parser.add_argument('--sources', type=str, default='',
                    help='Comma-separated source ids to include. Empty = legacy hardcoded set (mont,shen,qatar,tbx11k).')
```

- [ ] **Step 2: Add the `SoftAttnPool` branch inside `TBHeadT2`**

Locate `class TBHeadT2(nn.Module)` in `train_tb.py`. The current pooling component (likely named `self.zonal_softor` or similar) needs an alternate-pool branch. Modify the `__init__` to accept a `head_kind` arg and the `forward` to dispatch:

```python
# Inside TBHeadT2.__init__, after the existing patch-feature ingestion:
from heads.soft_attn_pool import SoftAttnPool
from mixstyle import MixStyle

if head_kind == 'soft-attn-pool':
    self.soft_attn_pool = SoftAttnPool(d_in=d_patch, d_hidden=128)
    self.zonal_softor = None
else:
    # keep existing ZonalSoftOR instantiation here
    self.soft_attn_pool = None
    # ... existing self.zonal_softor = ZonalSoftOR(...) ...

self.mixstyle = MixStyle(p=mixstyle_p) if mixstyle_p > 0.0 else nn.Identity()

# Inside TBHeadT2.forward, BEFORE the patch pooling step:
patches = self.mixstyle(patches)  # identity when p=0 or eval

# Replace the existing zonal_softor() call with a dispatch:
if self.soft_attn_pool is not None:
    pooled, attn_weights = self.soft_attn_pool(patches)
    self._last_attn = attn_weights  # store for diagnostic surfacing
else:
    pooled, zonal_scores = self.zonal_softor(patches, zones_mask)
    self._last_zonal_scores = zonal_scores
```

The implementer should adapt the dispatch to match the existing forward signature exactly.

- [ ] **Step 3: Pass the new flags through `train_head_t2()`**

In `train_head_t2()`, accept `head_kind` and `mixstyle_p` kwargs and forward them into the `TBHeadT2(...)` constructor.

- [ ] **Step 4: Pass the `--sources` flag into the data-loading routine**

Locate the data-loading routine in `train_tb.py` (likely loads `data/features.npz`). Add a branch:

```python
if args.sources:
    requested = set(args.sources.split(','))
    feature_files = []
    for src in requested:
        f = ROOT / 'data' / f'features_{src}.npz'
        if f.exists():
            feature_files.append(f)
        else:
            print(f'WARN: source {src} requested but {f} not found')
    # concatenate npz files into a single in-memory dataset
    # (cls, patches, txrv, y, source, patient_id arrays each)
else:
    # legacy single-file load from data/features.npz
    pass
```

- [ ] **Step 5: Add a smoke test**

```python
# training/test_softattn_lodo_smoke.py
"""Smoke test: 1-fold LODO training with SoftAttnPool + MixStyle on a tiny subset.

Goal: verify the new code path doesn't crash, produces non-NaN losses, and
converges to a sensible AUROC on a within-source train/val split.
This is NOT the P1 full-LODO measurement; it's a 5-minute sanity check.
"""
from __future__ import annotations
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_softattn_smoke_runs_without_error() -> None:
    """Run train_tb.py with --head soft-attn-pool --mixstyle-p 0.5 on a 1-fold subset."""
    result = subprocess.run(
        [
            str(ROOT / 'training' / '.venv' / 'bin' / 'python'),
            str(ROOT / 'training' / 'train_tb.py'),
            '--head', 'soft-attn-pool',
            '--mixstyle-p', '0.5',
            '--smoke',  # smoke flag: trains 1 fold, 5 epochs, 256 batch limit
        ],
        capture_output=True,
        text=True,
        timeout=600,
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
```

Also add a `--smoke` flag to `train_tb.py` that limits to 1 fold, 5 epochs, batch budget 256. Keep the smoke run under 10 minutes on M4.

- [ ] **Step 6: Run the smoke test**

```bash
training/.venv/bin/python training/test_softattn_lodo_smoke.py
```

Expected: passes within ~5-10 minutes; non-NaN AUROC printed.

- [ ] **Step 7: Commit**

```bash
git add training/train_tb.py training/test_softattn_lodo_smoke.py
git commit -m "feat(p1): wire SoftAttnPool + MixStyle into train_tb.py (--head, --mixstyle-p, --sources, --smoke)

Adds three CLI flags + a smoke test. Default behavior unchanged
(--head zonal-softor --mixstyle-p 0.0 --sources ''). P1 measurement run will
use --head soft-attn-pool --mixstyle-p 0.5 --sources
mont,shen,qatar,tbx11k,mendeley_pk[,padchest_tb].

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P1.4: Run the full P1 LODO training

**Files:**
- No source changes; pure run.

- [ ] **Step 1: Compose the run command**

```bash
HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
training/.venv/bin/python training/train_tb.py \
    --head soft-attn-pool \
    --mixstyle-p 0.5 \
    --sources mont,shen,qatar,tbx11k,mendeley_pk \
    --output data/tb_head_t2_softattn.pt \
    --threshold-output data/tb_threshold_t2_softattn.json
```

(Add `padchest_tb` to `--sources` if the BIMCV DUA has come through by this point.)

- [ ] **Step 2: Run the command (~3-4 hours on M4)**

The training script will run full LODO (one fold per source = 5 folds with the new Mendeley source; 6 with PadChest if available). For each fold it logs:
- fold name (e.g., "LODO held-out: mendeley_pk")
- training loss curve
- val AUROC + CI
- fold-specific T + thr@95sens
- artifact paths

- [ ] **Step 3: Sanity-check the per-fold AUROCs**

Look at the final stdout for the LODO mean AUROC + worst-fold AUROC. Expected:
- Mean LODO AUROC >= 0.88 (the baseline was 0.922; the surgical head change shouldn't catastrophically drop AUROC, and the added source diversity should help maintain or improve it).
- Worst-fold AUROC >= 0.80 (no source collapses to random).
- If LODO mean drops below 0.85 — **STOP**. The SoftAttnPool may have a bug; investigate before continuing.

- [ ] **Step 4: Commit the new head + threshold artifact**

```bash
git add data/tb_head_t2_softattn.pt data/tb_threshold_t2_softattn.json 2>/dev/null || true
# (these are gitignored; the commit is no-op for the binary files but documents the run)

# Commit the run log instead
training/.venv/bin/python training/train_tb.py --head soft-attn-pool --mixstyle-p 0.5 --sources mont,shen,qatar,tbx11k,mendeley_pk > docs/baselines/2026-05-25-p1-softattn.txt 2>&1 || true
git add docs/baselines/2026-05-25-p1-softattn.txt
git commit -m "chore(p1): full LODO training log for SoftAttnPool + MixStyle on 5-source corpus

Per-fold AUROCs + worst-fold sens recorded. Artifact at
data/tb_head_t2_softattn.pt (gitignored).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P1.5: Re-fit OOF cache + re-measure under the locked protocol

**Files:**
- No source changes; pure run.

- [ ] **Step 1: Regenerate the OOF logits using the SoftAttnPool head**

Modify `train_tb.py`'s LODO loop (if it doesn't already) to also write per-image OOF logits for the held-out fold. The output path should be `data/image_oof_logits_softattn.npz` (parallel to the existing `data/image_oof_logits.npz`).

- [ ] **Step 2: Run `measure_p0_baseline.py` against the SoftAttnPool head**

Modify the baseline script to accept a `--head-artifact` and `--oof-cache` flag. Run:

```bash
training/.venv/bin/python training/measure_p0_baseline.py \
    --head-artifact data/tb_head_t2_softattn.pt \
    --oof-cache data/image_oof_logits_softattn.npz \
    --output data/p1_softattn_measure.json
```

The locked T + thr come from `data/p0_locked_calibration.json` UNCHANGED — they are not re-fit.

- [ ] **Step 3: Compare P0 baseline vs P1 measurement**

```bash
training/.venv/bin/python -c "
import json
p0 = json.loads(open('data/p0_baseline.json').read())
p1 = json.loads(open('data/p1_softattn_measure.json').read())
print('              P0 baseline  P1 SoftAttnPool')
for k in ('strict_sens','strict_spec','effective_sens_with_abstain'):
    v0 = p0['cohen_blind'].get(k, 'n/a')
    v1 = p1['cohen_blind'].get(k, 'n/a')
    print(f'  cohen {k:30s}  {v0}  {v1}')
for k in ('strict_sens','strict_spec','abstain_rate'):
    v0 = p0['lodo_eval_slice'].get(k, 'n/a')
    v1 = p1['lodo_eval_slice'].get(k, 'n/a')
    print(f'  lodo  {k:30s}  {v0}  {v1}')
"
```

- [ ] **Step 4: Apply the GO/NO-GO/WATCH gate**

Per the steelman discipline:

```
GO:    Cohen strict_sens improvement >= +0.10 (at least 1 of the 4 M24 confident misses rescued)
       AND LODO 80% eval-slice sens unchanged or +0-2 pp
       AND No worst-fold AUROC catastrophe (< 0.80)

NO-GO: Cohen strict_sens worse or flat AND LODO sens worse
       → revert. SoftAttnPool didn't fix the M24 cause; try a different surgical change in P1.5.

WATCH: Cohen improves but LODO regresses slightly (or vice versa).
       Decision deferred to a follow-up plan that A/B-tests against M26 ABSTAIN-rule lift.
```

Record the verdict in `data/p1_softattn_measure.json` under a `verdict` key.

- [ ] **Step 5: Commit the measurement JSON**

```bash
git add data/p1_softattn_measure.json
git commit -m "chore(p1): measurement vs P0 baseline under locked protocol — SoftAttnPool + MixStyle 5-source

Verdict: GO / NO-GO / WATCH (filled by the measurement run).
Cohen strict_sens shift: [delta from P0]
LODO 80% eval-slice sens shift: [delta from P0]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task P1.6: Append P1 entry to CASE_STUDY + EXPERIMENT_LOG

**Files:**
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/docs/CASE_STUDY.md`
- Modify: `/Users/ahmadbilal/Downloads/hobby/TB detector/docs/EXPERIMENT_LOG.md`

- [ ] **Step 1: Append CASE_STUDY P1 entry**

Drop in (after P0.5, before maintenance log):

```markdown
## Milestone P1 — Surgical M24 Fix (SoftAttnPool + MixStyle on 5-source corpus) (2026-05-25)

The M24 diagnostic identified atypical-TB miss as caused by ZonalSoftOR's hard zone priors — mid-lung consolidative TB falls between fixed zones and gets attenuated by SoftOR even when the backbone clearly encoded the patch-level evidence. Wave 1.3 confirmed this is a published, attributable architectural cause and that learned non-zonal attention is the literature's preferred fix.

P1 changes exactly ONE lever: ZonalSoftOR -> SoftAttnPool (learned single-head attention over patch tokens, no fixed zones). Backbone unchanged (rad-dino-base). No multi-task TXRV head. No focal loss. No GLoRI multi-query. The point is to isolate the surgical-fix effect from any other architectural change so the measurement is interpretable.

Training corpus expanded from 4 to 5 sources (current 4 + Mendeley Kiran/Jabeen Pakistani via P0.5; PadChest TB-7-union still DUA-pending and joins as soon as credentials land). MixStyle feature-space augmentation (p=0.5) applied to patch tokens during training. All measurement under the P0 locked protocol — T + thr@95sens NOT re-fit on the evaluation surface.

**Measured numbers** (frozen P0 baseline left column; P1 right):

| Metric | P0 baseline | P1 SoftAttnPool |
|--------|---|---|
| Cohen blind strict sens | [filled by measure_p0_baseline.py] | [filled by measure_p1_softattn.py] |
| Cohen blind effective sens (with M26 abstain) | [same] | [same] |
| Cohen blind strict spec | [same] | [same] |
| LODO 80% eval-slice sens | [same] | [same] |
| LODO 80% eval-slice spec | [same] | [same] |
| Cohen abstain rate | [same] | [same] |
| LODO mean AUROC | 0.922 (M22 historical) | [from p1 fold logs] |
| LODO worst-fold AUROC | 0.825 (M22 Montgomery) | [from p1 fold logs] |

**Verdict:** [GO / NO-GO / WATCH per the steelman-discipline gate in P1.5]

**What I learned.** The single-lever discipline + locked protocol made this measurement honest in a way the original Option C never would have been. If SoftAttnPool wins on Cohen blind without regressing LODO 80% eval-slice, the win is attributable: not "we changed five things and it got better" but "we removed the hard zone priors and the M24-class miss got smaller, exactly where the architectural diagnosis predicted." If it loses, the architectural diagnosis was incomplete and we need a different surgical change. Either outcome teaches us something true. That's the value of the locked protocol; the value of the steelman is that it forced me to set the protocol up before measuring anything.
```

- [ ] **Step 2: Append the EXPERIMENT_LOG §C row**

```markdown
| 05-25 | **P1 SURGICAL M24 FIX** — `training/heads/soft_attn_pool.py` (single-head non-zonal learned attention, replaces ZonalSoftOR's hard zone priors) + `training/mixstyle.py` (feature-space MixStyle, p=0.5) wired into `train_tb.py` via `--head soft-attn-pool --mixstyle-p 0.5 --sources mont,shen,qatar,tbx11k,mendeley_pk`. Single architectural lever; backbone unchanged. Full LODO retrain + measurement under P0 locked protocol (T + thr NOT re-fit on evaluation surface). | Cohen strict_sens >= P0 + 0.10 (at least 1 of 4 M24 confident misses rescued) AND LODO 80% eval-slice sens unchanged or +0-2 pp. | [filled after measurement run] | No drift if measurement respects locked protocol (verified by the calibration JSON SHA being unchanged in the measurement output). | [GO / NO-GO / WATCH from P1.5 verdict] |
```

- [ ] **Step 3: Commit**

```bash
git add docs/CASE_STUDY.md docs/EXPERIMENT_LOG.md
git commit -m "docs(p1): CASE_STUDY P1 entry + EXPERIMENT_LOG row

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**

| Spec item from P0→P1 plan | Task implementing it |
|---|---|
| Locked OOD calibration split (deterministic, stratified) | P0.1 — `make_calibration_split`, seed=7, 20% strat by label+source |
| Locked T + thr@95sens fit ONCE, persisted | P0.2 — `fit_locked_calibration` → `data/p0_locked_calibration.json` |
| CXR-safe TTA, K=5, no rotation/crop | P0.3 — `training/tta.py` |
| TriageEngine opt-in `use_tta` + `use_locked_protocol` | P0.4 |
| P0 baseline measured ONCE on Cohen + LODO 80% | P0.5 — `measure_p0_baseline.py` |
| Mendeley PK download + index registration | P0.5.1, P0.5.2 |
| PadChest TB-7-union builder + DUA tracking | P0.5.3 |
| Re-dedup + re-extract on expanded sources | P0.5.4 |
| SoftAttnPool head module (non-zonal attention) | P1.1 |
| MixStyle feature-space augmentation | P1.2 |
| `train_tb.py` flag plumbing + smoke test | P1.3 |
| Full LODO retrain on 5-source corpus | P1.4 |
| Re-measure under locked protocol + GO/NO-GO gate | P1.5 |
| CASE_STUDY + EXPERIMENT_LOG entries for each phase | P0.6, P0.5.5, P1.6 |

All spec items covered. No gaps.

**2. Placeholder scan:**

- Three intentional placeholders in the docs entries: "[to be filled by measure_p0_baseline.py]", "[filled after measurement run]", "[GO / NO-GO / WATCH from P1.5 verdict]". These are not implementation placeholders — they're slots for empirical numbers from runs that happen DURING execution. The implementer fills them in after running the script, not from planning. Acceptable.
- One acceptable "[DATE — fill in when submitted]" for the BIMCV form submission tracking. Manual administrative step.
- No code-step placeholders. Every code block is complete + runnable.

**3. Type consistency:**

- `LockedCalibration` dataclass defined in P0.2; loaded by `TriageEngine` in P0.4 via `load_locked_calibration()`. Field names (`T`, `thr_at_95sens`, `borderline_low`, `s_inactive_escalate`, `asymmetric_evidence_thr`) consistent throughout.
- `K_PASSES = 5` constant in `tta.py` (P0.3); referenced in P0.4's `test_engine_supports_tta_mode` consistently.
- `SoftAttnPool.forward()` returns `(out, attn)` tuple (P1.1); referenced consistently in P1.3's dispatch (`pooled, attn_weights = self.soft_attn_pool(patches)`).
- `MixStyle` is `nn.Module` with `forward(x)` returning same shape (P1.2); used in P1.3 as `self.mixstyle = MixStyle(p=...) or nn.Identity()`.
- CLI flag names `--head`, `--mixstyle-p`, `--sources`, `--smoke` consistent between P1.3 and P1.4 invocations.
- Source ids consistent: `mont, shen, qatar, tbx11k, mendeley_pk, padchest_tb` used everywhere with the same spelling.

All consistent.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-25-global-maxima-p0-p1.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task; two-stage review (spec compliance, then code quality) between tasks; fast iteration; continuous execution.

**2. Inline Execution** — Execute tasks in this session using executing-plans; batch with checkpoints for review.

**Which approach?**
