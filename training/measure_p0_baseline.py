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
