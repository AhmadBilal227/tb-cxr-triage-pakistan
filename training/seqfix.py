"""T2-F: fix the old-scar false-positive rate (SeqFPR=0.842) by HARD-NEGATIVE MINING.

The active head confidently flags 84% of old/healed-TB scars as active. A post-hoc gate can't fix
that (the active signal is confidently wrong). The real fix: train the active head with the sequelae
probe AS NEGATIVES so it learns "old scar != active TB". We test it honestly:
  - split the 139 sequelae into train-negatives (70%) and a HELD-OUT scar test (30%) — SeqFPR is
    measured ONLY on scars the head never trained on.
  - BASELINE head (no scar negatives) vs FIXED head (+ scar negatives), same TB train/test split.
  - report SeqFPR drop AND the TB-sensitivity cost (the tradeoff to watch).
Caveat: the sequelae are TBX11K-provenance, so the scar-rejection learned here may not transfer to
other sites' scars (untested). Endpoint radiographic. Levers: fusion+zonal (the image-level decision).

    PYTORCH_ENABLE_MPS_FALLBACK=1 python training/seqfix.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from train_tb import (DATA, SEED, train_head_t2, predict_t2, predict_logits_t2, fit_temperature,
                      threshold_for_sensitivity, _sens_spec, clopper_pearson)

LEVERS = frozenset({"fusion", "zonal"})  # image-level decision; box is for localization, not this
SEQ_TEST_FRAC = 0.30


def _f(d, k):
    return d[k].astype("float32")


def main() -> None:
    rng = np.random.default_rng(SEED)
    main = np.load(DATA / "features.npz", allow_pickle=True)
    seq = np.load(DATA / "features_sequelae.npz", allow_pickle=True)
    keys = ("cls", "patches", "txrv", "zones")
    n_main = int(main["cls"].shape[0])
    n_seq = int(seq["cls"].shape[0])
    y_main = main["y"].astype("int64")

    # combined arrays: [main rows .. seq rows]; seq are negatives (label 0) for the active head
    arrs = {k: np.concatenate([_f(main, k), _f(seq, k)], axis=0) for k in keys}
    # train_head_t2 reads has_box/grid_label even with box lever off; sequelae have no boxes
    arrs["has_box"] = np.concatenate([main["has_box"].astype(bool), np.zeros(n_seq, dtype=bool)])
    arrs["grid_label"] = np.concatenate(
        [main["grid_label"].astype("float32"), np.zeros((n_seq, 8, 8), dtype="float32")], axis=0)
    y = np.concatenate([y_main, np.zeros(n_seq, dtype="int64")])
    main_idx = np.arange(n_main)
    seq_idx = np.arange(n_main, n_main + n_seq)

    # honest splits: TB test from the main binary; scar test held out of training
    tb_tr, tb_te = train_test_split(main_idx, test_size=0.2, stratify=y_main, random_state=SEED)
    seq_tr, seq_te = train_test_split(seq_idx, test_size=SEQ_TEST_FRAC, random_state=SEED)
    tr_base, va_base = train_test_split(tb_tr, test_size=0.15, stratify=y[tb_tr], random_state=SEED)
    fixed_train = np.concatenate([tb_tr, seq_tr])
    tr_fix, va_fix = train_test_split(fixed_train, test_size=0.15, stratify=y[fixed_train], random_state=SEED)

    def evaluate(tr, va, tag):
        model = train_head_t2(arrs, y, tr, va, LEVERS, seed=SEED)
        T = fit_temperature(predict_logits_t2(model, arrs, va), y[va])
        thr = threshold_for_sensitivity(y[va], predict_t2(model, arrs, va, T))  # cold-start @95% sens
        p_tb = predict_t2(model, arrs, tb_te, T)
        sens, lo, hi, spec = _sens_spec(y[tb_te], p_tb, thr)
        p_seq = predict_t2(model, arrs, seq_te, T)               # held-out scars
        k = int((p_seq >= thr).sum()); n = len(seq_te)
        s_lo, s_hi = clopper_pearson(k, n)
        print(f"[{tag}] TB-test sens={sens:.3f} [{lo:.2f}-{hi:.2f}] spec={spec:.3f} | "
              f"SeqFPR(held-out scar)={k}/{n}={k/n:.3f} [{s_lo:.2f}-{s_hi:.2f}] | thr={thr:.3f} T={T:.2f}")
        return sens, spec, k / n

    print(f"main={n_main} (pos {int((y_main==1).sum())}) | sequelae={n_seq} | "
          f"TB test={len(tb_te)} | scar test={len(seq_te)} | levers={sorted(LEVERS)}")
    print("=== BASELINE (active head, NO scar negatives) ===")
    b_sens, b_spec, b_fpr = evaluate(tr_base, va_base, "baseline")
    print("=== FIXED (active head + scar hard-negatives) ===")
    f_sens, f_spec, f_fpr = evaluate(tr_fix, va_fix, "fixed")
    print(f"\nSeqFPR {b_fpr:.3f} -> {f_fpr:.3f}  (delta {f_fpr-b_fpr:+.3f})  | "
          f"TB sens {b_sens:.3f} -> {f_sens:.3f} ({f_sens-b_sens:+.3f}), spec {b_spec:.3f} -> {f_spec:.3f}")


if __name__ == "__main__":
    main()
