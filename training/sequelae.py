"""Inactive-sequelae (active-vs-healed) probe metrics + the verdict gate. Blueprint §0.1, §2 E, §6.

THE PROBLEM. The dominant South-Asian FALSE-POSITIVE source is an OLD/HEALED TB scar (calcification,
fibrosis, volume loss) misread as ACTIVE disease. We have a 139-image inactive-sequelae specificity
probe (data/features_sequelae.npz; all TBX11K, the mislabeled-"latent" rows, deduped to UNIQUE images).
It is NEVER in features.npz, so it is automatically held out of the main TB head's training/eval.

NOTE ON THE DENOMINATOR. The blueprint anticipated 239 rows -> 169 unique. The current
data/index_tbx_latent.csv has 139 rows / 139 UNIQUE images, so the probe denominator is **139**, not
169. We report the real number (ethos: report real numbers).

WHAT THIS SCRIPT MEASURES (the honest deliverables):
  (a) SeqFPR — fraction of the 139 sequelae scored >= the MAIN TB head's operating threshold (does the
      TB head over-call old scar as active?), with a Clopper-Pearson CI.
  (b) active-vs-sequelae AUROC — (i) via the trained inactive head `s_inactive`, (ii) via the main TB
      prob. Honest ceiling ~0.82-0.88; **>0.90 => SUSPECT LEAKAGE** (probe + active boxes share TBX11K
      provenance), reported with a loud caveat. Cross-validated (the probe is tiny: 139).
  (c) net SeqFPR reduction when GATING with `s_inactive` (escalate-not-clear): how many over-called
      sequelae would be escalated to UNDETERMINED rather than waved through.

LEAKAGE CONFOUND (loud). Positives (the probe) and the negatives we train the inactive head on (active
TB) BOTH come substantially from TBX11K. A high AUROC can be a TBX11K-internal split, not a transferable
active-vs-healed signal. We therefore report TWO negative pools: TBX11K-only actives (hardest/honest,
same provenance) and all-source actives. The TBX11K-only number is the one to trust.

    PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 training/.venv/bin/python training/sequelae.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_tb import (SEED, TBHeadT2, activity_verdict, clopper_pearson, fit_temperature,
                      predict_inactive, predict_logits_t2, predict_t2, _seq_arrs_from,
                      train_inactive_head)

DATA = Path(__file__).resolve().parents[1] / "data"
FULL_LEVERS = frozenset({"fusion", "zonal", "box"})
N_FOLDS = 5
BOOTSTRAP_N = 2000
LEAK_SUSPECT_AUROC = 0.90  # > this => suspect leakage, NOT a win (Blueprint §2 E)
CEILING = (0.82, 0.88)     # honest active-vs-healed ceiling on a single frontal film


def _bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n: int = BOOTSTRAP_N) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], p[b]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _load_main_t2_head(npz_main: dict) -> tuple[TBHeadT2, float, float]:
    """Load the trained main T2 head + its calibration. Returns (model, temperature, threshold)."""
    cfg = json.load(open(DATA / "tb_threshold_t2.json"))
    model = TBHeadT2(npz_main["patches"].shape[2], npz_main["cls"].shape[1],
                     npz_main["txrv"].shape[1], FULL_LEVERS)
    from train_tb import DEVICE
    model.load_state_dict(torch.load(DATA / "tb_head_t2.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model, float(cfg["temperature"]), float(cfg["threshold"])


def main() -> None:
    npz_main = dict(np.load(DATA / "features.npz", allow_pickle=True))
    npz_seq = dict(np.load(DATA / "features_sequelae.npz", allow_pickle=True))
    n_seq = int(npz_seq["cls"].shape[0])
    print("ENDPOINT: radiographic inactive/sequelae PATTERN (NOT immunologic latent TB — that is "
          "radiographically silent). Research preview, not a device.")
    print(f"sequelae probe: {n_seq} UNIQUE images (denominator = {n_seq}; the blueprint's '169' is stale "
          f"— the current index_tbx_latent.csv has {n_seq} rows). Source: tbx11k_sequelae, all label-1-as-stored.")
    n_act_all = int((npz_main["y"] == 1).sum())
    n_act_tbx = int(((npz_main["y"] == 1) & (npz_main["source"].astype(str) == "tbx11k")).sum())
    print(f"active-TB negatives for the inactive head: all-source={n_act_all}, TBX11K-only={n_act_tbx}")

    # =====================================================================================
    # (a) SeqFPR — does the MAIN TB head over-call old scar as active?
    # =====================================================================================
    main_head, T_main, thr_main = _load_main_t2_head(npz_main)
    seq_idx = np.arange(n_seq)
    # score the 139 probe images with the main T2 head (build a sequelae-only arrs view)
    seq_arrs = {k: npz_seq[k].astype("float32") for k in ("cls", "patches", "txrv", "zones")}
    p_seq_main = predict_t2(main_head, seq_arrs, seq_idx, T_main)  # main TB prob on the sequelae
    k_fp = int((p_seq_main >= thr_main).sum())
    seqfpr = k_fp / n_seq
    lo, hi = clopper_pearson(k_fp, n_seq)
    print("\n=== (a) SeqFPR — main TB head over-calling old scar (operating threshold from "
          f"tb_threshold_t2.json @95% sens) ===")
    print(f"  threshold={thr_main:.3f}  temperature={T_main:.3f}")
    print(f"  SeqFPR = {k_fp}/{n_seq} = {seqfpr:.3f}  [95% Clopper-Pearson CI {lo:.3f}-{hi:.3f}]")
    print(f"  (fraction of the {n_seq} inactive/sequelae films the MAIN TB head flags as screen-positive; "
          f"these are the old-scar false alarms the inactive head must catch.)")

    # =====================================================================================
    # (b) active-vs-sequelae AUROC — inactive head `s_inactive` AND the main TB prob, cross-validated.
    #     Trained on BOTH negative pools (TBX11K-only = honest, all-source = optimistic).
    # =====================================================================================
    for tbx_only in (True, False):
        pool = "TBX11K-only (HONEST: same provenance as the probe)" if tbx_only else "all-source (optimistic)"
        arrs, y, src = _seq_arrs_from(npz_main, npz_seq, tbx_only=tbx_only)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        print(f"\n=== (b) active-vs-sequelae AUROC — negatives = {pool} ===")
        print(f"  set: {n_pos} sequelae (pos) vs {n_neg} active TB (neg)")
        # out-of-fold s_inactive (every sequelae image scored by a head that never trained on it)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof_s = np.full(len(y), np.nan, dtype="float32")
        for fold, (tr_all, te) in enumerate(skf.split(np.arange(len(y)), y)):
            # carve a small val slice off train for early stopping (stratified)
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold)
            tr_in, va_in = next(inner.split(tr_all, y[tr_all]))
            tr, va = tr_all[tr_in], tr_all[va_in]
            model = train_inactive_head(arrs, y, tr, va, seed=SEED + fold)
            oof_s[te] = predict_inactive(model, arrs, te)
        auc_s = roc_auc_score(y, oof_s)
        s_lo, s_hi = _bootstrap_auc_ci(y, oof_s)
        # main TB prob as the active-vs-sequelae discriminator. The main head separates sequelae from
        # active only INCIDENTALLY (it was trained for activity vs healthy, not active vs healed). We
        # build a zones-bearing view so the main T2 head can run, then score every row (sequelae rows
        # 0..n_pos-1, active rows after) with the main TB prob and take its AUROC for the sequelae label.
        act_zones = np.concatenate([npz_seq["zones"].astype("float32"),
                                    npz_main["zones"][_active_mask(npz_main, tbx_only)].astype("float32")], axis=0)
        full_arrs = {**{k: arrs[k] for k in ("cls", "patches", "txrv")}, "zones": act_zones}
        p_main_aligned = predict_t2(main_head, full_arrs, np.arange(len(y)), T_main)
        auc_main = roc_auc_score(y, p_main_aligned)
        m_lo, m_hi = _bootstrap_auc_ci(y, p_main_aligned)
        flag_s = "  !! SUSPECT LEAKAGE (>0.90)" if auc_s > LEAK_SUSPECT_AUROC else "  (within honest band)"
        print(f"  active-vs-sequelae AUROC via s_inactive = {auc_s:.3f} [95% CI {s_lo:.2f}-{s_hi:.2f}]{flag_s}")
        print(f"  active-vs-sequelae AUROC via MAIN TB prob = {auc_main:.3f} [95% CI {m_lo:.2f}-{m_hi:.2f}] "
              f"(the main head's prob separates sequelae from active only incidentally — it was never "
              f"trained for activity discrimination)")
        print(f"  honest ceiling for active-vs-healed on ONE frontal film ~{CEILING[0]:.2f}-{CEILING[1]:.2f}.")

        if tbx_only:
            # save the TBX11K-only inactive head trained on ALL the probe+TBX-active data (the deployable one)
            full = np.arange(len(y))
            inner = StratifiedKFold(n_splits=8, shuffle=True, random_state=SEED)
            tr_in, va_in = next(inner.split(full, y))
            model = train_inactive_head(arrs, y, full[tr_in], full[va_in])
            T_seq = fit_temperature(
                np.log(np.clip(predict_inactive(model, arrs, full[va_in]), 1e-7, 1 - 1e-7) /
                       np.clip(1 - predict_inactive(model, arrs, full[va_in]), 1e-7, 1 - 1e-7)), y[full[va_in]])
            torch.save(model.state_dict(), DATA / "tb_head_inactive.pt")
            json.dump({"temperature": T_seq, "seq_high": 0.5, "endpoint": "radiographic_inactive_sequelae_pattern",
                       "negatives": "tbx11k_active", "n_probe": n_seq,
                       "auroc_oof_tbx_only": auc_s, "auroc_ci": [s_lo, s_hi],
                       "note": "s_inactive RAISES old-scar suspicion; NEVER clears a flagged film. "
                               "Probe shares TBX11K provenance with active boxes — AUROC may not transfer off-site. "
                               "Not ONNX-exported into the app yet (follow-on)."},
                      open(DATA / "tb_inactive_meta.json", "w"))
            # keep the OOF s_inactive for the gating analysis below (probe rows only)
            global _OOF_S_PROBE
            _OOF_S_PROBE = oof_s[y == 1].copy()

    # =====================================================================================
    # (c) net SeqFPR reduction when GATING with s_inactive (escalate-not-clear).
    #     The gate forces UNDETERMINED on (main flags it OR near-threshold) AND high s_inactive — it never
    #     CLEARS a film. Here we measure: of the SeqFPR over-calls, how many would the gate ESCALATE to
    #     UNDETERMINED (i.e. NOT wave through as a confident active YES)? Uses out-of-fold s_inactive so no
    #     sequelae image is scored by a head that trained on it.
    # =====================================================================================
    cfg = json.load(open(DATA / "tb_threshold_t2.json"))
    tau_high = float(cfg["threshold"])
    tau_low = tau_high * 0.5  # a simple UNDETERMINED band below the operating point (illustrative; per-site)
    s_probe = _OOF_S_PROBE  # OOF s_inactive on the 139 probe images (TBX11K-only-trained head)
    print("\n=== (c) net effect of the escalate-not-clear gate on the SeqFPR over-calls ===")
    print(f"  gate: tau_high={tau_high:.3f} tau_low={tau_low:.3f} seq_high=0.5 (illustrative; re-fit per site)")
    verdicts = [activity_verdict(float(p), float(s), tau_low, tau_high) for p, s in zip(p_seq_main, s_probe)]
    yes = sum(1 for v in verdicts if v["verdict"] == "YES")
    undet = sum(1 for v in verdicts if v["verdict"] == "UNDETERMINED")
    no = sum(1 for v in verdicts if v["verdict"] == "NO")
    # how many of the ORIGINAL over-calls (p>=thr) does the gate keep as YES vs escalate to UNDETERMINED?
    overcall = p_seq_main >= tau_high
    overcall_yes = int(sum(1 for v, o in zip(verdicts, overcall) if o and v["verdict"] == "YES"))
    print(f"  pre-gate: {int(overcall.sum())}/{n_seq} sequelae over-called active (the SeqFPR).")
    print(f"  post-gate verdicts on the {n_seq} sequelae (strict escalate-not-clear): "
          f"YES={yes}  UNDETERMINED={undet}  NO={no}")
    print(f"  {overcall_yes}/{int(overcall.sum())} over-calls stay YES (strong active prob; the inactive head "
          f"CANNOT clear them — escalate-not-clear). The strict gate moves the BELOW-threshold films into "
          f"UNDETERMINED/NO, NEVER an over-call into a NO ('just old scar').")
    # ACHIEVABLE net specificity recovery: a high-s_inactive over-call in the EXTENDED REVIEW BAND
    # [tau_high, tau_review) is escalated YES->UNDETERMINED ("flag for old-scar review") instead of a
    # confident screen-positive. This RECOVERS specificity without ever CLEARING (UNDETERMINED is still a
    # test/review, not a NO). Above tau_review the strong active signal wins (reactivation in scar).
    tau_review = min(0.95, tau_high + 0.25)  # films just above threshold are the recoverable ones
    escalated = int(sum(1 for p, s in zip(p_seq_main, s_probe)
                        if (p >= tau_high) and (p < tau_review) and (s >= 0.5)))
    net_seqfpr_after = (k_fp - escalated) / n_seq
    print(f"\n  ACHIEVABLE net specificity recovery (review-band escalation, tau_review={tau_review:.3f}, "
          f"s_inactive>=0.5):")
    print(f"    {escalated}/{k_fp} over-calls moved YES->UNDETERMINED (flagged for old-scar review, not a "
          f"confident screen-positive).")
    print(f"    confident-YES SeqFPR {seqfpr:.3f} -> {net_seqfpr_after:.3f} "
          f"(delta {net_seqfpr_after - seqfpr:+.3f}). These films are still TESTED (UNDETERMINED), never CLEARED.")
    print("  The inactive head's primary job is the INVERSE direction: on a WEAK active signal it RAISES "
          "old-scar suspicion (NO -> UNDETERMINED), so a subtle reactivation in scar is reviewed, never cleared.")
    hi_conf = int((p_seq_main >= 0.95).sum())
    print(f"\n  HONEST FINDING (the load-bearing one): the main head is CONFIDENTLY wrong on old scar — "
          f"{hi_conf}/{n_seq} ({hi_conf / n_seq:.0%}) sequelae score p_active>=0.95 (median {np.median(p_seq_main):.3f}).")
    print("    => A post-hoc escalate-not-clear GATE cannot meaningfully cut SeqFPR (it must not override a")
    print("       confident active signal — reactivation hides in scar). The real SeqFPR fix is to feed")
    print("       s_inactive into the head as a SPECIFICITY FEATURE / recalibration input, or to HARD-")
    print("       NEGATIVE-MINE old scar into the ACTIVE head's training — NOT a downstream gate. The gate's")
    print("       value is the INVERSE safety direction (raise suspicion on weak-signal scar), not SeqFPR.")

    # demonstrate the escalate-not-clear rule on synthetic operating points (documented behavior)
    print("\n=== verdict-gate truth table (escalate-not-clear; the inactive head can RAISE but never CLEAR) ===")
    for p, s, label in [(0.95, 0.9, "strong active + old scar"), (0.95, 0.1, "strong active, no scar"),
                        (0.45, 0.9, "near-threshold + old scar"), (0.10, 0.9, "weak active + old scar"),
                        (0.10, 0.1, "weak active, no scar")]:
        v = activity_verdict(p, s, tau_low, tau_high)
        print(f"  p_active={p:.2f} s_inactive={s:.2f} ({label:28s}) -> {v['verdict']:13s} activity={v['activity']}")

    print("\n=== SHARED-PROVENANCE CAVEAT (carry into every claim) ===")
    print("  The 139-image probe and the active-TB BOXES both come from TBX11K. A high active-vs-sequelae")
    print("  AUROC may reflect a TBX11K-internal split, NOT a transferable active-vs-healed signal. The")
    print("  TBX11K-only-negatives number is the honest one; >0.90 is treated as SUSPECT, not a win. The")
    print("  endpoint is RADIOGRAPHIC; activity is often unprovable on one film. We never emit 'latent TB'.")


def _active_mask(npz_main: dict, tbx_only: bool) -> np.ndarray:
    m = npz_main["y"] == 1
    if tbx_only:
        m = m & (npz_main["source"].astype(str) == "tbx11k")
    return m


_OOF_S_PROBE: np.ndarray = np.array([])


if __name__ == "__main__":
    main()
