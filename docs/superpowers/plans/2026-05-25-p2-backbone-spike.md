# P2 — Foundation-Backbone Spike (frozen-embedding, evidence-gated) — Plan

**Status:** authored 2026-05-25 after the P1 NO-GO (a smarter pooling head did not improve external
generalization; PK AUROC 0.764→0.624 when the head swap was isolated). The P1 finding localized the
bottleneck to the **backbone representation and/or data diversity**, not the pooling layer. P2 tests the
*backbone-representation* half of that, as cheaply as possible, before any architecture commitment.

**Discipline carried from P0/P1 (non-negotiable):** one lever per rung; locked-protocol calibration
fit ONCE per model on the deterministic seed=7 cal slice and never re-tuned on the eval surface;
paired bootstrap; refuse to overclaim. Frame the result narrowly (see §Framing).

**GPT design review (2026-05-25, architecture template) folded in:** re-fit calibration per backbone
(do NOT reuse rad-dino's T/thr — the threshold-transfer trap caught in P1); patient-level bootstrap;
report pAUC + sensitivity at matched specificity (spec is the failure mode), not just AUROC; branch
ablations (TXRV-only / backbone-only / fused) because the fixed TXRV branch can mask a backbone
effect; pre-register the decision rule; demote the Pakistani set to *validation* when it is used to
CHOOSE a backbone (multiple-testing honesty).

---

## Framing (what this spike can and cannot claim)

CLAIM IT TESTS: "Under a frozen-feature, fixed-head-architecture protocol with per-model calibration,
does a different CXR foundation backbone improve external (Pakistani) transfer over rad-dino?"

IT DOES NOT CLAIM: "the backbone is/isn't the global bottleneck" (frozen-only; TXRV branch fixed; no
LoRA/fine-tune; preprocessing coupling). Record the narrow claim in CASE_STUDY, not the broad one.

---

## P2.0 — The cheapest discriminator FIRST (zero new extraction)

**Goal:** decide whether the rad-dino representation even *can* separate Pakistani TB better than the
deployed head achieves (0.78). If a simple linear probe on the ALREADY-CACHED rad-dino features hits
~0.78 too, the representation caps there → a backbone swap is well-motivated. If the probe scores much
higher, rad-dino has untapped separability the head isn't using → the lever is the head/training/
calibration, not the backbone, and we should NOT spend days on a new backbone.

**Inputs (all already on disk, no extraction):** `data/features.npz` (13,260 train rows, rad-dino
`cls` 768 + `txrv` 1042), `data/features_mendeley_pk.npz` (3,008 Pakistani holdout, same schema).

**Tasks:**
- `training/p2_linear_probe.py`: L2-logistic-regression probes trained on the 4-source corpus,
  evaluated on the held-out Pakistani set, three branches: (a) rad-dino CLS only, (b) TXRV only,
  (c) CLS⊕TXRV fused. Report patient-level bootstrap AUROC (95% CI) + pAUC in the high-specificity
  region (FPR ≤ 0.2) + sensitivity at spec=0.70 for each. Also report the SAME probes' LODO-eval-slice
  AUROC for an in-distribution reference.
- This reuses the locked-protocol split utilities; the probe is calibrated ONCE on the seed=7 cal
  slice if a threshold is needed, but the headline is threshold-free AUROC/pAUC.

**Decision gate P2.0:**
- If rad-dino CLS (or fused) linear-probe Pakistani AUROC ≈ 0.78 (within CI of the deployed head) →
  the representation caps at the external ceiling → PROCEED to P2.1 (backbone is the candidate lever).
- If the probe Pakistani AUROC is materially higher (≥ +0.05 over 0.78, CIs apart) → STOP the backbone
  spike; the rad-dino features already contain more external signal than the head extracts → the lever
  is head/training/calibration (revisit P1-class fixes or per-site recalibration), NOT a new backbone.
  Record this and surface to the coordinator.

---

## P2.1 — maira-2 frozen backbone swap (local, drop-in, runs in hours)

**Why maira-2 first:** `microsoft/rad-dino-maira-2` is already in `~/.cache/huggingface/hub/`, is a
drop-in identical geometry (37×37 patch grid → 8×8=64 pooled, 768-d CLS), so it reuses the ENTIRE
existing head + preprocessing + ablation harness with zero schema change. The prior maira-2 A/B was a
NO-GO but at N=123 (badly underpowered); this powers it properly on the full corpus + Pakistani.
Caveat acknowledged: maira-2 shares rad-dino's Microsoft DINO-family lineage, so the expected
information gain is modest — if it's a wash, that is itself evidence pointing to CheXFound (different
family/scale) as the higher-information move.

**Tasks:**
1. Extend `extract_features.py` with a `--backbone {rad-dino, rad-dino-maira-2}` flag (default
   rad-dino; preprocessing IDENTICAL across backbones — same resize/normalize/lung-crop, the GPT
   "preprocessing coupling" confound). Extract maira-2 features for the full 13,260 training corpus →
   `data/features_maira2_full.npz` and the 3,008 Pakistani holdout → `data/features_maira2_pk.npz`.
   Frozen-backbone read; Pakistani still never enters training.
2. Retrain the deployed head (zonal-softor, IDENTICAL recipe: same architecture, optimizer, seed,
   early-stop, 4-source mix, PRIOR-FIXED knobs) on the maira-2 features. ONE lever = the backbone.
3. Fit calibration (T + thr@95sens) ONCE per backbone on the seed=7 locked cal slice of THAT
   backbone's OOF. Do NOT reuse rad-dino's 0.6105. Freeze, then eval.
4. Evaluate rad-dino (config R = P1's config A, already measured) vs maira-2 (config M) on: held-out
   Pakistani + LODO-eval slice. Report patient-level paired bootstrap ΔAUROC (CI), pAUC (FPR≤0.2),
   sens@spec=0.70, spec@locked-thr, ECE. Branch ablations: backbone-only head vs fused (with TXRV).

**Pre-registered decision gate P2.1 (write this BEFORE running):**
- GO to deeper maira-2 investment / treat as a win ONLY IF external ΔAUROC ≥ +0.03 (paired CI excludes
  0) OR specificity improves materially at matched sensitivity (e.g. ≥ +0.05 spec at sens=0.90).
- Else NO-GO on maira-2 → proceed to P2.2 (CheXFound). Do NOT polish maira-2 past the gate.

---

## P2.2 — CheXFound (CONDITIONAL on P2.1 NO-GO; higher ceiling, more work)

Only if maira-2 is a wash. `DIAL-RPI/CheXFound` (Apache-2.0, ViT-L/16, 512px, 1.005M CXR pretrain),
reported +0.12 AUROC head-to-head vs rad-dino on TB. Requires: a network download (~ViT-L weights), a
32×32→8×8 re-pool adapter (and a fair adapter — a bad one makes CheXFound look artificially weak, the
GPT "token geometry" confound), and a CLS-dim change (ViT-L is 1024-d, the head's input dim must
adapt). Same protocol as P2.1 (per-model calibration, patient-level paired bootstrap, pAUC,
branch ablations, pre-registered ΔAUROC ≥ +0.03 gate). Detailed task breakdown authored only if P2.1
sends us here.

---

## Multiple-testing / holdout-integrity honesty

If both maira-2 AND CheXFound are evaluated on the Pakistani set and the winner is CHOSEN by it, the
Pakistani set has been used for model selection → its CI is optimistic and it is now a VALIDATION set,
not a clean test. Record this explicitly. If a backbone is adopted, the honest final external number
needs a fresh held-out site (PadChest/VinDr when DUAs land, or a re-split). Lead with this caveat in
the CASE_STUDY entry.

---

## Deliverables
- `training/p2_linear_probe.py` (+ a small test), `training/p2_backbone_ab.py` (extends p1_ablation.py
  harness with a backbone axis), `data/p2_*.json` results, `docs/baselines/2026-05-25-p2-*.txt` logs.
- CASE_STUDY P2 entry + EXPERIMENT_LOG §C row(s) with REAL patient-level-bootstrap numbers, the narrow
  framing, and the pre-registered gate outcome. Branch `feat/p2-backbone-spike`, NOT merged — the
  coordinator reviews + the backbone-adoption decision is a user gate.
