# Experiment & Drift Log

**Purpose:** an append-only ledger of every training run / pipeline change, recording **expected vs actual**
numbers so we can *see the moment we drift* from the correct pattern — a flattering jump (usually a leak),
a silent regression, or a broken invariant. CASE_STUDY.md is the narrative; `docs/baselines/*.txt` are the
raw dumps; **this file is the scoreboard + the tripwires.** Update it after every run or material change.

Endpoint is **radiographic-TB-pattern**, not bacteriological. Lead with sensitivity; report worst-fold + CIs.

---

## A. Expected patterns / invariants (the reference — drift = deviation from these)
- **Mean LODO AUROC band:** ~0.88–0.94 (commercial external band). Below ~0.85 = regression; **above ~0.95 on open data = suspect leakage, not a win.**
- **Per-fold trust:** Montgomery (n=58 pos) & Shenzhen are the *cleaner external* folds → lead with **worst-fold (Montgomery) recalibrated sensitivity + Clopper-Pearson CI**. Qatar/TBX11K are leakage-prone re-mix folds → treat their AUROC as an internal-ish upper bound.
- **Attention lever:** ≈ **+0.008 AUROC at full scale** (NOT the +0.04 seen on the 2,177 subset). A large attention gain at scale = re-investigate.
- **Operating point:** target 95% sensitivity (WHO optimum). At AUROC ~0.92 expect **spec ~0.55–0.70**; at 90% sens ~0.70–0.80.
- **PPV reality:** 1% prevalence ⇒ ~2% PPV (~50 confirmatory tests/flagged case); 9–15% (S-Asian presumptive) ⇒ ~20–40% PPV. PPV reported at prevalence, never balanced accuracy.
- **Site-leak canary:** high & expected (≈0.95 on head projection, ≈0.98 on cls+txrv) — a **diagnostic, not a target**. Do NOT optimize preprocessing to lower it.
- **Anti-shortcut floor:** permuted-label LODO AUROC **< 0.60** (memorization screen — necessary, NOT sufficient). True anti-shortcut evidence = lesion-localization mIoU/hit-rate above random + provenance-independent external validity.
- **Data invariants:** 0 NaN/inf in features; TXRV[:,1024:] are raw logits (~[-17,7]), not [0,1]; features↔labels 0 mismatches; LODO positives disjoint across sources (cross-fold leak ≤0.12%, all-negative).
- **Gates (CLAUDE.md):** `npm run build` clean · tests green · a11y ≥95 · no ≥90% sensitivity *claim* without ≥150 held-out positives.

## B. Drift triggers (if any fires → STOP and investigate before trusting the number)
1. Mean LODO AUROC **jumps > +0.05** vs the prior comparable run → suspect leakage / a preprocessing change that added a shortcut.
2. **Worst-fold (Montgomery) sensitivity drops** while the mean rises → the mean is hiding a regression on the honest fold.
3. **Sensitivity rises *with* rising site-leak** (or with falling localization mIoU) → a shortcut, not a real gain → reject the change.
4. **Permuted-label LODO AUROC ≥ 0.60** → features carry an exploitable shortcut → distrust the run.
5. A change is selected by a knob **tuned on the LODO test folds** → optimistic bias → re-run with prior-fixed / nested-CV weights.
6. ECE rises notably after a change (esp. with the box head / large LSE `r`) → calibration drift.
7. Build/tests/a11y red, or a doc claim exceeds the evidence (e.g., "guarantee", "active TB confirmed", a ≥90% claim without the positives) → honesty drift.

## C. Run ledger (append-only; newest at bottom)

| Date | Run / change | Expected | Actual | Drift? | Decision / note |
|---|---|---|---|---|---|
| 05-24 | M10 baseline subset (2,177; 3 src) fusion-only → +attention | commercial band 0.80–0.88 | **0.858 → 0.897** | on-band ✓ | recorded; optimistic (re-mix leakage, subsample) |
| 05-24 | M11 dual-report (same subset) | attention helps; Shenzhen cold-start low | 0.871 → 0.886; Shenzhen **34%→90%** w/ local recal | as-expected ✓ | dual cold-start/recal report adopted |
| 05-24 | M13 full 4-source (13,092) @92% sens | attention ~+0.04 (per subset) | 0.916 → **0.924**; attention only **+0.008** | **DRIFT (expectation)** | corrected: attention is marginal at scale; recorded the walk-back |
| 05-24 | M12 site-leak (harmonized) | reduce site separability | 1.000 → **0.945** | as-expected (modest) ✓ | site-leak kept as diagnostic, not target |
| 05-24 | M14 data audit + review swarm | data sound; "no shortcut" | data faithful ✓; site **98% recoverable**, prevalence 6.3× → "genuine signal" **overclaimed** | **DRIFT (claim)** | corrected M13 framing; localization is the real discriminator; 2 geometry bugs fixed |
| 05-24 | **T2 BASELINE** — re-extraction (letterbox geom + box/zone tensors) @95% sens, deterministic code | AUROC ≈0.92, no jump; spec ~0.55–0.70; worst-fold ≥0.90 | fusion **0.922** → +att **0.938** (+0.006/+0.014 vs M13, **no jump ✓**); worst-fold Shenzhen recal sens **0.949 [0.91-0.97]** ✓; site-leak 0.942 (stable); clean-fold cold-start sens 0.85–0.92 @ spec 0.72–0.93; PPV 9%→15.5% (6.5 tests/case) | **No** ✓ | ACCEPT as T2 baseline. grid_label/zones/has_box(660) cached → heads can train without re-extraction. attention +0.015 (still modest, consistent with M13). This is the "before" for every T2 improvement. |
| 05-24 | **T2 HEADS** (`feat/t2-heads`) — zonal soft-OR + box-evidence + pathology distillation, prior-fixed knobs (g_min=0.25, r0=5/cap8, λ_box=0.2, λ_distill=0.2). Same-run re-baseline: fusion **0.914** / +att **0.934**. | MODEST AUROC (heads buy spec/localization/trust, NOT big AUROC); >+0.05 jump = suspect; worst-fold sens preserved/↑; box lever's effect on NON-TBX11K folds shown | **T2[FULL] mean AUROC 0.925** (delta vs +att **−0.009**, no jump ✓); ablation: fusion+distill 0.912, +zonal 0.929, +box 0.929, FULL 0.925; **worst-fold Shenzhen recal sens 0.939→0.958** [0.92-0.98] ✓ (zonal soft-OR is the sensitivity lever, as designed); box lever's **non-box-fold AUROC 0.924 ≈ overall 0.925** (no TBX11K leak ✓); **localization (TBX11K LODO holdout, box loss never saw it): pointing-game 0.865 [0.84-0.89], mIoU-topk 0.604 [0.58-0.63] vs random floor 0.105/0.059 (+0.76/+0.55 lift)**. cold-start spec rose on most folds (zonal+box add specificity at the operating point). | **No** ✓ | ACCEPT. Levers that HELPED: zonal soft-OR (worst-fold sens +0.019, the headline) and box-evidence (localization + cold-start spec). Distillation alone: AUROC-neutral (−0.022) — it's a grounding/specificity anchor, not an AUROC lever (expected per blueprint §D). AUROC change is modest by design; the wins are worst-fold sensitivity, localization faithfulness, and operating-point specificity. Box supervision is 100% TBX11K → localization validated ONLY on TBX11K boxes (8×8 grid, relative-to-random metric, not absolute). |

## D. How to append (per run)
Add a row to §C with: what changed, the number you **expected before running**, the **actual**, a drift flag against §B,
and the decision. If a §B trigger fires, write the investigation outcome (leak found / corrected / accepted-with-caveat).
Mirror material milestones into CASE_STUDY.md (narrative) and the raw output into `docs/baselines/`.
