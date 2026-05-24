# Project: AI-Native TB Chest X-Ray Triage

Frontend-only (Vite + React 18 + TypeScript strict, **no `any`**), BYOK, IndexedDB. Three providers
called directly from the browser: Hugging Face (primary perception), Replicate (fallback), OpenAI
`gpt-5.5` (orchestration). Research preview — **not a medical device.**

## Non-negotiable ethos (carry this into every change)
- **Report real numbers.** Measure against ground truth (the `/validate` route + `scripts/accuracy-test*.mjs`). Lead with the honest metric (sensitivity is the safety-critical one for a screen), never a flattering one. The project's whole identity is intellectual honesty about model quality.
- **Don't overclaim (per the M9 expert panel — see CASE_STUDY.md).** The conformal layer does NOT "guarantee ≥90% sensitivity": its coverage is in-distribution + finite-sample and must be re-fit on labeled data from each deployment site, reported with a binomial CI. Open-dataset metrics are against **radiographic** labels, not bacteriological confirmation — say so. WHO 90/70 is a **floor**, not a target. No ≥90% sensitivity claim without ~150+ held-out TB positives. Don't surface "active vs latent TB" from a single film (latent TB is radiographically silent). Report PPV at deployment prevalence, not accuracy on balanced sets.
- **Fallback and degradation are always visible** to the user, never hidden.
- **The deterministic safety net wraps the LLM** — the model advises, guardrails decide; it can escalate but never clear a flagged case on weak evidence.
- Keep strict TS clean (`npm run build`), tests green (`npm test`), and a11y ≥95.

## Orientation
- Contract: `src/lib/types.ts` (`ClassifierResult` = `{ tb_prob, raw, provider_used, latency_ms }`).
- Providers: `src/lib/providers/` (`classify.ts` = the HF→Replicate fallback seam).
- Pipeline: `src/lib/pipeline/orchestrator.ts` (5 stages + screening policy + safety-net combine).
- Calibration: `src/lib/calibration.ts` (temperature/Platt + log-odds fusion + conformal; fit via `/validate` Calibrate).
- Perception training (offline, M4/MPS): `training/` + `docs/superpowers/plans/2026-05-24-tb-classifier-training-T1.md`.
- Roadmap: `docs/superpowers/plans/2026-05-24-perception-module.md` (the path to ≥90%).

## REQUIRED: maintain the case study
`docs/CASE_STUDY.md` is a first-person engineering narrative kept for the user's portfolio.
**After each meaningful milestone, append a dated first-person entry** (and update the "Maintenance log"
list at the bottom). An entry states: what I set out to do, the key decisions and tradeoffs, the
**measured result with real numbers**, and what I learned. Match the existing voice (reflective, honest,
portfolio-grade — not a dry changelog). Never overwrite history; only append. If a later result
contradicts an earlier claim, record the correction rather than editing the past entry.

## REQUIRED: maintain the drift log
`docs/EXPERIMENT_LOG.md` is the expected-vs-actual scoreboard + drift tripwires (catches the moment we
diverge from correct patterns). **After every training run or material pipeline change, append a row to
§C**: what changed, the number you EXPECTED *before* running, the ACTUAL, a drift flag against §B, and the
decision. If any §B trigger fires (LODO AUROC jump >+0.05, worst-fold sensitivity drop, sensitivity rising
*with* site-leak/falling localization, permuted-label LODO ≥0.60, a knob tuned on the LODO test folds, or a
doc claim exceeding the evidence) — STOP, investigate, and record the outcome before trusting the number.
