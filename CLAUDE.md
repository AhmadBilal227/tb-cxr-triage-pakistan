# Project: AI-Native TB Chest X-Ray Triage

Frontend-only (Vite + React 18 + TypeScript strict, **no `any`**), BYOK, IndexedDB. Three providers
called directly from the browser. Research preview — **not a medical device.**

**Perception path as of M21 (2026-05-24):** OpenAI `gpt-5.5` vision via the Responses API's
structured-output mode IS the PRIMARY perception (see `src/lib/pipeline/vlmTriage.ts`). Hugging
Face / Replicate classifier heads remain best-effort fallbacks but the free hf-inference router
no longer hosts `microsoft/rad-dino` or the M1 default TB heads, so on the deployed app today
they are inert. The local validated ONNX heads in `public/models/` (M19: AUROC 0.922 LODO) are
on disk but cannot execute in the browser without a feature-extraction backbone (Phase B gap).
This is the honest tradeoff M21 made: deployability up, measured accuracy down vs the offline
trained heads.

## Non-negotiable ethos (carry this into every change)
- **Report real numbers.** Measure against ground truth (the `/validate` route + `scripts/accuracy-test*.mjs`). Lead with the honest metric (sensitivity is the safety-critical one for a screen), never a flattering one. The project's whole identity is intellectual honesty about model quality.
- **Don't overclaim (per the M9 expert panel — see CASE_STUDY.md).** The conformal layer does NOT "guarantee ≥90% sensitivity": its coverage is in-distribution + finite-sample and must be re-fit on labeled data from each deployment site, reported with a binomial CI. Open-dataset metrics are against **radiographic** labels, not bacteriological confirmation — say so. WHO 90/70 is a **floor**, not a target. No ≥90% sensitivity claim without ~150+ held-out TB positives. Don't surface "active vs latent TB" from a single film (latent TB is radiographically silent). Report PPV at deployment prevalence, not accuracy on balanced sets.
- **Fallback and degradation are always visible** to the user, never hidden.
- **The deterministic safety net wraps the LLM** — the model advises, guardrails decide; it can escalate but never clear a flagged case on weak evidence.
- Keep strict TS clean (`npm run build`), tests green (`npm test`), and a11y ≥95.

## Orientation
- Contract: `src/lib/types.ts` (`ClassifierResult` = `{ tb_prob, raw, provider_used, latency_ms }`; `Adjudication.perception_path` + `vlm_audit` since M21).
- Providers: `src/lib/providers/` (`classify.ts` = the HF→Replicate fallback seam; `openai.ts` = Responses API including structured-output / json_schema mode used by M21).
- Pipeline: `src/lib/pipeline/orchestrator.ts` (M21: gpt-5.5 vision PRIMARY via vlmTriage; HF heads auxiliary; safety-net combine via applyVlmEscalation).
- VLM triage (M21): `src/lib/pipeline/vlmTriage.ts` (the `submit_triage` JSON schema + boring policy prompt + forced-abstain rails + borderline-band predicate) and `src/lib/pipeline/vlmEscalation.ts` (path-specific escalation, SEPARATE 0.5 threshold from the ONNX path's 0.7126 — DO NOT mix).
- Sequelae escalation (M19, Phase B): `src/lib/pipeline/sequelaeEscalation.ts` — interface only today; consumes `s_inactive` from `public/models/sequelae_head.onnx` once a browser-side feature pathway exists.
- Calibration: `src/lib/calibration.ts` (temperature/Platt + log-odds fusion + conformal; fit via `/validate` Calibrate). On the VLM path the calibration is bypassed — the VLM score is uncalibrated by definition.
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
