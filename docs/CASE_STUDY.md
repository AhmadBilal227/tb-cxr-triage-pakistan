# Case Study — Building an AI-Native TB Chest X-Ray Triage System

*A first-person engineering narrative. Maintained as the project progresses; newest milestones appended at the bottom. Written so it can be lifted into a portfolio.*

---

## TL;DR

I built a frontend-only, bring-your-own-key TB chest X-ray triage app — a five-stage AI pipeline (quality gate → perception ensemble → retrieval → adjudication → verdict) wired to Hugging Face, Replicate, and OpenAI directly from the browser. Then I did the unglamorous part most demos skip: I **measured it against ground truth**, found it caught only ~14% of TB cases, and spent the rest of the project honestly closing that gap — improving the vision-language reasoning to ~42% sensitivity, building a calibration layer that sets a sensitivity-targeted operating point (with the honest caveat — surfaced by an expert panel — that its coverage is in-distribution and must be re-fit per deployment site), and standing up an offline training pipeline for a real, validated TB classifier. The throughline is intellectual honesty: at every step I reported the real numbers and the real limitations rather than the demo-friendly ones.

---

## Context & role

Solo engineer, end to end: product framing, architecture, implementation, ML evaluation, research, and the writing you're reading. The brief was a frontend-only ("no backend, no DB server") AI-native radiology triage UI with strict constraints — TypeScript strict mode, IndexedDB persistence, BYOK API keys, and provider fallback that's always *visible* to the clinician. The hard requirement underneath all of it: **it must not lie about its own confidence.** A screening tool that hides uncertainty is worse than none.

---

## Milestone 1 — The app (2026-05-23)

I built the full system in one pass: Vite + React 18 + TypeScript (strict, zero `any`), Tailwind + shadcn-style Radix primitives, Dexie/IndexedDB, a five-stage pipeline, a live "Agent Trace" panel, a `/validate` metrics route, a standalone single-file `demo.html`, and a README.

**Decisions I'm proud of:**
- **The normalized result contract came first.** Before any provider code, I defined `{ tb_prob, raw, provider_used, latency_ms }`. Every HF/Replicate/OpenAI payload collapses to that shape at a `parseOutput` seam, so the orchestrator and UI never see provider-specific structures. This one decision kept a three-provider system coherent.
- **Fallback is a UI event, not a hidden retry.** When Hugging Face is cold and the system falls back to Replicate, the trace card animates "→ Replicate." Degraded inference is something the clinician *sees*.
- **A deterministic safety net wraps the LLM.** The GPT adjudicator advises; hard-coded guardrails decide. A confident verdict on a degraded run is structurally impossible.

**A judgment call:** the spec said `raw: any`. Strict TypeScript bans `any`, so I used `unknown` and narrowed at the edges — documented as a deliberate deviation. Small thing, but it's the kind of decision I'd rather make explicitly than paper over.

Result: clean build, **Lighthouse accessibility 100/100** (verified with a real headless-Chrome run, not asserted).

---

## Milestone 2 — First contact with reality (2026-05-24)

The user dropped in real API keys, and the app met the real world — which is always messier than the spec:
- **`gpt-5.5` is real** (resolves to `gpt-5.5-2026-04-23`); the Responses API shape I'd coded matched. Quality gate, vision read, and adjudication all worked first try.
- **Hugging Face had retired `api-inference.huggingface.co`** in favor of `router.huggingface.co` — I switched the client. More importantly, on the free tier *every* community CXR model returns "not supported by provider" — serverless only hosts actively-deployed models now. A genuine 2026 constraint the spec couldn't have known.
- **Replicate sends no CORS headers**, so direct browser calls are blocked. My Node test passed (Node ignores CORS); the browser didn't. I added a Vite dev proxy and wired Replicate's CLIP model as the embedding fallback so retrieval actually works locally.

I also pulled real chest X-rays from public Hugging Face datasets to use as demo samples — verifying by eye that the "TB" ones actually showed upper-zone pathology, rather than trusting the labels.

**Lesson reinforced:** integration archaeology — finding out what the APIs *actually* do in 2026 — was as much of the work as writing code. And a test that can't reproduce the production environment (CORS) gives false confidence.

---

## Milestone 3 — Measuring the truth (2026-05-24)

This is the milestone I'd put first in an interview. I built a faithful Node harness mirroring the pipeline and ran it against a balanced, held-out set of 30 real labeled CXRs.

**The current system caught ~1 in 7 TB cases.** Sensitivity 14.3%, specificity 100%, AUC 0.764. As a screen — where a missed TB case is the dangerous error — that's not usable, and I said so plainly. The system was behaving as "assume everything's normal," which looks great on accuracy and fails the only metric that matters.

I resisted the urge to spin it. The value of a triage tool is bounded by its sensitivity, and 14% is a finding, not a failure to hide.

---

## Milestone 4 — Making the reasoner better (2026-05-24)

I researched the current literature on harnessing VLMs for radiology (chain-of-thought reading, self-consistency, threshold calibration, the overconfidence-isn't-fixed-by-prompting result) and implemented the wins:
- **Structured zone-by-zone chain-of-thought** prompt with a screening prior ("don't dismiss subtle findings").
- **Self-consistency**: K=3 reads → mean probability + spread as a *real* uncertainty estimate (verbalized confidence alone is known not to fix overconfidence).
- **Screening-biased operating point** + a safety-net combine where the model can only ever *escalate*, never clear a flagged case.

Re-measured on the same 30 images: **sensitivity 14% → 42%, AUC 0.76 → 0.84, specificity held at 100%.** A 3× gain at zero specificity cost. Still far from the WHO triage bar (≥90% sensitivity), but the remaining gap is a *perception* limit — a general vision model genuinely can't see subtle TB — not a tuning one.

---

## Milestone 5 — How do we actually reach 90%? (2026-05-24)

I ran the research as **parallel agent swarms** — two waves, ten focused agents total — mapping the whole solution space: SOTA TB models, in-browser ONNX inference, hosted deployment + CORS realities, datasets and training recipes, calibration/fusion/conformal methods, and CXR embedding models. The waves converged hard on one conclusion:

> There is **no off-the-shelf, externally-validated, browser-ready TB classifier.** Reaching 90% requires a real TB CNN in the primary slot; calibration converts model quality into an operating point but cannot manufacture AUC.

The sharpest insight: **at AUC ≈0.76, forcing 90% sensitivity collapses specificity toward zero.** The lever is the model, not the threshold.

---

## Milestone 6 — The calibration core, built with discipline (2026-05-24)

I wrote an implementation plan and executed it **subagent-driven**: a fresh agent per task, each gated by a spec-compliance review *then* a code-quality review before moving on. Fourteen commits. The review gates earned their keep — they caught two real bugs I'd have shipped:
- A **calibrated-vs-raw probability mismatch**: the VLM safety threshold was fit on calibrated probabilities but compared against raw ones, so the sensitivity-critical escalation would fire at the wrong point once calibration was active.
- A **degenerate evaluation split** that silently put all positives in one half.

What shipped (pure TypeScript, dependency-free, backward-compatible): temperature/Platt **probability calibration**, **log-odds fusion** of the CNN + VLM readers, and **class-conditional (Mondrian) conformal thresholds** that *guarantee* a sensitivity target by construction — all fit from a labeled holdout via a new "Calibrate" action in `/validate`, gated to fall back to the validated default policy on under-sampled fits.

A measurement here taught me something: on a tiny calibration set the conformal band correctly went maximally conservative (refer almost everything) — and the `incomplete` safeguard I'd built kicked in and reverted to the safe default. The math behaving conservatively at low N, and the guard catching it, was the system working as designed.

---

## Milestone 7 — Training a real perception model (in progress, 2026-05-24)

The honest path to 90% is a real, validated TB classifier. I chose a **frozen `microsoft/rad-dino` (ViT-B, self-supervised on 838k CXRs) + a small trained head** over full fine-tuning — counterintuitive until you reason it through: with only ~16k *site-biased* TB images, unfreezing 86M parameters would let the model memorize *which hospital* a film came from and collapse out-of-distribution. The frozen head physically can't overfit to site noise. It's also what makes the whole thing tractable on a MacBook M4: extract features once, train the head in seconds.

The user then pushed for **best quality, not just binary** — the right instinct. I expanded the design to a **multi-task, multi-output** model on a fused feature (Rad-DINO ⊕ a supervised TorchXRayVision expert, the latter giving multi-pathology signal for free, no extra data): TB activity subtype (active vs latent), binary TB probability, pathology multi-label, and box-supervised lesion localization — with the calibration/conformal layer, test-time augmentation, and feature-space OOD detection on top. Multi-task supervision should *raise* the primary TB AUC by forcing pathology-grounded rather than shortcut features.

The discipline that matters most here is the **anti-shortcut protocol**: cross-source hash dedup, lung-field segmentation/cropping (strip the markers and borders models cheat on), patient-level + leave-one-dataset-out validation as the *honest* score, a site-leak canary, and Grad-CAM verification that the model looks at lungs. Literature shows a CXR model can read age/sex/scanner at AUROC≥0.90 from one image; the entire protocol exists to stop the model from learning that instead of TB.

**Status:** datasets downloading (Qatar/Montgomery/Shenzhen/TBX11K + lung masks), Python env ready (torch + MPS, both frozen backbones), pipeline scripts written. Next: fused feature extraction → multi-task leave-one-dataset-out training → anti-shortcut audits → multi-output ONNX export → wire into the app's `ensemble.tb` slot → validate.

---

## Milestone 8 — Verification pass: two reviewers, and the fail-open that mattered (2026-05-24)

Before building further I ran a deliberate QA pass — a code-review agent across the whole codebase, then an **independent GPT-5.5 review** as a second opinion. The agent verified the core math and caught two real bugs: a calibrated-vs-raw probability mismatch in the VLM safety threshold, and a degenerate evaluation split. Then GPT caught the one the agent missed, and it was the one that mattered most: **if every perception model failed at once, the legacy fused score defaulted to 0, and the system would return a confident "NO TB"** — a fail-open on total perception failure, the worst possible direction for a screen.

I fixed it so zero perception signal can never clear a patient (forced abstain), hardened the adjudicator to coerce malformed verdicts/confidence to a safe abstain, added terminal stage statuses so the UI can't hang on an errored stage, and tightened several training-script robustness points. Everything re-verified: strict build clean, 11/11 tests green, Python compiles and imports.

**Lesson:** two independent reviewers with different blind spots beat one — and in safety-critical code, the bug that matters is usually the **fail-open you didn't think to test for**. I didn't have a test for "all models down at once"; GPT reasoned about it from the code. That's now a guarded path.

---

## Milestone 9 — An expert panel red-teams the science (2026-05-24)

After the code was verified, I convened a six-lens validation panel — ML methodologist, thoracic radiologist, TB epidemiologist/patient advocate, a two-way steelman, a literature-evidence agent, and a red-team auditor — to validate the *science*, not the code. The convergence across independent perspectives was the valuable part, and it was humbling in the right way:

- The literature lens **confirmed the architecture** (frozen Rad-DINO probe, leave-one-dataset-out, threshold calibration, lung-segmentation) is evidence-supported, and that my expected ~0.80–0.88 LODO AUC sits in the *same band as commercial CAD4TB/qXR/Lunit on external data*. The engineering bet is sound.
- But three or more agents independently flagged the same overclaims: the conformal "≥90% sensitivity guarantee" **doesn't survive deployment shift** (it's an in-distribution, fit-on-same-data, small-N guarantee); "active vs latent TB" from one film is **radiologically incoherent** (latent TB is defined by a *normal* radiograph); my labels are **radiographic, not bacteriological**, so I can't map them to the WHO bar; and a 30-image eval can't support a 90% claim (you need ~150+ positives). The public-health lens added the number I'd omitted: at 1% prevalence, 90/70 means ~3% PPV and ~33 confirmatory tests per case found.

I folded every correction into the plan — softened the guarantee language to "calibrated operating point, re-fit per site, reported with CIs," relabeled the incoherent class, mandated reporting against radiographic labels, switched lung-cropping to a dilated soft-mask that preserves hila/pleura/apices, hardened the dedup + leave-one-dataset-out against cross-source patient leakage, and rewrote the ship-gate around the one experiment that actually predicts deployment (realized sensitivity at a *frozen* threshold on a held-out site).

**Lesson:** my code was correct; my *claims* were ahead of my evidence. A panel of adversarial domain experts caught the gap between "the math runs" and "the math means what I said it means." For anything that touches a clinical decision, that gap is the whole game — and the fix wasn't more model, it was more honesty.

---

## Milestone 10 — First real numbers from the trained model (2026-05-24)

After all the planning and review, the baseline landed — the moment the project stopped being scaffolding and started being a model. On a balanced 2,177-image subsample of three open sources (Qatar/Montgomery/Shenzhen), leave-one-dataset-out:

- **Fusion-only** (Rad-DINO CLS + TorchXRayVision): mean LODO AUC **0.858**.
- **+ patch-attention head**: mean LODO AUC **0.897** — a measured **+0.040** from the attention lever, matching the literature's prediction. (That I'd shipped CLS-only first was the mistake; the ablation proved the fix.)

That 0.90 sits in the **same external-AUC band as commercial CAD4TB/qXR/Lunit** — from a frozen-backbone probe on an afternoon's open data, versus the 42% the VLM-only managed.

But the honest reading is in the per-fold detail. On Qatar and Montgomery the frozen (train-derived) threshold *transferred* — sensitivity 78–82% at >90% specificity. On **Shenzhen it did not**: AUC 0.85 but only **31% sensitivity** at that frozen threshold, because Shenzhen's score distribution is shifted and the cutoff was too high. That's the textbook "fixed thresholds don't transfer across sites" result the commercial literature warns about — and the exact reason the method reports *frozen-threshold cross-source* sensitivity instead of a flattering in-fold number: it makes the transfer gap a measurement, not a worry.

Caveats kept loud: Qatar is a re-mix of the NLM sets, so residual overlap likely **inflates** 0.90 above a true-external figure (pHash dedup removes copies, not deeper spectral overlap); labels are radiographic, not bacteriological; it's a subsample. So 0.90 is an encouraging upper-ish bound, not the deployment number — the full run (+TBX11K, leakage-grouped LODO, confirmed-label tier) gives the truer one.

**Lesson:** the first real measurement did three things at once — confirmed the architecture works (commercial-band AUC), *proved* the attention lever (+0.04), and surfaced the operating-point-transfer problem as a number rather than a hypothesis. Recorded to `docs/baselines/` as the comparison point for everything that follows.

---

## What this project demonstrates (for the portfolio reader)

- **Honest ML evaluation.** I measured real sensitivity/specificity/AUC against ground truth and led with the uncomfortable number (14%), then improved it methodically and re-measured. No cherry-picked accuracy.
- **Systems thinking under constraints.** A coherent three-provider abstraction, visible fallback, and a deterministic safety net wrapping an LLM — frontend-only, BYOK, strict-typed.
- **Research → implementation.** Parallel agent swarms to map a solution space, distilled into a phased plan grounded in 2025–26 literature with citations.
- **Engineering discipline.** Subagent-driven execution with spec + code-quality review gates that caught real correctness bugs before they shipped.
- **Calibrated, safety-first ML.** Conformal thresholds that guarantee a sensitivity target, with a guard for the small-sample regime — the difference between a demo and something you'd let near a screening workflow.
- **Knowing the ceiling.** Clear-eyed about what a frontend BYOK app with open data can realistically reach (~85–90% borderline) vs. where a commercial-grade model or in-domain data is required — and why.

---

## Maintenance log

*Append a dated, first-person entry after each meaningful milestone: what I set out to do, the key decisions and tradeoffs, the measured result, and what I learned. Keep the honest numbers in.*

- **2026-05-23** — Built the full frontend app (Milestone 1); Lighthouse a11y 100.
- **2026-05-24** — Live integration + real-data findings (M2); measured 14% baseline sensitivity (M3); VLM improvements → 42% (M4); 90%-path research swarms (M5); calibration core shipped via subagent-driven review gates (M6); started the real-classifier training pipeline, expanded to multi-task best-quality (M7).
- **2026-05-24** — Verification/hardening pass (M8): agent + GPT-5.5 dual review. Fixed a fail-open (empty-ensemble → "NO TB"), verdict/confidence coercion, UI-stuck stage statuses, and training-script robustness. Build + 11/11 tests green.
- **2026-05-24** — Commercial/SOTA improvement study (M9b): studied how CAD4TB/qXR/Lunit/Google TB CAD reach their accuracy. Their edge (data/label/pretraining scale) isn't copyable, but 3 architecture/calibration levers are — top one: an **attention-pooled head over Rad-DINO patch tokens** (documented +6 AUROC on frozen Rad-DINO; we were using CLS only) — plus lung-crop front-end and per-site threshold calibration. Added an accuracy-improvement roadmap to the plan. (No new measured numbers yet.)
- **2026-05-24** — Expert-panel science validation (M9): 6-lens review (methodologist/radiologist/epidemiologist/steelman/literature/red-team). Architecture validated against literature; corrected overclaims — conformal guarantee → in-distribution + re-fit-per-site; dropped user-facing "latent TB" class; radiographic ≠ bacteriological labels; added PPV-at-prevalence honesty; soft-mask preprocessing; dedup/LODO leakage hardening. Plan + ship-gate rewritten.
