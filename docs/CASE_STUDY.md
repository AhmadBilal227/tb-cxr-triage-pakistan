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

## Milestone 11 — The Shenzhen "collapse" was a threshold, not the model (2026-05-24)

The baseline's Shenzhen fold looked alarming: 31% sensitivity. The instinct is to throw data at it. I checked first whether it was a *discrimination* problem or a *calibration* one — because the AUC (0.82–0.85) didn't match a 31% sensitivity. A model that ranks TB vs normal at 0.82 AUC but catches only a third of cases isn't weak; it's sitting at the wrong operating point.

So I changed the honest scoreboard: every LODO fold now reports **two** sensitivities — **cold-start** (the frozen train-derived threshold, no local adaptation) and **+ local recalibration** (threshold fit on a small disjoint slice of the held-out site, evaluated on the rest, simulating "deploy with a little local labeled data"). The result settled it:

| Held-out site | Cold-start sens | + local recalibration | Spec cost |
|---|---|---|---|
| Shenzhen | 33.9% | **89.5%** [0.85–0.93] | 99% → 41% |
| Montgomery | 67.2% | 97.6% (n=41, wide CI) | 96% → 20% |
| Qatar | 78.8% | 92.4% | 95% → 59% |

Shenzhen wasn't broken — its threshold hadn't travelled. With a local calibration slice it hits 90% sensitivity, exactly what AUC 0.82 should permit. But the dual report also makes the *price* honest: reaching 90% sensitivity on Shenzhen drops specificity to 41% — a flood of false positives — because at this AUC you cannot have both. That specificity collapse is the real argument for more and better data: a higher AUC is what buys you 90% sensitivity *cheaply* (at acceptable specificity), and no amount of threshold tuning substitutes for it.

I also caught that head training was unseeded (~0.02 AUC run-to-run; the attention-mean wobbled 0.897 → 0.886 between runs) and pinned the seed, while noting the truthful figure is a band, not a point.

**Lesson:** when a safety-critical metric looks catastrophic, separate discrimination (AUC) from operating point (sensitivity at a threshold) *before* reaching for more data. Reporting the pair — cold-start floor and recalibrated-achievable, with the specificity each costs — tells the whole truth where a single number would have lied in one direction or the other.

---

## Milestone 12 — A reliability audit, and the day the honest numbers got harder (2026-05-24)

One line from the user — *"are the datasets normalized, will the different fields mess it up?"* — became the most important correction of the project. I ran a **site-leak canary** (can a trivial classifier name the *source dataset* from the frozen features?). It scored **balanced accuracy 1.000** against a 0.333 chance. The features were perfectly site-separable, which meant every LODO number to date was suspect: the head could be reading scanner/resolution, not TB.

So I put the whole pipeline through a **four-agent literature audit** (data harmonization, evaluation methodology, ML/calibration math, foundation-model input contracts) and then had **gpt-5.5 steelman** my proposed fixes. The convergent verdict was humbling and clarifying.

**The plumbing was right.** No register-token bug in Rad-DINO, the patch-grid slice and the mean/std were correct, the attention-MIL math and the conformal index were sound. The foundation held.

**But two of my own planned fixes were wrong turns,** and the steelman caught them. I wanted to **Gaussian-blur** images to erase the resolution shortcut — gpt-5.5 pointed out I'd be suppressing the exact high-frequency signal TB sensitivity depends on (miliary nodules, faint apical infiltrates), trading away the safety-critical metric to make a diagnostic number prettier. And **chasing the site-canary to chance** was optimizing the wrong thing: foundation features always encode acquisition; the goal is external validity and threshold transfer, not site-invisible features. I cut both. The discipline that mattered most was deleting my own clever idea.

**What I kept and shipped:** dropped CLAHE for a monotonic normalization closer to Rad-DINO's pretraining; antialiased resolution standardization (no blur); detect-and-*log* inversion (no silent pixel-flipping — medical-grade needs the DICOM tag); true TorchXRayVision logits; a cross-source **provenance match-graph** in dedup with a leak-guard so re-mixed wrappers can't span the train/test split, and cross-label clusters raised as a data-quality alarm; dropped label smoothing (it was distorting the very probability we fuse); **temperature calibration with ECE reported**; **bootstrap AUROC CIs**; and **PPV/NPV with confirmatory-tests-per-case at 1–2% prevalence**.

That last number matters most and flatters least: at a 0.93-sensitivity operating point, **1% prevalence gives ~2% PPV and ~48 confirmatory tests per flagged case**. A model can post 0.90 AUROC and still send 48 people for confirmation per true case found. That's the honest cost of screening at low prevalence, and it's in the report now.

**The hardest correction was semantic, not numerical:** my labels are *radiographic*, not microbiological. The system detects a *radiographic pattern associated with TB labels* — not confirmed *active* TB. I reframed the endpoint everywhere to say exactly that.

**And the "medical-grade" question.** The user asked for a medical-grade system. The honest answer, which gpt-5.5 put bluntly and I have to record: you cannot get there by patching a frontend BYOK prototype on open radiographic data. Medical-grade needs microbiological ground truth, prospective multi-site validation, a locked server-side model (not a browser calling third-party APIs that can change under you), an LLM kept *out* of the decision path, and a QMS/regulatory program (ISO 13485, IEC 62304, ISO 14971, an FDA/CE pathway). What this project can honestly be: *built with medical-grade engineering discipline, clearly labeled not a device, with a documented path to clearance.* I'd rather ship that sentence than overclaim.

Measured after the fixes (2,177-image subset, backbones unchanged): fusion-only **0.871 → +attention 0.905** LODO AUROC — the attention lever survived every correction. The full 4-source re-run on the harmonized pipeline is the next number.

**Lesson:** a one-line user question was worth more than any metric. The site-leak canary is the cheapest honest instrument I added — and reliability came from being willing to delete my own clever fix when the steelman showed it traded sensitivity for a prettier diagnostic.

---

## Milestone 13 — The full validated run, and the attention lever deflates (2026-05-24)

The audit-fixed pipeline ran end to end on all four sources — **13,092 deduplicated images**, harmonized preprocessing, the dup-cluster leak-guard on, bootstrap CIs, prevalence-aware reporting. The honest 4-source numbers:

- **Fusion-only mean LODO AUROC 0.916; fusion+attention 0.924.** Strong, in the commercial external band — but read it through the leak flags, not the mean.
- **The attention-over-patches lever deflated to +0.008.** It looked like the headline win on the 2,177-image subset (+0.034), and the commercial literature credited patch-attention with +6 AUROC. At full scale it's inside the bootstrap CIs — marginal. More data made the CLS+TorchXRayVision fusion strong enough on its own that attention added almost nothing (it still helped TBX11K +0.024 but was neutral-to-negative on Montgomery). I'm recording the deflation, not the earlier hope: at this scale, attention pooling is a rounding error here, not the lever I'd billed since M9b.
- **Shenzhen's cold-start "collapse" from M11 is gone** — 34% sensitivity at a frozen threshold → **79%** on the full run. Harmonized intensity + dropping label smoothing + temperature scaling + more training data fixed the threshold transfer that a local recalibration had previously been needed to rescue.
- **Site-leak fell 1.000 → 0.945** (4 sources, chance 0.25): the features are still highly site-separable, exactly as the steelman warned — so I trust the cleaner-external folds (Montgomery AUROC **0.88**, Shenzhen **0.94**) and treat the re-mix folds (Qatar/TBX11K, 0.94) as leakage-inflated.
- **The deployment number, on the cleaner externals** (sens 0.94, spec 0.50): at 1% prevalence, **1.9% PPV and ~54 confirmatory tests per flagged case**; cross-site calibration still imperfect (Montgomery ECE 0.21). A radiographic-pattern detector, not an active-TB diagnosis.

**Lesson:** the subset lied about the attention lever; the full run + bootstrap CIs caught it. The cheapest insurance against a flattering result is more held-out data and a confidence interval — plus the willingness to write down "the thing I thought mattered, doesn't."

---

## Milestone 14 — A review swarm, and the correction I owed (2026-05-24)

Before building the T2 "sharpening" heads, I put the work through a deliberate gauntlet: gpt-5.5 + 4 code/methodology reviewers (debugger, ML engineer, code-quality, architecture) + 3 data auditors (form, original→vectorized correctness, training-fitness). It caught two silent bugs and one of *my* overclaims.

**Two coordinate bugs the unit tests couldn't see.** The Rad-DINO processor does shortest-edge-resize **+ center-crop** to 518², so for non-square crops the patch tokens covered only the centre square while the box/zone labels were built over the full cropped rectangle — a systematic misregistration that would have corrupted exactly the apical/costophrenic supervision, and ran clean through 9 square-crop unit tests. Fixed by letterboxing image+masks+boxes to a common square. The second was subtler: the token-order assert was a *tautology* (it only proved numpy reshapes row-major, not that Rad-DINO *emits* row-major). I added a real empirical test — feed a white-top/black-bottom image through the live model — and it **confirmed Rad-DINO is row-major** (token norms separate ~6.4× more along rows than columns). The box/zone geometry is now verified, not assumed.

**The correction I owed.** In M13 I read the label-randomization result (0.505 ≈ chance) as proof the model uses "genuine TB signal, not a site shortcut." The methodology reviewer showed that doesn't follow, and the data auditor proved why with numbers: **site is 98% recoverable from the features and TB prevalence swings 6.3× across sources** (tbx11k 8% → shenzhen 51%). Because site and prevalence are confounded, a model exploiting "this looks like Shenzhen → 51% prior" would *also* collapse under label permutation — so 0.505 only rules out label *memorization*, not site-keyed shortcutting. The honest position: the randomization probe is necessary-not-sufficient, and the **lesion-localization harness + provenance-independent external validity are the real discriminators**. I'd rather record the walk-back than keep the flattering reading.

**What the data audit also settled — some of it reassuring.** The vectorized data is well-formed (no NaN/inf; the 18 TorchXRayVision values are真 logits, not sigmoid'd) and the original→features combination is faithful (0 label/alignment mismatches across 13,092 rows). And LODO is a *genuine* external test: the TB **positives are disjoint across sources** (Qatar-positive vs NLM max cosine 0.78; cross-fold leakage ≤0.12% and entirely negatives) — the re-mix-leakage fear is smaller than I'd assumed at the positive level. The standing risk is the site-confound, not positive leakage.

**The plan got re-sequenced.** I had labelled the box-supervised head "the #1 lever." The ML review (and my own §7) say it isn't: all box supervision is 100% TBX11K (a leakage-prone fold), so it buys interpretability and a faithful evidence map, not external recall. The real external-sensitivity levers, re-ranked: **the native 37×37 patch grid + letterbox** (our 8×8 pool averages a TB focus into a 65px cell, discarding the subtle/miliary/subclinical signal that caps sensitivity), then the **lower-zone-floored zonal soft-OR**, then **active-vs-healed** for usable specificity. And a process guardrail: with T2's many loss-weight knobs, tuning any of them against the LODO test folds would optimistically bias the headline — so weights are prior-fixed / nested-CV'd, never tuned on the test fold.

**Safety correction to the verdict.** A confident **NO** on a subclinically-positive, normal-looking film is the one place the verdict layer can falsely reassure with nothing in the image to flag (~40–50% of prevalent TB is subclinical). No guardrail can catch it, so every NO now carries a hard-wired *"cannot rule out subclinical/early TB — test symptomatic patients regardless."*

**Lesson:** a review swarm earns its cost when it deletes your own clever idea (the blur, twice), catches a bug your tests structurally can't (square-only crops), and makes you walk back a conclusion you'd already written down (0.505 ≠ "no shortcut"). The honesty infrastructure works only if you let it correct *you*, not just the code.

---

## What this project demonstrates (for the portfolio reader)

- **Honest ML evaluation.** I measured real sensitivity/specificity/AUC against ground truth and led with the uncomfortable number (14%), then improved it methodically and re-measured. No cherry-picked accuracy.
- **Systems thinking under constraints.** A coherent three-provider abstraction, visible fallback, and a deterministic safety net wrapping an LLM — frontend-only, BYOK, strict-typed.
- **Research → implementation.** Parallel agent swarms to map a solution space, distilled into a phased plan grounded in 2025–26 literature with citations.
- **Engineering discipline.** Subagent-driven execution with spec + code-quality review gates that caught real correctness bugs before they shipped.
- **Calibrated, safety-first ML.** Conformal thresholds that *target* a sensitivity level (finite-sample, in-distribution coverage — re-fit per site, reported with CIs; **not** a guarantee under deployment shift), with a guard for the small-sample regime — the difference between a demo and something you'd let near a screening workflow.
- **Knowing the ceiling.** Clear-eyed about what a frontend BYOK app with open *radiographic-labeled* data can realistically reach vs. where microbiological labels, a locked server-side model, and clinical validation are required for an actual medical-grade claim — and honest that this is a research preview, **not a medical device**.

---

## Maintenance log

*Append a dated, first-person entry after each meaningful milestone: what I set out to do, the key decisions and tradeoffs, the measured result, and what I learned. Keep the honest numbers in.*

- **2026-05-23** — Built the full frontend app (Milestone 1); Lighthouse a11y 100.
- **2026-05-24** — Live integration + real-data findings (M2); measured 14% baseline sensitivity (M3); VLM improvements → 42% (M4); 90%-path research swarms (M5); calibration core shipped via subagent-driven review gates (M6); started the real-classifier training pipeline, expanded to multi-task best-quality (M7).
- **2026-05-24** — Verification/hardening pass (M8): agent + GPT-5.5 dual review. Fixed a fail-open (empty-ensemble → "NO TB"), verdict/confidence coercion, UI-stuck stage statuses, and training-script robustness. Build + 11/11 tests green.
- **2026-05-24** — Commercial/SOTA improvement study (M9b): studied how CAD4TB/qXR/Lunit/Google TB CAD reach their accuracy. Their edge (data/label/pretraining scale) isn't copyable, but 3 architecture/calibration levers are — top one: an **attention-pooled head over Rad-DINO patch tokens** (documented +6 AUROC on frozen Rad-DINO; we were using CLS only) — plus lung-crop front-end and per-site threshold calibration. Added an accuracy-improvement roadmap to the plan. (No new measured numbers yet.)
- **2026-05-24** — First trained-model baseline (M10): 2,177-img 3-source LODO. Fusion-only AUC 0.858 → +attention 0.897 (validates the attention head). Recorded to `docs/baselines/`. Strong but optimistic (Qatar re-mix leakage, radiographic labels, subsampled).
- **2026-05-24** — Shenzhen fix (M11): the 31% sensitivity was threshold-transfer, not model quality. Added a dual sensitivity report (cold-start vs + local recalibration); Shenzhen 34% → 90% with a local calibration slice (spec cost 99% → 41%). Seeded training. TBX11K landed (11,702 imgs → ~16k full set).
- **2026-05-24** — Full validated 4-source run (M13): 13,092 imgs, harmonized + leak-guarded. Mean LODO AUROC fusion-only 0.916 → +attention **0.924**, but the attention lever **deflated to +0.008** at scale (was +0.034 on the subset; literature claimed +6) — recorded the correction. Shenzhen cold-start recovered 34% → **79%**. Site-leak 1.000 → 0.945. Cleaner-external folds: Montgomery 0.88, Shenzhen 0.94. Deployment: 1% prevalence ⇒ 1.9% PPV, ~54 tests/flagged case (radiographic endpoint).
- **2026-05-24** — Review swarm (M14): gpt-5.5 + 4 method/code reviewers + 3 data auditors on the T2 work. Caught 2 silent geometry bugs (center-crop misregistration → letterbox fix; tautological token-order assert → real test, **confirmed Rad-DINO row-major**). **Corrected M13's overclaim**: label-randomization 0.505 only rules out *memorization*, not site-shortcutting (site 98% recoverable, prevalence swings 6.3× across sources) — localization is the real anti-shortcut discriminator. Reassuring: LODO is genuinely external (TB positives disjoint across sources, cross-fold leak ≤0.12% all-negative). Data verified well-formed + faithfully combined (0 mismatches; TXRV are真 logits). Re-sequenced the plan (37×37+letterbox > box head; box supervision is 100% TBX11K → interpretability not recall); prior-fixed loss weights (no tuning on test folds); hard-wired subclinical disclaimer on every NO. Code fixes: determinism, recal K-splits, skip counters, worst-fold reporting.
- **2026-05-24** — Literature swarm (10 agents, Waves 1–2) → **TB feature-SHARPENING blueprint** (`docs/superpowers/plans/2026-05-24-tb-sharpening-blueprint.md`). Direction: sharpen TB-specific features (not domain-generalize). Plan: box-supervised spatial evidence head (TBX11K boxes), lower-zone-floored zonal pooling (don't anchor upper-lobe — diabetic lower-zone TB is the top South-Asian miss), pathology grounding + distillation, active-vs-healed sequelae head, and a validation harness (localization mIoU/hit-rate vs boxes + Adebayo label-randomization shortcut test). Corrections: TBX11K "latent" = inactive-sequelae *misnomer* (probe = 169 unique imgs); TBX11K boxes are usable; TXRV has no calcification logit. No measured numbers yet — blueprint stage.
- **2026-05-24** — Reliability audit + steelman (M12): the site-leak canary scored **1.000** (perfect), exposing a scanner/resolution shortcut. 4-agent literature audit + gpt-5.5 steelman. Verified the foundation-model plumbing correct; fixed preprocessing (drop CLAHE→monotonic norm, antialiased resolution, detect-only inversion, true TXRV logits, same input to both backbones) and eval (provenance match-graph leak-guard, drop label smoothing, temperature+ECE, bootstrap AUC CIs, PPV-at-prevalence). **Cut two wrong turns** (Gaussian blur, chasing the canary to chance). Reframed endpoint to "radiographic-TB-pattern" (not active TB). Recorded the honest medical-grade gap (frontend BYOK + radiographic labels ≠ device). +attention held: 0.871 → 0.905 AUROC; at sens 0.93/spec 0.56, 1% prevalence ⇒ 2.1% PPV, ~48 tests/flagged case.
- **2026-05-24** — Expert-panel science validation (M9): 6-lens review (methodologist/radiologist/epidemiologist/steelman/literature/red-team). Architecture validated against literature; corrected overclaims — conformal guarantee → in-distribution + re-fit-per-site; dropped user-facing "latent TB" class; radiographic ≠ bacteriological labels; added PPV-at-prevalence honesty; soft-mask preprocessing; dedup/LODO leakage hardening. Plan + ship-gate rewritten.
