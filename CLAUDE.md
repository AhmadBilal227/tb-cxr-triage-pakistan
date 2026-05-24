# Project: AI-Native TB Chest X-Ray Triage

Frontend-only (Vite + React 18 + TypeScript strict, **no `any`**), BYOK, IndexedDB. Two runtime
providers called directly from the browser (**OpenAI** + an optional **Replicate** BYO slot) plus the
M22 **local-mode FastAPI server** running the validated trained model. Research preview — **not a
medical device.**

**Perception path as of M23 (2026-05-25 — HF removed):** TWO paths, chosen at runtime by
`settings.localMode`.

- **LOCAL MODE (M22, primary on the user's machine):** when `settings.localMode === true` AND the FastAPI
  server at `settings.localServerUrl` (default `http://localhost:8001` in dev, `:8000` in prod) is
  reachable, the **validated pipeline** runs (Rad-DINO + TorchXRayVision + `TBHeadT2` +
  `InactiveSequelaeHead` under their fitted calibration constants from `data/tb_threshold_t2.json` +
  `data/tb_inactive_meta.json`). `tb_prob` comes back **calibrated under T**; the verdict is decided
  at the validated `thr_at_95sens=0.6105`; the M19 `applySequelaeEscalation` consumes `s_inactive`;
  gpt-5.5 vision is reduced to a BORDERLINE second-opinion verifier that fires only on
  `tb_prob ∈ [0.35, 0.65]` OR `s_inactive ≥ 0.7126`, and forces ABSTAIN on disagreement (never
  overrides a confident verdict). Single-source-of-truth Python module: `training/triage_core.py`.
  CORS narrow (localhost:5173 + 127.0.0.1:5173 only). `Adjudication.perception_path =
  'local-onnx-via-server'`.

- **VLM-PRIMARY (M21, deployed-app default):** the deployed Netlify app cannot reach a user's
  localhost, so it stays on the M21 path: OpenAI `gpt-5.5` vision via the Responses API's
  structured-output mode IS the primary perception (see `src/lib/pipeline/vlmTriage.ts`). The local
  validated ONNX heads in `public/models/` (M19: AUROC 0.922 LODO) are on disk but cannot execute in
  the browser without a feature-extraction backbone (Phase B gap).

**M23 cleanup (2026-05-25 — HF gone):** Hugging Face was removed from the runtime path entirely.
The free hf-inference router dropped every default classifier we relied on (`Owos/tb-classifier`,
`keremberke/yolov8m-chest-xray-classification`) and the backbones (`microsoft/rad-dino`,
`torchxrayvision/...`). We had a strictly better path on user machines (M22 local mode) and a working
deployed default (M21 VLM), so HF in the browser was dead weight that surfaced red errors. The HF
provider client, settings field, status badge, model-id overrides, and orchestrator auxiliary stages
were all deleted. The `training/` directory still loads Rad-DINO + TXRV via the HF Python library
**offline** from `~/.cache/huggingface/` — that is a file-system read at engine startup, not a
runtime API call.

The orchestrator falls through from LOCAL to VLM-PRIMARY if the local server is unreachable, so a
user who toggles "Local mode" but forgets to start the server still gets a usable run.

## Non-negotiable ethos (carry this into every change)
- **Report real numbers.** Measure against ground truth (the `/validate` route + `scripts/accuracy-test*.mjs`). Lead with the honest metric (sensitivity is the safety-critical one for a screen), never a flattering one. The project's whole identity is intellectual honesty about model quality.
- **Don't overclaim (per the M9 expert panel — see CASE_STUDY.md).** The conformal layer does NOT "guarantee ≥90% sensitivity": its coverage is in-distribution + finite-sample and must be re-fit on labeled data from each deployment site, reported with a binomial CI. Open-dataset metrics are against **radiographic** labels, not bacteriological confirmation — say so. WHO 90/70 is a **floor**, not a target. No ≥90% sensitivity claim without ~150+ held-out TB positives. Don't surface "active vs latent TB" from a single film (latent TB is radiographically silent). Report PPV at deployment prevalence, not accuracy on balanced sets.
- **Fallback and degradation are always visible** to the user, never hidden.
- **The deterministic safety net wraps the LLM** — the model advises, guardrails decide; it can escalate but never clear a flagged case on weak evidence.
- Keep strict TS clean (`npm run build`), tests green (`npm test`), and a11y ≥95.

## Orientation
- Contract: `src/lib/types.ts` (`ClassifierResult` = `{ tb_prob, raw, provider_used, latency_ms }`; `Provider = 'replicate' | 'openai' | 'local-triage'` post-M23; `Adjudication.perception_path` = `'local-onnx-via-server' | 'vlm-primary' | 'onnx-primary' | 'perception-unavailable'`; `Settings.localMode` + `localServerUrl` since M22; M23 removed `Settings.hfToken` + the HF model-id override fields).
- Providers: `src/lib/providers/` (`classify.ts` = Replicate-only embedding seam post-M23 — `classifyWithFallback` was deleted with HF; `openai.ts` = Responses API including structured-output / json_schema; **`localTriage.ts` (M22)** = POST `/triage` to the local FastAPI server with four-way error tagging in `providerStatusStore`).
- Pipeline: `src/lib/pipeline/orchestrator.ts` — M22+M23 single-path perception: when `settings.localMode === true` AND the local server is reachable, `localTriage` IS the primary perception and gpt-5.5 vision is a BORDERLINE verifier (consistency-check disagreement forces ABSTAIN); otherwise the M21 VLM-primary path runs. There is no longer an "ensemble" — M23 deleted the HF auxiliary stages.
- Local triage core (M22, server-side): `training/triage_core.py` (single source of truth — `TriageEngine` warm-loaded once, calibration constants READ from JSON not pasted, preprocessing IMPORTED from `extract_features.py` not paraphrased), `training/triage_cli.py` (--human / --json / --include-gpt), `training/server.py` (FastAPI, narrow CORS).
- VLM triage (M21, deployed-app fallback): `src/lib/pipeline/vlmTriage.ts` (`submit_triage` schema + boring policy prompt + forced-abstain rails + borderline-band predicate) and `src/lib/pipeline/vlmEscalation.ts` (path-specific escalation, SEPARATE 0.5 threshold from the M19 ONNX 0.7126 — DO NOT mix).
- Sequelae escalation (M19): `src/lib/pipeline/sequelaeEscalation.ts` — pure rule, consumed on the M22 local-mode path now that `s_inactive` is produced by the local server. (Browser-direct ONNX still gated on Phase B feature pathway.)
- Calibration: `src/lib/calibration.ts` (temperature/Platt + log-odds fusion + conformal; fit via `/validate` Calibrate). On the M21 VLM-primary path it is bypassed (the VLM score is uncalibrated). On the M22 local-mode path the head's `tb_prob` is ALREADY calibrated under T server-side and the verdict uses the validated `thr_at_95sens=0.6105`.
- Perception training (offline, M4/MPS): `training/` + `docs/superpowers/plans/2026-05-24-tb-classifier-training-T1.md`.
- Roadmap: `docs/superpowers/plans/2026-05-24-perception-module.md` (the path to ≥90%).

**M22 run commands (local mode):**
```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
  training/.venv/bin/python -m uvicorn training.server:app --port 8000
# in another terminal:
npm run dev
# then in the browser: Settings → toggle "Local mode" ON
```

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
