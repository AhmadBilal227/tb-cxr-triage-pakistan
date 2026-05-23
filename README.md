# TB Triage — AI-native chest X-ray triage (research preview)

> **Research preview. Not a medical device. Not for diagnostic use.**
> Every output is produced by third-party AI models called directly from your browser.
> Do not use it to make clinical decisions.

A **frontend-only** TB chest-X-ray triage app. No backend server, no database server. All API
calls go directly from the browser via `fetch()`. API keys are **BYOK** (bring your own key),
stored in `localStorage`. Client persistence is **IndexedDB** (via Dexie).

**Providers:** Hugging Face is the **primary** perception layer, Replicate is the **per-stage
fallback**, OpenAI (`gpt-5.5`) is used for **orchestration only** (quality gate, vision read,
adjudication).

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # typechecks (strict, no `any`) + production bundle
```

A dependency-free single-file version is in **[`demo.html`](./demo.html)** — open it directly in a
browser (Stages 1, 2, 4; see note inside).

---

## Pipeline

| Stage | What | Provider | Fallback |
|---|---|---|---|
| 1 — Quality gate | "Is this a frontal CXR suitable for TB screening?" Halts on failure. | GPT-5.5 vision | none (→ `gpt-5.5-instant`) |
| 2 — Perception ensemble (parallel) | A) TB classifier (w 0.5) · B) general CXR (w 0.2) · C) GPT-5.5 vision read (w 0.3) | HF / HF / OpenAI | Replicate / Replicate / none |
| 3 — Retrieval (kNN, k=5) | CXR-Foundation embedding → cosine kNN over labeled corpus | HF Inference Endpoint | Replicate CLIP |
| 4 — Adjudicator | Single streamed GPT-5.5 call over all signals; deterministic auto-abstain guardrails | GPT-5.5 | none (→ instant) |
| 5 — Verdict | TB SUSPECTED / NO TB / UNCERTAIN — REFER + confidence ring | — | — |

The **Agent Trace** panel (right rail) streams each stage live with its provider badge
(`HF` green / `Replicate` amber / `OpenAI` indigo), latency, and expandable raw JSON. When a stage
falls back, a `→ Replicate` transition animates on that card — **fallback is never hidden**.

### Auto-abstain guardrails (deterministic, override the model)

The verdict is forced to **abstain** if any of:

- model confidence `< 75`
- ensemble std `> 0.2`
- top-1 retrieval similarity `< 0.6` **and** ensemble disagreement `> 0.3`
- `≥ 2` stages fell back to Replicate (degraded inference quality)

Thresholds live in `src/lib/defaults.ts` (`ABSTAIN_RULES`).

---

## 1. BYOK setup

Open **Settings** (gear icon or `Cmd/Ctrl-K → Settings`) and paste your keys. Get them here:

- **OpenAI** — <https://platform.openai.com/api-keys> (required: quality gate, VLM, adjudicator)
- **Hugging Face** — <https://huggingface.co/settings/tokens> (required: primary perception)
- **Replicate** — <https://replicate.com/account/api-tokens> (optional: enables per-stage fallback)

> Keys live in `localStorage`. Any JS on the page can read them. **Do not use production keys.**
> If the Replicate token is empty, fallback is disabled and a visible warning is shown; a cold or
> failing HF model will then fail that stage instead of falling back.

---

## 2. Hugging Face Inference API quirks

- **Cold starts / 503.** Serverless models unload when idle. The first call returns
  `503` with `{ "estimated_time": <seconds> }`. This app parses `estimated_time`, waits that long
  plus a buffer, and retries — **max 3 retries**. If the *projected* total wait exceeds **20s**, it
  fails fast and falls back to Replicate rather than blocking the UI (`src/lib/providers/hf.ts`).
- **Free-tier rate limits.** The free serverless tier is rate-limited and intended for light use;
  expect `429`s under load. A PRO account or a dedicated Inference Endpoint raises these limits.
- **Why `google/cxr-foundation` needs a paid Inference Endpoint.** Large/gated foundation models are
  **not** served on free serverless. To get CXR-Foundation embeddings you must deploy your own
  **Inference Endpoint** (<https://ui.endpoints.huggingface.co/>) and paste its URL into
  Settings → *CXR Embedding — HF Inference Endpoint URL*. The app POSTs the image bytes to that URL
  and extracts the embedding vector (it tolerates flat, nested, mean-poolable, and `{embedding:[…]}`
  shapes). If you don't have one, configure a Replicate CLIP fallback instead, or Stage 3 is skipped
  with an inline banner.

---

## 3. (Optional) Deploy a TB classifier to Replicate via Cog as fallback

If you want a real Replicate fallback for the TB classifier, publish a TB CNN with
[Cog](https://github.com/replicate/cog). A minimal example is in
[`replicate-cog-example/`](./replicate-cog-example/):

- [`cog.yaml`](./replicate-cog-example/cog.yaml) — build config
- [`predict.py`](./replicate-cog-example/predict.py) — wraps a torchvision ResNet, returns
  `[{label, score}]` with a `"Tuberculosis"` class (matches the app's parser)

```bash
pip install cog
cog build -t my-tb-classifier
cog push r8.im/<your-username>/tb-classifier
```

Paste the resulting **slug** and **version hash** into Settings → *TB Classifier — Replicate slug* /
*…version hash*. Replicate predictions are async: the app creates a prediction then polls
`urls.get` until `succeeded`/`failed`, capped at **60s** with exponential backoff.

---

## 4. Labeled set CSV format

Both the RAG corpus import and the validation route accept a folder containing image files **and** a
CSV. The CSV has a header and one row per image:

```csv
filename,label
MCUCXR_0001_0.png,0
MCUCXR_0101_1.png,1
```

- `filename` — basename of the image (folder paths are stripped on match).
- `label` — `1` = TB, `0` = not TB. Rows with any other value are skipped.

See [`examples/labels.example.csv`](./examples/labels.example.csv). On import, each matched image is
embedded (if an embedding provider is configured) and stored in IndexedDB with its label, becoming a
kNN candidate. Unmatched rows and embedding failures are reported in the status line.

The **Montgomery** and **Shenzhen** sets (NIH/NLM) follow exactly this naming convention
(`*_0` = normal, `*_1` = TB), so a CSV like the example drops in directly.

---

## 5. Validation flow

Go to **`/validate`** (or `Cmd/Ctrl-K → Validate`):

1. Choose a holdout folder (images + a `filename,label` CSV).
2. **Run** — each image runs through the full pipeline sequentially with a progress bar.
3. Metrics are computed live:
   - **Accuracy, Sensitivity, Specificity** over *decided* cases (abstains/halted excluded).
   - **AUC** via rank-based Mann–Whitney U over the ensemble weighted score (`src/lib/metrics.ts`).
   - **Confusion matrix** (TP/FP/TN/FN), plus abstain/halted counts.
4. Export the report as **JSON** or **PDF** — both embed the disclaimer, model versions, and
   timestamps.

This route is fully implemented (not stubbed) and makes real API calls per image, so a large holdout
set will consume API quota and take time (HF cold starts + Replicate polling dominate latency).

---

## 6. Known limits

- **Correlated VLM errors.** Stage 2C (GPT-5.5 vision) and Stage 4 (GPT-5.5 adjudicator) share a
  model family, so their errors are correlated — the ensemble is less independent than the three
  weights imply. The adjudicator can rationalize a wrong VLM read.
- **CLIP retrieval ≪ CXR-Foundation.** The Replicate CLIP embedding fallback is a generic image
  encoder; its CXR retrieval quality is materially worse than CXR-Foundation. Low top-1 similarity is
  expected with CLIP, which the abstain guardrail partly accounts for.
- **Polling latency.** Replicate is create-then-poll; a fallback adds seconds. HF cold starts add up
  to the 20s budget before fallback fires.
- **Fallback impact on accuracy.** Each Replicate fallback signals degraded inference; `≥2` forces an
  abstain. Session fallback rate is logged and shown in every export — **a high fallback rate means a
  degraded run and must be visible in any review.**
- **CORS.** This is frontend-only. OpenAI and HF return browser-usable CORS headers. **Replicate's
  REST API may not** send permissive CORS headers for all accounts/edges, so a direct browser call
  can be blocked; the failure surfaces in the trace. If you hit this, a thin same-origin proxy is the
  usual workaround (out of scope for a no-backend build).
- **Synthetic "Try sample".** The empty-state sample is procedurally drawn, **not** a real
  radiograph, and the quality gate may correctly reject it. Use a real CXR (e.g. Montgomery) for
  meaningful results.

---

## VLM as a calibrated safety net (and additive to the CNN)

The GPT-5.5 vision member is tuned to be a *sensitive safety net*, not a confident
classifier. Techniques applied (grounded in 2025–26 literature):

- **Structured zone-by-zone chain-of-thought** prompt with a screening prior
  ([X-Ray-CoT](https://arxiv.org/abs/2508.12455), [GPT-4V prompt eng.](https://arxiv.org/pdf/2312.04344)),
  returning a directly verbalized `tb_probability`.
- **Self-consistency** (`SELF_CONSISTENCY_K` reads): mean = probability, spread =
  real uncertainty — single-prompt confidence does not fix overconfidence
  ([arXiv 2604.02543](https://arxiv.org/abs/2604.02543)).
- **Screening-biased calibrated thresholds** (`SCREENING_POLICY` in `defaults.ts`):
  low bar to flag, high bar to clear — the calibration lever that moved generic-VLM
  TB F1 0.48→0.77 ([arXiv 2510.00411](https://arxiv.org/pdf/2510.00411)). Fit on a
  holdout via `/validate`.
- **Safety-net combine** (`mostCautious` in `orchestrator.ts`): final verdict =
  most cautious of {policy, model, guardrails}. Neither the model nor a degraded run
  can *clear* a flagged case. When the CNN comes online, probabilities fuse by weight,
  but the VLM can still escalate on its own (`vlmSafetyThreshold`) to catch CNN false
  negatives — strictly **additive** on the sensitivity axis, never a veto.

Measured on a 30-image balanced held-out slice (`scripts/accuracy-test*.mjs`),
GPT-5.5-only (no live CNN): **sensitivity 14% → 42%, AUC 0.76 → 0.84, specificity held
at 100%.** Still far below the WHO triage bar (≥90% sensitivity) — the remaining gap is
a *perception* ceiling that a real, validated TB CNN in the 0.5-weight slot is meant to
close. (`demo.html` still uses the simpler single-read prompt.)

## Design decisions / defaults (documented)

- **`raw: unknown`, not `any`.** Strict TS bans `any`; the spec's `raw: any` is implemented as
  `unknown` and narrowed at the edges. No `any` anywhere.
- **Ensemble weights** 0.5 / 0.2 / 0.3; renormalized over members that actually returned.
- **shadcn/ui** is implemented as Radix + Tailwind + `cva` primitives in `src/components/ui/`
  (that is what shadcn *is* — copied components, not a dependency).
- **State**: a dependency-free `useSyncExternalStore` settings store; pipeline state via `useReducer`
  folding the live event stream.
- **`demo.html`** implements Stages 1, 2, 4 (RAG omitted — it needs a corpus + IndexedDB).
- **Replicate auth** uses `Authorization: Bearer <token>`.

## Project layout

```
src/lib/types.ts              normalized result contract (the seam)
src/lib/providers/            hf.ts · replicate.ts · openai.ts · classify.ts · parsers.ts
src/lib/pipeline/             orchestrator.ts · stageConfigs.ts · prompts.ts
src/lib/db.ts                 Dexie (labeledCases + caseHistory)
src/lib/metrics.ts            accuracy / sensitivity / specificity / AUC
src/hooks/usePipeline.ts      event stream → render state
src/components/               AgentTrace, StageCard, VerdictCard, DropCanvas, SettingsDrawer, …
src/routes/Validate.tsx       /validate
demo.html                     single-file plain-JS pipeline
```
