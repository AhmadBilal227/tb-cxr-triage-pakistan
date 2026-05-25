/**
 * Live smoke for the M24 gpt-interpreter provider (Milestone 24).
 *
 * Runs the local engine via `training/.venv/bin/python -m training.triage_cli` (single fresh
 * load), feeds the resulting JSON into the gpt-interpreter, and prints the model's
 * clinician_report. NOT a regression check — costs cents per run, and the model output
 * is the load-bearing artifact for the M24 EXPERIMENT_LOG row and the CASE_STUDY entry.
 *
 * Usage:
 *   VITE_OPENAI_KEY=... node scripts/verify-gpt-interpreter.mjs public/samples/tb-sample-1.jpg
 *
 * Requires:
 *   - the local FastAPI server is RUNNING on http://localhost:8000 (we POST the image bytes
 *     and read back the TriageResult with the M24 enrichment fields)
 *   - a valid OpenAI API key in env as VITE_OPENAI_KEY
 */
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const SERVER = process.env.LOCAL_SERVER_URL ?? 'http://localhost:8000';
const sampleArg = process.argv[2];
if (!sampleArg) {
  console.error('usage: node scripts/verify-gpt-interpreter.mjs <path-to-cxr>');
  process.exit(1);
}
const samplePath = resolve(sampleArg);
const apiKey = process.env.VITE_OPENAI_KEY ?? process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error('VITE_OPENAI_KEY not set in env');
  process.exit(1);
}

const bytes = await readFile(samplePath);

// ---- 1. POST /triage to the local server. -----------------------------------
const form = new FormData();
const blob = new Blob([bytes], { type: 'image/jpeg' });
form.append('file', blob, 'cxr.jpg');
const triageRes = await fetch(`${SERVER}/triage`, { method: 'POST', body: form });
if (!triageRes.ok) {
  console.error(`/triage failed ${triageRes.status}:`, await triageRes.text());
  process.exit(1);
}
const local = await triageRes.json();

console.log('=== LOCAL TRIAGE RESULT (validated model) ===');
console.log(`verdict:               ${local.verdict}`);
console.log(`tb_prob:               ${local.tb_prob.toFixed(6)}`);
console.log(`decided_at_threshold:  ${local.decided_at_threshold.toFixed(4)}`);
console.log(`s_inactive:            ${local.s_inactive.toFixed(4)}`);
console.log(`crop_box:              ${JSON.stringify(local.crop_box)}`);
console.log(`inversion_detected:    ${local.inversion_detected}`);
console.log('zonal_scores:');
for (const [k, v] of Object.entries(local.zonal_scores ?? {})) {
  console.log(`  ${k.padEnd(10)} = ${v.toFixed(4)}`);
}
console.log('txrv top-5:');
const top5 = Object.entries(local.txrv_pathologies ?? {})
  .sort((a, b) => b[1] - a[1]).slice(0, 5);
for (const [k, v] of top5) console.log(`  ${k.padEnd(30)} = ${v.toFixed(4)}`);

// ---- 2. Build the evidence message and call gpt-5.5 vision. -----------------
// We can't import the TS provider from node directly without a build step, so we
// re-implement the evidence-message + prompt + schema here. Keep in sync with
// src/lib/providers/gptInterpreter.ts.
const PROMPT = [
  'You are translating the structured output of a validated TB-triage pipeline (Rad-DINO + TorchXRayVision + TBHeadT2 + InactiveSequelaeHead, LODO AUROC 0.922) into a short clinician-style narrative.',
  'You are NOT diagnosing the image. The pipeline already produced the verdict, the calibrated tb_prob, and the per-zone probabilities. Your job is to put the pipeline\'s OWN evidence into a 1-2 sentence paragraph and a small differential.',
  'STRICT RULES:',
  '- Use ONLY zones the pipeline flagged in the per-zone probabilities (the highest probabilities are the regions to mention by name).',
  '- Use ONLY pathology features the pipeline\'s TXRV head ranked in the top scores (do NOT introduce findings the pipeline did not score).',
  '- Do NOT contradict the pipeline verdict. If the verdict is "tb", describe the evidence consistent with that verdict; if "no_tb", describe the absence of evidence; if "abstain", describe the borderline / ambiguity.',
  '- top_differential_alternatives ranks alternatives by FIT TO THE PIPELINE EVIDENCE, not fit to the image alone.',
  '- key_regions MUST be a subset of the zone keys the user provides (e.g. ["upper_r","hilar"]).',
  '- confidence_qualifier reflects the calibrated tb_prob (high near 0/1, moderate in mid-band, low in the borderline band 0.35-0.65).',
  '- refusal_or_limitation is non-null ONLY when the image data the pipeline preprocessed is degraded or you cannot honestly map its scores into a narrative.',
  'No demographic priors. No inferred clinical history. No claim that your narrative is calibrated. No chain-of-thought — output ONLY the schema fields.',
].join(' ');

const evidenceLines = [];
evidenceLines.push('VALIDATED PIPELINE OUTPUT (translate this, do not re-diagnose):');
evidenceLines.push(`verdict: ${local.verdict}`);
evidenceLines.push(`tb_prob (calibrated under T=${local.audit.calibration.T.toFixed(3)}): ${local.tb_prob.toFixed(4)}`);
evidenceLines.push(`decided_at_threshold: ${local.decided_at_threshold.toFixed(4)}`);
evidenceLines.push(`s_inactive (scar/sequelae score, under T_seq=${local.audit.calibration.T_sequelae.toFixed(3)}): ${local.s_inactive.toFixed(4)}`);
if (local.safety_net_applied) evidenceLines.push(`safety_net_applied: ${local.safety_net_applied}`);
if (local.zonal_scores) {
  const sorted = Object.entries(local.zonal_scores).sort((a, b) => b[1] - a[1]);
  evidenceLines.push('per-zone calibrated TB probabilities (key: probability):');
  for (const [k, v] of sorted) evidenceLines.push(`  ${k}: ${v.toFixed(4)}`);
}
if (local.txrv_pathologies) {
  const top = Object.entries(local.txrv_pathologies).sort((a, b) => b[1] - a[1]).slice(0, 5);
  evidenceLines.push('top-5 TorchXRayVision findings (label: probability — input features, NOT independent diagnoses):');
  for (const [k, v] of top) evidenceLines.push(`  ${k}: ${v.toFixed(4)}`);
}
if (local.image_quality?.warnings?.length) {
  evidenceLines.push('preprocessing warnings:');
  for (const w of local.image_quality.warnings) evidenceLines.push(`  - ${w}`);
}
evidenceLines.push('');
evidenceLines.push('Now produce the clinician_report JSON. Mention zones by name; do not invent zone keys not listed above. Differential alternatives must be grounded in the TXRV findings or the scar/sequelae score, not in pixels.');

const fullPrompt = `${PROMPT}\n\n${evidenceLines.join('\n')}`;

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    finding: { type: 'string' },
    key_regions: { type: 'array', items: { type: 'string' } },
    confidence_qualifier: { type: 'string', enum: ['low', 'moderate', 'high'] },
    top_differential_alternatives: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          label: { type: 'string' },
          likelihood: { type: 'string', enum: ['consider', 'less_likely'] },
          rationale: { type: 'string' },
        },
        required: ['label', 'likelihood', 'rationale'],
      },
    },
    refusal_or_limitation: { type: ['string', 'null'] },
  },
  required: [
    'finding', 'key_regions', 'confidence_qualifier',
    'top_differential_alternatives', 'refusal_or_limitation',
  ],
};

// Base64 the image bytes for the Responses API data URL.
const dataUrl = `data:image/jpeg;base64,${bytes.toString('base64')}`;

const body = {
  model: 'gpt-5.5',
  input: [
    {
      role: 'user',
      content: [
        { type: 'input_text', text: fullPrompt },
        { type: 'input_image', image_url: dataUrl },
      ],
    },
  ],
  text: { format: { type: 'json_schema', name: 'clinician_report', schema: SCHEMA, strict: true } },
};
const start = performance.now();
const resp = await fetch('https://api.openai.com/v1/responses', {
  method: 'POST',
  headers: {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(body),
});
const latency = ((performance.now() - start) / 1000).toFixed(2);
if (!resp.ok) {
  console.error(`/v1/responses failed ${resp.status}:`, await resp.text());
  process.exit(1);
}
const env = await resp.json();
const text = env.output_text ?? env.output?.[0]?.content?.[0]?.text ?? '';
let report;
try {
  report = JSON.parse(text);
} catch (e) {
  console.error('failed to parse model output:', text);
  throw e;
}
console.log('\n=== GPT-5.5 CLINICIAN REPORT ===');
console.log(`model_id_from_response: ${env.model}`);
console.log(`latency: ${latency}s`);
console.log(`finding: ${report.finding}`);
console.log(`key_regions: ${JSON.stringify(report.key_regions)}`);
console.log(`confidence_qualifier: ${report.confidence_qualifier}`);
console.log(`refusal_or_limitation: ${report.refusal_or_limitation}`);
console.log('top_differential_alternatives:');
for (const d of report.top_differential_alternatives ?? []) {
  console.log(`  - ${d.label} (${d.likelihood}): ${d.rationale}`);
}
