#!/usr/bin/env node
/**
 * One-off live verification (Milestone 21) that the Responses API's
 * structured-output / json_schema mode actually accepts our
 * `submit_triage` schema, and returns the model_id we record in the audit
 * trail. Reads VITE_OPENAI_KEY from .env.local. NOT committed to CI;
 * gitignored output is intentional.
 *
 * Usage: node scripts/verify-vlm-schema.mjs [image_path]
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

function loadEnvLocal() {
  try {
    const text = readFileSync('.env.local', 'utf8');
    const env = {};
    for (const line of text.split(/\n/)) {
      const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
      if (m) env[m[1]] = m[2];
    }
    return env;
  } catch {
    return {};
  }
}

const env = loadEnvLocal();
const apiKey = env.VITE_OPENAI_KEY;
if (!apiKey) {
  console.error('Missing VITE_OPENAI_KEY in .env.local');
  process.exit(1);
}

const imagePath = process.argv[2] ?? 'public/samples/tb-sample-1.jpg';
const fullPath = resolve(imagePath);
const bytes = readFileSync(fullPath);
const b64 = bytes.toString('base64');
const ext = (imagePath.split('.').pop() ?? 'jpg').toLowerCase();
const mime = ext === 'png' ? 'image/png' : 'image/jpeg';
const dataUrl = `data:${mime};base64,${b64}`;

const VLM_TRIAGE_SCHEMA = {
  name: 'submit_triage',
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      image_quality: { type: 'string', enum: ['diagnostic', 'limited', 'nondiagnostic'] },
      projection: { type: 'string', enum: ['pa_ap', 'lateral', 'unknown'] },
      tb_screen_result: {
        type: 'string',
        enum: ['screen_positive', 'screen_negative', 'abstain'],
      },
      tb_score_uncalibrated: { type: 'number' },
      confidence_band: { type: 'string', enum: ['low', 'medium', 'high'] },
      scar_shape_score_uncalibrated: { type: 'number' },
      mimic_features_present: { type: 'array', items: { type: 'string' } },
      abnormality_localization: { type: 'array', items: { type: 'string' } },
      safety_flags: { type: 'array', items: { type: 'string' } },
      short_rationale: { type: 'string' },
      refusal_or_limitation: { type: ['string', 'null'] },
      model_version_seen_by_client: { type: 'string' },
    },
    required: [
      'image_quality',
      'projection',
      'tb_screen_result',
      'tb_score_uncalibrated',
      'confidence_band',
      'scar_shape_score_uncalibrated',
      'mimic_features_present',
      'abnormality_localization',
      'safety_flags',
      'short_rationale',
      'refusal_or_limitation',
      'model_version_seen_by_client',
    ],
  },
};

const VLM_TRIAGE_PROMPT_PRIMARY = [
  'You are a structured-output triage helper for a research preview tuberculosis screen.',
  'This is NOT diagnosis. Output ONLY the schema fields; do not narrate.',
  'When uncertain or when image quality is limited or nondiagnostic, choose ABSTAIN rather than screen_negative.',
  'Do not assume demographic priors. Do not infer clinical history from the image.',
  'Do not claim your scores are calibrated.',
  'tb_score_uncalibrated is a soft 0..1 indicator only; it has not been validated against labels.',
  'scar_shape_score_uncalibrated is a separate heuristic for healed fibrosis / pleural thickening / volume loss.',
  'mimic_features_present should list specific visible patterns (e.g. apical_fibrosis, pleural_thickening, healed_scar, calcified_granuloma) when seen.',
  'safety_flags should include pediatric, portable_ap, lateral_only, artifact, rotation, or other quality concerns.',
  'Keep short_rationale to one or two sentences, citing visible evidence; do not include step-by-step reasoning.',
].join(' ');

async function callOnce(model) {
  const start = Date.now();
  const body = {
    model,
    input: [
      {
        role: 'user',
        content: [
          { type: 'input_text', text: VLM_TRIAGE_PROMPT_PRIMARY },
          { type: 'input_image', image_url: dataUrl },
        ],
      },
    ],
    text: {
      format: {
        type: 'json_schema',
        name: VLM_TRIAGE_SCHEMA.name,
        schema: VLM_TRIAGE_SCHEMA.schema,
        strict: true,
      },
    },
  };
  const res = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const latency = Date.now() - start;
  const status = res.status;
  const text = await res.text();
  let json = null;
  try { json = JSON.parse(text); } catch {}
  return { status, latency, json, raw: text };
}

console.log('=== M21 VLM SCHEMA LIVE VERIFICATION ===');
console.log(`image: ${imagePath} (${bytes.length} bytes)`);
const { status, latency, json, raw } = await callOnce('gpt-5.5');
console.log(`HTTP ${status} (latency ${latency}ms)`);
if (status !== 200) {
  console.log('--- raw body (first 1200 chars) ---');
  console.log(raw.slice(0, 1200));
  process.exit(2);
}

console.log(`model returned: ${json.model}`);
console.log(`id: ${json.id}`);
// Walk to the output_text
let outText = json.output_text ?? '';
if (!outText && Array.isArray(json.output)) {
  for (const item of json.output) {
    if (item.type === 'message' && Array.isArray(item.content)) {
      for (const c of item.content) {
        if (c.type === 'output_text' && typeof c.text === 'string') outText += c.text;
      }
    }
  }
}
console.log('--- output_text ---');
console.log(outText);
console.log('--- parsed ---');
try {
  const parsed = JSON.parse(outText);
  console.log(JSON.stringify(parsed, null, 2));
  console.log('--- audit pins we would record ---');
  console.log({
    model_id_from_response: json.model,
    schema_name: VLM_TRIAGE_SCHEMA.name,
  });
} catch (e) {
  console.log('Could not parse output_text:', e.message);
  process.exit(3);
}
