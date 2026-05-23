/**
 * Live integration smoke test against the configured BYOK providers.
 *   node scripts/integration-test.mjs
 * Reads keys from .env.local. Verifies the exact request shapes the app uses.
 */
import { readFile } from 'node:fs/promises';

const env = Object.fromEntries(
  (await readFile(new URL('../.env.local', import.meta.url), 'utf8'))
    .split('\n')
    .filter((l) => l.includes('=') && !l.trim().startsWith('#'))
    .map((l) => [l.slice(0, l.indexOf('=')).trim(), l.slice(l.indexOf('=') + 1).trim()]),
);
const OPENAI = env.VITE_OPENAI_KEY;
const HF = env.VITE_HF_TOKEN;

function extractText(j) {
  if (typeof j.output_text === 'string' && j.output_text) return j.output_text;
  let t = '';
  for (const o of j.output ?? []) for (const c of o.content ?? []) if (c.type === 'output_text') t += c.text;
  return t;
}

async function openaiText(model) {
  const res = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: { Authorization: `Bearer ${OPENAI}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      input: [{ role: 'user', content: [{ type: 'input_text', text: 'Reply JSON only: {"ok": true}' }] }],
      text: { format: { type: 'json_object' } },
    }),
  });
  const body = await res.text();
  return { status: res.status, body };
}

async function openaiVision(model, dataUrl) {
  const res = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: { Authorization: `Bearer ${OPENAI}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      input: [{ role: 'user', content: [
        { type: 'input_text', text: 'Is this a frontal chest X-ray? Reply JSON only: {is_cxr: bool, quality: "good"|"poor"|"unreadable", reason: string}' },
        { type: 'input_image', image_url: dataUrl },
      ] }],
      text: { format: { type: 'json_object' } },
    }),
  });
  return { status: res.status, body: await res.text() };
}

async function hf(model, bytes) {
  const res = await fetch(`https://api-inference.huggingface.co/models/${model}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${HF}`, Accept: 'application/json', 'Content-Type': 'image/jpeg' },
    body: bytes,
  });
  return { status: res.status, body: await res.text() };
}

const log = (label, r) => {
  console.log(`\n### ${label}  [HTTP ${r.status}]`);
  console.log(r.body.slice(0, 600));
};

console.log('=== OpenAI: model probe (gpt-5.5) ===');
let r = await openaiText('gpt-5.5');
log('responses gpt-5.5 (text)', r);
if (r.status !== 200) {
  console.log('\n>>> gpt-5.5 not OK; trying gpt-5.5-instant');
  log('responses gpt-5.5-instant', await openaiText('gpt-5.5-instant'));
}

const img = await readFile(new URL('../public/samples/tb-sample-1.jpg', import.meta.url));
const dataUrl = `data:image/jpeg;base64,${img.toString('base64')}`;

console.log('\n=== OpenAI: vision quality gate on tb-sample-1.jpg ===');
log('vision gpt-5.5', await openaiVision('gpt-5.5', dataUrl));

console.log('\n=== HF: Owos/tb-classifier on tb-sample-1.jpg ===');
log('hf Owos/tb-classifier', await hf('Owos/tb-classifier', img));
