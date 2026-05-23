/**
 * Improved harness (v2): tests the proposed VLM upgrades on the SAME test slice as v1.
 *   node scripts/accuracy-test-v2.mjs [N_PER_CLASS]
 *
 * Changes vs v1 (current app):
 *  1. Structured chain-of-thought, zone-by-zone prompt with a SCREENING prior
 *     ("don't dismiss subtle findings; lean toward flagging when uncertain").
 *  2. Verbalized calibrated probability (tb_probability 0..1) instead of tb_likely+confidence.
 *  3. Self-consistency: K samples -> mean probability + spread (real uncertainty).
 *  4. Screening-biased decision policy: low bar to FLAG, high bar to CLEAR; uncertain -> refer.
 *  5. Safety-net combine: GPT adjudicator can only ESCALATE severity, never clear a flagged case.
 */
import { readFile, writeFile } from 'node:fs/promises';

const env = Object.fromEntries(
  (await readFile(new URL('../.env.local', import.meta.url), 'utf8'))
    .split('\n').filter((l) => l.includes('=') && !l.trim().startsWith('#'))
    .map((l) => [l.slice(0, l.indexOf('=')).trim(), l.slice(l.indexOf('=') + 1).trim()]),
);
const OPENAI = env.VITE_OPENAI_KEY;
const MODEL = 'gpt-5.5', FALLBACK = 'gpt-5.5-instant';
const N = Number(process.argv[2] ?? 15);
const K = 3; // self-consistency samples

// Screening-biased policy (sensitivity-first). Fit these on a labeled set via /validate.
const POLICY = { tbFlag: 0.30, negClear: 0.15, maxClearUncertainty: 0.15 };

const QUALITY_PROMPT =
  'Is this a frontal chest X-ray suitable for TB screening? Reply JSON only: {is_cxr: bool, quality: "good"|"poor"|"unreadable", reason: string}';

const VLM_PROMPT_V2 = [
  'You are a sensitive safety-net reader for tuberculosis screening in a high-prevalence setting.',
  'Systematically inspect each lung zone (right and left upper, mid, lower) plus the hila, mediastinum and pleura.',
  'TB-associated signs: upper-zone consolidation, cavitation, miliary/nodular pattern, hilar or',
  'mediastinal lymphadenopathy, pleural effusion, fibrosis or volume loss, tree-in-bud.',
  'Reason step by step about each zone, then give a calibrated probability that this radiograph shows tuberculosis.',
  'This is a SCREEN: subtle or equivocal findings matter — do NOT dismiss them, and when genuinely',
  'uncertain lean toward a higher probability rather than clearing the patient.',
  'Respond JSON only: {reasoning: string, findings: string[], tb_probability: number between 0 and 1, image_quality: "good"|"limited"|"unreadable"}',
].join(' ');

function extractText(j) {
  if (typeof j.output_text === 'string' && j.output_text) return j.output_text;
  let t = ''; for (const o of j.output ?? []) for (const c of o.content ?? []) if (c.type === 'output_text') t += c.text; return t;
}
function extractJSON(text) { const s = text.indexOf('{'), e = text.lastIndexOf('}'); return JSON.parse(text.slice(s, e + 1)); }
async function openaiJSON(prompt, dataUrl) {
  for (const model of [MODEL, FALLBACK]) {
    try {
      const res = await fetch('https://api.openai.com/v1/responses', {
        method: 'POST', headers: { Authorization: `Bearer ${OPENAI}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, input: [{ role: 'user', content: [{ type: 'input_text', text: prompt }, ...(dataUrl ? [{ type: 'input_image', image_url: dataUrl }] : [])] }], text: { format: { type: 'json_object' } } }),
      });
      if (!res.ok) throw new Error(`OpenAI ${res.status}`);
      return extractJSON(extractText(await res.json()));
    } catch (e) { if (model === FALLBACK) throw e; }
  }
}
const clamp = (n) => Math.max(0, Math.min(1, n));
const severity = { no_tb: 0, abstain: 1, tb: 2 };
const fromSeverity = ['no_tb', 'abstain', 'tb'];

async function runOne(bytes) {
  const dataUrl = `data:image/jpeg;base64,${bytes.toString('base64')}`;
  const q = await openaiJSON(QUALITY_PROMPT, dataUrl);
  if (!q.is_cxr || q.quality === 'unreadable') return { halted: true, verdict: null, score: null };

  // Self-consistency: K VLM samples
  const samples = await Promise.allSettled(Array.from({ length: K }, () => openaiJSON(VLM_PROMPT_V2, dataUrl)));
  const probs = samples.filter((s) => s.status === 'fulfilled').map((s) => clamp(Number(s.value.tb_probability)));
  if (!probs.length) throw new Error('all VLM samples failed');
  const pVlm = probs.reduce((a, b) => a + b, 0) / probs.length;
  const mean = pVlm;
  const u = Math.sqrt(probs.reduce((a, b) => a + (b - mean) ** 2, 0) / probs.length); // spread = uncertainty
  const findings = samples.find((s) => s.status === 'fulfilled')?.value.findings ?? [];

  // Screening policy (VLM-only here; fused prob == pVlm)
  let policyVerdict;
  if (pVlm >= POLICY.tbFlag) policyVerdict = 'tb';
  else if (pVlm <= POLICY.negClear && u <= POLICY.maxClearUncertainty) policyVerdict = 'no_tb';
  else policyVerdict = 'abstain';

  // GPT adjudicator (can only escalate)
  const adjPrompt = [
    'You are a cautious TB screening adjudicator and safety net. A perception reader assessed the chest X-ray.',
    `Mean TB probability over ${K} reads: ${pVlm.toFixed(3)} (spread ${u.toFixed(3)}).`,
    `Findings: ${JSON.stringify(findings)}.`,
    'This is a screen; missing TB is far worse than a false alarm. Respond JSON only:',
    '{verdict: "tb"|"no_tb"|"abstain", confidence: 0-100, rationale: string}',
  ].join(' ');
  let modelVerdict = 'abstain';
  try { modelVerdict = (await openaiJSON(adjPrompt, dataUrl)).verdict ?? 'abstain'; } catch { /* keep policy */ }

  // Safety-net combine: take the MORE cautious (higher severity) of policy vs model.
  const verdict = fromSeverity[Math.max(severity[policyVerdict] ?? 1, severity[modelVerdict] ?? 1)];
  return { halted: false, verdict, score: pVlm, uncertainty: u, policyVerdict, modelVerdict };
}

async function rows(off, len) {
  const ds = 'darthPanda%2Fcombined-unknown-pneumonia-and-tuberculosis';
  const d = await (await fetch(`https://datasets-server.huggingface.co/rows?dataset=${ds}&config=default&split=train&offset=${off}&length=${len}`)).json();
  return d.rows.map((r) => ({ label: r.row.label, img: Object.values(r.row).find((v) => v && typeof v === 'object' && v.src)?.src }));
}
const get = async (u) => Buffer.from(await (await fetch(u)).arrayBuffer());
function auc(items) {
  const pos = items.filter((i) => i.label === 1), neg = items.filter((i) => i.label === 0);
  if (!pos.length || !neg.length) return NaN;
  const s = [...items].sort((a, b) => a.score - b.score); const r = []; let i = 0;
  while (i < s.length) { let j = i; while (j + 1 < s.length && s[j + 1].score === s[i].score) j++; const v = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[k] = v; i = j + 1; }
  let sr = 0; s.forEach((it, idx) => { if (it.label === 1) sr += r[idx]; });
  return (sr - (pos.length * (pos.length + 1)) / 2) / (pos.length * neg.length);
}

const run = async () => {
  console.log(`v2: ${N} TB + ${N} NORMAL, K=${K} self-consistency, screening policy ${JSON.stringify(POLICY)}\n`);
  const tb = (await rows(8500, N + 4)).filter((r) => r.label === 2 && r.img).slice(0, N).map((r) => ({ ...r, truth: 1 }));
  const norm = (await rows(300, N + 8)).filter((r) => r.label === 0 && r.img).slice(0, N).map((r) => ({ ...r, truth: 0 }));
  const set = [...tb, ...norm];
  const results = []; let done = 0; const queue = [...set];
  async function worker() {
    while (queue.length) {
      const item = queue.shift();
      try { const bytes = await get(item.img); const r = await runOne(bytes); results.push({ truth: item.truth, ...r }); done++;
        process.stdout.write(`  [${done}/${set.length}] ${item.truth ? 'TB ' : 'NEG'} -> ${r.halted ? 'HALT' : r.verdict.padEnd(6)} p=${r.score?.toFixed(2) ?? '-'} u=${r.uncertainty?.toFixed(2) ?? '-'} (policy ${r.policyVerdict ?? '-'} / gpt ${r.modelVerdict ?? '-'})\n`);
      } catch (e) { results.push({ truth: item.truth, halted: true, verdict: null, score: null }); done++; process.stdout.write(`  [${done}/${set.length}] ${item.truth ? 'TB' : 'NEG'} -> ERR ${e.message.slice(0,50)}\n`); }
    }
  }
  await Promise.all(Array.from({ length: 4 }, worker));

  let tp = 0, fp = 0, tn = 0, fn = 0, nAb = 0, nH = 0;
  for (const r of results) {
    if (r.halted) { nH++; continue; }
    if (r.verdict === 'abstain') { nAb++; continue; }
    const pred = r.verdict === 'tb' ? 1 : 0;
    if (pred && r.truth) tp++; else if (pred && !r.truth) fp++; else if (!pred && !r.truth) tn++; else fn++;
  }
  const dec = tp + fp + tn + fn;
  const pct = (x) => Number.isNaN(x) ? 'n/a' : (x * 100).toFixed(1) + '%';
  const acc = dec ? (tp + tn) / dec : NaN, sens = (tp + fn) ? tp / (tp + fn) : NaN, spec = (tn + fp) ? tn / (tn + fp) : NaN;
  const a = auc(results.filter((r) => r.score != null && !r.halted).map((r) => ({ score: r.score, label: r.truth })));
  console.log('\n========== v2 RESULTS ==========');
  console.log(`Decided ${dec}  Abstain ${nAb}  Halt/err ${nH}`);
  console.log(`Confusion: TP=${tp} FP=${fp} TN=${tn} FN=${fn}`);
  console.log(`Accuracy ${pct(acc)} | Sensitivity ${pct(sens)} | Specificity ${pct(spec)} | AUC ${Number.isNaN(a) ? 'n/a' : a.toFixed(3)}`);
  console.log('================================');
  await writeFile(new URL('../accuracy-report-v2.json', import.meta.url), JSON.stringify({ policy: POLICY, K, results }, null, 2));
};
run().catch((e) => { console.error(e); process.exit(1); });
