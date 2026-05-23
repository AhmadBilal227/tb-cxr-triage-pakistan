/**
 * Accuracy harness: mirrors src/lib/pipeline/orchestrator.ts against the live APIs
 * to measure the CURRENT system's TB triage performance on a labeled test set.
 *
 *   node scripts/accuracy-test.mjs [N_PER_CLASS]
 *
 * Faithful to the app: GPT-5.5 quality gate -> ensemble (HF classifiers error on
 * free tier, GPT-5.5 vision is the live member) -> GPT-5.5 adjudicator -> guardrails.
 * RAG is run with an EMPTY corpus (a fresh user's default), so retrieval contributes
 * nothing and the retrieval abstain rule cannot fire — exactly as in the app on first use.
 */
import { readFile, writeFile } from 'node:fs/promises';

const env = Object.fromEntries(
  (await readFile(new URL('../.env.local', import.meta.url), 'utf8'))
    .split('\n').filter((l) => l.includes('=') && !l.trim().startsWith('#'))
    .map((l) => [l.slice(0, l.indexOf('=')).trim(), l.slice(l.indexOf('=') + 1).trim()]),
);
const OPENAI = env.VITE_OPENAI_KEY;
const HF = env.VITE_HF_TOKEN;
const MODEL = 'gpt-5.5';
const FALLBACK = 'gpt-5.5-instant';
const N = Number(process.argv[2] ?? 12); // per class

// ---- prompts copied verbatim from src/lib/pipeline/prompts.ts ----
const QUALITY_PROMPT =
  'Is this a frontal chest X-ray suitable for TB screening? Reply JSON only: {is_cxr: bool, quality: "good"|"poor"|"unreadable", reason: string}';
const VLM_PROMPT =
  'Are TB-associated findings (upper-zone consolidation, cavitation, miliary nodules, hilar lymphadenopathy) present? Respond JSON: {tb_likely: bool, findings: string[], confidence: 0..1}';
const ABSTAIN = { minConfidence: 75, maxStd: 0.2, minTop1: 0.6, maxDisagreeLowSim: 0.3, maxReplicate: 2 };

function extractText(j) {
  if (typeof j.output_text === 'string' && j.output_text) return j.output_text;
  let t = '';
  for (const o of j.output ?? []) for (const c of o.content ?? []) if (c.type === 'output_text') t += c.text;
  return t;
}
function extractJSON(text) {
  const s = text.indexOf('{'), e = text.lastIndexOf('}');
  return JSON.parse(text.slice(s, e + 1));
}
async function openaiJSON(prompt, dataUrl) {
  for (const model of [MODEL, FALLBACK]) {
    try {
      const res = await fetch('https://api.openai.com/v1/responses', {
        method: 'POST',
        headers: { Authorization: `Bearer ${OPENAI}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          input: [{ role: 'user', content: [
            { type: 'input_text', text: prompt },
            ...(dataUrl ? [{ type: 'input_image', image_url: dataUrl }] : []),
          ] }],
          text: { format: { type: 'json_object' } },
        }),
      });
      if (!res.ok) throw new Error(`OpenAI ${res.status}: ${(await res.text()).slice(0, 120)}`);
      return extractJSON(extractText(await res.json()));
    } catch (e) {
      if (model === FALLBACK) throw e;
    }
  }
}

// ---- faithful single-image pipeline ----
async function runOne(bytes) {
  const dataUrl = `data:image/jpeg;base64,${bytes.toString('base64')}`;

  // Stage 1 — quality gate
  const q = await openaiJSON(QUALITY_PROMPT, dataUrl);
  if (!q.is_cxr || q.quality === 'unreadable') {
    return { halted: true, reason: q.reason, verdict: null, score: null };
  }

  // Stage 2 — ensemble. HF TB + General classifiers error on free serverless
  // (verified: 400 "not supported by provider hf-inference"); only the VLM returns.
  const v = await openaiJSON(VLM_PROMPT, dataUrl);
  const vlmProb = Math.max(0, Math.min(1, v.tb_likely ? v.confidence : 1 - v.confidence));
  const members = [{ id: 'vlm', tb_prob: vlmProb, provider: 'openai' }];
  const probs = members.map((m) => m.tb_prob);
  const weighted = vlmProb; // single member -> weight renormalizes to 1
  const std = 0;
  const disagreement = 0;
  const replicateFallbacks = 0;

  // Stage 3 — RAG: empty corpus on a fresh install -> no neighbors
  const neighbors = [];

  // Stage 4 — adjudicator
  const adjPrompt = [
    'You are a cautious TB chest X-ray triage adjudicator. You are shown the query',
    'chest X-ray image plus the outputs of three independent perception models, a',
    'weighted ensemble score, and the most similar labeled cases retrieved by',
    'embedding similarity. Decide a triage verdict. Be conservative: abstain when',
    'signals conflict or evidence is weak.', '',
    'ENSEMBLE MEMBERS:', JSON.stringify(members, null, 2), '',
    `WEIGHTED ENSEMBLE SCORE (0..1): ${weighted.toFixed(3)}`,
    `ENSEMBLE STD: ${std.toFixed(3)}`,
    `ENSEMBLE DISAGREEMENT (max-min): ${disagreement.toFixed(3)}`, '',
    'NEAREST LABELED CASES (cosine similarity):',
    neighbors.length ? JSON.stringify(neighbors) : '(none — retrieval skipped or empty corpus)', '',
    `PROVIDERS USED PER STAGE: quality:openai, ensemble.vlm:openai`,
    `STAGES THAT FELL BACK TO REPLICATE: ${replicateFallbacks}`, '',
    'Respond JSON only:',
    '{verdict: "tb"|"no_tb"|"abstain", confidence: 0-100, rationale: string (<=3 sentences), abstain_reason?: string}',
  ].join('\n');
  const adj = await openaiJSON(adjPrompt, dataUrl);

  // deterministic guardrails (faithful)
  const reasons = [];
  if (adj.confidence < ABSTAIN.minConfidence) reasons.push('confidence');
  if (std > ABSTAIN.maxStd) reasons.push('std');
  const top1 = neighbors[0]?.similarity ?? 1;
  if (top1 < ABSTAIN.minTop1 && disagreement > ABSTAIN.maxDisagreeLowSim) reasons.push('retrieval');
  if (replicateFallbacks >= ABSTAIN.maxReplicate) reasons.push('replicate');
  const verdict = reasons.length ? 'abstain' : adj.verdict;

  return { halted: false, verdict, confidence: adj.confidence, score: weighted, vlmProb, autoAbstain: reasons };
}

// ---- dataset: balanced TB vs NORMAL from darthPanda (held-out slice) ----
async function rows(off, len) {
  const ds = 'darthPanda%2Fcombined-unknown-pneumonia-and-tuberculosis';
  const u = `https://datasets-server.huggingface.co/rows?dataset=${ds}&config=default&split=train&offset=${off}&length=${len}`;
  const d = await (await fetch(u)).json();
  return d.rows.map((r) => {
    const row = r.row;
    const img = Object.values(row).find((v) => v && typeof v === 'object' && v.src)?.src;
    return { label: row.label, img };
  });
}
async function get(url) {
  return Buffer.from(await (await fetch(url)).arrayBuffer());
}

// rank-based ROC AUC
function auc(items) {
  const pos = items.filter((i) => i.label === 1), neg = items.filter((i) => i.label === 0);
  if (!pos.length || !neg.length) return NaN;
  const sorted = [...items].sort((a, b) => a.score - b.score);
  const ranks = []; let i = 0;
  while (i < sorted.length) {
    let j = i; while (j + 1 < sorted.length && sorted[j + 1].score === sorted[i].score) j++;
    const r = (i + j) / 2 + 1; for (let k = i; k <= j; k++) ranks[k] = r; i = j + 1;
  }
  let sr = 0; sorted.forEach((it, idx) => { if (it.label === 1) sr += ranks[idx]; });
  return (sr - (pos.length * (pos.length + 1)) / 2) / (pos.length * neg.length);
}

const run = async () => {
  console.log(`Building balanced test set: ${N} TB + ${N} NORMAL (held-out slice)...`);
  const tb = (await rows(8500, N + 4)).filter((r) => r.label === 2 && r.img).slice(0, N).map((r) => ({ ...r, truth: 1 }));
  const norm = (await rows(300, N + 8)).filter((r) => r.label === 0 && r.img).slice(0, N).map((r) => ({ ...r, truth: 0 }));
  const set = [...tb, ...norm];
  console.log(`Got ${tb.length} TB + ${norm.length} NORMAL = ${set.length} images. Running pipeline...\n`);

  const results = [];
  let done = 0;
  const POOL = 3;
  const queue = [...set];
  async function worker() {
    while (queue.length) {
      const item = queue.shift();
      try {
        const bytes = await get(item.img);
        const r = await runOne(bytes);
        results.push({ truth: item.truth, ...r });
        done++;
        process.stdout.write(`  [${done}/${set.length}] truth=${item.truth ? 'TB' : 'NEG'} -> ${r.halted ? 'HALTED' : r.verdict} (conf ${r.confidence ?? '-'}, score ${r.score?.toFixed(2) ?? '-'})\n`);
      } catch (e) {
        results.push({ truth: item.truth, halted: true, error: e.message, verdict: null, score: null });
        done++;
        process.stdout.write(`  [${done}/${set.length}] truth=${item.truth ? 'TB' : 'NEG'} -> ERROR ${e.message.slice(0, 60)}\n`);
      }
    }
  }
  await Promise.all(Array.from({ length: POOL }, worker));

  // metrics (faithful to src/lib/metrics.ts)
  let tp = 0, fp = 0, tn = 0, fn = 0, nAbstain = 0, nHalted = 0;
  for (const r of results) {
    if (r.halted) { nHalted++; continue; }
    if (r.verdict === 'abstain' || r.verdict == null) { nAbstain++; continue; }
    const pred = r.verdict === 'tb' ? 1 : 0;
    if (pred === 1 && r.truth === 1) tp++;
    else if (pred === 1 && r.truth === 0) fp++;
    else if (pred === 0 && r.truth === 0) tn++;
    else fn++;
  }
  const decided = tp + fp + tn + fn;
  const acc = decided ? (tp + tn) / decided : NaN;
  const sens = tp + fn ? tp / (tp + fn) : NaN;
  const spec = tn + fp ? tn / (tn + fp) : NaN;
  const aucItems = results.filter((r) => r.score != null && !r.halted).map((r) => ({ score: r.score, label: r.truth }));
  const a = auc(aucItems);
  const pct = (x) => (Number.isNaN(x) ? 'n/a' : (x * 100).toFixed(1) + '%');

  const report = {
    generatedAt: new Date().toISOString(),
    config: 'current system (GPT-5.5 quality+vision+adjudicator; HF CXR classifiers down on free tier; RAG empty corpus)',
    n_total: results.length, n_decided: decided, n_abstain: nAbstain, n_halted: nHalted,
    confusion: { tp, fp, tn, fn },
    accuracy: acc, sensitivity: sens, specificity: spec, auc: a,
  };
  console.log('\n========== RESULTS ==========');
  console.log(`Test set:     ${results.length} images (${tb.length} TB / ${norm.length} NEG)`);
  console.log(`Decided:      ${decided}   Abstained: ${nAbstain}   Halted/err: ${nHalted}`);
  console.log(`Confusion:    TP=${tp} FP=${fp} TN=${tn} FN=${fn}`);
  console.log(`Accuracy:     ${pct(acc)}  (of decided cases)`);
  console.log(`Sensitivity:  ${pct(sens)}  (TB recall — the safety-critical metric)`);
  console.log(`Specificity:  ${pct(spec)}`);
  console.log(`AUC:          ${Number.isNaN(a) ? 'n/a' : a.toFixed(3)}  (on VLM ensemble score)`);
  console.log('=============================');
  await writeFile(new URL('../accuracy-report.json', import.meta.url), JSON.stringify({ report, results }, null, 2));
  console.log('saved accuracy-report.json');
};
run().catch((e) => { console.error(e); process.exit(1); });
