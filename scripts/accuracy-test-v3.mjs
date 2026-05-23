/**
 * Measurement harness (v3): measures calibrated/conformal decision policy vs fixed policy.
 *   node scripts/accuracy-test-v3.mjs [N_PER_CLASS]
 *
 * Changes vs v2:
 *  1. Splits the built set 50/50 into calibration (even index) and test (odd index) halves.
 *  2. Runs the full pipeline on ALL images, collecting pVlm + label.
 *  3. Fits conformal thresholds (tauLow / tauHigh) on the calibration half using the same
 *     math as src/lib/calibration.ts fitConformalThresholds (alphaSens=0.92, gammaSpec=0.10).
 *  4. On the test half, derives the verdict from the fitted band:
 *       score >= tauHigh  -> tb
 *       score <  tauLow   -> no_tb
 *       else              -> abstain
 *     The GPT adjudicator severity-max combine from v2 is applied on top (can only escalate).
 *  5. Prints TWO result blocks on the test half:
 *       (a) baseline  – v2 fixed policy thresholds (tbFlag=0.30, negClear=0.15)
 *       (b) calibrated – fitted tauLow / tauHigh
 *  6. Saves accuracy-report-v3.json with both blocks + fitted thresholds.
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

// v2 fixed screening policy (baseline for before/after comparison)
const POLICY = { tbFlag: 0.30, negClear: 0.15, maxClearUncertainty: 0.15 };

// Conformal config matching src/lib/calibration.ts fitConformalThresholds defaults
const CONFORMAL_CFG = { alphaSens: 0.92, gammaSpec: 0.10 };

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
          input: [{ role: 'user', content: [{ type: 'input_text', text: prompt }, ...(dataUrl ? [{ type: 'input_image', image_url: dataUrl }] : [])] }],
          text: { format: { type: 'json_object' } },
        }),
      });
      if (!res.ok) throw new Error(`OpenAI ${res.status}`);
      return extractJSON(extractText(await res.json()));
    } catch (e) {
      if (model === FALLBACK) throw e;
    }
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
  const samples = await Promise.allSettled(
    Array.from({ length: K }, () => openaiJSON(VLM_PROMPT_V2, dataUrl)),
  );
  const probs = samples
    .filter((s) => s.status === 'fulfilled')
    .map((s) => clamp(Number(s.value.tb_probability)));
  if (!probs.length) throw new Error('all VLM samples failed');
  const pVlm = probs.reduce((a, b) => a + b, 0) / probs.length;
  const mean = pVlm;
  const u = Math.sqrt(probs.reduce((a, b) => a + (b - mean) ** 2, 0) / probs.length);
  const findings = samples.find((s) => s.status === 'fulfilled')?.value.findings ?? [];

  // GPT adjudicator
  const adjPrompt = [
    'You are a cautious TB screening adjudicator and safety net. A perception reader assessed the chest X-ray.',
    `Mean TB probability over ${K} reads: ${pVlm.toFixed(3)} (spread ${u.toFixed(3)}).`,
    `Findings: ${JSON.stringify(findings)}.`,
    'This is a screen; missing TB is far worse than a false alarm. Respond JSON only:',
    '{verdict: "tb"|"no_tb"|"abstain", confidence: 0-100, rationale: string}',
  ].join(' ');
  let modelVerdict = 'abstain';
  try {
    modelVerdict = (await openaiJSON(adjPrompt, dataUrl)).verdict ?? 'abstain';
  } catch { /* keep abstain */ }

  return { halted: false, score: pVlm, uncertainty: u, findings, modelVerdict };
}

// ---- Dataset helpers -------------------------------------------------------

async function rows(off, len) {
  const ds = 'darthPanda%2Fcombined-unknown-pneumonia-and-tuberculosis';
  const d = await (
    await fetch(
      `https://datasets-server.huggingface.co/rows?dataset=${ds}&config=default&split=train&offset=${off}&length=${len}`,
    )
  ).json();
  return d.rows.map((r) => ({
    label: r.row.label,
    img: Object.values(r.row).find((v) => v && typeof v === 'object' && v.src)?.src,
  }));
}
const get = async (u) => Buffer.from(await (await fetch(u)).arrayBuffer());

// ---- Metrics ---------------------------------------------------------------

function auc(items) {
  const pos = items.filter((i) => i.label === 1);
  const neg = items.filter((i) => i.label === 0);
  if (!pos.length || !neg.length) return NaN;
  const s = [...items].sort((a, b) => a.score - b.score);
  const r = [];
  let i = 0;
  while (i < s.length) {
    let j = i;
    while (j + 1 < s.length && s[j + 1].score === s[i].score) j++;
    const v = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[k] = v;
    i = j + 1;
  }
  let sr = 0;
  s.forEach((it, idx) => { if (it.label === 1) sr += r[idx]; });
  return (sr - (pos.length * (pos.length + 1)) / 2) / (pos.length * neg.length);
}

function evalResults(results) {
  let tp = 0, fp = 0, tn = 0, fn = 0, nAb = 0, nH = 0;
  for (const r of results) {
    if (r.halted) { nH++; continue; }
    if (r.verdict === 'abstain') { nAb++; continue; }
    const pred = r.verdict === 'tb' ? 1 : 0;
    if (pred && r.truth) tp++;
    else if (pred && !r.truth) fp++;
    else if (!pred && !r.truth) tn++;
    else fn++;
  }
  const dec = tp + fp + tn + fn;
  const acc = dec ? (tp + tn) / dec : NaN;
  const sens = (tp + fn) ? tp / (tp + fn) : NaN;
  const spec = (tn + fp) ? tn / (tn + fp) : NaN;
  const a = auc(
    results
      .filter((r) => r.score != null && !r.halted)
      .map((r) => ({ score: r.score, label: r.truth })),
  );
  return { tp, fp, tn, fn, nAb, nH, dec, acc, sens, spec, auc: a };
}

// ---- Conformal threshold fitting (mirrors calibration.ts fitConformalThresholds) -----------

function fitConformalThresholds(scores, labels, cfg) {
  const { alphaSens, gammaSpec } = cfg;
  const beta = 1 - alphaSens;

  const pos = scores.filter((_, i) => labels[i] === 1).sort((a, b) => a - b);
  const neg = scores.filter((_, i) => labels[i] === 0).sort((a, b) => a - b);
  const nPos = pos.length;
  const nNeg = neg.length;

  let tauLow = 0;
  if (nPos > 0) {
    const k = Math.floor(beta * (nPos + 1));
    tauLow = k <= 0 ? 0 : (pos[k - 1] ?? 0);
  }

  let tauHigh = 1;
  if (nNeg > 0) {
    const m = Math.floor(gammaSpec * (nNeg + 1));
    tauHigh = m <= 0 ? 1 : (neg[nNeg - m] ?? 1);
  }

  tauHigh = Math.max(tauHigh, tauLow); // never invert the band
  tauLow = clamp(tauLow, 0, 1);
  tauHigh = clamp(tauHigh, 0, 1);

  return { tauLow, tauHigh, nPos, nNeg };
}

// ---- Verdict derivation ----------------------------------------------------

/** v2 fixed policy verdict (baseline). */
function baselineVerdict(score, uncertainty, modelVerdict) {
  let policyVerdict;
  if (score >= POLICY.tbFlag) policyVerdict = 'tb';
  else if (score <= POLICY.negClear && uncertainty <= POLICY.maxClearUncertainty) policyVerdict = 'no_tb';
  else policyVerdict = 'abstain';
  // Safety-net combine: take the more cautious (higher severity).
  return fromSeverity[Math.max(severity[policyVerdict] ?? 1, severity[modelVerdict] ?? 1)];
}

/** Conformal band verdict, GPT adjudicator can only escalate. */
function calibratedVerdict(score, modelVerdict, tauLow, tauHigh) {
  let bandVerdict;
  if (score >= tauHigh) bandVerdict = 'tb';
  else if (score < tauLow) bandVerdict = 'no_tb';
  else bandVerdict = 'abstain';
  // Safety-net combine: take the more cautious (higher severity).
  return fromSeverity[Math.max(severity[bandVerdict] ?? 1, severity[modelVerdict] ?? 1)];
}

// ---- Main ------------------------------------------------------------------

const run = async () => {
  console.log(`v3: ${N} TB + ${N} NORMAL, K=${K} self-consistency`);
  console.log(`    Calibration: even indices | Test: odd indices`);
  console.log(`    Conformal cfg: alphaSens=${CONFORMAL_CFG.alphaSens}, gammaSpec=${CONFORMAL_CFG.gammaSpec}`);
  console.log(`    Baseline policy: ${JSON.stringify(POLICY)}\n`);

  // Fetch the combined set (same offsets as v2 for reproducibility)
  const tbRows = (await rows(8500, N + 4)).filter((r) => r.label === 2 && r.img).slice(0, N).map((r) => ({ ...r, truth: 1 }));
  const normRows = (await rows(300, N + 8)).filter((r) => r.label === 0 && r.img).slice(0, N).map((r) => ({ ...r, truth: 0 }));

  // Interleave TB and NORMAL so index parity splits each class evenly
  const allItems = [];
  for (let i = 0; i < Math.max(tbRows.length, normRows.length); i++) {
    if (i < tbRows.length) allItems.push({ ...tbRows[i], origClass: 'TB' });
    if (i < normRows.length) allItems.push({ ...normRows[i], origClass: 'NORMAL' });
  }

  // Run pipeline on ALL images
  const rawResults = new Array(allItems.length).fill(null);
  let done = 0;
  const queue = allItems.map((item, idx) => ({ item, idx }));

  async function worker() {
    while (queue.length) {
      const { item, idx } = queue.shift();
      try {
        const bytes = await get(item.img);
        const r = await runOne(bytes);
        rawResults[idx] = { truth: item.truth, origClass: item.origClass, splitIdx: idx, ...r };
        done++;
        process.stdout.write(
          `  [${done}/${allItems.length}] ${item.truth ? 'TB ' : 'NEG'} (${idx % 2 === 0 ? 'CAL' : 'TST'}) -> ${r.halted ? 'HALT' : `p=${r.score?.toFixed(3) ?? '-'}`}\n`,
        );
      } catch (e) {
        rawResults[idx] = { truth: item.truth, origClass: item.origClass, splitIdx: idx, halted: true, verdict: null, score: null };
        done++;
        process.stdout.write(
          `  [${done}/${allItems.length}] ${item.truth ? 'TB' : 'NEG'} (${idx % 2 === 0 ? 'CAL' : 'TST'}) -> ERR ${e.message.slice(0, 50)}\n`,
        );
      }
    }
  }
  await Promise.all(Array.from({ length: 4 }, worker));

  // Split by index parity
  const calItems = rawResults.filter((r, i) => i % 2 === 0 && r != null && !r.halted && r.score != null);
  const tstItems = rawResults.filter((r, i) => i % 2 !== 0 && r != null);

  console.log(`\nCalibration half: ${calItems.length} scored items`);
  console.log(`Test half: ${tstItems.length} items`);

  // Fit conformal thresholds on the calibration half
  const calScores = calItems.map((r) => r.score);
  const calLabels = calItems.map((r) => r.truth);
  const { tauLow, tauHigh, nPos: calNPos, nNeg: calNNeg } = fitConformalThresholds(
    calScores,
    calLabels,
    CONFORMAL_CFG,
  );

  console.log(`\nFitted conformal thresholds:`);
  console.log(`  tauLow  = ${tauLow.toFixed(4)}  (from ${calNPos} positives)`);
  console.log(`  tauHigh = ${tauHigh.toFixed(4)}  (from ${calNNeg} negatives)`);

  // Derive both verdicts for test-half items
  const baselineResults = tstItems.map((r) => {
    if (r.halted) return { truth: r.truth, halted: true, verdict: null, score: null };
    const v = baselineVerdict(r.score, r.uncertainty ?? 0, r.modelVerdict ?? 'abstain');
    return { truth: r.truth, halted: false, verdict: v, score: r.score };
  });
  const calibratedResults = tstItems.map((r) => {
    if (r.halted) return { truth: r.truth, halted: true, verdict: null, score: null };
    const v = calibratedVerdict(r.score, r.modelVerdict ?? 'abstain', tauLow, tauHigh);
    return { truth: r.truth, halted: false, verdict: v, score: r.score };
  });

  // Compute metrics
  const bMetrics = evalResults(baselineResults);
  const cMetrics = evalResults(calibratedResults);

  const pct = (x) => (Number.isNaN(x) ? 'n/a' : (x * 100).toFixed(1) + '%');
  const fmt = (m) =>
    `Decided ${m.dec}  Abstain ${m.nAb}  Halt/err ${m.nH}\n` +
    `  Confusion: TP=${m.tp} FP=${m.fp} TN=${m.tn} FN=${m.fn}\n` +
    `  Accuracy ${pct(m.acc)} | Sensitivity ${pct(m.sens)} | Specificity ${pct(m.spec)} | AUC ${Number.isNaN(m.auc) ? 'n/a' : m.auc.toFixed(3)}`;

  console.log('\n========== v3 TEST-HALF RESULTS (baseline fixed policy) ==========');
  console.log(fmt(bMetrics));
  console.log('==================================================================');

  console.log('\n========== v3 TEST-HALF RESULTS (calibrated conformal) ===========');
  console.log(`  tauLow=${tauLow.toFixed(4)}  tauHigh=${tauHigh.toFixed(4)}  (calNPos=${calNPos}  calNNeg=${calNNeg})`);
  console.log(fmt(cMetrics));
  console.log('==================================================================');

  // Save report
  const report = {
    version: 3,
    N,
    K,
    conformalCfg: CONFORMAL_CFG,
    baselinePolicy: POLICY,
    calibration: {
      nItems: calItems.length,
      nPos: calNPos,
      nNeg: calNNeg,
      tauLow,
      tauHigh,
    },
    testHalf: {
      nItems: tstItems.length,
      baseline: {
        policy: POLICY,
        ...bMetrics,
      },
      calibrated: {
        tauLow,
        tauHigh,
        ...cMetrics,
      },
    },
    rawResults,
  };
  await writeFile(
    new URL('../accuracy-report-v3.json', import.meta.url),
    JSON.stringify(report, null, 2),
  );
  console.log('\nSaved accuracy-report-v3.json');
};

run().catch((e) => { console.error(e); process.exit(1); });
