/**
 * Puppeteer E2E: validated local model running behind the existing UI.
 *
 * Pre-conditions (orchestrated by the launcher bash):
 *   - FastAPI server on http://localhost:8001 (training/server.py)
 *   - Vite dev server on http://localhost:5173 (npm run dev)
 *
 * Flow:
 *   1. Open the app
 *   2. Seed localStorage with localMode=true + localServerUrl=:8001 (the storage
 *      key is `tb-triage.settings.v1`, deep-merged with DEFAULT_SETTINGS on load).
 *      This bypasses the click-toggle-type-wait UI sequence — that route is
 *      fragile across drawer animations + portal mounts; we don't need it for
 *      this test, we just need the validated-model perception path to be ON.
 *   3. Reload — the app picks up the seed and re-pings /health.
 *   4. Dispatch a synthetic `DataTransfer` drop with tb-sample-1.jpg.
 *   5. Wait for `[data-testid="local-mode-disclosure"]` (proves the local
 *      perception path won — VLM disclosure is on a different testid).
 *   6. Assert verdict text + perception-path indicator + disclosure content.
 *   7. Reload, drop normal-sample-1.jpg, assert NO TB.
 *   8. Screenshot at every key step.
 */
import puppeteer from 'puppeteer';
import { readFileSync, mkdirSync } from 'node:fs';

const FRONTEND = 'http://localhost:5173';
const SERVER_URL = 'http://localhost:8001';
const ROOT = '/Users/ahmadbilal/Downloads/hobby/TB detector';
const SAMPLES = `${ROOT}/public/samples`;
const OUT = '/tmp/e2e-screenshots';
const STORAGE_KEY = 'tb-triage.settings.v1';

mkdirSync(OUT, { recursive: true });

const results = [];
const assert = (cond, msg) => {
  results.push({ ok: !!cond, msg });
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${msg}`);
  if (!cond) process.exitCode = 1;
};

async function dropFile(page, filename, contentType) {
  const b64 = readFileSync(`${SAMPLES}/${filename}`).toString('base64');
  await page.evaluate(
    async ({ b64, name, type }) => {
      const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const file = new File([bin.buffer], name, { type });
      const dt = new DataTransfer();
      dt.items.add(file);
      const fire = (kind) =>
        document.dispatchEvent(
          new DragEvent(kind, { bubbles: true, cancelable: true, dataTransfer: dt }),
        );
      fire('dragenter');
      fire('dragover');
      fire('drop');
    },
    { b64, name: filename, type: contentType },
  );
}

async function dumpDiagnostics(page, label) {
  const path = `${OUT}/diag-${label}.png`;
  await page.screenshot({ path, fullPage: true });
  const testids = await page.evaluate(() =>
    [...document.querySelectorAll('[data-testid]')].map((e) => e.getAttribute('data-testid')),
  );
  console.log(`  [diag] screenshot: ${path}`);
  console.log(`  [diag] testids visible: ${JSON.stringify(testids)}`);
}

const browser = await puppeteer.launch({
  headless: 'new',
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 1000 });
  page.on('pageerror', (err) => console.log(`  [page-uncaught] ${err.message}`));
  page.on('console', (msg) => {
    if (msg.type() === 'error') console.log(`  [page-error] ${msg.text()}`);
  });

  console.log('=== 1. open app + seed localStorage with Local mode ON ===');
  await page.goto(FRONTEND, { waitUntil: 'networkidle2', timeout: 30000 });
  await page.evaluate(
    ({ key, url }) => {
      // Deep-merged on load with DEFAULT_SETTINGS in src/store/settings.ts:loadInitial.
      // We only need to override the local-mode toggle + URL — everything else inherits.
      localStorage.setItem(key, JSON.stringify({ localMode: true, localServerUrl: url }));
    },
    { key: STORAGE_KEY, url: SERVER_URL },
  );
  console.log('  seeded localStorage; reloading…');
  await page.reload({ waitUntil: 'networkidle2', timeout: 30000 });

  // Sanity: settings persisted across reload
  const seeded = await page.evaluate((k) => JSON.parse(localStorage.getItem(k) ?? '{}'), STORAGE_KEY);
  assert(seeded.localMode === true, 'localStorage seeded: localMode is true after reload');
  assert(
    seeded.localServerUrl === SERVER_URL,
    `localStorage seeded: localServerUrl is ${SERVER_URL}`,
  );
  await page.screenshot({ path: `${OUT}/01-app-loaded.png`, fullPage: true });

  console.log('\n=== 2. drop TB sample (synthetic DataTransfer on document) ===');
  await dropFile(page, 'tb-sample-1.jpg', 'image/jpeg');

  console.log('\n=== 3. wait for local-mode-disclosure (proves local path won) ===');
  try {
    await page.waitForSelector('[data-testid="local-mode-disclosure"]', { timeout: 60000 });
  } catch {
    await dumpDiagnostics(page, 'tb-no-local-disclosure');
    throw new Error('local-mode-disclosure never rendered for TB sample');
  }
  await new Promise((r) => setTimeout(r, 1500));

  const tbVerdict = await page.$eval('.text-3xl', (el) => el.textContent?.trim());
  const tbPath = await page.$eval('[data-testid="perception-path-indicator"]', (el) =>
    el.textContent?.trim(),
  );
  const tbDisclosure = await page.$eval('[data-testid="local-mode-disclosure"]', (el) =>
    el.textContent?.trim(),
  );
  const vlmDisclosure = await page.$('[data-testid="vlm-primary-disclosure"]');
  console.log(`  verdict:    "${tbVerdict}"`);
  console.log(`  path:       "${tbPath}"`);
  console.log(`  validated-model disclosure: ${tbDisclosure?.slice(0, 60)}…`);

  assert(tbVerdict === 'TB SUSPECTED', `TB sample verdict = "TB SUSPECTED" (got "${tbVerdict}")`);
  assert(
    tbPath?.toLowerCase().includes('local'),
    `perception-path-indicator mentions "local" (got "${tbPath}")`,
  );
  assert(tbDisclosure?.includes('Rad-DINO'), 'disclosure mentions Rad-DINO');
  assert(
    tbDisclosure?.includes('TorchXRayVision'),
    'disclosure mentions TorchXRayVision (validated model)',
  );
  assert(!vlmDisclosure, 'VLM-primary disclosure is NOT shown (the local path won)');
  await page.screenshot({ path: `${OUT}/02-tb-verdict.png`, fullPage: true });

  console.log('\n=== 4. reload + drop normal sample (settings persist) ===');
  await page.reload({ waitUntil: 'networkidle2' });
  await new Promise((r) => setTimeout(r, 800));
  await dropFile(page, 'normal-sample-1.jpg', 'image/jpeg');

  try {
    await page.waitForSelector('[data-testid="local-mode-disclosure"]', { timeout: 60000 });
  } catch {
    await dumpDiagnostics(page, 'normal-no-local-disclosure');
    throw new Error('local-mode-disclosure never rendered for normal sample');
  }
  await new Promise((r) => setTimeout(r, 1500));

  const noVerdict = await page.$eval('.text-3xl', (el) => el.textContent?.trim());
  const noPath = await page.$eval('[data-testid="perception-path-indicator"]', (el) =>
    el.textContent?.trim(),
  );
  console.log(`  verdict:    "${noVerdict}"`);
  console.log(`  path:       "${noPath}"`);
  assert(noVerdict === 'NO TB', `normal sample verdict = "NO TB" (got "${noVerdict}")`);
  assert(noPath?.toLowerCase().includes('local'), 'perception-path-indicator mentions "local"');
  await page.screenshot({ path: `${OUT}/03-normal-verdict.png`, fullPage: true });

  console.log('\n=== screenshots ===');
  console.log(`  ${OUT}/01-app-loaded.png`);
  console.log(`  ${OUT}/02-tb-verdict.png`);
  console.log(`  ${OUT}/03-normal-verdict.png`);
} finally {
  await browser.close();
}

const passed = results.filter((r) => r.ok).length;
const total = results.length;
console.log(`\n=== ${passed}/${total} assertions passed ===`);
if (process.exitCode) {
  console.log('FAILED — see assertions above');
} else {
  console.log('ALL PASS');
}
