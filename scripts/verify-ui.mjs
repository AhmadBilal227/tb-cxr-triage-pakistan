/**
 * Visual verification harness. Seeds settings (accepted disclaimer + BYO
 * OpenAI key from the OPENAI_KEY env var — NEVER hardcoded here) so the full
 * analyzed flow runs, then screenshots: verdict + canvas overlays, collapsed
 * summary bar, lightbox, and the unified report. Screenshots -> /tmp/tbverify.
 *
 * Run: OPENAI_KEY=sk-... node scripts/verify-ui.mjs
 */
import puppeteer from 'puppeteer';
import { mkdirSync } from 'node:fs';

const OUT = '/tmp/tbverify';
mkdirSync(OUT, { recursive: true });
const URL = 'http://localhost:5173';
const KEY = process.env.OPENAI_KEY ?? '';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 160)); });
  page.on('pageerror', (e) => errors.push('PAGEERR ' + e.message.slice(0, 160)));

  // Seed settings BEFORE app scripts run; the store deep-merges with defaults.
  await page.evaluateOnNewDocument((key) => {
    localStorage.setItem(
      'tb-triage.settings.v1',
      JSON.stringify({ acceptedDisclaimer: true, openaiKey: key, localMode: true, localServerUrl: 'http://localhost:8001' }),
    );
  }, KEY);

  await page.goto(URL, { waitUntil: 'networkidle2', timeout: 30000 });
  await sleep(400);

  // Pick the first real demo sample.
  await page.evaluate(() => {
    const s = [...document.querySelectorAll('button[title]')].find((b) => /·/.test(b.getAttribute('title') || ''));
    s?.click();
  });

  // Wait for the verdict (the always-present caveat line).
  await page.waitForFunction(
    () => /RADIOGRAPHIC TB SCREEN/i.test(document.body.innerText),
    { timeout: 45000 },
  );
  await sleep(1200);
  await page.screenshot({ path: `${OUT}/10-verdict-overlays.png` });
  console.log('shot 10-verdict-overlays');

  const overlays = await page.evaluate(() => ({
    heatmap: !!document.querySelector('[data-testid="heatmap-overlay"]'),
    zones: !!document.querySelector('[data-testid="zone-overlay"]'),
    toolbarToggles: [...document.querySelectorAll('[data-testid="viewer-toolbar"] button')]
      .filter((b) => /heatmap|zone/i.test(b.getAttribute('aria-label') || ''))
      .map((b) => ({ label: b.getAttribute('aria-label'), disabled: b.disabled, pressed: b.getAttribute('aria-pressed') })),
    verdict: document.body.innerText.match(/NO TB|TB SUSPECTED|UNCERTAIN/i)?.[0] ?? '?',
  }));
  console.log('overlays:', JSON.stringify(overlays));

  // Collapse the verdict to the sticky summary bar.
  await page.evaluate(() => document.querySelector('[aria-label="Collapse findings to use the viewer"]')?.click());
  await sleep(600);
  await page.screenshot({ path: `${OUT}/11-collapsed.png` });
  const collapsed = await page.evaluate(() => !!document.querySelector('[data-testid="verdict-summary-bar"]'));
  console.log('shot 11-collapsed; summary-bar present:', collapsed);

  // Re-expand, then open the lightbox from the heatmap fullscreen control.
  await page.evaluate(() => document.querySelector('[data-testid="verdict-summary-bar"]')?.click());
  await sleep(500);
  await page.evaluate(() => document.querySelector('[data-testid="box-evidence-fullscreen"]')?.click());
  await page.waitForSelector('[data-testid="image-lightbox"]', { timeout: 8000 }).catch(() => {});
  await sleep(900);
  await page.screenshot({ path: `${OUT}/12-lightbox.png` });
  const lightbox = await page.evaluate(() => ({
    open: !!document.querySelector('[data-testid="image-lightbox"]'),
    heatmap: !!document.querySelector('[data-testid="image-lightbox"] [data-testid="heatmap-overlay"]'),
    zoneChips: document.querySelectorAll('[data-testid="image-lightbox"] [data-testid^="zone-"]').length,
    panel: !!document.querySelector('[data-testid="lightbox-side-panel"]'),
  }));
  console.log('shot 12-lightbox:', JSON.stringify(lightbox));

  await page.keyboard.press('Escape');
  await sleep(500);

  // Generate the unified report (real OpenAI call).
  await page.evaluate(() => document.querySelector('[data-testid="clinician-report-generate"]')?.click());
  const gotReport = await page
    .waitForSelector('[data-testid="clinician-report-view"]', { timeout: 45000 })
    .then(() => true)
    .catch(() => false);
  console.log('report ready:', gotReport);
  if (gotReport) {
    await page.evaluate(() => document.querySelector('[data-testid="clinician-report-view"]')?.click());
    await page.waitForSelector('[data-testid="clinician-report-modal"]', { timeout: 8000 }).catch(() => {});
    await sleep(900);
    await page.screenshot({ path: `${OUT}/13-report.png` });
    const report = await page.evaluate(() => ({
      headline: document.querySelector('[data-testid="clinician-report-headline"]')?.innerText?.slice(0, 140) ?? null,
      hasSupportDevices: !!document.querySelector('[data-testid="clinician-report-support-devices"]'),
      hasIncidentals: !!document.querySelector('[data-testid="clinician-report-incidentals"]'),
      schema: /gpt-interpreter-v3/.test(document.body.innerText),
    }));
    console.log('report:', JSON.stringify(report));
  } else {
    await page.screenshot({ path: `${OUT}/13-report-FAILED.png` });
    const err = await page.evaluate(() => document.querySelector('[data-testid="clinician-report-error"]')?.innerText ?? '(no error node)');
    console.log('report generation did not complete:', err.slice(0, 200));
  }

  console.log('console errors:', JSON.stringify(errors.slice(0, 8)));
  console.log('DONE');
} catch (err) {
  console.log('SCRIPT ERROR:', err.message);
} finally {
  await browser.close();
}
