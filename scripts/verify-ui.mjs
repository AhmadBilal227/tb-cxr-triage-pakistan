/**
 * Probe: read the actual transform/filter the viewer applies after clicking
 * its controls, and the pipeline DOM state. Ground truth over eyeballing.
 */
import puppeteer from 'puppeteer';

const URL = 'http://localhost:5173';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 160)); });
  page.on('pageerror', (e) => errors.push('PAGEERR ' + e.message.slice(0, 160)));

  await page.goto(URL, { waitUntil: 'networkidle2', timeout: 30000 });
  await page.waitForSelector('input[type=checkbox]', { timeout: 10000 });
  await page.click('input[type=checkbox]');
  await page.evaluate(() => {
    [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Continue')?.click();
  });
  await sleep(500);

  await page.evaluate(() => {
    const s = [...document.querySelectorAll('button[title]')].find((b) => /·/.test(b.getAttribute('title') || ''));
    s?.click();
  });
  await page.waitForSelector('[data-testid="viewer-toolbar"]', { timeout: 15000 });
  await sleep(500);

  const readTransform = () =>
    page.evaluate(() => {
      const img = document.querySelector('img[alt="Chest radiograph under analysis"]');
      const wrap = img?.parentElement;
      return {
        transform: wrap?.style.transform ?? '(none)',
        imgFilter: img?.style.filter ?? '(none)',
        pct: document.querySelector('[data-testid="viewer-toolbar"]')?.innerText?.match(/\d+%/)?.[0] ?? '?',
      };
    });

  console.log('before:', JSON.stringify(await readTransform()));
  await page.click('[aria-label="Zoom in"]');
  await page.click('[aria-label="Zoom in"]');
  await sleep(300);
  console.log('after 2x zoom-in:', JSON.stringify(await readTransform()));
  await page.click('[aria-label="Invert grayscale"]');
  await sleep(300);
  console.log('after invert:', JSON.stringify(await readTransform()));

  // Pipeline DOM state after giving it time.
  await sleep(5000);
  const pipeline = await page.evaluate(() => {
    const txt = document.body.innerText;
    return {
      hasVerdict: /RADIOGRAPHIC TB SCREEN|NO TB|TB SUSPECTED|UNCERTAIN/i.test(txt),
      hasHalt: /halt|API key|unavailable/i.test(txt),
      hasNoKeysBanner: /no .*key|set .*key|add .*key/i.test(txt),
      bodySample: txt.replace(/\s+/g, ' ').slice(0, 300),
    };
  });
  console.log('pipeline:', JSON.stringify(pipeline));
  console.log('console errors:', JSON.stringify(errors.slice(0, 6)));
} catch (err) {
  console.log('SCRIPT ERROR:', err.message);
} finally {
  await browser.close();
}
