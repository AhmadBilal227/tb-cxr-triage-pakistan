/**
 * One-off tooling: download public-domain / CC chest X-rays from Wikimedia Commons
 * into public/samples/ for use as in-app demo samples. Not part of the app bundle.
 *
 *   node scripts/fetch-samples.mjs
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const API = 'https://commons.wikimedia.org/w/api.php';
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(root, 'public', 'samples');

async function api(params) {
  const url = new URL(API);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  url.searchParams.set('format', 'json');
  url.searchParams.set('origin', '*');
  const res = await fetch(url, { headers: { 'User-Agent': 'tb-triage-demo/0.1 (research)' } });
  if (!res.ok) throw new Error(`API ${res.status} for ${url}`);
  return res.json();
}

async function imageInfo(titles) {
  const data = await api({
    action: 'query',
    titles: titles.join('|'),
    prop: 'imageinfo',
    iiprop: 'url|size|extmetadata',
  });
  const pages = Object.values(data.query?.pages ?? {});
  return pages
    .map((p) => {
      const ii = p.imageinfo?.[0];
      if (!ii) return null;
      const ext = ii.extmetadata ?? {};
      const strip = (s) => (s ? String(s).replace(/<[^>]+>/g, '').trim() : '');
      return {
        title: p.title,
        url: ii.url,
        width: ii.width,
        height: ii.height,
        mime: ii.mime,
        license: strip(ext.LicenseShortName?.value),
        artist: strip(ext.Artist?.value),
        descUrl: ii.descriptionurl,
      };
    })
    .filter(Boolean);
}

async function categoryFiles(cat) {
  const data = await api({
    action: 'query',
    list: 'categorymembers',
    cmtitle: cat,
    cmtype: 'file',
    cmlimit: '80',
  });
  return (data.query?.categorymembers ?? []).map((m) => m.title);
}

async function download(url, dest) {
  const res = await fetch(url, { headers: { 'User-Agent': 'tb-triage-demo/0.1 (research)' } });
  if (!res.ok) throw new Error(`download ${res.status} for ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(dest, buf);
  return buf.length;
}

const run = async () => {
  await mkdir(outDir, { recursive: true });

  // --- TB candidates from the category, pick the largest jpg/png ---
  const tbTitles = await categoryFiles('Category:X-rays of lung tuberculosis');
  const tbInfo = (await imageInfo(tbTitles.slice(0, 60)))
    .filter((i) => /jpe?g|png/i.test(i.mime))
    .sort((a, b) => b.width * b.height - a.width * a.height);

  // --- Normal chest X-ray ---
  const normalInfo = await imageInfo([
    'File:Normal posteroanterior (PA) chest radiograph (X-ray).jpg',
    'File:Chest X-ray plain film normal.jpg',
  ]);

  const picks = [];
  if (tbInfo[0]) picks.push({ ...tbInfo[0], label: 1, slug: 'tb-sample-1' });
  if (tbInfo[1]) picks.push({ ...tbInfo[1], label: 1, slug: 'tb-sample-2' });
  const normal = normalInfo.sort((a, b) => b.width * b.height - a.width * a.height)[0];
  if (normal) picks.push({ ...normal, label: 0, slug: 'normal-sample-1' });

  const manifest = [];
  for (const p of picks) {
    const ext = p.mime.includes('png') ? 'png' : 'jpg';
    const file = `${p.slug}.${ext}`;
    const bytes = await download(p.url, join(outDir, file));
    manifest.push({
      file,
      label: p.label,
      width: p.width,
      height: p.height,
      title: p.title,
      license: p.license,
      attribution: p.artist,
      source: p.descUrl,
    });
    console.log(`saved ${file}  ${p.width}x${p.height}  ${(bytes / 1024).toFixed(0)}KB  [${p.license}]`);
  }

  await writeFile(join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 2));
  console.log(`\nmanifest.json written with ${manifest.length} samples`);
};

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
