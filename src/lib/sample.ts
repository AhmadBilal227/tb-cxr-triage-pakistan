/**
 * Synthetic chest-X-ray-like sample for the "Try sample" empty-state action.
 *
 * NOTE: This is procedurally drawn, NOT a real radiograph. It exists so the UI
 * and pipeline wiring can be exercised without shipping licensed medical imagery.
 * The Stage 1 quality gate may (correctly) flag it as not a clinical CXR. For real
 * evaluation, import the Montgomery/Shenzhen sets — see the README.
 */
export async function makeSampleCXR(): Promise<{ blob: Blob; name: string }> {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('canvas 2d context unavailable');

  // Dark radiographic background
  const bg = ctx.createRadialGradient(size / 2, size / 2, 40, size / 2, size / 2, size * 0.7);
  bg.addColorStop(0, '#2a2a2a');
  bg.addColorStop(1, '#050505');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, size, size);

  // Two lung fields (lighter, semi-transparent ellipses)
  ctx.fillStyle = 'rgba(150,150,150,0.35)';
  for (const cx of [size * 0.34, size * 0.66]) {
    ctx.beginPath();
    ctx.ellipse(cx, size * 0.52, size * 0.16, size * 0.26, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  // Spine + mediastinum
  ctx.fillStyle = 'rgba(200,200,200,0.25)';
  ctx.fillRect(size * 0.48, size * 0.2, size * 0.04, size * 0.6);

  // Faint rib arcs
  ctx.strokeStyle = 'rgba(220,220,220,0.12)';
  ctx.lineWidth = 2;
  for (let i = 0; i < 6; i++) {
    const y = size * 0.32 + i * size * 0.07;
    ctx.beginPath();
    ctx.moveTo(size * 0.2, y);
    ctx.quadraticCurveTo(size * 0.5, y - 18, size * 0.8, y);
    ctx.stroke();
  }

  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png'),
  );
  return { blob, name: 'synthetic-sample.png' };
}
