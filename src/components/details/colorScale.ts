/**
 * Perceptually-uniform color scale for the M24 evidence panels.
 *
 * Choice: VIRIDIS (purple → blue → green → yellow). Reasons:
 *   - Monotonic in luminance — bright cells stand out without hue confusion.
 *   - Colorblind-friendly (deuteranopia / protanopia / tritanopia).
 *   - Same family used by matplotlib/d3 viridis; widely recognized in
 *     medical imaging visualization.
 *
 * The exact stops come from the cubehelix-derived viridis polynomial; we keep a
 * 11-stop linear interpolation here (dependency-free, runs in the same
 * server-side renderToStaticMarkup pass the existing tests use).
 */

const VIRIDIS_STOPS: ReadonlyArray<readonly [number, number, number]> = [
  [68, 1, 84],     // 0.0
  [72, 35, 116],   // 0.1
  [64, 67, 135],   // 0.2
  [52, 94, 141],   // 0.3
  [41, 120, 142],  // 0.4
  [32, 144, 140],  // 0.5
  [34, 167, 132],  // 0.6
  [68, 190, 112],  // 0.7
  [121, 209, 81],  // 0.8
  [189, 222, 38],  // 0.9
  [253, 231, 36],  // 1.0
];

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

/** Map a [0,1] score to an `rgb(r,g,b)` CSS string via the viridis ramp. */
export function viridis(v: number): string {
  const t = clamp01(v) * (VIRIDIS_STOPS.length - 1);
  const i = Math.min(VIRIDIS_STOPS.length - 2, Math.floor(t));
  const f = t - i;
  const a = VIRIDIS_STOPS[i]!;
  const b = VIRIDIS_STOPS[i + 1]!;
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r}, ${g}, ${bl})`;
}

/** Severity bucket for the TXRV pathology chips (green/amber/red, three thresholds). */
export function severityClass(v: number): 'low' | 'mid' | 'high' {
  if (v >= 0.5) return 'high';
  if (v >= 0.2) return 'mid';
  return 'low';
}
