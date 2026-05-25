/**
 * Single source of truth for the verdict triad — used by Tailwind config
 * (mirrored manually as `verdict-tb` / `verdict-clear` / `verdict-uncertain`
 * keys), the ImageLightbox zone-score color mapping, the PDF generator, and
 * the verdict card header tint.
 *
 * If you change a hex value here, mirror it in `tailwind.config.js`. Both
 * files name the same constants in their comments so a future contributor
 * grepping for `#C8102E` lands in both places.
 *
 * The triad is meaning-bearing per DESIGN.md's "Verdict Color Reservation
 * Rule": these colors appear only on verdicts, status, and danger states.
 * Never use them as decorative accents.
 */
export const VERDICT_HEX = {
  tb: '#C8102E',
  no_tb: '#00754A',
  abstain: '#F59E0B',
} as const;

/** RGB tuples for jsPDF (which takes [r, g, b] component arrays). */
export const VERDICT_RGB = {
  tb: [200, 16, 46],
  no_tb: [0, 117, 74],
  abstain: [245, 158, 11],
} as const satisfies Record<keyof typeof VERDICT_HEX, [number, number, number]>;

/**
 * Map a [0, 1] zone-probability score to one of the three verdict-triad
 * hexes. Used by ImageLightbox zone labels. Thresholds match the de-facto
 * "low / mid / high" breakdown the validated head's calibrated probabilities
 * fall into post-T scaling.
 */
export function scoreToTriadHex(v: number): string {
  const s = Math.max(0, Math.min(1, v));
  if (s < 0.2) return VERDICT_HEX.no_tb;
  if (s < 0.5) return VERDICT_HEX.abstain;
  return VERDICT_HEX.tb;
}
