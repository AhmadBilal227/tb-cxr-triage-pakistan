/**
 * Shared evidence overlays for the X-ray viewer (in-canvas + lightbox).
 *
 * HeatmapOverlay  — the 8x8 BoxEvidence grid (viridis), absolute over the image.
 * HeatmapLegend   — a small low→high gradient key.
 * ZoneOverlay     — the seven anatomical zones as high-contrast labeled chips
 *                   over thin severity-tinted regions. Replaces the earlier
 *                   dark unreadable pills.
 *
 * Both overlays render inside the viewer's transformed layer, so they pan and
 * zoom locked to the image. Honesty framing (NOT a radiologist annotation)
 * lives with the toggle, not on the pixels.
 */
import { viridis } from '../details/colorScale';
import { scoreToTriadHex } from '@/lib/colors';

const NEAR_ZERO_FLOOR = 0.05;

/** 8x8 BoxEvidence heatmap, absolutely positioned to fill the image box. */
export function HeatmapOverlay({
  grid,
}: {
  grid: ReadonlyArray<ReadonlyArray<number>>;
}): JSX.Element {
  const safe: ReadonlyArray<ReadonlyArray<number>> =
    Array.isArray(grid) && grid.length === 8 && grid.every((r) => Array.isArray(r) && r.length === 8)
      ? grid
      : Array.from({ length: 8 }, () => Array<number>(8).fill(0));

  return (
    <div
      className="pointer-events-none absolute inset-0 grid"
      style={{
        gridTemplateColumns: 'repeat(8, minmax(0, 1fr))',
        gridTemplateRows: 'repeat(8, minmax(0, 1fr))',
      }}
      data-testid="heatmap-overlay"
    >
      {safe.map((row, r) =>
        row.map((v, c) => {
          const score = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
          const alpha = score < NEAR_ZERO_FLOOR ? 0 : 0.5 * (0.3 + 0.7 * score);
          return (
            <div
              key={`${r}-${c}`}
              data-testid={`heatmap-cell-${r}-${c}`}
              style={{ background: viridis(score), opacity: alpha }}
              title={`row ${r}, col ${c}: ${score.toFixed(3)}`}
            />
          );
        }),
      )}
    </div>
  );
}

/** Compact low→high viridis key, for placement near the heatmap toggle. */
export function HeatmapLegend(): JSX.Element {
  const ramp = `linear-gradient(90deg, ${viridis(0.05)}, ${viridis(0.3)}, ${viridis(0.55)}, ${viridis(0.8)}, ${viridis(1)})`;
  return (
    <div className="flex items-center gap-1.5" data-testid="heatmap-legend">
      <span className="font-mono text-[9px] uppercase tracking-wider text-white/55">low</span>
      <div className="h-1.5 w-20 rounded-full" style={{ background: ramp }} />
      <span className="font-mono text-[9px] uppercase tracking-wider text-white/55">
        high TB-evidence
      </span>
    </div>
  );
}

/**
 * Heuristic fractional rectangles per zone key. NOT precise lobar boundaries;
 * visual anchors only. {x,y,w,h} in [0,1] over the image box.
 */
const ZONE_RECTS: Record<
  string,
  { x: number; y: number; w: number; h: number; abbr: string }
> = {
  upper_r: { x: 0.11, y: 0.1, w: 0.34, h: 0.22, abbr: 'R Upper' },
  upper_l: { x: 0.55, y: 0.1, w: 0.34, h: 0.22, abbr: 'L Upper' },
  mid_r: { x: 0.11, y: 0.34, w: 0.34, h: 0.22, abbr: 'R Mid' },
  mid_l: { x: 0.55, y: 0.34, w: 0.34, h: 0.22, abbr: 'L Mid' },
  lower_r: { x: 0.11, y: 0.58, w: 0.34, h: 0.22, abbr: 'R Lower' },
  lower_l: { x: 0.55, y: 0.58, w: 0.34, h: 0.22, abbr: 'L Lower' },
  hilar: { x: 0.4, y: 0.34, w: 0.2, h: 0.18, abbr: 'Hilar' },
};

/**
 * Seven anatomical zones. Each renders a thin severity-tinted region plus a
 * solid, high-contrast label chip (dot + abbreviation + percentage) so it
 * reads cleanly over any radiograph density. Severity color from the verdict
 * triad (green low / amber mid / red high).
 */
export function ZoneOverlay({
  scores,
}: {
  scores: Record<string, number>;
}): JSX.Element {
  return (
    <div className="pointer-events-none absolute inset-0" data-testid="zone-overlay">
      {Object.entries(scores).map(([k, v]) => {
        const z = ZONE_RECTS[k];
        if (!z || typeof v !== 'number') return null;
        const color = scoreToTriadHex(v);
        return (
          <div
            key={k}
            data-testid={`zone-${k}`}
            className="absolute"
            style={{
              left: `${z.x * 100}%`,
              top: `${z.y * 100}%`,
              width: `${z.w * 100}%`,
              height: `${z.h * 100}%`,
            }}
          >
            <div
              className="h-full w-full rounded-md"
              style={{ border: `1px solid ${color}66`, background: `${color}14` }}
            />
            <span className="absolute left-1 top-1 inline-flex items-center gap-1 rounded bg-black/75 px-1.5 py-0.5 font-mono text-[10px] leading-none text-white">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
              {z.abbr} {(Math.max(0, Math.min(1, v)) * 100).toFixed(0)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
