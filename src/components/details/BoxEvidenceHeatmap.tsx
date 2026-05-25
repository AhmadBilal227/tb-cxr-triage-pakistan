/**
 * BoxEvidenceHeatmap — M24 evidence panel #1.
 *
 * Renders the 8x8 per-cell sigmoid'd box-evidence map that the trained
 * BoxEvidence head (LSE-LBA pool's input) emits and that the engine used to
 * discard. Overlays the heatmap on a cropped-CXR base layer so the user sees
 * WHERE the trained model's evidence concentrated.
 *
 * Visual choices (locked, document any change):
 *   - VIRIDIS palette (perceptually uniform, colorblind-friendly).
 *   - Cells with score < 0.05 render at near-zero alpha (don't add visual
 *     noise on near-empty cells; the model's threshold of clinical interest
 *     is well above this floor).
 *   - Caption explicitly says NOT a radiologist annotation; not an ROI in
 *     the clinical sense. This is honest framing of what BoxEvidence
 *     actually localizes (Li et al. 1803.07703 LSE-LBA pooling target).
 *
 * Accessibility: each cell renders an `<title>` so screen-readers expose
 * `row,col: probability`. The container has a stable test id.
 */
import { viridis } from './colorScale';

export interface BoxEvidenceHeatmapProps {
  /** Row-major 8x8 array of sigmoid'd box-evidence probabilities in [0,1]. */
  grid: ReadonlyArray<ReadonlyArray<number>>;
  /** Object URL or data URL of the source CXR (rendered as base layer). Optional — the
   *  heatmap is also legible standalone, and the orchestrator does not currently thread
   *  the source-image URL through. */
  imageUrl?: string;
  /** Optional caption override; defaults to the M24 honest-framing line. */
  caption?: string;
}

const DEFAULT_CAPTION =
  'Box-evidence overlay — where the trained model sees TB-suggestive patterns. NOT a radiologist annotation; not a region of interest in the clinical sense.';

const NEAR_ZERO_FLOOR = 0.05;

export function BoxEvidenceHeatmap({
  grid,
  imageUrl,
  caption = DEFAULT_CAPTION,
}: BoxEvidenceHeatmapProps): JSX.Element {
  // Defensive — if a malformed grid slips through, render an empty panel rather than crash.
  const safeGrid: ReadonlyArray<ReadonlyArray<number>> =
    Array.isArray(grid) && grid.length === 8 && grid.every((r) => Array.isArray(r) && r.length === 8)
      ? grid
      : Array.from({ length: 8 }, () => Array<number>(8).fill(0));

  return (
    <div data-testid="box-evidence-heatmap" className="space-y-2">
      <div
        className="relative mx-auto aspect-square w-full max-w-[256px] overflow-hidden rounded-md border border-border bg-black"
        role="img"
        aria-label="Box-evidence heatmap, 8 by 8 grid"
      >
        {imageUrl && (
          // eslint-disable-next-line @next/next/no-img-element -- not a Next app
          <img
            src={imageUrl}
            alt="Cropped CXR base layer"
            className="absolute inset-0 h-full w-full object-cover opacity-70"
            data-testid="box-evidence-base"
          />
        )}
        <div
          className="absolute inset-0 grid"
          style={{ gridTemplateColumns: 'repeat(8, minmax(0, 1fr))', gridTemplateRows: 'repeat(8, minmax(0, 1fr))' }}
        >
          {safeGrid.map((row, r) =>
            row.map((v, c) => {
              const score = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
              // Below the floor, render near-transparent so quiet cells don't add noise.
              const alpha = score < NEAR_ZERO_FLOOR ? 0.0 : 0.55 * (0.3 + 0.7 * score);
              const bg = viridis(score);
              return (
                <div
                  key={`${r}-${c}`}
                  data-testid={`box-cell-${r}-${c}`}
                  className="border border-white/5"
                  style={{ background: bg, opacity: alpha }}
                  title={`row ${r}, col ${c}: ${score.toFixed(3)}`}
                />
              );
            }),
          )}
        </div>
      </div>
      <p
        className="text-[10px] leading-snug text-muted/80"
        data-testid="box-evidence-caption"
      >
        {caption}
      </p>
    </div>
  );
}
