/**
 * ImageLightbox — full-screen X-ray viewer with optional evidence overlays.
 *
 * Triggered from the BoxEvidenceHeatmap's "fullscreen" affordance inside the
 * VerdictCard. Provides progressive disclosure: image-only on first view, with
 * toggles to reveal the BoxEvidence heatmap, the seven anatomical zone labels,
 * and the TXRV pathology badges. The chrome auto-fades after 2.5s of mouse
 * inactivity so the radiologist can read the film without UI clutter; any
 * pointer motion brings it back.
 *
 * The toggles are deliberately scoped to "evidence the validated pipeline
 * actually computed" — we never overlay annotations the model did not produce.
 * This is the same anchoring discipline as the gpt-interpreter prompt: the
 * lightbox shows the model's reasoning, never invents radiographic content.
 *
 * Hover/tap on any heatmap cell exposes its per-cell sigmoid score via the
 * cell's `title` attribute (screen-reader accessible) and a transient on-image
 * pill overlay (mouse only).
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Dialog, FullscreenContent } from '../ui/dialog';
import { Button } from '../ui/button';
import { Activity, Crosshair, ListTree, Maximize2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { viridis } from './colorScale';
import { scoreToTriadHex } from '@/lib/colors';

const NEAR_ZERO_FLOOR = 0.05;

/**
 * Approximate fractional rectangles over a frontal CXR for the seven zone
 * keys the pipeline scores. These are heuristic visual anchors, NOT precise
 * lobar boundaries — a real CXR lung field segmentation would be needed for
 * exact placement. Used only to position label pills near the right region.
 *
 *   {x, y, w, h} in [0, 1] over the image bounding box.
 */
const ZONE_RECTS: Record<string, { x: number; y: number; w: number; h: number; label: string }> = {
  upper_l: { x: 0.55, y: 0.10, w: 0.34, h: 0.22, label: 'Left upper' },
  upper_r: { x: 0.11, y: 0.10, w: 0.34, h: 0.22, label: 'Right upper' },
  mid_l: { x: 0.55, y: 0.34, w: 0.34, h: 0.22, label: 'Left mid' },
  mid_r: { x: 0.11, y: 0.34, w: 0.34, h: 0.22, label: 'Right mid' },
  lower_l: { x: 0.55, y: 0.58, w: 0.34, h: 0.22, label: 'Left lower' },
  lower_r: { x: 0.11, y: 0.58, w: 0.34, h: 0.22, label: 'Right lower' },
  hilar: { x: 0.40, y: 0.32, w: 0.20, h: 0.18, label: 'Hilar' },
};

/** TB-relevant TXRV pathology subset for the badge strip; matches asymmetricEvidence.ts. */
const TB_RELEVANT_PATHOLOGIES: readonly string[] = [
  'Lung Opacity',
  'Effusion',
  'Lung Lesion',
  'Infiltration',
  'Consolidation',
  'Fibrosis',
  'Pleural Thickening',
  'Atelectasis',
];

export interface ImageLightboxProps {
  open: boolean;
  onOpenChange: (next: boolean) => void;
  imageUrl: string | null;
  /** 8x8 BoxEvidence grid (sigmoid'd in [0,1]); null if local pathway did not emit it. */
  boxGrid?: ReadonlyArray<ReadonlyArray<number>> | null;
  /** Per-zone TB probabilities (calibrated) from the validated head. */
  zonalScores?: Record<string, number> | null;
  /** TXRV 18-class pathology scores. */
  txrvPathologies?: Record<string, number> | null;
  /** Verdict label shown in the title chip. */
  verdictLabel?: string;
}

export function ImageLightbox({
  open,
  onOpenChange,
  imageUrl,
  boxGrid,
  zonalScores,
  txrvPathologies,
  verdictLabel,
}: ImageLightboxProps): JSX.Element {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showZones, setShowZones] = useState(true);
  const [showPathologies, setShowPathologies] = useState(true);
  const [chromeVisible, setChromeVisible] = useState(true);
  const hideTimer = useRef<number | null>(null);

  // Auto-hide chrome after 2.5s of mouse inactivity so the radiologist can
  // read the film without UI clutter. Any pointer motion brings it back.
  useEffect(() => {
    if (!open) return;
    const reset = (): void => {
      setChromeVisible(true);
      if (hideTimer.current !== null) window.clearTimeout(hideTimer.current);
      hideTimer.current = window.setTimeout(() => setChromeVisible(false), 2500);
    };
    reset();
    window.addEventListener('mousemove', reset);
    return () => {
      window.removeEventListener('mousemove', reset);
      if (hideTimer.current !== null) window.clearTimeout(hideTimer.current);
    };
  }, [open]);

  const sortedPathologies = useMemo(() => {
    if (!txrvPathologies) return [];
    return Object.entries(txrvPathologies)
      .filter(([k]) => TB_RELEVANT_PATHOLOGIES.includes(k))
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
  }, [txrvPathologies]);

  const sortedZones = useMemo(() => {
    if (!zonalScores) return [];
    return Object.entries(zonalScores).sort((a, b) => b[1] - a[1]);
  }, [zonalScores]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <FullscreenContent
        data-testid="image-lightbox"
        className="bg-black"
        aria-describedby={undefined}
      >
        {/* ===================== top chrome ===================== */}
        <header
          className={cn(
            'absolute inset-x-0 top-0 z-10 flex items-center gap-3 border-b border-white/10 bg-black/60 px-5 py-3 backdrop-blur transition-opacity duration-200',
            chromeVisible ? 'opacity-100' : 'opacity-0',
          )}
        >
          <Maximize2 className="h-4 w-4 text-white/60" />
          <span className="font-mono text-[11px] uppercase tracking-wider text-white/80">
            Chest radiograph — evidence overlay
          </span>
          {verdictLabel && (
            <span className="rounded-md border border-white/20 bg-white/5 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-white/80">
              {verdictLabel}
            </span>
          )}
          <div className="ml-auto flex items-center gap-1 pr-10">
            <ToggleButton
              active={showHeatmap}
              icon={<Activity className="h-3.5 w-3.5" />}
              onClick={() => setShowHeatmap((v) => !v)}
              testid="lightbox-toggle-heatmap"
            >
              Heatmap
            </ToggleButton>
            <ToggleButton
              active={showZones}
              icon={<Crosshair className="h-3.5 w-3.5" />}
              onClick={() => setShowZones((v) => !v)}
              testid="lightbox-toggle-zones"
            >
              Zones
            </ToggleButton>
            <ToggleButton
              active={showPathologies}
              icon={<ListTree className="h-3.5 w-3.5" />}
              onClick={() => setShowPathologies((v) => !v)}
              testid="lightbox-toggle-pathologies"
            >
              Findings
            </ToggleButton>
          </div>
        </header>

        {/* ===================== main canvas ===================== */}
        <div className="relative flex h-full w-full items-center justify-center overflow-hidden">
          {imageUrl ? (
            <div className="relative h-full w-full flex items-center justify-center">
              <div className="relative max-h-full max-w-full">
                <img
                  src={imageUrl}
                  alt="Chest radiograph"
                  className="block max-h-[95vh] max-w-[90vw] object-contain"
                  data-testid="lightbox-image"
                />
                {showHeatmap && boxGrid && <HeatmapOverlay grid={boxGrid} />}
                {showZones && zonalScores && <ZoneLabelsOverlay scores={zonalScores} />}
              </div>
            </div>
          ) : (
            <p className="text-sm text-white/40">No image available.</p>
          )}

          {/* ===================== side panel (pathologies + zones) ===================== */}
          {showPathologies && (sortedPathologies.length > 0 || sortedZones.length > 0) && (
            <aside
              className={cn(
                // Hidden below md (cramped on phones); the toggles still
                // gate it, this just removes it from narrow viewports.
                'absolute right-0 top-0 hidden h-full w-64 overflow-y-auto scroll-thin border-l border-white/10 bg-black/60 px-4 py-16 backdrop-blur transition-opacity duration-200 md:block',
                chromeVisible ? 'opacity-100' : 'opacity-30',
              )}
              data-testid="lightbox-side-panel"
            >
              {sortedZones.length > 0 && (
                <div className="space-y-2">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-white/50">
                    Per-zone TB probability
                  </div>
                  <ul className="space-y-1">
                    {sortedZones.map(([k, v]) => (
                      <ZoneRow key={k} zoneKey={k} value={v} />
                    ))}
                  </ul>
                </div>
              )}
              {sortedPathologies.length > 0 && (
                <div className="mt-6 space-y-2">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-white/50">
                    Top TXRV findings (TB-relevant)
                  </div>
                  <ul className="space-y-1">
                    {sortedPathologies.map(([k, v]) => (
                      <li
                        key={k}
                        className="flex items-center justify-between gap-2 rounded border border-white/10 bg-white/5 px-2 py-1.5"
                      >
                        <span className="text-[11px] text-white/85">{k}</span>
                        <span className="font-mono text-[10px] text-white/70">
                          {v.toFixed(3)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <p className="mt-6 text-[10px] leading-snug text-white/40">
                Anchored to the validated pipeline's outputs. NOT a radiologist annotation;
                values are model confidences, not clinical measurements.
              </p>
            </aside>
          )}
        </div>
      </FullscreenContent>
    </Dialog>
  );
}

// =====================================================================
// HeatmapOverlay — same viridis cell logic as BoxEvidenceHeatmap, scaled
// to the lightbox image bounds.
// =====================================================================
function HeatmapOverlay({ grid }: { grid: ReadonlyArray<ReadonlyArray<number>> }): JSX.Element {
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
      data-testid="lightbox-heatmap-overlay"
    >
      {safe.map((row, r) =>
        row.map((v, c) => {
          const score = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
          const alpha = score < NEAR_ZERO_FLOOR ? 0 : 0.45 * (0.3 + 0.7 * score);
          return (
            <div
              key={`${r}-${c}`}
              data-testid={`lightbox-cell-${r}-${c}`}
              className="pointer-events-auto border border-white/5"
              style={{ background: viridis(score), opacity: alpha }}
              title={`row ${r}, col ${c}: ${score.toFixed(3)}`}
            />
          );
        }),
      )}
    </div>
  );
}

// =====================================================================
// ZoneLabelsOverlay — labels positioned over each zone region of the
// image. Uses approximate fractional rectangles, NOT precise lobar
// segmentation. Color reflects the calibrated zone probability.
// =====================================================================
function ZoneLabelsOverlay({ scores }: { scores: Record<string, number> }): JSX.Element {
  return (
    <div className="pointer-events-none absolute inset-0">
      {Object.entries(scores).map(([k, v]) => {
        const rect = ZONE_RECTS[k];
        if (!rect) return null;
        const color = scoreToTriadHex(v);
        return (
          <div
            key={k}
            data-testid={`lightbox-zone-${k}`}
            className="absolute flex items-end justify-start"
            style={{
              left: `${rect.x * 100}%`,
              top: `${rect.y * 100}%`,
              width: `${rect.w * 100}%`,
              height: `${rect.h * 100}%`,
            }}
          >
            <span
              className="m-1 rounded-sm px-1.5 py-0.5 font-mono text-[10px] backdrop-blur"
              style={{ background: 'rgba(0,0,0,0.55)', color, border: `1px solid ${color}88` }}
            >
              {rect.label} {(v * 100).toFixed(0)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ZoneRow({ zoneKey, value }: { zoneKey: string; value: number }): JSX.Element {
  const label = ZONE_RECTS[zoneKey]?.label ?? zoneKey;
  const pct = Math.max(0, Math.min(1, value));
  return (
    <li className="space-y-0.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[11px] text-white/85">{label}</span>
        <span className="font-mono text-[10px] text-white/70">{value.toFixed(3)}</span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded bg-white/10">
        <div className="h-full" style={{ width: `${pct * 100}%`, background: viridis(pct) }} />
      </div>
    </li>
  );
}

function ToggleButton({
  active,
  icon,
  onClick,
  testid,
  children,
}: {
  active: boolean;
  icon: React.ReactNode;
  onClick: () => void;
  testid: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={onClick}
      aria-pressed={active}
      data-testid={testid}
      className={cn('text-white/70 hover:text-white', active && 'text-white')}
    >
      {icon}
      <span className="hidden sm:inline">{children}</span>
    </Button>
  );
}
