/**
 * ImageLightbox — fullscreen radiograph viewer. Wraps the shared XRayViewer
 * (zoom / pan / fit / invert + the BoxEvidence heatmap and zone overlays) and
 * adds a side panel with the per-zone probabilities and top TB-relevant TXRV
 * findings. URL-bound by the parent (?lightbox) for back-button parity.
 *
 * Overlays here use the shared components, so the earlier unreadable dark
 * zone pills are gone (now high-contrast labeled chips).
 */
import { useMemo, useState } from 'react';
import { Dialog, FullscreenContent, DialogTitle } from '../ui/dialog';
import { Button } from '../ui/button';
import { PanelRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { viridis } from './colorScale';
import { XRayViewer } from '../viewer/XRayViewer';

const VERDICT_CHIP_COLOR: Record<string, string> = {
  'TB SUSPECTED': '#C8102E',
  'NO TB': '#00754A',
  'UNCERTAIN — REFER': '#F59E0B',
};

const ZONE_LABEL: Record<string, string> = {
  upper_r: 'Right upper',
  upper_l: 'Left upper',
  mid_r: 'Right mid',
  mid_l: 'Left mid',
  lower_r: 'Right lower',
  lower_l: 'Left lower',
  hilar: 'Hilar',
};

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
  boxGrid?: ReadonlyArray<ReadonlyArray<number>> | null;
  zonalScores?: Record<string, number> | null;
  txrvPathologies?: Record<string, number> | null;
  cropBox?: { x: number; y: number; w: number; h: number } | null;
  verdictLabel?: string;
}

export function ImageLightbox({
  open,
  onOpenChange,
  imageUrl,
  boxGrid,
  zonalScores,
  txrvPathologies,
  cropBox,
  verdictLabel,
}: ImageLightboxProps): JSX.Element {
  const [showPanel, setShowPanel] = useState(true);
  const chipColor = verdictLabel ? VERDICT_CHIP_COLOR[verdictLabel] : undefined;

  const sortedZones = useMemo(
    () => (zonalScores ? Object.entries(zonalScores).sort((a, b) => b[1] - a[1]) : []),
    [zonalScores],
  );
  const sortedPathologies = useMemo(
    () =>
      txrvPathologies
        ? Object.entries(txrvPathologies)
            .filter(([k]) => TB_RELEVANT_PATHOLOGIES.includes(k))
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
        : [],
    [txrvPathologies],
  );

  const hasPanelData = sortedZones.length > 0 || sortedPathologies.length > 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <FullscreenContent data-testid="image-lightbox" className="bg-black" aria-describedby={undefined}>
        <header className="flex shrink-0 items-center gap-3 border-b border-white/10 px-5 py-3">
          {chipColor && verdictLabel && (
            <span
              className="rounded-md px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider"
              style={{ color: chipColor, background: `${chipColor}1f`, border: `1px solid ${chipColor}55` }}
            >
              {verdictLabel}
            </span>
          )}
          <DialogTitle className="text-sm font-semibold tracking-tight text-white">
            Radiograph viewer
          </DialogTitle>
          <div className="ml-auto flex items-center gap-2 pr-10">
            {hasPanelData && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowPanel((s) => !s)}
                aria-pressed={showPanel}
                data-testid="lightbox-toggle-panel"
                className={cn('text-white/70 hover:text-white', showPanel && 'text-white')}
              >
                <PanelRight className="h-3.5 w-3.5" /> Evidence
              </Button>
            )}
          </div>
        </header>

        <div className="relative flex h-full w-full overflow-hidden">
          <div className="min-w-0 flex-1">
            {imageUrl ? (
              <XRayViewer
                imageUrl={imageUrl}
                boxGrid={boxGrid}
                zonalScores={zonalScores}
                cropBox={cropBox}
                overlaysReady
                imageClassName="max-h-[82vh] max-w-full"
              />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-white/40">
                No image available.
              </div>
            )}
          </div>

          {showPanel && hasPanelData && (
            <aside
              className="hidden w-64 shrink-0 overflow-y-auto scroll-thin border-l border-white/10 bg-black/40 px-4 py-4 md:block"
              data-testid="lightbox-side-panel"
            >
              {sortedZones.length > 0 && (
                <div className="space-y-2">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-white/55">
                    Per-zone TB probability
                  </div>
                  <ul className="space-y-1.5">
                    {sortedZones.map(([k, v]) => {
                      const pct = Math.max(0, Math.min(1, v));
                      return (
                        <li key={k} className="space-y-0.5">
                          <div className="flex items-baseline justify-between gap-2">
                            <span className="text-[11px] text-white/85">{ZONE_LABEL[k] ?? k}</span>
                            <span className="font-mono text-[10px] text-white/70">{v.toFixed(3)}</span>
                          </div>
                          <div className="h-1 w-full overflow-hidden rounded bg-white/10">
                            <div className="h-full" style={{ width: `${pct * 100}%`, background: viridis(pct) }} />
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
              {sortedPathologies.length > 0 && (
                <div className="mt-6 space-y-2">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-white/55">
                    Top TXRV findings (TB-relevant)
                  </div>
                  <ul className="space-y-1">
                    {sortedPathologies.map(([k, v]) => (
                      <li
                        key={k}
                        className="flex items-center justify-between gap-2 rounded border border-white/10 bg-white/5 px-2 py-1.5"
                      >
                        <span className="text-[11px] text-white/85">{k}</span>
                        <span className="font-mono text-[10px] text-white/70">{v.toFixed(3)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <p className="mt-6 text-[10px] leading-snug text-white/45">
                Anchored to the validated pipeline's outputs. NOT a radiologist annotation; values
                are model confidences, not clinical measurements.
              </p>
            </aside>
          )}
        </div>
      </FullscreenContent>
    </Dialog>
  );
}
