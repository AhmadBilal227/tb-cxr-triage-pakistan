/**
 * XRayViewer — pannable / zoomable radiograph viewer with the radiology
 * baseline control set, used by the main canvas (DropCanvas) and the
 * fullscreen lightbox.
 *
 * Controls:
 *   - scroll wheel        zoom toward the cursor
 *   - +/- buttons         zoom toward center
 *   - click-drag          pan (only when zoomed in)
 *   - double-click        reset to fit
 *   - fit button          reset to fit (scale 1, centered)
 *   - invert button       grayscale invert (filter on the image only)
 *   - heatmap / zones      overlay toggles, disabled until `overlaysReady`
 *
 * The image + overlays live inside one transformed layer so the BoxEvidence
 * heatmap and zone chips stay locked to the radiograph through zoom and pan.
 * Transform uses translate + scale only (composited; no layout animation).
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { Minus, Plus, Maximize, Contrast, Activity, Crosshair } from 'lucide-react';
import { cn, clamp } from '@/lib/utils';
import { HeatmapOverlay, HeatmapLegend, ZoneOverlay } from './overlays';

const MIN_SCALE = 1;
const MAX_SCALE = 8;

interface ViewTransform {
  scale: number;
  tx: number;
  ty: number;
}

export interface XRayViewerProps {
  imageUrl: string;
  alt?: string;
  boxGrid?: ReadonlyArray<ReadonlyArray<number>> | null;
  zonalScores?: Record<string, number> | null;
  /** When false, the overlay toggles render disabled ("after analysis"). */
  overlaysReady?: boolean;
  /** Tailwind size cap for the image, e.g. 'max-h-[70vh]'. */
  imageClassName?: string;
  className?: string;
}

const INITIAL: ViewTransform = { scale: 1, tx: 0, ty: 0 };

export function XRayViewer({
  imageUrl,
  alt = 'Chest radiograph under analysis',
  boxGrid,
  zonalScores,
  overlaysReady = false,
  imageClassName = 'max-h-[70vh] max-w-full',
  className,
}: XRayViewerProps): JSX.Element {
  const [view, setView] = useState<ViewTransform>(INITIAL);
  const [inverted, setInverted] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showZones, setShowZones] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);

  const reset = useCallback(() => setView(INITIAL), []);

  // Zoom by `factor` keeping the point (cx, cy) — measured from the container
  // center — stationary. Functional update so rapid wheel events compose.
  const applyZoom = useCallback((factor: number, cx: number, cy: number) => {
    setView((v) => {
      const ns = clamp(v.scale * factor, MIN_SCALE, MAX_SCALE);
      if (ns === v.scale) return v;
      if (ns === MIN_SCALE) return INITIAL;
      const ratio = ns / v.scale;
      return { scale: ns, tx: cx - ratio * (cx - v.tx), ty: cy - ratio * (cy - v.ty) };
    });
  }, []);

  // Native non-passive wheel listener so we can preventDefault (React's
  // onWheel is passive and cannot block page scroll).
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent): void => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      applyZoom(e.deltaY < 0 ? 1.12 : 1 / 1.12, cx, cy);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [applyZoom]);

  // Reset transform whenever the source image changes.
  useEffect(() => {
    setView(INITIAL);
    setInverted(false);
  }, [imageUrl]);

  const onPointerDown = (e: React.PointerEvent): void => {
    if (view.scale <= 1) return;
    dragRef.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent): void => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    dragRef.current = { x: e.clientX, y: e.clientY };
    setView((v) => ({ ...v, tx: v.tx + dx, ty: v.ty + dy }));
  };
  const onPointerUp = (): void => {
    dragRef.current = null;
  };

  const zoomed = view.scale > 1;
  const hasHeatmap = Boolean(boxGrid && boxGrid.length > 0);
  const hasZones = Boolean(zonalScores && Object.keys(zonalScores).length > 0);

  return (
    <div
      ref={containerRef}
      className={cn('relative h-full w-full overflow-hidden select-none', className)}
      onDoubleClick={reset}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={onPointerUp}
      style={{ cursor: zoomed ? (dragRef.current ? 'grabbing' : 'grab') : 'default', touchAction: 'none' }}
    >
      <div className="absolute inset-0 flex items-center justify-center p-6">
        <div
          className="relative"
          style={{
            transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`,
            transformOrigin: 'center center',
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element -- not a Next app */}
          <img
            src={imageUrl}
            alt={alt}
            draggable={false}
            className={cn('block rounded-lg border border-border object-contain', imageClassName)}
            style={{ filter: inverted ? 'invert(1)' : undefined }}
          />
          {overlaysReady && showHeatmap && hasHeatmap && boxGrid && (
            <HeatmapOverlay grid={boxGrid} />
          )}
          {overlaysReady && showZones && hasZones && zonalScores && (
            <ZoneOverlay scores={zonalScores} />
          )}
        </div>
      </div>

      {/* Heatmap legend, top-center, only while the heatmap is shown. */}
      {overlaysReady && showHeatmap && hasHeatmap && (
        <div className="pointer-events-none absolute left-1/2 top-3 -translate-x-1/2 rounded-md border border-border bg-surface/85 px-2 py-1 backdrop-blur">
          <HeatmapLegend />
        </div>
      )}

      {/* Floating control toolbar, bottom-center. */}
      <div
        className="absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-0.5 rounded-lg border border-border bg-surface/90 px-1.5 py-1 shadow-lg backdrop-blur"
        data-testid="viewer-toolbar"
      >
        <ToolBtn label="Zoom out" onClick={() => applyZoom(1 / 1.25, 0, 0)}>
          <Minus className="h-3.5 w-3.5" />
        </ToolBtn>
        <span className="min-w-[3.5ch] text-center font-mono text-[10px] tabular-nums text-muted">
          {Math.round(view.scale * 100)}%
        </span>
        <ToolBtn label="Zoom in" onClick={() => applyZoom(1.25, 0, 0)}>
          <Plus className="h-3.5 w-3.5" />
        </ToolBtn>

        <Divider />

        <ToolBtn label="Fit to window" onClick={reset} disabled={view.scale === 1 && view.tx === 0 && view.ty === 0}>
          <Maximize className="h-3.5 w-3.5" />
        </ToolBtn>
        <ToolBtn label="Invert grayscale" onClick={() => setInverted((i) => !i)} active={inverted}>
          <Contrast className="h-3.5 w-3.5" />
        </ToolBtn>

        <Divider />

        <ToolBtn
          label={overlaysReady ? 'Toggle BoxEvidence heatmap' : 'Heatmap available after analysis'}
          onClick={() => setShowHeatmap((s) => !s)}
          active={overlaysReady && showHeatmap && hasHeatmap}
          disabled={!overlaysReady || !hasHeatmap}
        >
          <Activity className="h-3.5 w-3.5" />
        </ToolBtn>
        <ToolBtn
          label={overlaysReady ? 'Toggle zone overlay' : 'Zones available after analysis'}
          onClick={() => setShowZones((s) => !s)}
          active={overlaysReady && showZones && hasZones}
          disabled={!overlaysReady || !hasZones}
        >
          <Crosshair className="h-3.5 w-3.5" />
        </ToolBtn>
      </div>
    </div>
  );
}

function Divider(): JSX.Element {
  return <span className="mx-0.5 h-4 w-px bg-border" aria-hidden />;
}

function ToolBtn({
  label,
  onClick,
  children,
  active,
  disabled,
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
  active?: boolean;
  disabled?: boolean;
}): JSX.Element {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      aria-pressed={active}
      title={label}
      className={cn(
        'inline-flex h-7 w-7 items-center justify-center rounded transition-colors',
        'text-muted hover:bg-surface-2 hover:text-offwhite',
        'disabled:pointer-events-none disabled:opacity-30',
        active && 'bg-provider-openai/15 text-provider-openai',
      )}
    >
      {children}
    </button>
  );
}
