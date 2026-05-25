import { ImageUp, FlaskConical } from 'lucide-react';
import { cn } from '@/lib/utils';
import { XRayViewer } from './viewer/XRayViewer';

export interface SampleEntry {
  file: string;
  label: 0 | 1;
  labelText: string;
  source: string;
  note: string;
}

export function DropCanvas({
  imageUrl,
  samples,
  onBrowse,
  onSample,
  onPickSample,
  boxGrid,
  zonalScores,
  overlaysReady,
}: {
  imageUrl: string | null;
  samples: SampleEntry[];
  onBrowse: () => void;
  onSample: () => void;
  onPickSample: (file: string) => void;
  /** BoxEvidence 8x8 grid + per-zone scores; drive the in-viewer overlays. */
  boxGrid?: ReadonlyArray<ReadonlyArray<number>> | null;
  zonalScores?: Record<string, number> | null;
  /** Overlay toggles render disabled until analysis has produced the evidence. */
  overlaysReady?: boolean;
}): JSX.Element {
  if (!imageUrl) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 p-8 text-center">
        <ImageUp className="h-10 w-10 text-muted" strokeWidth={1.5} />
        <div>
          <p className="text-lg font-medium text-offwhite">Drop a chest X-ray.</p>
          <p className="mt-1 text-sm text-muted">Drag anywhere on this canvas, or</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onBrowse}
            className="rounded-md border border-border px-3 py-1.5 text-sm text-offwhite hover:bg-surface-2"
          >
            Browse files
          </button>
          <button
            onClick={onSample}
            className="inline-flex items-center gap-1.5 text-sm text-muted hover:text-offwhite"
          >
            <FlaskConical className="h-3.5 w-3.5" /> Synthetic sample
          </button>
        </div>

        {samples.length > 0 && (
          <div className="mt-4 w-full max-w-md">
            <p className="mb-2 text-[11px] uppercase tracking-wide text-muted">Real demo X-rays</p>
            <div className="flex justify-center gap-3">
              {samples.map((s) => (
                <button
                  key={s.file}
                  onClick={() => onPickSample(s.file)}
                  className="group w-24 overflow-hidden rounded-md border border-border hover:border-provider-openai"
                  title={`${s.labelText} · ${s.source}`}
                >
                  <div className="relative aspect-square bg-ink">
                    <img src={`/samples/${s.file}`} alt={s.labelText} className="h-full w-full object-cover" />
                    <span
                      className={cn(
                        'absolute left-0 top-0 px-1 text-[9px] font-mono text-white',
                        s.label === 1 ? 'bg-verdict-tb' : 'bg-verdict-clear',
                      )}
                    >
                      {s.labelText}
                    </span>
                  </div>
                </button>
              ))}
            </div>
            <p className="mt-2 text-[10px] text-muted">
              Real CXRs from public Hugging Face datasets · research/educational use only.
            </p>
          </div>
        )}
      </div>
    );
  }

  return (
    <XRayViewer
      imageUrl={imageUrl}
      boxGrid={boxGrid}
      zonalScores={zonalScores}
      overlaysReady={overlaysReady}
      imageClassName="max-h-[70vh] max-w-full"
    />
  );
}
