import { useState } from 'react';
import { motion } from 'framer-motion';
import { ImageUp, Sparkles } from 'lucide-react';
import type { BBox } from '@/lib/providers/parsers';
import { cn } from '@/lib/utils';

export interface SampleEntry {
  file: string;
  label: 0 | 1;
  labelText: string;
  source: string;
  note: string;
}

export function DropCanvas({
  imageUrl,
  boxes,
  samples,
  onBrowse,
  onSample,
  onPickSample,
}: {
  imageUrl: string | null;
  boxes: BBox[];
  samples: SampleEntry[];
  onBrowse: () => void;
  onSample: () => void;
  onPickSample: (file: string) => void;
}): JSX.Element {
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);

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
            className="inline-flex items-center gap-1.5 text-sm text-muted/70 hover:text-muted"
          >
            <Sparkles className="h-3.5 w-3.5" /> Synthetic sample
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
            <p className="mt-2 text-[10px] text-muted/60">
              Real CXRs from public Hugging Face datasets · research/educational use only.
            </p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex h-full items-center justify-center p-6">
      <motion.div
        layout
        initial={{ scale: 0.96, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 200, damping: 24 }}
        className="relative max-h-full max-w-full"
      >
        <img
          src={imageUrl}
          alt="Chest X-ray under analysis"
          className="max-h-[70vh] max-w-full rounded-lg border border-border object-contain"
          onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        />
        {/* Real detection boxes only — never faked heatmaps */}
        {nat && boxes.length > 0 && (
          <svg
            className="pointer-events-none absolute inset-0 h-full w-full"
            viewBox={`0 0 ${nat.w} ${nat.h}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {boxes.map((b, i) => (
              <g key={i}>
                <rect
                  x={b.x}
                  y={b.y}
                  width={b.w}
                  height={b.h}
                  fill="none"
                  stroke="#F59E0B"
                  strokeWidth={Math.max(2, nat.w * 0.004)}
                />
                {b.label && (
                  <text x={b.x} y={b.y - 4} fill="#F59E0B" style={{ fontSize: nat.w * 0.03 }} className="font-mono">
                    {b.label} {b.score ? (b.score * 100).toFixed(0) + '%' : ''}
                  </text>
                )}
              </g>
            ))}
          </svg>
        )}
      </motion.div>
    </div>
  );
}
