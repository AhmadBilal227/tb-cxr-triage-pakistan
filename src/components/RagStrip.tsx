import type { RagNeighbor } from '@/lib/types';
import { cn } from '@/lib/utils';

/** Horizontal thumbnail strip of retrieved nearest-neighbor cases with similarity %. */
export function RagStrip({ neighbors }: { neighbors: RagNeighbor[] }): JSX.Element {
  if (neighbors.length === 0) {
    return <p className="text-[11px] text-muted">No labeled cases in corpus yet — import a labeled set to enable retrieval evidence.</p>;
  }
  return (
    <div className="flex gap-2 overflow-x-auto scroll-thin pb-1">
      {neighbors.map((n, i) => (
        <figure key={`${n.filename}-${i}`} className="w-20 shrink-0">
          <div className="relative aspect-square overflow-hidden rounded border border-border bg-ink">
            <img src={n.thumbUrl} alt={n.filename} className="h-full w-full object-cover" />
            <span
              className={cn(
                'absolute left-0 top-0 px-1 text-[9px] font-mono',
                n.label === 1 ? 'bg-verdict-tb text-white' : 'bg-verdict-clear text-white',
              )}
            >
              {n.label === 1 ? 'TB' : 'NEG'}
            </span>
          </div>
          <figcaption className="mt-1 text-center font-mono text-[10px] tabular-nums text-muted">
            {(n.similarity * 100).toFixed(0)}%
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
