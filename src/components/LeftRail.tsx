import { useEffect, useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { PanelLeftClose, PanelLeftOpen, History } from 'lucide-react';
import { listHistory } from '@/lib/db';
import type { CaseHistoryRecord, Verdict } from '@/lib/types';
import { cn } from '@/lib/utils';

const VERDICT_BORDER: Record<Verdict, string> = {
  tb: 'border-verdict-tb',
  no_tb: 'border-verdict-clear',
  abstain: 'border-verdict-uncertain',
};

/**
 * Manage one object URL per history thumbnail, revoking on change/unmount.
 * Previously the URLs were minted inline in render and never revoked — a slow
 * Blob-URL leak that grew with every history mutation over a session.
 */
function useThumbnailUrls(history: CaseHistoryRecord[]): Map<string, string> {
  const [urls, setUrls] = useState<Map<string, string>>(() => new Map());
  useEffect(() => {
    const next = new Map<string, string>();
    for (const rec of history) next.set(rec.id, URL.createObjectURL(rec.blob));
    setUrls(next);
    return () => {
      for (const url of next.values()) URL.revokeObjectURL(url);
    };
  }, [history]);
  return urls;
}

export function LeftRail({
  open,
  onToggle,
  onSelect,
}: {
  open: boolean;
  onToggle: () => void;
  onSelect: (rec: CaseHistoryRecord) => void;
}): JSX.Element {
  const history = useLiveQuery(() => listHistory(50), [], [] as CaseHistoryRecord[]);
  const thumbUrls = useThumbnailUrls(history);

  return (
    <aside
      className={cn(
        // Hidden below md — the history rail is a desktop affordance. Below
        // md the canvas owns the full width; history stays reachable via the
        // command palette's "New case" / re-drop flow.
        'hidden h-full flex-col border-r border-border bg-surface transition-all duration-200 md:flex',
        open ? 'md:w-44' : 'md:w-12',
      )}
    >
      <div className="flex items-center justify-between border-b border-border px-2 py-3">
        {open && (
          <span className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-muted">
            <History className="h-3.5 w-3.5" /> History
          </span>
        )}
        <button
          onClick={onToggle}
          className="text-muted hover:text-offwhite"
          aria-label={open ? 'Collapse history' : 'Expand history'}
        >
          {open ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeftOpen className="h-4 w-4" />}
        </button>
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto scroll-thin p-2">
        {history.length === 0 && open && (
          <p className="px-1 text-[11px] text-muted">No cases yet.</p>
        )}
        {history.map((rec) => (
          <button
            key={rec.id}
            onClick={() => onSelect(rec)}
            className={cn(
              'block w-full overflow-hidden rounded border bg-surface-2 text-left',
              rec.verdict ? VERDICT_BORDER[rec.verdict] : 'border-border',
            )}
            title={`${rec.imageName} — ${rec.verdict ?? 'incomplete'}`}
          >
            <img
              src={thumbUrls.get(rec.id)}
              alt={rec.imageName}
              className={cn('w-full object-cover', open ? 'h-20' : 'h-8')}
            />
            {open && (
              <div className="px-1.5 py-1">
                <div className="truncate text-[10px] text-offwhite">{rec.imageName}</div>
                <div className="font-mono text-[9px] text-muted">
                  {rec.verdict ?? '—'} {rec.confidence != null ? `· ${rec.confidence}` : ''}
                </div>
              </div>
            )}
          </button>
        ))}
      </div>
    </aside>
  );
}
