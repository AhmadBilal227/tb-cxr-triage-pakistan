import { useEffect, useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
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
 * URLs are minted in an effect (never inline in render) so they are revoked
 * on cleanup — preventing the Blob-URL leak that grows with every history
 * mutation over a session.
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

/**
 * The case-history thumbnail list. Shared by the desktop LeftRail (md+) and
 * the mobile off-canvas drawer (below md) so history is reachable on every
 * viewport. `expanded` controls thumbnail height + metadata visibility —
 * the collapsed desktop rail passes false; the mobile drawer passes true.
 */
export function HistoryList({
  onSelect,
  expanded,
}: {
  onSelect: (rec: CaseHistoryRecord) => void;
  expanded: boolean;
}): JSX.Element {
  const history = useLiveQuery(() => listHistory(50), [], [] as CaseHistoryRecord[]);
  const thumbUrls = useThumbnailUrls(history);

  return (
    <div className="flex-1 space-y-2 overflow-y-auto scroll-thin p-2">
      {history.length === 0 && expanded && (
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
            className={cn('w-full object-cover', expanded ? 'h-20' : 'h-8')}
          />
          {expanded && (
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
  );
}
