import { PanelLeftClose, PanelLeftOpen, History } from 'lucide-react';
import type { CaseHistoryRecord } from '@/lib/types';
import { cn } from '@/lib/utils';
import { HistoryList } from './HistoryList';

/**
 * Desktop history rail (md+). The collapse toggle controls the expanded /
 * icon-only width. Below md this rail is hidden; mobile history is served by
 * the off-canvas drawer in App.tsx (both render the shared HistoryList).
 */
export function LeftRail({
  open,
  onToggle,
  onSelect,
}: {
  open: boolean;
  onToggle: () => void;
  onSelect: (rec: CaseHistoryRecord) => void;
}): JSX.Element {
  return (
    <aside
      className={cn(
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

      <HistoryList onSelect={onSelect} expanded={open} />
    </aside>
  );
}
