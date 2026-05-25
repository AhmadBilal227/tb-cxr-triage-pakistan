import { ChevronUp } from 'lucide-react';
import type { Adjudication, Verdict } from '@/lib/types';
import { VERDICT_HEX } from '@/lib/colors';

const VERDICT_LABEL: Record<Verdict, string> = {
  tb: 'TB SUSPECTED',
  no_tb: 'NO TB',
  abstain: 'UNCERTAIN — REFER',
};

/**
 * Slim sticky bar shown when the verdict card is collapsed, so the X-ray
 * viewer can take the full canvas. Click anywhere to re-expand the full
 * findings. Eager (not lazy) since it must render the instant the user
 * collapses.
 */
export function VerdictSummaryBar({
  adjudication,
  onExpand,
}: {
  adjudication: Adjudication;
  onExpand: () => void;
}): JSX.Element {
  const color = VERDICT_HEX[adjudication.verdict];
  return (
    <button
      type="button"
      onClick={onExpand}
      data-testid="verdict-summary-bar"
      aria-label="Expand findings"
      className="sticky bottom-0 z-10 flex w-full items-center gap-3 border-t border-border bg-surface/95 px-6 py-2.5 text-left backdrop-blur transition-colors hover:bg-surface-2"
    >
      <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} aria-hidden />
      <span className="text-sm font-semibold tracking-tight" style={{ color }}>
        {VERDICT_LABEL[adjudication.verdict]}
      </span>
      <span className="font-mono text-[10px] text-muted">confidence {adjudication.confidence}</span>
      {adjudication.abstain_reason && (
        <span className="hidden truncate text-[11px] text-muted sm:inline">
          · {adjudication.abstain_reason}
        </span>
      )}
      <span className="ml-auto flex items-center gap-1 font-mono text-[10px] uppercase tracking-wide text-muted">
        Expand <ChevronUp className="h-4 w-4" />
      </span>
    </button>
  );
}
