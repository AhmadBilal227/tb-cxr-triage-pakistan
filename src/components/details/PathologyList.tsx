/**
 * PathologyList — M24 evidence panel #3.
 *
 * Sortable chip list of the 18 TorchXRayVision named-finding probabilities the
 * trained head's fusion lever consumes. Default sort: probability descending.
 *
 * Color semantics:
 *   - green (low):   p <= 0.2  — finding not flagged by the backbone
 *   - amber (mid):   0.2 < p < 0.5  — moderate signal worth surfacing
 *   - red (high):    p >= 0.5  — strong signal the head saw
 *
 * Honest framing: these are FEATURE inputs to the TB head — not independent
 * diagnoses. The TXRV DenseNet emits raw logits which we sigmoid (the engine
 * runs the DenseNet with `op_threshs=None` precisely so the raw logits feed
 * the TB head; this UI just sigmoids them for human eyes).
 */
import type { TxrvPathologies } from '@/lib/providers/localTriage';
import { severityClass } from './colorScale';

export interface PathologyListProps {
  pathologies: TxrvPathologies;
}

const SEVERITY_STYLE: Record<'low' | 'mid' | 'high', string> = {
  low: 'border-verdict-clear/40 bg-verdict-clear/10 text-verdict-clear',
  mid: 'border-verdict-uncertain/40 bg-verdict-uncertain/10 text-verdict-uncertain',
  high: 'border-verdict-tb/40 bg-verdict-tb/10 text-verdict-tb',
};

export function PathologyList({ pathologies }: PathologyListProps): JSX.Element {
  const items = Object.entries(pathologies)
    .filter(([, v]) => typeof v === 'number' && Number.isFinite(v))
    .sort((a, b) => b[1] - a[1]);

  return (
    <div data-testid="pathology-list" className="space-y-1.5">
      <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
        TorchXRayVision pathology scores
      </div>
      <p className="text-[10px] leading-snug text-muted">
        Other findings the perception backbone sees, used as input features to the TB head. Not
        independent diagnoses.
      </p>
      {items.length === 0 ? (
        <p className="text-[11px] text-muted">No pathology scores in this run.</p>
      ) : (
        <ul className="flex flex-wrap gap-1.5" data-testid="pathology-chip-list">
          {items.map(([label, value]) => {
            const sev = severityClass(value);
            return (
              <li
                key={label}
                data-testid={`pathology-chip-${label.replace(/\s+/g, '-')}`}
                data-severity={sev}
                className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] ${SEVERITY_STYLE[sev]}`}
                title={`${label}: ${value.toFixed(4)}`}
              >
                <span className="truncate">{label}</span>
                <span className="font-mono opacity-80">{value.toFixed(2)}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
