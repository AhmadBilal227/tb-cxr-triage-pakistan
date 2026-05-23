import { useState, type ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, AlertCircle } from 'lucide-react';
import type { Provider, StageStatus } from '@/lib/types';
import { cn, fmtLatency } from '@/lib/utils';
import { ProviderBadge } from './ProviderBadge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';

const STATUS_DOT: Record<StageStatus, string> = {
  queued: 'bg-status-queued',
  running: 'bg-status-running animate-pulse',
  fallback: 'bg-status-fallback animate-pulse',
  done: 'bg-status-done',
  error: 'bg-status-error',
  skipped: 'bg-muted',
};

const STATUS_LABEL: Record<StageStatus, string> = {
  queued: 'queued',
  running: 'running',
  fallback: 'falling back',
  done: 'done',
  error: 'error',
  skipped: 'skipped',
};

export interface StageCardProps {
  title: string;
  status: StageStatus;
  provider?: Provider | null;
  latencyMs?: number | null;
  fellBack?: boolean;
  error?: string;
  note?: string;
  raw?: unknown;
  children?: ReactNode;
}

export function StageCard({
  title,
  status,
  provider,
  latencyMs,
  fellBack,
  error,
  note,
  raw,
  children,
}: StageCardProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const showProvider = provider !== undefined && (status === 'done' || status === 'fallback' || status === 'error');

  return (
    <div
      className={cn(
        'rounded-lg border bg-surface-2 p-3 transition-colors',
        status === 'error' ? 'border-status-error/40' : 'border-border',
        status === 'running' && 'border-status-running/40',
      )}
    >
      <div className="flex items-center gap-2">
        <span className={cn('h-2 w-2 shrink-0 rounded-full', STATUS_DOT[status])} aria-hidden />
        <h3 className="flex-1 text-sm font-medium text-offwhite">{title}</h3>
        {showProvider && <ProviderBadge provider={provider ?? null} />}
        {latencyMs != null && status !== 'running' && (
          <span className="font-mono text-[10px] tabular-nums text-muted">{fmtLatency(latencyMs)}</span>
        )}
      </div>

      <div className="mt-1 flex items-center gap-2 pl-4">
        <span className="font-mono text-[10px] uppercase tracking-wide text-muted">
          {STATUS_LABEL[status]}
        </span>
        {/* Animated fallback transition the clinician can see happen */}
        <AnimatePresence>
          {fellBack && (
            <motion.span
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              className="font-mono text-[10px] text-provider-replicate"
            >
              → Replicate
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* Running: shimmer skeleton mirroring the result shape (no spinner) */}
      {status === 'running' && (
        <div className="mt-3 space-y-2 pl-4">
          <div className="skeleton h-1.5 w-full" />
          <div className="skeleton h-1.5 w-2/3" />
        </div>
      )}

      {note && status !== 'running' && (
        <p className="mt-2 pl-4 text-[11px] leading-relaxed text-muted">{note}</p>
      )}

      {error && (
        <div className="mt-2 flex items-start gap-1.5 pl-4 text-[11px] text-status-error">
          <AlertCircle className="mt-0.5 h-3 w-3 shrink-0" />
          <span className="break-words">{error}</span>
        </div>
      )}

      {status !== 'running' && children && <div className="mt-3 pl-4">{children}</div>}

      {raw != null && status !== 'running' && (
        <Collapsible open={open} onOpenChange={setOpen} className="mt-2 pl-4">
          <CollapsibleTrigger className="flex items-center gap-1 text-[10px] text-muted hover:text-offwhite">
            <ChevronRight className={cn('h-3 w-3 transition-transform', open && 'rotate-90')} />
            raw
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="mono mt-1 max-h-48 overflow-auto rounded bg-ink p-2 text-[10px] text-muted scroll-thin">
              {JSON.stringify(raw, null, 2)}
            </pre>
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  );
}
