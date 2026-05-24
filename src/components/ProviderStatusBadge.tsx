import { CheckCircle2, MinusCircle, XCircle } from 'lucide-react';
import type { Provider } from '@/lib/types';
import type { ProviderStatus } from '@/store/providerStatus';

/**
 * Tiny per-provider status indicator used in the Settings drawer.
 *
 * Shows nothing while the provider has never been called and has no key set
 * (the no-keys banner already covers that case). Once it has been called or
 * short-circuited by missing config, reports the LAST result so the user can
 * see WHICH key is the actual problem without DevTools.
 *
 * Status TEXT is stable across runs so tests can match on substrings; the
 * human REASON carried in `status.note` is the actionable detail.
 *
 * Extracted to a standalone module so tests can render it via
 * react-dom/server without needing to instantiate the Radix Dialog tree.
 */
export function ProviderStatusBadge({
  provider,
  hasKey,
  status,
}: {
  provider: Provider;
  hasKey: boolean;
  status: ProviderStatus;
}): JSX.Element | null {
  if (status.state === 'unknown' && !hasKey) return null;
  if (status.state === 'unknown' && hasKey) {
    return (
      <span
        data-testid={`provider-status-${provider}`}
        className="inline-flex items-center gap-1 rounded-md border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-muted"
      >
        <MinusCircle className="h-3 w-3" /> {provider}: configured · not yet called
      </span>
    );
  }
  if (status.state === 'not-configured') {
    return (
      <span
        data-testid={`provider-status-${provider}`}
        className="inline-flex items-center gap-1 rounded-md border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-muted"
      >
        <MinusCircle className="h-3 w-3" /> {provider}: not configured
      </span>
    );
  }
  if (status.state === 'ok') {
    return (
      <span
        data-testid={`provider-status-${provider}`}
        className="inline-flex items-center gap-1 rounded-md border border-verdict-clear/40 bg-verdict-clear/10 px-1.5 py-0.5 font-mono text-[10px] text-verdict-clear"
      >
        <CheckCircle2 className="h-3 w-3" /> {provider}: ok{status.note ? ` · ${status.note}` : ''}
      </span>
    );
  }
  return (
    <span
      data-testid={`provider-status-${provider}`}
      className="inline-flex items-center gap-1 rounded-md border border-verdict-tb/40 bg-verdict-tb/10 px-1.5 py-0.5 font-mono text-[10px] text-verdict-tb"
    >
      <XCircle className="h-3 w-3" /> {provider}: last call {status.state}
      {status.note ? ` · ${status.note}` : ''}
    </span>
  );
}
