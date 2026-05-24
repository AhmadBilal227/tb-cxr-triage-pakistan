import type { Provider } from '@/lib/types';
import { cn } from '@/lib/utils';

const LABELS: Record<Provider, string> = {
  replicate: 'Replicate',
  openai: 'OpenAI',
  'local-triage': 'Local',
};

// Fills/borders keep the brand color; text uses brighter on-dark variants so 10px
// badge text clears WCAG AA (4.5:1) on the dark surface.
const STYLES: Record<Provider, string> = {
  replicate: 'bg-provider-replicate/15 text-[#FBBF24] border-provider-replicate/50',
  openai: 'bg-provider-openai/15 text-[#A5B4FC] border-provider-openai/50',
  // M22 local-mode: neutral muted-blue badge (no brand to invoke). Stays
  // distinguishable from the remote providers.
  'local-triage': 'bg-[#1e293b]/40 text-[#94A3B8] border-[#475569]/50',
};

/** Provider badge surfaced on every result card so fallback is always visible. */
export function ProviderBadge({ provider }: { provider: Provider | null }): JSX.Element {
  if (!provider) {
    return (
      <span className="inline-flex items-center rounded border border-border px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-wide text-muted">
        none
      </span>
    );
  }
  return (
    <span
      className={cn(
        'inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-wide',
        STYLES[provider],
      )}
    >
      {LABELS[provider]}
    </span>
  );
}
