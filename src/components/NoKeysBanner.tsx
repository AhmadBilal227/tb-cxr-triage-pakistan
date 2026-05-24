import { KeyRound } from 'lucide-react';
import { Button } from './ui/button';
import { useSettings, deriveCapabilities } from '@/store/settings';

/**
 * BYOK contract banner.
 *
 * Surfaces a calm, top-of-app message when the user has NO provider key set
 * (no OpenAI, no HF, no Replicate). Without this banner, a fresh visitor hits
 * the perception pipeline, sees nothing happen visibly, and has no path to
 * recovery short of opening DevTools — exactly the failure mode observed in
 * the production deploy at tb-triage-research.netlify.app on 2026-05-24.
 *
 * Reuses the existing surface tone (subtle, monospaced, no emoji). Returns
 * null when at least one key is set, so it disappears the moment the user
 * configures something.
 *
 * Split into a thin hook-using wrapper (default export) plus a pure
 * presentational view (`NoKeysBannerView`) so tests can render the markup
 * via react-dom/server without needing a node-env localStorage shim.
 */
export function NoKeysBannerView({
  hasAny,
  onOpenSettings,
}: {
  hasAny: boolean;
  onOpenSettings: () => void;
}): JSX.Element | null {
  if (hasAny) return null;
  return (
    <div
      role="status"
      data-testid="no-keys-banner"
      className="flex items-center justify-center gap-3 border-b border-provider-openai/30 bg-provider-openai/10 px-4 py-1.5 text-center text-[11px] text-offwhite/90"
    >
      <KeyRound className="h-3.5 w-3.5 shrink-0 text-provider-openai" />
      <span>
        Add at least one API key in <span className="font-semibold">Settings</span> — perception runs in
        your browser using your keys (BYOK). No keys, no perception.
      </span>
      <Button variant="outline" size="sm" onClick={onOpenSettings}>
        Open Settings
      </Button>
    </div>
  );
}

export function NoKeysBanner({ onOpenSettings }: { onOpenSettings: () => void }): JSX.Element | null {
  const settings = useSettings();
  const caps = deriveCapabilities(settings);
  // M23 removed Hugging Face. Local mode (the validated trained-model path) is
  // also a valid "configured" state — when it's on the user has perception even
  // without any remote keys, so the BYOK banner stays silent.
  const hasAny = caps.hasOpenAI || caps.hasReplicate || settings.localMode;
  return <NoKeysBannerView hasAny={hasAny} onOpenSettings={onOpenSettings} />;
}
