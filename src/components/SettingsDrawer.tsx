import { useEffect, useState, type ReactNode } from 'react';
import { AlertTriangle, ChevronRight, Cpu, KeyRound } from 'lucide-react';
import { Dialog, DrawerContent, DialogTitle } from './ui/dialog';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { Button } from './ui/button';
import { settingsStore, useSettings, deriveCapabilities } from '@/store/settings';
import { useProviderStatus } from '@/store/providerStatus';
import { ProviderStatusBadge } from './ProviderStatusBadge';
import { cn } from '@/lib/utils';
import { localHealth, type LocalHealth } from '@/lib/providers/localTriage';

function Field({
  label,
  value,
  onChange,
  placeholder,
  type = 'text',
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  hint?: ReactNode;
}): JSX.Element {
  return (
    <label className="block space-y-1">
      <span className="text-[11px] font-medium text-offwhite">{label}</span>
      <input
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
        autoComplete="off"
        className="w-full rounded-md border border-border bg-ink px-2.5 py-1.5 font-mono text-xs text-offwhite placeholder:text-muted/50 focus-visible:border-provider-openai"
      />
      {hint && <span className="block text-[10px] leading-snug text-muted">{hint}</span>}
    </label>
  );
}

export function SettingsDrawer({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
}): JSX.Element {
  const s = useSettings();
  const caps = deriveCapabilities(s);
  const providerStatus = useProviderStatus();
  const [overridesOpen, setOverridesOpen] = useState(false);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DrawerContent>
        <div className="space-y-5 p-5">
          <DialogTitle className="flex items-center gap-2">
            <KeyRound className="h-4 w-4" /> Settings
          </DialogTitle>

          {/* localStorage warning (always visible) */}
          <div className="flex items-start gap-2 rounded-md border border-verdict-uncertain/30 bg-verdict-uncertain/10 p-2.5 text-[11px] text-verdict-uncertain">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>Keys live in localStorage. Any JS on this page can read them. Do not use production keys.</span>
          </div>

          <div className="space-y-3">
            <div className="space-y-1">
              <Field
                label="OpenAI API key"
                type="password"
                value={s.openaiKey}
                onChange={(v) => settingsStore.set({ openaiKey: v })}
                placeholder="sk-..."
                hint="Used for the quality gate AND the gpt-5.5 vision primary (or borderline verifier when local mode is on)."
              />
              <ProviderStatusBadge provider="openai" hasKey={caps.hasOpenAI} status={providerStatus.openai} />
            </div>
            <div className="space-y-1">
              <Field
                label="Replicate API token (optional)"
                type="password"
                value={s.replicateToken}
                onChange={(v) => settingsStore.set({ replicateToken: v })}
                placeholder="r8_..."
                hint="Optional — enables a BYO Replicate classifier slot and the Replicate CLIP embedding for retrieval. Leave empty if you only use local mode + gpt-5.5 vision."
              />
              <ProviderStatusBadge provider="replicate" hasKey={caps.hasReplicate} status={providerStatus.replicate} />
            </div>
          </div>

          {/* Local mode (Milestone 22) */}
          <LocalModeSection />

          {/* Model overrides (collapsed by default) */}
          <Collapsible open={overridesOpen} onOpenChange={setOverridesOpen}>
            <CollapsibleTrigger className="flex w-full items-center gap-1.5 border-t border-border pt-3 text-[11px] font-medium uppercase tracking-wide text-muted hover:text-offwhite">
              <ChevronRight className={cn('h-3.5 w-3.5 transition-transform', overridesOpen && 'rotate-90')} />
              Model overrides
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-3 pt-3">
              <div className="grid grid-cols-2 gap-2">
                <Field
                  label="TB Classifier — Replicate slug"
                  value={s.overrides.tbClassifierReplicate}
                  onChange={(v) => settingsStore.setOverride({ tbClassifierReplicate: v })}
                  placeholder="owner/tb-cnn"
                  hint="Optional BYO classifier. Push a TB CNN via Cog (see README) or paste a published slug."
                />
                <Field
                  label="…version hash"
                  value={s.overrides.tbClassifierReplicateVersion}
                  onChange={(v) => settingsStore.setOverride({ tbClassifierReplicateVersion: v })}
                  placeholder="version id"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <Field
                  label="Embedding — Replicate CLIP slug"
                  value={s.overrides.embeddingReplicate}
                  onChange={(v) => settingsStore.setOverride({ embeddingReplicate: v })}
                  placeholder="e.g. krthr/clip-embeddings"
                />
                <Field
                  label="…version hash"
                  value={s.overrides.embeddingReplicateVersion}
                  onChange={(v) => settingsStore.setOverride({ embeddingReplicateVersion: v })}
                  placeholder="version id"
                />
              </div>

              {!caps.hasEmbedding && (
                <p className="text-[10px] text-provider-replicate">
                  No embedding provider configured — Stage 3 (retrieval) will be skipped with an inline banner.
                </p>
              )}
            </CollapsibleContent>
          </Collapsible>

          <div className="flex justify-between border-t border-border pt-3">
            <Button variant="ghost" size="sm" onClick={() => settingsStore.reset()}>
              Reset to defaults
            </Button>
            <Button size="sm" onClick={() => onOpenChange(false)}>
              Done
            </Button>
          </div>
        </div>
      </DrawerContent>
    </Dialog>
  );
}

/**
 * Milestone 22 — LOCAL MODE section.
 *
 * Toggle + URL input + per-provider status badge + a one-shot health probe that
 * fires when the user flips the toggle ON. Health output is small and
 * audit-flavored: `engine: ready · model_sha: ... · git: ...` or a clear
 * "engine: unreachable · run uvicorn ..." line.
 */
function LocalModeSection(): JSX.Element {
  const s = useSettings();
  const providerStatus = useProviderStatus();
  const [health, setHealth] = useState<LocalHealth | null>(null);
  const [pinging, setPinging] = useState(false);

  // Ping /health when local mode is toggled on, or when the URL changes while on.
  useEffect(() => {
    if (!s.localMode) {
      setHealth(null);
      return;
    }
    let cancelled = false;
    setPinging(true);
    void localHealth(s.localServerUrl).then((h) => {
      if (cancelled) return;
      setHealth(h);
      setPinging(false);
    });
    return () => {
      cancelled = true;
    };
  }, [s.localMode, s.localServerUrl]);

  return (
    <div className="space-y-2 border-t border-border pt-3">
      <div className="flex items-center justify-between">
        <span className="inline-flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted">
          <Cpu className="h-3.5 w-3.5" /> Local mode (Milestone 22)
        </span>
        <label className="inline-flex cursor-pointer items-center gap-2 text-[11px] text-offwhite">
          <input
            type="checkbox"
            data-testid="local-mode-toggle"
            checked={s.localMode}
            onChange={(e) => settingsStore.set({ localMode: e.target.checked })}
            className="h-3.5 w-3.5 cursor-pointer"
          />
          {s.localMode ? 'on' : 'off'}
        </label>
      </div>
      <p className="text-[10px] leading-snug text-muted">
        Run the trained Rad-DINO + TorchXRayVision + TBHeadT2 pipeline locally on your machine
        (FastAPI server at the URL below). When ON and reachable, this is the PRIMARY perception
        and gpt-5.5 vision becomes a borderline second-opinion verifier. When OFF (or unreachable),
        the M21 VLM-primary path runs unchanged. On a new external site (Pakistani cohort, 3,008
        images) the model measured AUROC 0.78, sens 0.75, spec 0.68 at the shipped operating point —
        the honest field estimate. The in-distribution LODO numbers (AUROC 0.92, sens 0.80,
        spec 0.91 on 13,092 held-out predictions) are an upper bound, not what to expect at your
        site (see CASE_STUDY M15-M22 + Track A).
      </p>
      <p
        className="rounded-md border border-provider-openai/30 bg-provider-openai/5 px-2 py-1.5 text-[10px] leading-snug text-offwhite/90"
        data-testid="recalibrate-onboarding"
      >
        New deployment site? Calibrate to your local cases first —{' '}
        <a
          href="/validate"
          className="font-medium text-provider-openai underline underline-offset-2 hover:opacity-80"
          data-testid="recalibrate-link"
        >
          Validate &amp; recalibrate
        </a>
        . The shipped threshold/temperature were fit on the training cohorts and should be re-fit
        on labeled local cases before trusting sensitivity/specificity.
      </p>
      <Field
        label="Local server URL"
        value={s.localServerUrl}
        onChange={(v) => settingsStore.set({ localServerUrl: v })}
        placeholder="http://localhost:8000"
      />
      <ProviderStatusBadge
        provider="local-triage"
        hasKey={s.localMode}
        status={providerStatus['local-triage']}
      />
      {s.localMode && (
        <div className="rounded-md border border-border bg-surface-2 p-2 font-mono text-[10px] leading-relaxed text-muted">
          {pinging && <span>pinging {s.localServerUrl}/health …</span>}
          {!pinging && health?.ok && (
            <span data-testid="local-health-ok">
              engine: ready · model_sha: {health.model_sha.slice(0, 22)}… · git:{' '}
              {health.git_sha.slice(0, 7)} · T={health.calibration.T.toFixed(4)}
            </span>
          )}
          {!pinging && health && !health.ok && (
            <span data-testid="local-health-down" className="text-provider-replicate">
              engine: unreachable — {health.reason.slice(0, 120)}
              <br />
              <span className="text-muted">{health.hint}</span>
            </span>
          )}
        </div>
      )}
    </div>
  );
}
