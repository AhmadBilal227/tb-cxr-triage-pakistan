import { useState, type ReactNode } from 'react';
import { AlertTriangle, ChevronRight, KeyRound } from 'lucide-react';
import { Dialog, DrawerContent, DialogTitle } from './ui/dialog';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { Button } from './ui/button';
import { settingsStore, useSettings, deriveCapabilities } from '@/store/settings';
import { cn } from '@/lib/utils';

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
            <Field
              label="OpenAI API key"
              type="password"
              value={s.openaiKey}
              onChange={(v) => settingsStore.set({ openaiKey: v })}
              placeholder="sk-..."
              hint="Used for quality gate, VLM read, and adjudication (orchestration only)."
            />
            <Field
              label="Hugging Face token"
              type="password"
              value={s.hfToken}
              onChange={(v) => settingsStore.set({ hfToken: v })}
              placeholder="hf_..."
              hint="Primary perception layer (serverless Inference API)."
            />
            <Field
              label="Replicate API token (optional)"
              type="password"
              value={s.replicateToken}
              onChange={(v) => settingsStore.set({ replicateToken: v })}
              placeholder="r8_..."
            />
            {!caps.hasReplicate && (
              <div className="flex items-start gap-2 rounded-md border border-border bg-surface-2 p-2 text-[10px] text-muted">
                <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-provider-replicate" />
                <span>No Replicate token — per-stage fallback is disabled. If an HF model is cold or errors, that stage will fail instead of falling back.</span>
              </div>
            )}
          </div>

          {/* Model overrides (collapsed by default) */}
          <Collapsible open={overridesOpen} onOpenChange={setOverridesOpen}>
            <CollapsibleTrigger className="flex w-full items-center gap-1.5 border-t border-border pt-3 text-[11px] font-medium uppercase tracking-wide text-muted hover:text-offwhite">
              <ChevronRight className={cn('h-3.5 w-3.5 transition-transform', overridesOpen && 'rotate-90')} />
              Model overrides
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-3 pt-3">
              <Field
                label="TB Classifier — HF model"
                value={s.overrides.tbClassifierHf}
                onChange={(v) => settingsStore.setOverride({ tbClassifierHf: v })}
              />
              <div className="grid grid-cols-2 gap-2">
                <Field
                  label="TB Classifier — Replicate slug"
                  value={s.overrides.tbClassifierReplicate}
                  onChange={(v) => settingsStore.setOverride({ tbClassifierReplicate: v })}
                  placeholder="owner/tb-cnn"
                  hint="Push a TB CNN via Cog (see README) or paste a published slug."
                />
                <Field
                  label="…version hash"
                  value={s.overrides.tbClassifierReplicateVersion}
                  onChange={(v) => settingsStore.setOverride({ tbClassifierReplicateVersion: v })}
                  placeholder="version id"
                />
              </div>

              <Field
                label="General CXR — HF model"
                value={s.overrides.generalCxrHf}
                onChange={(v) => settingsStore.setOverride({ generalCxrHf: v })}
              />
              <div className="grid grid-cols-2 gap-2">
                <Field
                  label="General CXR — Replicate slug"
                  value={s.overrides.generalCxrReplicate}
                  onChange={(v) => settingsStore.setOverride({ generalCxrReplicate: v })}
                  placeholder="owner/cxr-cls"
                />
                <Field
                  label="…version hash"
                  value={s.overrides.generalCxrReplicateVersion}
                  onChange={(v) => settingsStore.setOverride({ generalCxrReplicateVersion: v })}
                  placeholder="version id"
                />
              </div>

              <Field
                label="CXR Embedding — HF Inference Endpoint URL"
                value={s.overrides.embeddingEndpointUrl}
                onChange={(v) => settingsStore.setOverride({ embeddingEndpointUrl: v })}
                placeholder="https://xxxx.endpoints.huggingface.cloud"
                hint="google/cxr-foundation is NOT on free serverless — paste your dedicated Inference Endpoint URL. See README."
              />
              <div className="grid grid-cols-2 gap-2">
                <Field
                  label="Embedding — Replicate CLIP slug"
                  value={s.overrides.embeddingReplicate}
                  onChange={(v) => settingsStore.setOverride({ embeddingReplicate: v })}
                  placeholder="e.g. CLIP-ViT-L"
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
