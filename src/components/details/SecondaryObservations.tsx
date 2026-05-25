/**
 * SecondaryObservations — Phase B post-M24 evidence panel.
 *
 * Sibling to ClinicianReport (the radiology-report narrative) but for
 * NON-TB side information the trained head ignores by design. Same
 * idle / loading / ready / error state machine; same conservative
 * presentation; same JSON-schema-grounded provider call.
 *
 * Render layout (ready state): four labeled sections (Image quality,
 * Support devices, Cardiomediastinal, Other incidentals) + a Limitations
 * sub-list + the load-bearing "VLM observation — not validated,
 * advisory only" disclosure pinned at the bottom.
 *
 * Empty categories collapse — when nothing was found in a category, the
 * section reads "None observed" rather than displaying an empty list.
 */
import { useEffect, useRef, useState } from 'react';
import { Eye, Loader2 } from 'lucide-react';
import { Button } from '../ui/button';
import {
  GPT_SECONDARY_SCHEMA_VERSION,
  gptSecondary,
  type SecondaryObservations as SecondaryObservationsData,
} from '@/lib/providers/gptSecondary';
import type { LocalTriageResult } from '@/lib/providers/localTriage';

export interface SecondaryObservationsProps {
  apiKey: string;
  primaryModel: string;
  fallbackModel: string;
  imageDataUrl: string;
  localResult: LocalTriageResult;
}

interface ObservationsState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  data: SecondaryObservationsData | null;
  modelId: string | null;
  latencyMs: number | null;
  error: string | null;
  startedAt: number | null;
}

const INITIAL: ObservationsState = {
  status: 'idle',
  data: null,
  modelId: null,
  latencyMs: null,
  error: null,
  startedAt: null,
};

const EXPECTED_LATENCY_MS = 10_000;
const PROGRESS_CAP_PCT = 90;

export function SecondaryObservations({
  apiKey,
  primaryModel,
  fallbackModel,
  imageDataUrl,
  localResult,
}: SecondaryObservationsProps): JSX.Element {
  const [state, setState] = useState<ObservationsState>(INITIAL);

  const onGenerate = async (): Promise<void> => {
    setState({ ...INITIAL, status: 'loading', startedAt: Date.now() });
    try {
      const call = await gptSecondary({
        apiKey,
        primaryModel,
        fallbackModel,
        imageDataUrl,
        localResult,
      });
      setState({
        status: 'ready',
        data: call.observations,
        modelId: call.audit.model_id_from_response,
        latencyMs: call.latencyMs,
        error: null,
        startedAt: null,
      });
    } catch (err) {
      setState({
        status: 'error',
        data: null,
        modelId: null,
        latencyMs: null,
        error: (err as Error).message,
        startedAt: null,
      });
    }
  };

  if (state.status === 'idle') {
    return (
      <div data-testid="secondary-observations-idle" className="space-y-2">
        <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
          secondary observations (vlm — non-tb side information)
        </div>
        <p className="text-[10px] leading-snug text-muted/80">
          Optional. Runs gpt-5.5 vision on the X-ray a second time to surface what the TB-specific
          head ignores: image quality, support devices, cardiomediastinal notes, and non-TB
          incidentals. Does NOT influence the verdict.
        </p>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            void onGenerate();
          }}
          disabled={!apiKey}
          data-testid="secondary-observations-generate"
        >
          <Eye className="h-3.5 w-3.5" />
          {apiKey ? 'Run secondary observations (GPT)' : 'Run (set OpenAI key in Settings first)'}
        </Button>
      </div>
    );
  }

  if (state.status === 'loading') {
    return <LoadingState startedAt={state.startedAt} onCancel={() => setState(INITIAL)} />;
  }

  if (state.status === 'error') {
    return (
      <div data-testid="secondary-observations-error" className="space-y-1">
        <div className="font-mono text-[10px] uppercase tracking-wide text-verdict-tb">
          secondary observations failed
        </div>
        <p className="text-[10px] text-verdict-tb/90">{state.error}</p>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            void onGenerate();
          }}
        >
          Retry
        </Button>
      </div>
    );
  }

  return (
    <SecondaryObservationsReadyView
      observations={state.data!}
      modelId={state.modelId}
      latencyMs={state.latencyMs}
      onRegenerate={() => void onGenerate()}
    />
  );
}

// =====================================================================
function LoadingState({
  startedAt,
  onCancel,
}: {
  startedAt: number | null;
  onCancel: () => void;
}): JSX.Element {
  const [elapsedMs, setElapsedMs] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (startedAt === null) return;
    const tick = (): void => {
      setElapsedMs(Date.now() - startedAt);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [startedAt]);

  const t = Math.max(0, elapsedMs) / EXPECTED_LATENCY_MS;
  const eased = 1 - Math.exp(-3 * t);
  const progress = Math.min(PROGRESS_CAP_PCT, PROGRESS_CAP_PCT * eased);
  const elapsedSeconds = (elapsedMs / 1000).toFixed(1);

  return (
    <div data-testid="secondary-observations-loading" className="space-y-2">
      <div className="flex items-center gap-2">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-provider-openai" />
        <span className="font-mono text-[10px] uppercase tracking-wide text-muted">
          running secondary observations…
        </span>
        <span className="ml-auto font-mono text-[10px] text-muted">
          {elapsedSeconds}s / ~10s
        </span>
      </div>
      <div
        className="h-1.5 w-full overflow-hidden rounded bg-surface-2"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress)}
        aria-label="Running secondary observations"
      >
        <div
          className="h-full bg-provider-openai transition-[width] duration-200 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="flex items-center justify-between">
        <p className="text-[10px] text-muted/70">
          Single call to gpt-5.5 vision. Cap at 90% until response arrives.
        </p>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

// =====================================================================
export interface SecondaryObservationsReadyViewProps {
  observations: SecondaryObservationsData;
  modelId: string | null;
  latencyMs: number | null;
  onRegenerate?: () => void;
}

export function SecondaryObservationsReadyView({
  observations,
  modelId,
  latencyMs,
  onRegenerate,
}: SecondaryObservationsReadyViewProps): JSX.Element {
  const o = observations;
  const totalFindings =
    (o.image_quality.adequate ? 0 : 1) +
    o.support_devices.length +
    o.cardiomediastinal_notes.length +
    o.other_incidentals.length;

  return (
    <div data-testid="secondary-observations-ready" className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
          secondary observations · {totalFindings === 0 ? 'no flags' : `${totalFindings} flag${totalFindings === 1 ? '' : 's'}`}
        </div>
        {onRegenerate && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onRegenerate}
            data-testid="secondary-observations-regenerate"
          >
            Regenerate
          </Button>
        )}
      </div>

      <div
        className="space-y-3 rounded-md border border-border bg-surface-2 p-3 text-[12px] leading-relaxed text-offwhite/90"
        data-testid="secondary-observations-body"
      >
        <Section
          label="Image quality"
          empty={o.image_quality.adequate && o.image_quality.concerns.length === 0 ? 'Adequate for interpretation.' : null}
          dataTestid="secondary-image-quality"
        >
          {!o.image_quality.adequate && o.image_quality.concerns.length === 0 && (
            <p className="text-[11px] text-verdict-uncertain">Image flagged inadequate without specific concerns listed.</p>
          )}
          {o.image_quality.concerns.length > 0 && <BulletList items={o.image_quality.concerns} />}
        </Section>

        <Section
          label="Support devices"
          empty={o.support_devices.length === 0 ? 'None observed.' : null}
          dataTestid="secondary-support-devices"
        >
          <BulletList items={o.support_devices} />
        </Section>

        <Section
          label="Cardiomediastinal notes"
          empty={o.cardiomediastinal_notes.length === 0 ? 'Unremarkable.' : null}
          dataTestid="secondary-cardiomediastinal"
        >
          <BulletList items={o.cardiomediastinal_notes} />
        </Section>

        <Section
          label="Other incidentals (non-TB)"
          empty={o.other_incidentals.length === 0 ? 'None observed.' : null}
          dataTestid="secondary-other-incidentals"
        >
          <BulletList items={o.other_incidentals} />
        </Section>

        {o.limitations.length > 0 && (
          <Section label="Limitations" empty={null} dataTestid="secondary-limitations">
            <BulletList items={o.limitations} muted />
          </Section>
        )}
      </div>

      <p
        className="text-[10px] leading-snug text-muted/70"
        data-testid="secondary-observations-disclosure"
      >
        Generated by {modelId ?? 'gpt-5.5'} vision as a secondary observation pass (schema{' '}
        {GPT_SECONDARY_SCHEMA_VERSION}; latency{' '}
        {latencyMs ? `${(latencyMs / 1000).toFixed(1)}s` : 'n/a'}). VLM observation — not
        validated, advisory only. Does not influence the verdict.
      </p>
    </div>
  );
}

function Section({
  label,
  empty,
  children,
  dataTestid,
}: {
  label: string;
  empty: string | null;
  children: React.ReactNode;
  dataTestid: string;
}): JSX.Element {
  return (
    <div className="space-y-1" data-testid={dataTestid}>
      <div className="font-mono text-[10px] uppercase tracking-wide text-muted">{label}</div>
      {empty ? <p className="text-[11px] text-muted/80">{empty}</p> : children}
    </div>
  );
}

function BulletList({ items, muted }: { items: string[]; muted?: boolean }): JSX.Element {
  return (
    <ul
      className={`ml-4 list-disc space-y-0.5 text-[11px] ${
        muted ? 'text-muted/80' : 'text-offwhite/85'
      }`}
    >
      {items.map((it, i) => (
        <li key={i}>{it}</li>
      ))}
    </ul>
  );
}
