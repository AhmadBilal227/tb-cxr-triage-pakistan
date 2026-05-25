/**
 * ClinicianReport — M24 evidence panel #4 (GPT-as-interpreter).
 *
 * Async, on-demand narrative from gpt-5.5 vision that TRANSLATES the validated
 * model's structured output into a 1-2 sentence clinician-style paragraph plus
 * a small differential. The pipeline verdict is FIXED; this component does NOT
 * change it.
 *
 * Cost / latency: ~$0.01-0.04 per call, ~5-15s. Lazy by default — the user
 * clicks "Generate clinician report" to fire it. The result is cached in
 * component state so collapsing+expanding the Details panel does not re-bill.
 *
 * Disclosure (always visible above the rendered narrative): names the model
 * (gpt-5.5-2026-04-23 echoed from the response), names the schema version,
 * and says explicitly "Narrative interpretation only — does not change the
 * verdict." That sentence is load-bearing for the M24 honesty contract.
 */
import { useState } from 'react';
import { Sparkles } from 'lucide-react';
import { Button } from '../ui/button';
import type { LocalTriageResult } from '@/lib/providers/localTriage';
import {
  GPT_INTERPRETER_SCHEMA_VERSION,
  gptInterpreter,
  type ClinicianReport as ClinicianReportData,
} from '@/lib/providers/gptInterpreter';

export interface ClinicianReportProps {
  apiKey: string;
  primaryModel: string;
  fallbackModel: string;
  imageDataUrl: string;
  localResult: LocalTriageResult;
}

interface ReportState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  data: ClinicianReportData | null;
  modelId: string | null;
  latencyMs: number | null;
  error: string | null;
}

const INITIAL: ReportState = {
  status: 'idle',
  data: null,
  modelId: null,
  latencyMs: null,
  error: null,
};

export function ClinicianReport({
  apiKey,
  primaryModel,
  fallbackModel,
  imageDataUrl,
  localResult,
}: ClinicianReportProps): JSX.Element {
  const [state, setState] = useState<ReportState>(INITIAL);

  const onGenerate = async (): Promise<void> => {
    setState({ ...INITIAL, status: 'loading' });
    try {
      const call = await gptInterpreter({
        apiKey,
        primaryModel,
        fallbackModel,
        imageDataUrl,
        localResult,
      });
      setState({
        status: 'ready',
        data: call.report,
        modelId: call.audit.model_id_from_response,
        latencyMs: call.latencyMs,
        error: null,
      });
    } catch (err) {
      setState({
        status: 'error',
        data: null,
        modelId: null,
        latencyMs: null,
        error: (err as Error).message,
      });
    }
  };

  if (state.status === 'idle') {
    return (
      <div data-testid="clinician-report-idle" className="space-y-2">
        <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
          clinician report (gpt translates the local model's output)
        </div>
        <p className="text-[10px] leading-snug text-muted/80">
          Optional. Calls gpt-5.5 vision once with the validated model's structured output as input
          (top-5 zones + top-5 TXRV findings + verdict + threshold). The narrative is bound by the
          local verdict; the model cannot disagree or invent regions.
        </p>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            void onGenerate();
          }}
          disabled={!apiKey}
          data-testid="clinician-report-generate"
        >
          <Sparkles className="h-3.5 w-3.5" />
          {apiKey ? 'Generate clinician report (GPT)' : 'Generate (set OpenAI key in Settings first)'}
        </Button>
      </div>
    );
  }

  if (state.status === 'loading') {
    return (
      <div data-testid="clinician-report-loading" className="space-y-1">
        <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
          generating clinician report…
        </div>
        <div className="h-2 w-32 animate-pulse rounded bg-surface-2" />
        <p className="text-[10px] text-muted/70">
          Single call to gpt-5.5 vision. ~5-15 seconds. Does not change the verdict.
        </p>
      </div>
    );
  }

  if (state.status === 'error') {
    return (
      <div data-testid="clinician-report-error" className="space-y-1">
        <div className="font-mono text-[10px] uppercase tracking-wide text-verdict-tb">
          clinician report failed
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

  // state.status === 'ready'
  return (
    <ClinicianReportReadyView
      report={state.data!}
      modelId={state.modelId}
      latencyMs={state.latencyMs}
    />
  );
}

/**
 * Exported for unit-test surface. Renders the READY-state body of
 * `<ClinicianReport />` without depending on the async-fetch state machine.
 * The disclosure line at the bottom is load-bearing for the M24 honesty
 * contract — it MUST always render with this exact "Narrative interpretation
 * only — does not change the verdict." framing.
 */
export interface ClinicianReportReadyViewProps {
  report: ClinicianReportData;
  modelId: string | null;
  latencyMs: number | null;
}

export function ClinicianReportReadyView({
  report,
  modelId,
  latencyMs,
}: ClinicianReportReadyViewProps): JSX.Element {
  const r = report;
  return (
    <div data-testid="clinician-report-ready" className="space-y-2">
      <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
        clinician report (gpt narrative)
      </div>
      <p
        className="rounded-md border border-border bg-surface-2 p-2 text-[12px] leading-relaxed text-offwhite/90"
        data-testid="clinician-report-finding"
      >
        {r.finding}
      </p>
      {r.refusal_or_limitation && (
        <p className="text-[10px] text-verdict-uncertain" data-testid="clinician-report-limitation">
          Limitation: {r.refusal_or_limitation}
        </p>
      )}
      {r.key_regions.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="font-mono text-[10px] uppercase tracking-wide text-muted">
            key regions:
          </span>
          {r.key_regions.map((z) => (
            <span
              key={z}
              data-testid={`clinician-report-region-${z}`}
              className="inline-flex items-center rounded-full border border-border bg-surface-2 px-2 py-0.5 font-mono text-[10px] text-offwhite/80"
            >
              {z}
            </span>
          ))}
        </div>
      )}
      <div className="space-y-1">
        <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
          top differential alternatives (ranked by fit to pipeline evidence)
        </div>
        <ul className="space-y-1 text-[11px]" data-testid="clinician-report-differential">
          {r.top_differential_alternatives.map((d, i) => (
            <li
              key={i}
              className="rounded-md border border-border/60 bg-surface-2/60 px-2 py-1"
              data-testid={`clinician-report-diff-${i}`}
            >
              <span className="font-medium text-offwhite/90">{d.label}</span>
              <span
                className={`ml-2 font-mono text-[9px] uppercase tracking-wider ${
                  d.likelihood === 'consider' ? 'text-verdict-uncertain' : 'text-muted'
                }`}
              >
                {d.likelihood}
              </span>
              <div className="mt-0.5 text-[10px] leading-snug text-offwhite/70">{d.rationale}</div>
            </li>
          ))}
        </ul>
      </div>
      <p
        className="text-[10px] leading-snug text-muted/70"
        data-testid="clinician-report-disclosure"
      >
        Generated by {modelId ?? 'gpt-5.5'} vision from the validated model's structured
        output (schema {GPT_INTERPRETER_SCHEMA_VERSION}; confidence band {r.confidence_qualifier};
        latency {latencyMs ? `${(latencyMs / 1000).toFixed(1)}s` : 'n/a'}). Narrative
        interpretation only — does not change the verdict.
      </p>
    </div>
  );
}
