/**
 * ClinicianReport — M24 evidence panel #4, v2 radiologist + view-modal flow.
 *
 * Async, on-demand structured radiology report from gpt-5.5 vision that
 * TRANSLATES the validated model's structured output into RSNA-style
 * Findings / Impression / Recommendation sections using Fleischner Society
 * terminology. The pipeline verdict is FIXED; this component does NOT
 * change it.
 *
 * UI flow (user-requested):
 *   1. IDLE     — CTA "Generate radiology report (GPT)"
 *   2. LOADING  — visible progress: spinner + elapsed time + animated bar
 *                 (capped at 90% pending response, since we don't get
 *                 streaming progress events).
 *   3. READY    — single "View report" button. The full report renders ONLY
 *                 inside the URL-bound full-screen modal (cleaner host card,
 *                 reading experience is room-to-breathe).
 *   4. ERROR    — error message + retry.
 *
 * Cost / latency: ~$0.01-0.04 per call, ~5-15s. The result is cached in
 * component state so closing+reopening the modal does not re-bill.
 *
 * Disclosure (always visible inside the modal, above the report): names the
 * model echoed from the response, the schema version, and the load-bearing
 * "Narrative interpretation only — does not change the verdict." sentence.
 */
import { useEffect, useRef, useState } from 'react';
import { Download, FileText, Loader2 } from 'lucide-react';
import { Button } from '../ui/button';
import type { Adjudication } from '@/lib/types';
import type { LocalTriageResult } from '@/lib/providers/localTriage';
import {
  GPT_INTERPRETER_SCHEMA_VERSION,
  gptInterpreter,
  type ClinicianReport as ClinicianReportData,
} from '@/lib/providers/gptInterpreter';
import { useUrlBoundOverlay } from '@/hooks/useUrlBoundOverlay';
import { ClinicianReportModal } from './ClinicianReportModal';

export interface ClinicianReportProps {
  apiKey: string;
  primaryModel: string;
  fallbackModel: string;
  imageDataUrl: string;
  localResult: LocalTriageResult;
  /** Required for the modal + PDF export. */
  adjudication?: Adjudication;
}

interface ReportState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  data: ClinicianReportData | null;
  modelId: string | null;
  latencyMs: number | null;
  error: string | null;
  /** Wall-clock ms since fetch started; used to drive the loading bar + counter. */
  startedAt: number | null;
}

const INITIAL: ReportState = {
  status: 'idle',
  data: null,
  modelId: null,
  latencyMs: null,
  error: null,
  startedAt: null,
};

/** Expected single-call duration in ms. The progress bar asymptotes here. */
const EXPECTED_LATENCY_MS = 10_000;
/** Progress cap before the response actually arrives. */
const PROGRESS_CAP_PCT = 90;

export function ClinicianReport({
  apiKey,
  primaryModel,
  fallbackModel,
  imageDataUrl,
  localResult,
  adjudication,
}: ClinicianReportProps): JSX.Element {
  const [state, setState] = useState<ReportState>(INITIAL);
  const [modalOpen, setModalOpen] = useUrlBoundOverlay('report');

  const onGenerate = async (): Promise<void> => {
    setState({ ...INITIAL, status: 'loading', startedAt: Date.now() });
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
        startedAt: null,
      });
      // Auto-open the modal the first time the report becomes ready, so
      // the user does not have to click twice. They can close and re-open
      // anytime via the persistent "View report" button.
      if (adjudication) setModalOpen(true);
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

  // -----------------------------------------------------------------
  // IDLE
  // -----------------------------------------------------------------
  if (state.status === 'idle') {
    return (
      <div data-testid="clinician-report-idle" className="space-y-2">
        <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
          radiology report (gpt translates the local model's output)
        </div>
        <p className="text-[10px] leading-snug text-muted">
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
          <FileText className="h-3.5 w-3.5" />
          {apiKey ? 'Generate radiology report (GPT)' : 'Generate (set OpenAI key in Settings first)'}
        </Button>
      </div>
    );
  }

  // -----------------------------------------------------------------
  // LOADING — visible progress (elapsed time + animated bar)
  // -----------------------------------------------------------------
  if (state.status === 'loading') {
    return (
      <LoadingState startedAt={state.startedAt} onCancel={() => setState(INITIAL)} />
    );
  }

  // -----------------------------------------------------------------
  // ERROR
  // -----------------------------------------------------------------
  if (state.status === 'error') {
    return (
      <div data-testid="clinician-report-error" className="space-y-1">
        <div className="font-mono text-[10px] uppercase tracking-wide text-verdict-tb">
          radiology report failed
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

  // -----------------------------------------------------------------
  // READY — single "View report" button. Modal carries the actual report.
  // -----------------------------------------------------------------
  const onDownloadInline = (): void => {
    if (!state.data || !adjudication) return;
    // Imported lazily because PDF generation pulls in jspdf (~370 kB).
    void import('./clinicianReportPdf').then((m) =>
      m.downloadClinicianReportPdf({
        report: state.data!,
        adjudication,
        localResult,
        imageDataUrl,
        modelId: state.modelId,
        latencyMs: state.latencyMs,
      }),
    );
  };

  return (
    <div data-testid="clinician-report-ready-summary" className="space-y-2">
      <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
        radiology report ready
      </div>
      <p className="text-[10px] leading-snug text-muted">
        Drafted by {state.modelId ?? 'gpt-5.5'} in{' '}
        {state.latencyMs ? `${(state.latencyMs / 1000).toFixed(1)}s` : 'n/a'}. Narrative
        interpretation only — does not change the verdict.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setModalOpen(true)}
          disabled={!adjudication}
          data-testid="clinician-report-view"
        >
          <FileText className="h-3.5 w-3.5" /> View report
        </Button>
        {adjudication && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onDownloadInline}
            data-testid="clinician-report-download-pdf"
          >
            <Download className="h-3.5 w-3.5" /> PDF
          </Button>
        )}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            void onGenerate();
          }}
        >
          Regenerate
        </Button>
      </div>

      {state.data && adjudication && (
        <ClinicianReportModal
          open={modalOpen}
          onOpenChange={setModalOpen}
          report={state.data}
          modelId={state.modelId}
          latencyMs={state.latencyMs}
          adjudication={adjudication}
          localResult={localResult}
          imageDataUrl={imageDataUrl}
        />
      )}
    </div>
  );
}

// =====================================================================
// LoadingState — spinner + elapsed counter + animated progress bar
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

  // Easing toward the cap: progress = cap * (1 - exp(-3 * t / expected))
  // Reaches ~85% at the expected duration, asymptotes to the cap.
  const t = Math.max(0, elapsedMs) / EXPECTED_LATENCY_MS;
  const eased = 1 - Math.exp(-3 * t);
  const progress = Math.min(PROGRESS_CAP_PCT, PROGRESS_CAP_PCT * eased);
  const elapsedSeconds = (elapsedMs / 1000).toFixed(1);

  return (
    <div data-testid="clinician-report-loading" className="space-y-2">
      <div className="flex items-center gap-2">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-provider-openai" />
        <span className="font-mono text-[10px] uppercase tracking-wide text-muted">
          drafting radiology report…
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
        aria-label="Drafting radiology report"
      >
        <div
          className="h-full bg-provider-openai transition-[width] duration-200 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="flex items-center justify-between">
        <p className="text-[10px] text-muted">
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
// Read-only presentational view, exported for unit tests + the modal.
// The disclosure line is load-bearing for the M24 honesty contract — it
// MUST always render with this exact "Narrative interpretation only —
// does not change the verdict." framing.
// =====================================================================
export interface ClinicianReportReadyViewProps {
  report: ClinicianReportData;
  modelId: string | null;
  latencyMs: number | null;
  /** When provided, renders a "Download PDF" button next to the section heading. */
  onDownload?: () => void;
}

export function ClinicianReportReadyView({
  report,
  modelId,
  latencyMs,
  onDownload,
}: ClinicianReportReadyViewProps): JSX.Element {
  const r = report;
  return (
    <div data-testid="clinician-report-ready" className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
          radiology report (gpt narrative)
        </div>
        {onDownload && (
          <Button
            variant="outline"
            size="sm"
            onClick={onDownload}
            data-testid="clinician-report-ready-download-pdf"
          >
            <Download className="h-3.5 w-3.5" /> Download PDF
          </Button>
        )}
      </div>

      <div
        className="space-y-3 rounded-md border border-border bg-surface-2 p-3 text-[12px] leading-relaxed text-offwhite/90"
        data-testid="clinician-report-body"
      >
        <ReportSection label="Technique">{r.technique}</ReportSection>
        <ReportSection label="Comparison">{r.comparison}</ReportSection>

        <div className="space-y-1.5" data-testid="clinician-report-findings">
          <SectionLabel>Findings</SectionLabel>
          <SubFinding label="Lungs and airways">{r.findings.lungs_and_airways}</SubFinding>
          <SubFinding label="Pleura">{r.findings.pleura}</SubFinding>
          <SubFinding label="Cardiomediastinum">{r.findings.cardiomediastinum}</SubFinding>
          <SubFinding label="Bones and soft tissues">{r.findings.bones_and_soft_tissues}</SubFinding>
        </div>

        <div className="space-y-1" data-testid="clinician-report-impression">
          <SectionLabel>Impression</SectionLabel>
          <ol className="ml-4 list-decimal space-y-1">
            {r.impression.map((it, i) => (
              <li key={i} data-testid={`clinician-report-impression-${i}`}>
                <span>{it.statement}</span>
                {it.likelihood !== 'primary' && (
                  <span
                    className={`ml-1.5 font-mono text-[10px] uppercase tracking-wider ${
                      it.likelihood === 'consider' ? 'text-verdict-uncertain' : 'text-muted'
                    }`}
                  >
                    [{it.likelihood === 'consider' ? 'consider' : 'less likely'}]
                  </span>
                )}
              </li>
            ))}
          </ol>
        </div>

        <ReportSection label="Recommendation">{r.recommendation}</ReportSection>

        {r.limitations.length > 0 && (
          <div className="space-y-1" data-testid="clinician-report-limitations">
            <SectionLabel>Limitations</SectionLabel>
            <ul className="ml-4 list-disc space-y-0.5 text-[11px] text-offwhite/80">
              {r.limitations.map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ul>
          </div>
        )}

        {r.key_regions.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            <span className="font-mono text-[10px] uppercase tracking-wide text-muted">
              regions cited:
            </span>
            {r.key_regions.map((z) => (
              <span
                key={z}
                data-testid={`clinician-report-region-${z}`}
                className="inline-flex items-center rounded-full border border-border bg-surface px-2 py-0.5 font-mono text-[10px] text-offwhite/80"
              >
                {z}
              </span>
            ))}
          </div>
        )}
      </div>

      <p
        className="text-[10px] leading-snug text-muted"
        data-testid="clinician-report-disclosure"
      >
        Generated by {modelId ?? 'gpt-5.5'} vision from the validated model's structured output
        (schema {GPT_INTERPRETER_SCHEMA_VERSION}; latency{' '}
        {latencyMs ? `${(latencyMs / 1000).toFixed(1)}s` : 'n/a'}). Narrative interpretation only —
        does not change the verdict.
      </p>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <div className="font-mono text-[10px] uppercase tracking-wide text-muted">{children}</div>
  );
}

function ReportSection({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <div className="space-y-0.5">
      <SectionLabel>{label}</SectionLabel>
      <p className="text-[12px] leading-relaxed text-offwhite/90">{children}</p>
    </div>
  );
}

function SubFinding({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <p className="text-[12px] leading-relaxed text-offwhite/90">
      <span className="font-medium text-offwhite">{label}: </span>
      {children}
    </p>
  );
}
