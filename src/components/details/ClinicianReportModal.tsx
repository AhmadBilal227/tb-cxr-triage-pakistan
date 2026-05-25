/**
 * ClinicianReportModal — full-screen viewer for the v2 radiologist report.
 *
 * The host component (ClinicianReport) hands this modal the already-generated
 * report + audit pins. The modal itself does NO fetching; it only renders +
 * provides the PDF download affordance + close X.
 *
 * Layout:
 *   sticky header band      verdict chip + title + Download PDF + Close X
 *   scrollable body         ClinicianReportReadyView (re-used from inline)
 *   footer band             audit pins (model id, schema version, latency)
 *
 * The Dialog is URL-bound by the parent via `?report=open` (the parent owns
 * the open state through useUrlBoundOverlay so the browser back button
 * closes the modal cleanly).
 */
import { Dialog, FullscreenContent, DialogTitle } from '../ui/dialog';
import { Button } from '../ui/button';
import { Download } from 'lucide-react';
import type { Adjudication, Verdict } from '@/lib/types';
import type { LocalTriageResult } from '@/lib/providers/localTriage';
import type { ClinicianReport as ClinicianReportData } from '@/lib/providers/gptInterpreter';
import { ClinicianReportReadyView } from './ClinicianReport';

const VERDICT_CHIP: Record<Verdict, { label: string; color: string }> = {
  tb: { label: 'TB SUSPECTED', color: '#C8102E' },
  no_tb: { label: 'NO TB', color: '#00754A' },
  abstain: { label: 'UNCERTAIN — REFER', color: '#F59E0B' },
};

export interface ClinicianReportModalProps {
  open: boolean;
  onOpenChange: (next: boolean) => void;
  report: ClinicianReportData;
  modelId: string | null;
  latencyMs: number | null;
  adjudication: Adjudication;
  localResult: LocalTriageResult;
  imageDataUrl: string;
}

export function ClinicianReportModal({
  open,
  onOpenChange,
  report,
  modelId,
  latencyMs,
  adjudication,
  localResult,
  imageDataUrl,
}: ClinicianReportModalProps): JSX.Element {
  const chip = VERDICT_CHIP[adjudication.verdict];

  const onDownload = (): void => {
    // jspdf is ~370 kB minified; lazy-load so the main bundle stays lean.
    void import('./clinicianReportPdf').then((m) =>
      m.downloadClinicianReportPdf({
        report,
        adjudication,
        localResult,
        imageDataUrl,
        modelId,
        latencyMs,
      }),
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <FullscreenContent
        data-testid="clinician-report-modal"
        aria-describedby={undefined}
      >
        {/* Sticky header band */}
        <header className="flex shrink-0 items-center gap-3 border-b border-border bg-surface px-5 py-3">
          <span
            className="rounded-md px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider"
            style={{ color: chip.color, background: `${chip.color}14`, border: `1px solid ${chip.color}40` }}
          >
            {chip.label}
          </span>
          <DialogTitle className="text-sm font-semibold tracking-tight">
            Radiology Report — TB Triage
          </DialogTitle>
          <div className="ml-auto flex items-center gap-2 pr-10">
            <Button
              variant="outline"
              size="sm"
              onClick={onDownload}
              data-testid="clinician-report-modal-download-pdf"
            >
              <Download className="h-3.5 w-3.5" /> Download PDF
            </Button>
          </div>
        </header>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto scroll-thin px-5 py-5">
          <div className="mx-auto max-w-2xl">
            <ClinicianReportReadyView report={report} modelId={modelId} latencyMs={latencyMs} />
          </div>
        </div>
      </FullscreenContent>
    </Dialog>
  );
}
