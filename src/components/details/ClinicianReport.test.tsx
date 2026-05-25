/**
 * ClinicianReport — M24 component test (v2 radiologist schema, view-modal flow).
 *
 * Vitest runs in the node env (no jsdom) so we cannot dispatch click events
 * across the async fetch state machine. Strategy:
 *   - Assert the IDLE state's static markup directly (the only public state
 *     reachable on first render with no async work done yet).
 *   - Assert the READY presentational layer via `ClinicianReportReadyView`,
 *     which the modal + ClinicianReport both compose.
 *
 * Because the live component uses `useUrlBoundOverlay` (depends on react-
 * router-dom's `useSearchParams`), every render of the full component must
 * be wrapped in a router context. The presentational view (ReadyView) has
 * no hook dependencies and renders standalone.
 *
 * The pin is the M24 honesty contract: the disclosure line at the bottom of
 * the ready state MUST always carry the literal
 *   "Narrative interpretation only — does not change the verdict."
 * framing alongside the model id + schema version. If a test fails because
 * that string drifted, fix the COMPONENT, not the test.
 */
import { describe, it, expect } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { ClinicianReport, ClinicianReportReadyView } from './ClinicianReport';
import type { LocalTriageResult } from '@/lib/providers/localTriage';
import type {
  ClinicianReportProps,
} from './ClinicianReport';
import type { ClinicianReport as ClinicianReportData } from '@/lib/providers/gptInterpreter';

function mockLocal(): LocalTriageResult {
  return {
    tb_prob: 0.7,
    tb_logit: 1.2,
    s_inactive: 0.1,
    verdict: 'tb',
    decided_at_threshold: 0.6105,
    safety_net_applied: null,
    image_quality: { warnings: [] },
    latency_ms: { total: 320 },
    audit: {
      model_id: 'tb_head_t2',
      model_sha: 'sha256:deadbeef',
      calibration: { T: 1.5915, thr_at_95sens: 0.6105, T_sequelae: 1.1313 },
      git_sha: 'abc1234',
      version: 1,
      timestamp: '2026-05-25T00:00:00.000Z',
    },
    zonal_scores: { upper_r: 0.42, hilar: 0.04 },
    txrv_pathologies: { 'Lung Opacity': 0.51 },
  };
}

const REPORT: ClinicianReportData = {
  technique: 'Single frontal chest radiograph; AI-assisted TB triage pipeline.',
  comparison: 'No prior studies available for comparison.',
  findings: {
    lungs_and_airways:
      'Moderate parenchymal opacity in the right upper lobe with apical predominance, consistent with the validated head’s screen-positive call.',
    pleura: 'The pleural spaces are clear.',
    cardiomediastinum: 'The cardiomediastinal silhouette is unremarkable.',
    bones_and_soft_tissues: 'No osseous abnormality is identified on this projection.',
  },
  impression: [
    {
      statement:
        'Radiographic findings raise concern for active pulmonary tuberculosis. Bacteriological confirmation (sputum AFB smear, culture, or NAAT) is recommended.',
      likelihood: 'primary',
    },
    {
      statement: 'Bacterial pneumonia.',
      likelihood: 'consider',
    },
    {
      statement: 'Post-inflammatory scarring.',
      likelihood: 'less_likely',
    },
  ],
  recommendation:
    'Recommend sputum AFB smear, culture, and NAAT; clinical evaluation for TB risk factors and symptomatic assessment.',
  key_regions: ['upper_r'],
  limitations: [
    'Single-view frontal radiograph; lateral and CT may yield additional information.',
    'Radiographic features are not diagnostic of microbiological status.',
  ],
};

function renderInRouter(props: ClinicianReportProps): string {
  return renderToStaticMarkup(
    <MemoryRouter initialEntries={['/']}>
      <ClinicianReport {...props} />
    </MemoryRouter>,
  );
}

describe('ClinicianReport — idle state', () => {
  it('renders the Generate report CTA when an OpenAI key is present', () => {
    const html = renderInRouter({
      apiKey: 'sk-fake',
      primaryModel: 'gpt-5.5-2026-04-23',
      fallbackModel: 'gpt-5.5-mini-2026-04-23',
      imageDataUrl: 'data:image/png;base64,xxx',
      localResult: mockLocal(),
    });
    expect(html).toContain('data-testid="clinician-report-idle"');
    expect(html).toContain('data-testid="clinician-report-generate"');
    expect(html).toContain('Generate radiology report (GPT)');
    expect(html).not.toMatch(/<button[^>]*\bdisabled=""/);
  });

  it('disables the CTA + swaps the label when no OpenAI key is configured', () => {
    const html = renderInRouter({
      apiKey: '',
      primaryModel: 'gpt-5.5-2026-04-23',
      fallbackModel: 'gpt-5.5-mini-2026-04-23',
      imageDataUrl: 'data:image/png;base64,xxx',
      localResult: mockLocal(),
    });
    expect(html).toContain('Generate (set OpenAI key in Settings first)');
    expect(html).toMatch(/<button[^>]*\bdisabled=""/);
  });

  it('shows the "bound by the local verdict" framing in the idle setup copy', () => {
    const html = renderInRouter({
      apiKey: 'sk-fake',
      primaryModel: 'gpt-5.5-2026-04-23',
      fallbackModel: 'gpt-5.5-mini-2026-04-23',
      imageDataUrl: 'data:image/png;base64,xxx',
      localResult: mockLocal(),
    });
    expect(html).toContain('bound by the local verdict');
    expect(html).toContain('cannot disagree or invent regions');
  });
});

describe('ClinicianReport — ready state (ClinicianReportReadyView)', () => {
  it('renders the RSNA-style structured sections (findings + impression + recommendation + limitations)', () => {
    const html = renderToStaticMarkup(
      <ClinicianReportReadyView report={REPORT} modelId="gpt-5.5-2026-04-23" latencyMs={4321} />,
    );
    expect(html).toContain('data-testid="clinician-report-body"');
    expect(html).toContain('data-testid="clinician-report-findings"');
    expect(html).toContain('data-testid="clinician-report-impression"');
    expect(html).toContain('data-testid="clinician-report-limitations"');
    // Section labels render
    expect(html).toMatch(/Technique/);
    expect(html).toMatch(/Comparison/);
    expect(html).toMatch(/Findings/);
    expect(html).toMatch(/Impression/);
    expect(html).toMatch(/Recommendation/);
    expect(html).toMatch(/Limitations/);
    // Findings sub-section labels
    expect(html).toContain('Lungs and airways');
    expect(html).toContain('Pleura');
    expect(html).toContain('Cardiomediastinum');
    expect(html).toContain('Bones and soft tissues');
    // Impression items numbered + non-primary likelihood badges rendered
    expect(html).toContain('data-testid="clinician-report-impression-0"');
    expect(html).toContain('data-testid="clinician-report-impression-1"');
    expect(html).toContain('data-testid="clinician-report-impression-2"');
    expect(html).toContain('consider');
    expect(html).toContain('less likely');
    // Region chip from key_regions
    expect(html).toContain('data-testid="clinician-report-region-upper_r"');
  });

  it('renders the M24 honesty disclosure with model id, schema version, and the "interpretation only" framing', () => {
    const html = renderToStaticMarkup(
      <ClinicianReportReadyView report={REPORT} modelId="gpt-5.5-2026-04-23" latencyMs={4321} />,
    );
    expect(html).toContain('data-testid="clinician-report-disclosure"');
    expect(html).toContain('gpt-5.5-2026-04-23');
    expect(html).toContain('gpt-interpreter-v2');
    expect(html).toContain('Narrative interpretation only');
    expect(html).toContain('does not change the verdict');
    expect(html).toContain('4.3s');
  });

  it('does NOT render a "primary" likelihood badge on the first impression item (it is the anchor)', () => {
    const html = renderToStaticMarkup(
      <ClinicianReportReadyView report={REPORT} modelId="gpt-5.5-2026-04-23" latencyMs={4321} />,
    );
    const idx0 = html.indexOf('data-testid="clinician-report-impression-0"');
    const idx1 = html.indexOf('data-testid="clinician-report-impression-1"');
    expect(idx0).toBeGreaterThan(-1);
    expect(idx1).toBeGreaterThan(idx0);
    const firstItemHtml = html.slice(idx0, idx1);
    expect(firstItemHtml).not.toMatch(/\[primary\]/i);
  });

  it('renders the Download PDF button only when an onDownload handler is wired (ReadyView)', () => {
    const withHandler = renderToStaticMarkup(
      <ClinicianReportReadyView
        report={REPORT}
        modelId="gpt-5.5-2026-04-23"
        latencyMs={4321}
        onDownload={(): void => undefined}
      />,
    );
    expect(withHandler).toContain('data-testid="clinician-report-ready-download-pdf"');

    const noHandler = renderToStaticMarkup(
      <ClinicianReportReadyView report={REPORT} modelId="gpt-5.5-2026-04-23" latencyMs={4321} />,
    );
    expect(noHandler).not.toContain('data-testid="clinician-report-ready-download-pdf"');
  });

  it('falls back to "gpt-5.5" in the disclosure when modelId is null and renders "n/a" latency', () => {
    const html = renderToStaticMarkup(
      <ClinicianReportReadyView report={REPORT} modelId={null} latencyMs={null} />,
    );
    expect(html).toContain('Generated by gpt-5.5');
    expect(html).toContain('n/a');
  });
});
