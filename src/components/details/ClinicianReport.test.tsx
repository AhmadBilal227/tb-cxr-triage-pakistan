/**
 * ClinicianReport — M24 component test.
 *
 * Vitest runs in the node env (no jsdom) so we cannot dispatch a click event
 * across the async fetch state machine. Strategy: assert the static markup of
 * the IDLE state directly, and the READY state via the exported
 * `ClinicianReportReadyView` sub-component (a pure presentational view the
 * production `ClinicianReport` composes once its fetch resolves).
 *
 * The pin is the M24 honesty contract: the disclosure line at the bottom of
 * the ready state MUST always carry the literal
 *   "Narrative interpretation only — does not change the verdict."
 * framing alongside the model id + schema version. If the test fails because
 * that string drifted, fix the COMPONENT, not the test.
 */
import { describe, it, expect } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  ClinicianReport,
  ClinicianReportReadyView,
} from './ClinicianReport';
import type { LocalTriageResult } from '@/lib/providers/localTriage';
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
  finding:
    'Upper-right zone shows moderate parenchymal opacity consistent with the validated head’s screen-positive call.',
  key_regions: ['upper_r'],
  confidence_qualifier: 'moderate',
  top_differential_alternatives: [
    {
      label: 'Bacterial pneumonia',
      likelihood: 'consider',
      rationale: 'TXRV Lung Opacity 0.51 with apical predominance is non-specific.',
    },
    {
      label: 'Healed granuloma',
      likelihood: 'less_likely',
      rationale: 'Scar/sequelae score is low (0.10).',
    },
  ],
  refusal_or_limitation: null,
};

describe('ClinicianReport — idle state', () => {
  it('renders the Generate report CTA when an OpenAI key is present', () => {
    const html = renderToStaticMarkup(
      <ClinicianReport
        apiKey="sk-fake"
        primaryModel="gpt-5.5-2026-04-23"
        fallbackModel="gpt-5.5-mini-2026-04-23"
        imageDataUrl="data:image/png;base64,xxx"
        localResult={mockLocal()}
      />,
    );
    expect(html).toContain('data-testid="clinician-report-idle"');
    expect(html).toContain('data-testid="clinician-report-generate"');
    expect(html).toContain('Generate clinician report (GPT)');
    // CTA is enabled — no boolean `disabled=""` attribute on the rendered <button>.
    // (The tailwind class "disabled:opacity-50" is a class selector, not an attribute.)
    expect(html).not.toMatch(/<button[^>]*\bdisabled=""/);
  });

  it('disables the CTA + swaps the label when no OpenAI key is configured', () => {
    const html = renderToStaticMarkup(
      <ClinicianReport
        apiKey=""
        primaryModel="gpt-5.5-2026-04-23"
        fallbackModel="gpt-5.5-mini-2026-04-23"
        imageDataUrl="data:image/png;base64,xxx"
        localResult={mockLocal()}
      />,
    );
    expect(html).toContain('Generate (set OpenAI key in Settings first)');
    expect(html).toMatch(/<button[^>]*\bdisabled=""/);
  });

  it('shows the "bound by the local verdict" framing in the idle setup copy', () => {
    const html = renderToStaticMarkup(
      <ClinicianReport
        apiKey="sk-fake"
        primaryModel="gpt-5.5-2026-04-23"
        fallbackModel="gpt-5.5-mini-2026-04-23"
        imageDataUrl="data:image/png;base64,xxx"
        localResult={mockLocal()}
      />,
    );
    expect(html).toContain('bound by the local verdict');
    expect(html).toContain('cannot disagree or invent regions');
  });
});

describe('ClinicianReport — ready state (ClinicianReportReadyView)', () => {
  it('renders the finding, differential, and key-regions structured fields', () => {
    const html = renderToStaticMarkup(
      <ClinicianReportReadyView
        report={REPORT}
        modelId="gpt-5.5-2026-04-23"
        latencyMs={4321}
      />,
    );
    expect(html).toContain('data-testid="clinician-report-finding"');
    expect(html).toContain('data-testid="clinician-report-differential"');
    expect(html).toContain('data-testid="clinician-report-diff-0"');
    expect(html).toContain('data-testid="clinician-report-diff-1"');
    expect(html).toContain('data-testid="clinician-report-region-upper_r"');
    expect(html).toContain('Bacterial pneumonia');
    expect(html).toContain('Healed granuloma');
    expect(html).toContain('consider');
    expect(html).toContain('less_likely');
  });

  it('renders the M24 honesty disclosure with model id, schema version, and the "interpretation only" framing', () => {
    const html = renderToStaticMarkup(
      <ClinicianReportReadyView
        report={REPORT}
        modelId="gpt-5.5-2026-04-23"
        latencyMs={4321}
      />,
    );
    expect(html).toContain('data-testid="clinician-report-disclosure"');
    // Model id echoed (so the audit trail surfaces the actual served model)
    expect(html).toContain('gpt-5.5-2026-04-23');
    // Schema version pinned
    expect(html).toContain('gpt-interpreter-v1');
    // Load-bearing literal sentence
    expect(html).toContain('Narrative interpretation only');
    expect(html).toContain('does not change the verdict');
    // Latency formatted as seconds with 1dp
    expect(html).toContain('4.3s');
  });

  it('omits the limitation line when the model did not refuse or limit', () => {
    const html = renderToStaticMarkup(
      <ClinicianReportReadyView
        report={REPORT}
        modelId="gpt-5.5-2026-04-23"
        latencyMs={4321}
      />,
    );
    expect(html).not.toContain('data-testid="clinician-report-limitation"');
  });

  it('surfaces a refusal_or_limitation as its own row when present', () => {
    const limited: ClinicianReportData = {
      ...REPORT,
      refusal_or_limitation: 'Image quality limited — cannot reliably localize.',
    };
    const html = renderToStaticMarkup(
      <ClinicianReportReadyView
        report={limited}
        modelId="gpt-5.5-2026-04-23"
        latencyMs={4321}
      />,
    );
    expect(html).toContain('data-testid="clinician-report-limitation"');
    expect(html).toContain('Image quality limited');
  });

  it('falls back to "gpt-5.5" in the disclosure when modelId is null and renders "n/a" latency', () => {
    const html = renderToStaticMarkup(
      <ClinicianReportReadyView report={REPORT} modelId={null} latencyMs={null} />,
    );
    expect(html).toContain('Generated by gpt-5.5');
    expect(html).toContain('n/a');
  });
});
