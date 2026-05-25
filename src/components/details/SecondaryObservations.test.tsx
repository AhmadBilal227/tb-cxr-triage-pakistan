/**
 * SecondaryObservations — Phase B component test.
 *
 * Same pattern as ClinicianReport.test.tsx: render the IDLE state via the
 * full component, and the READY state via the exported presentational
 * sub-view. The honesty pin is the literal disclosure sentence
 * "VLM observation — not validated, advisory only".
 */
import { describe, it, expect } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  SecondaryObservations,
  SecondaryObservationsReadyView,
} from './SecondaryObservations';
import type { LocalTriageResult } from '@/lib/providers/localTriage';
import type { SecondaryObservations as Data } from '@/lib/providers/gptSecondary';

function mockLocal(): LocalTriageResult {
  return {
    tb_prob: 0.32,
    tb_logit: -0.7,
    s_inactive: 0.18,
    verdict: 'no_tb',
    decided_at_threshold: 0.6105,
    safety_net_applied: null,
    image_quality: { warnings: [] },
    latency_ms: { total: 240 },
    audit: {
      model_id: 'tb_head_t2',
      model_sha: 'sha256:deadbeef',
      calibration: { T: 1.5915, thr_at_95sens: 0.6105, T_sequelae: 1.1313 },
      git_sha: 'abc1234',
      version: 1,
      timestamp: '2026-05-25T00:00:00.000Z',
    },
    zonal_scores: { upper_r: 0.12, hilar: 0.03 },
    txrv_pathologies: { 'Lung Opacity': 0.21 },
  };
}

const EMPTY: Data = {
  image_quality: { adequate: true, concerns: [] },
  support_devices: [],
  cardiomediastinal_notes: [],
  other_incidentals: [],
  limitations: [],
};

const WITH_FINDINGS: Data = {
  image_quality: {
    adequate: false,
    concerns: ['AP projection magnifies the cardiac silhouette', 'patient rotated to the left'],
  },
  support_devices: ['ET tube tip approximately at the carina', 'right subclavian central line'],
  cardiomediastinal_notes: ['cardiac silhouette borderline enlarged'],
  other_incidentals: ['old healed left rib fracture'],
  limitations: ['lateral view not available for further characterization'],
};

describe('SecondaryObservations — idle state', () => {
  it('renders the Run CTA when an OpenAI key is present', () => {
    const html = renderToStaticMarkup(
      <SecondaryObservations
        apiKey="sk-fake"
        primaryModel="gpt-5.5-2026-04-23"
        fallbackModel="gpt-5.5-mini-2026-04-23"
        imageDataUrl="data:image/png;base64,xxx"
        localResult={mockLocal()}
      />,
    );
    expect(html).toContain('data-testid="secondary-observations-idle"');
    expect(html).toContain('data-testid="secondary-observations-generate"');
    expect(html).toContain('Run secondary observations (GPT)');
    expect(html).not.toMatch(/<button[^>]*\bdisabled=""/);
  });

  it('disables the CTA when no OpenAI key is configured', () => {
    const html = renderToStaticMarkup(
      <SecondaryObservations
        apiKey=""
        primaryModel="gpt-5.5-2026-04-23"
        fallbackModel="gpt-5.5-mini-2026-04-23"
        imageDataUrl="data:image/png;base64,xxx"
        localResult={mockLocal()}
      />,
    );
    expect(html).toContain('Run (set OpenAI key in Settings first)');
    expect(html).toMatch(/<button[^>]*\bdisabled=""/);
  });

  it('explicitly names that it does NOT influence the verdict', () => {
    const html = renderToStaticMarkup(
      <SecondaryObservations
        apiKey="sk-fake"
        primaryModel="gpt-5.5-2026-04-23"
        fallbackModel="gpt-5.5-mini-2026-04-23"
        imageDataUrl="data:image/png;base64,xxx"
        localResult={mockLocal()}
      />,
    );
    expect(html).toContain('Does NOT influence the verdict');
  });
});

describe('SecondaryObservations — ready state (empty findings)', () => {
  it('renders all four sections with their empty-state phrasing', () => {
    const html = renderToStaticMarkup(
      <SecondaryObservationsReadyView
        observations={EMPTY}
        modelId="gpt-5.5-2026-04-23"
        latencyMs={4321}
      />,
    );
    expect(html).toContain('data-testid="secondary-observations-ready"');
    expect(html).toContain('data-testid="secondary-image-quality"');
    expect(html).toContain('data-testid="secondary-support-devices"');
    expect(html).toContain('data-testid="secondary-cardiomediastinal"');
    expect(html).toContain('data-testid="secondary-other-incidentals"');
    // Empty-state phrasing
    expect(html).toContain('Adequate for interpretation');
    expect(html).toContain('None observed');
    expect(html).toContain('Unremarkable');
    // Summary counter
    expect(html).toMatch(/no flags/i);
  });

  it('renders the disclosure pin with the load-bearing "advisory only" framing', () => {
    const html = renderToStaticMarkup(
      <SecondaryObservationsReadyView
        observations={EMPTY}
        modelId="gpt-5.5-2026-04-23"
        latencyMs={4321}
      />,
    );
    expect(html).toContain('data-testid="secondary-observations-disclosure"');
    expect(html).toContain('VLM observation');
    expect(html).toContain('not validated');
    expect(html).toContain('advisory only');
    expect(html).toContain('Does not influence the verdict');
    expect(html).toContain('gpt-secondary-v1');
    expect(html).toContain('4.3s');
  });
});

describe('SecondaryObservations — ready state (with findings)', () => {
  it('renders each finding as a bullet under the right section', () => {
    const html = renderToStaticMarkup(
      <SecondaryObservationsReadyView
        observations={WITH_FINDINGS}
        modelId="gpt-5.5-2026-04-23"
        latencyMs={4321}
      />,
    );
    // Image quality concerns
    expect(html).toContain('AP projection magnifies the cardiac silhouette');
    expect(html).toContain('patient rotated to the left');
    // Support devices
    expect(html).toContain('ET tube tip approximately at the carina');
    expect(html).toContain('right subclavian central line');
    // Cardiomediastinal
    expect(html).toContain('cardiac silhouette borderline enlarged');
    // Other incidentals
    expect(html).toContain('old healed left rib fracture');
    // Limitations section appears (only when non-empty)
    expect(html).toContain('data-testid="secondary-limitations"');
    expect(html).toContain('lateral view not available');
  });

  it('counts non-empty flags in the summary header', () => {
    const html = renderToStaticMarkup(
      <SecondaryObservationsReadyView
        observations={WITH_FINDINGS}
        modelId="gpt-5.5-2026-04-23"
        latencyMs={4321}
      />,
    );
    // 1 (image quality not adequate) + 2 (support devices) + 1 (cardio) + 1 (other) = 5
    expect(html).toMatch(/5 flags/);
  });

  it('omits the limitations section when the model returned an empty list', () => {
    const html = renderToStaticMarkup(
      <SecondaryObservationsReadyView
        observations={EMPTY}
        modelId="gpt-5.5-2026-04-23"
        latencyMs={4321}
      />,
    );
    expect(html).not.toContain('data-testid="secondary-limitations"');
  });
});
