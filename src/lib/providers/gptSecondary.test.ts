/**
 * gptSecondary provider tests (Phase B post-M24).
 *
 * Load-bearing properties:
 *   1. Prompt forbids TB diagnosis (the stage's whole point — keep GPT out
 *      of the verdict).
 *   2. Validator returns the safe-default shape on malformed input (every
 *      category present, empty arrays where the model omitted them).
 *   3. Missing OpenAI key raises before any network call.
 *
 * Strategy: mock fetch (not openaiJSON) — same pattern as the M24
 * gptInterpreter test suite.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  GPT_SECONDARY_PROMPT,
  GPT_SECONDARY_PROMPT_HASH,
  GPT_SECONDARY_SCHEMA_HASH,
  GPT_SECONDARY_SCHEMA_VERSION,
  gptSecondary,
  validateAndNormalize,
  type SecondaryObservations,
} from './gptSecondary';
import type { LocalTriageResult } from './localTriage';

function mockLocal(over: Partial<LocalTriageResult> = {}): LocalTriageResult {
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
    ...over,
  };
}

function mockEnvelope(obs: SecondaryObservations): Record<string, unknown> {
  return {
    model: 'gpt-5.5-2026-04-23',
    output: [
      {
        type: 'message',
        content: [{ type: 'output_text', text: JSON.stringify(obs) }],
      },
    ],
  };
}

function wellFormed(over: Partial<SecondaryObservations> = {}): SecondaryObservations {
  return {
    image_quality: { adequate: true, concerns: [] },
    support_devices: [],
    cardiomediastinal_notes: [],
    other_incidentals: [],
    limitations: [],
    ...over,
  };
}

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('GPT_SECONDARY constants', () => {
  it('prompt hash is a deterministic 8-hex fingerprint', () => {
    expect(GPT_SECONDARY_PROMPT_HASH).toMatch(/^[0-9a-f]{8}$/);
    expect(GPT_SECONDARY_SCHEMA_HASH).toMatch(/^[0-9a-f]{8}$/);
  });

  it('schema version is the pinned constant we audit against', () => {
    expect(GPT_SECONDARY_SCHEMA_VERSION).toBe('gpt-secondary-v1');
  });

  it('the prompt explicitly forbids re-diagnosing TB (stage is non-TB only)', () => {
    expect(GPT_SECONDARY_PROMPT).toMatch(/do not diagnose tb/i);
    expect(GPT_SECONDARY_PROMPT).toMatch(/wrong-by-omission over invention/i);
    // No claim of clinical-grade calibration
    expect(GPT_SECONDARY_PROMPT).not.toMatch(/calibrated/i);
  });

  it('the prompt names the four observation categories', () => {
    expect(GPT_SECONDARY_PROMPT).toMatch(/image_quality/);
    expect(GPT_SECONDARY_PROMPT).toMatch(/support_devices/);
    expect(GPT_SECONDARY_PROMPT).toMatch(/cardiomediastinal_notes/);
    expect(GPT_SECONDARY_PROMPT).toMatch(/other_incidentals/);
  });
});

describe('validateAndNormalize', () => {
  it('returns the safe-default shape on a fully-empty payload', () => {
    const o = validateAndNormalize({});
    expect(o.image_quality.adequate).toBe(true);
    expect(o.image_quality.concerns).toEqual([]);
    expect(o.support_devices).toEqual([]);
    expect(o.cardiomediastinal_notes).toEqual([]);
    expect(o.other_incidentals).toEqual([]);
    expect(o.limitations).toEqual([]);
  });

  it('passes through valid arrays untouched', () => {
    const input: SecondaryObservations = {
      image_quality: { adequate: false, concerns: ['AP projection magnifies cardiac silhouette'] },
      support_devices: ['ET tube tip approximately at the carina'],
      cardiomediastinal_notes: ['cardiac silhouette borderline enlarged'],
      other_incidentals: ['old healed left rib fracture'],
      limitations: ['lateral view not available for further characterization'],
    };
    const o = validateAndNormalize(input);
    expect(o).toEqual(input);
  });

  it('drops empty strings + caps each category at 20 items (defensive)', () => {
    const oversized = {
      image_quality: { adequate: true, concerns: ['', '', 'real concern'] },
      support_devices: Array.from({ length: 30 }, (_, i) => `device ${i}`),
      cardiomediastinal_notes: [],
      other_incidentals: [],
      limitations: [],
    };
    const o = validateAndNormalize(oversized);
    expect(o.image_quality.concerns).toEqual(['real concern']);
    expect(o.support_devices).toHaveLength(20);
  });

  it('defaults image_quality.adequate to true when missing/malformed', () => {
    const o = validateAndNormalize({
      image_quality: { adequate: 'yes', concerns: ['something'] },
      support_devices: [],
      cardiomediastinal_notes: [],
      other_incidentals: [],
      limitations: [],
    });
    expect(o.image_quality.adequate).toBe(true);
    expect(o.image_quality.concerns).toEqual(['something']);
  });
});

describe('gptSecondary — happy path', () => {
  it('returns parsed observations + audit pins from a well-formed response', async () => {
    const r = await runSecondary(
      wellFormed({
        support_devices: ['ECG leads on chest wall'],
        cardiomediastinal_notes: ['cardiac silhouette unremarkable'],
      }),
    );
    expect(r.observations.support_devices).toEqual(['ECG leads on chest wall']);
    expect(r.observations.cardiomediastinal_notes).toEqual(['cardiac silhouette unremarkable']);
    expect(r.audit.prompt_hash).toBe(GPT_SECONDARY_PROMPT_HASH);
    expect(r.audit.schema_version).toBe(GPT_SECONDARY_SCHEMA_VERSION);
    expect(r.audit.model_id_from_response).toBe('gpt-5.5-2026-04-23');
  });
});

describe('gptSecondary — error paths', () => {
  it('throws OpenAIError when the API key is missing', async () => {
    await expect(
      gptSecondary({
        apiKey: '',
        primaryModel: 'gpt-5.5',
        fallbackModel: 'gpt-5.5-instant',
        imageDataUrl: 'data:image/png;base64,iVBORw0K',
        localResult: mockLocal(),
      }),
    ).rejects.toThrow(/API key missing/i);
  });

  it('throws when the model returns a non-object payload', async () => {
    const envelope = {
      model: 'gpt-5.5',
      output: [{ type: 'message', content: [{ type: 'output_text', text: '"bare string"' }] }],
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(envelope), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
    await expect(
      gptSecondary({
        apiKey: 'sk-test',
        primaryModel: 'gpt-5.5',
        fallbackModel: 'gpt-5.5-instant',
        imageDataUrl: 'data:image/png;base64,iVBORw0K',
        localResult: mockLocal(),
      }),
    ).rejects.toThrow();
  });
});

async function runSecondary(obs: SecondaryObservations): Promise<Awaited<ReturnType<typeof gptSecondary>>> {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      new Response(JSON.stringify(mockEnvelope(obs)), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    ),
  );
  return gptSecondary({
    apiKey: 'sk-test',
    primaryModel: 'gpt-5.5',
    fallbackModel: 'gpt-5.5-instant',
    imageDataUrl: 'data:image/png;base64,iVBORw0K',
    localResult: mockLocal(),
  });
}
