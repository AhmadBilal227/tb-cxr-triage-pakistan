/**
 * gptInterpreter provider tests (Milestone 24).
 *
 * The provider's load-bearing properties:
 *   1. It calls the Responses API with the gpt-interpreter prompt + strict
 *      JSON schema; the evidence message is built from the local result.
 *   2. Validator normalizes/clamps the response: enums fall back to safe
 *      defaults; key_regions is filtered to the SUBSET of zone keys we
 *      passed in (so the model cannot invent zone names).
 *   3. Missing OpenAI key raises before any network call.
 *
 * Strategy: mock fetch (not openaiJSON) — covers the full path including
 * envelope parse, schema validation, and our subset filter. Mirrors the
 * mock pattern in vlmTriage.test.ts.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  GPT_INTERPRETER_PROMPT,
  GPT_INTERPRETER_PROMPT_HASH,
  GPT_INTERPRETER_SCHEMA_HASH,
  GPT_INTERPRETER_SCHEMA_VERSION,
  buildEvidenceMessage,
  gptInterpreter,
  type ClinicianReport,
} from './gptInterpreter';
import type { LocalTriageResult } from './localTriage';

function mockLocal(over: Partial<LocalTriageResult> = {}): LocalTriageResult {
  return {
    tb_prob: 0.9999,
    tb_logit: 16.99,
    s_inactive: 0.046,
    verdict: 'tb',
    decided_at_threshold: 0.6105,
    safety_net_applied: null,
    image_quality: { warnings: [] },
    latency_ms: { harmonize: 6, seg: 116, rad_dino: 117, txrv: 62, heads: 4, total: 317 },
    audit: {
      model_id: 'tb_head_t2',
      model_sha: 'sha256:66846e47e242f4d1',
      calibration: { T: 1.5915, thr_at_95sens: 0.6105, T_sequelae: 1.1313 },
      git_sha: 'deadbee',
      version: 1,
      timestamp: '2026-05-25T00:00:00.000Z',
    },
    zonal_scores: {
      upper_r: 0.406,
      mid_r: 0.027,
      lower_r: 0.005,
      upper_l: 0.163,
      mid_l: 0.038,
      lower_l: 0.0006,
      hilar: 0.037,
    },
    txrv_pathologies: {
      'Lung Opacity': 0.5183,
      Consolidation: 0.2542,
      Pneumonia: 0.2529,
      Atelectasis: 0.1,
    },
    ...over,
  };
}

function mockResponsesEnvelope(report: ClinicianReport): Record<string, unknown> {
  // Shape mirrors what postResponses returns: an output array carrying a single
  // assistant message whose content is an output_text block carrying the JSON.
  return {
    model: 'gpt-5.5-2026-04-23',
    output: [
      {
        type: 'message',
        content: [{ type: 'output_text', text: JSON.stringify(report) }],
      },
    ],
  };
}

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('buildEvidenceMessage', () => {
  it('puts the verdict, calibration, and top-5 zones + top-5 findings in the message', () => {
    const local = mockLocal();
    const msg = buildEvidenceMessage(local);
    expect(msg).toContain('verdict: tb');
    expect(msg).toContain('tb_prob (calibrated under T=1.591)');
    expect(msg).toContain('decided_at_threshold: 0.6105');
    // Zones sorted desc — upper_r at the top
    expect(msg.indexOf('upper_r: 0.4060')).toBeGreaterThan(0);
    expect(msg.indexOf('upper_r: 0.4060')).toBeLessThan(msg.indexOf('hilar: 0.0370'));
    // Top-5 TXRV findings present
    expect(msg).toContain('Lung Opacity: 0.5183');
    expect(msg).toContain('Consolidation: 0.2542');
  });

  it('surfaces safety_net_applied + image_quality warnings when present', () => {
    const local = mockLocal({
      safety_net_applied: 'scar-shape pattern flagged for re-read',
      image_quality: { warnings: ['suspected MONOCHROME1 polarity'] },
    });
    const msg = buildEvidenceMessage(local);
    expect(msg).toContain('safety_net_applied: scar-shape pattern flagged for re-read');
    expect(msg).toContain('suspected MONOCHROME1 polarity');
  });

  it('omits zone / pathology blocks when the local result lacks them', () => {
    const local = mockLocal({ zonal_scores: undefined, txrv_pathologies: undefined });
    const msg = buildEvidenceMessage(local);
    expect(msg).not.toContain('per-zone calibrated TB probabilities');
    expect(msg).not.toContain('top-5 TorchXRayVision findings');
  });
});

describe('GPT_INTERPRETER constants', () => {
  it('prompt hash is a deterministic 8-hex fingerprint', () => {
    expect(GPT_INTERPRETER_PROMPT_HASH).toMatch(/^[0-9a-f]{8}$/);
    expect(GPT_INTERPRETER_SCHEMA_HASH).toMatch(/^[0-9a-f]{8}$/);
  });

  it('the prompt forbids re-diagnosis and contradiction', () => {
    expect(GPT_INTERPRETER_PROMPT).toMatch(/not diagnosing the image/i);
    expect(GPT_INTERPRETER_PROMPT).toMatch(/do not contradict the pipeline verdict/i);
    // Anti-anchoring: no LODO numbers in the prompt body
    expect(GPT_INTERPRETER_PROMPT).not.toMatch(/0\.8/); // no specific sensitivity figure
    expect(GPT_INTERPRETER_PROMPT).not.toMatch(/0\.91[12]/); // no specific specificity figure
  });

  it('schema version is the pinned constant we audit against', () => {
    expect(GPT_INTERPRETER_SCHEMA_VERSION).toBe('gpt-interpreter-v1');
  });
});

describe('gptInterpreter — happy path', () => {
  it('returns a parsed ClinicianReport + audit pins from a well-formed response', async () => {
    const wellFormed: ClinicianReport = {
      finding: 'Patchy upper-right consolidation; tb_prob 0.9999 above the 0.6105 threshold.',
      key_regions: ['upper_r', 'upper_l'],
      confidence_qualifier: 'high',
      top_differential_alternatives: [
        { label: 'pneumonia', likelihood: 'less_likely', rationale: 'Pneumonia 0.25 below the dominant Lung Opacity 0.52' },
        { label: 'atelectasis', likelihood: 'less_likely', rationale: 'Atelectasis 0.10 — not flagged by the pipeline' },
      ],
      refusal_or_limitation: null,
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(mockResponsesEnvelope(wellFormed)), { status: 200, headers: { 'Content-Type': 'application/json' } }),
      ),
    );
    const r = await gptInterpreter({
      apiKey: 'sk-test',
      primaryModel: 'gpt-5.5',
      fallbackModel: 'gpt-5.5-instant',
      imageDataUrl: 'data:image/png;base64,iVBORw0K',
      localResult: mockLocal(),
    });
    expect(r.report.finding).toBe(wellFormed.finding);
    expect(r.report.key_regions).toEqual(['upper_r', 'upper_l']);
    expect(r.report.confidence_qualifier).toBe('high');
    expect(r.report.top_differential_alternatives).toHaveLength(2);
    expect(r.audit.prompt_hash).toBe(GPT_INTERPRETER_PROMPT_HASH);
    expect(r.audit.schema_version).toBe(GPT_INTERPRETER_SCHEMA_VERSION);
    expect(r.audit.model_id_from_response).toBe('gpt-5.5-2026-04-23');
  });
});

describe('gptInterpreter — defensive normalization', () => {
  it('filters key_regions to the subset of zones we passed in (model cannot invent zone keys)', async () => {
    const drifted: ClinicianReport = {
      finding: 'note',
      // 'made_up_zone' is NOT in zonal_scores; must be dropped.
      key_regions: ['upper_r', 'made_up_zone', 'hilar'],
      confidence_qualifier: 'moderate',
      top_differential_alternatives: [],
      refusal_or_limitation: null,
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(mockResponsesEnvelope(drifted)), { status: 200, headers: { 'Content-Type': 'application/json' } }),
      ),
    );
    const r = await gptInterpreter({
      apiKey: 'sk-test',
      primaryModel: 'gpt-5.5',
      fallbackModel: 'gpt-5.5-instant',
      imageDataUrl: 'data:image/png;base64,iVBORw0K',
      localResult: mockLocal(),
    });
    expect(r.report.key_regions).toEqual(['upper_r', 'hilar']);
  });

  it('confidence_qualifier falls back to "low" on an unknown enum value (safe default)', async () => {
    const drifted = {
      finding: 'x',
      key_regions: [],
      confidence_qualifier: 'extremely_high', // not in enum
      top_differential_alternatives: [],
      refusal_or_limitation: null,
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(mockResponsesEnvelope(drifted as unknown as ClinicianReport)), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
    const r = await gptInterpreter({
      apiKey: 'sk-test',
      primaryModel: 'gpt-5.5',
      fallbackModel: 'gpt-5.5-instant',
      imageDataUrl: 'data:image/png;base64,iVBORw0K',
      localResult: mockLocal(),
    });
    expect(r.report.confidence_qualifier).toBe('low');
  });

  it('caps top_differential_alternatives at 3 entries', async () => {
    const overFull: ClinicianReport = {
      finding: 'x',
      key_regions: [],
      confidence_qualifier: 'moderate',
      top_differential_alternatives: [
        { label: 'a', likelihood: 'consider', rationale: 'r' },
        { label: 'b', likelihood: 'less_likely', rationale: 'r' },
        { label: 'c', likelihood: 'consider', rationale: 'r' },
        { label: 'd', likelihood: 'consider', rationale: 'r' },
        { label: 'e', likelihood: 'consider', rationale: 'r' },
      ],
      refusal_or_limitation: null,
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(mockResponsesEnvelope(overFull)), { status: 200, headers: { 'Content-Type': 'application/json' } }),
      ),
    );
    const r = await gptInterpreter({
      apiKey: 'sk-test',
      primaryModel: 'gpt-5.5',
      fallbackModel: 'gpt-5.5-instant',
      imageDataUrl: 'data:image/png;base64,iVBORw0K',
      localResult: mockLocal(),
    });
    expect(r.report.top_differential_alternatives.map((d) => d.label)).toEqual(['a', 'b', 'c']);
  });
});

describe('gptInterpreter — error paths', () => {
  it('throws OpenAIError when the API key is missing', async () => {
    await expect(
      gptInterpreter({
        apiKey: '',
        primaryModel: 'gpt-5.5',
        fallbackModel: 'gpt-5.5-instant',
        imageDataUrl: 'data:image/png;base64,iVBORw0K',
        localResult: mockLocal(),
      }),
    ).rejects.toThrow(/API key missing/i);
  });

  it('throws when the model returns a non-object payload', async () => {
    // Envelope shape is valid, but `output_text` is a JSON string literal, not an object.
    const envelope = {
      model: 'gpt-5.5',
      output: [{ type: 'message', content: [{ type: 'output_text', text: '"just a bare string"' }] }],
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(envelope), { status: 200, headers: { 'Content-Type': 'application/json' } }),
      ),
    );
    // extractJSON will fail to find '{', which throws OpenAIError before we ever see the data.
    await expect(
      gptInterpreter({
        apiKey: 'sk-test',
        primaryModel: 'gpt-5.5',
        fallbackModel: 'gpt-5.5-instant',
        imageDataUrl: 'data:image/png;base64,iVBORw0K',
        localResult: mockLocal(),
      }),
    ).rejects.toThrow();
  });
});
