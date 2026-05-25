/**
 * gptInterpreter provider tests (Milestone 24, v2 radiologist schema).
 *
 * The provider's load-bearing properties:
 *   1. It calls the Responses API with the gpt-interpreter prompt + strict
 *      JSON schema; the evidence message is built from the local result.
 *   2. Validator normalizes/clamps the response: enums fall back to safe
 *      defaults; key_regions is filtered to the SUBSET of zone keys we
 *      passed in (so the model cannot invent zone names); the two
 *      mandatory limitation lines are injected if the model omitted them.
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

function wellFormedReport(over: Partial<ClinicianReport> = {}): ClinicianReport {
  return {
    technique: 'Single frontal chest radiograph; AI-assisted TB triage pipeline.',
    comparison: 'No prior studies available for comparison.',
    findings: {
      lungs_and_airways:
        'Patchy parenchymal opacity in the right upper lobe with suggestion of cavitation; distribution is consistent with apical / posterior upper-lobe TB pattern.',
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
    ],
    recommendation:
      'Recommend sputum AFB smear, culture, and NAAT; clinical evaluation for TB risk factors and symptomatic assessment.',
    key_regions: ['upper_r', 'upper_l'],
    limitations: [
      'Single-view frontal radiograph; lateral and CT may yield additional information.',
      'Radiographic features are not diagnostic of microbiological status.',
    ],
    ...over,
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

  it('the prompt forbids re-diagnosis and contradiction (v2 radiologist tone)', () => {
    // v2 phrasing: "You do NOT diagnose, disagree, or contradict the pipeline."
    expect(GPT_INTERPRETER_PROMPT).toMatch(/do not diagnose/i);
    expect(GPT_INTERPRETER_PROMPT).toMatch(/disagree.*contradict/i);
    // Anti-anchoring: no LODO performance numbers in the prompt body. The
    // prompt may reference the word AUROC inside a "do NOT mention" negative
    // constraint, but must never anchor a specific value the model could echo.
    expect(GPT_INTERPRETER_PROMPT).not.toMatch(/0\.8(00|0)?\b/); // no specific sensitivity figure
    expect(GPT_INTERPRETER_PROMPT).not.toMatch(/0\.91[12]/); // no specific specificity figure
    expect(GPT_INTERPRETER_PROMPT).not.toMatch(/auroc[^.,]*0\./i); // no "AUROC of 0.92x"
  });

  it('schema version is the pinned constant we audit against (v2)', () => {
    expect(GPT_INTERPRETER_SCHEMA_VERSION).toBe('gpt-interpreter-v2');
  });
});

describe('gptInterpreter — happy path', () => {
  it('returns a parsed ClinicianReport + audit pins from a well-formed response', async () => {
    const r = await runInterpreter(wellFormedReport());
    expect(r.report.technique).toMatch(/frontal chest radiograph/i);
    expect(r.report.findings.lungs_and_airways).toMatch(/right upper lobe/i);
    expect(r.report.impression).toHaveLength(2);
    expect(r.report.impression[0]?.likelihood).toBe('primary');
    expect(r.report.impression[1]?.likelihood).toBe('consider');
    expect(r.report.recommendation).toMatch(/sputum/i);
    expect(r.report.key_regions).toEqual(['upper_r', 'upper_l']);
    expect(r.audit.prompt_hash).toBe(GPT_INTERPRETER_PROMPT_HASH);
    expect(r.audit.schema_version).toBe(GPT_INTERPRETER_SCHEMA_VERSION);
    expect(r.audit.model_id_from_response).toBe('gpt-5.5-2026-04-23');
  });
});

describe('gptInterpreter — defensive normalization', () => {
  it('filters key_regions to the subset of zones we passed in (model cannot invent zone keys)', async () => {
    const drifted = wellFormedReport({
      // 'made_up_zone' is NOT in zonal_scores; must be dropped.
      key_regions: ['upper_r', 'made_up_zone', 'hilar'],
    });
    const r = await runInterpreter(drifted);
    expect(r.report.key_regions).toEqual(['upper_r', 'hilar']);
  });

  it('impression likelihood falls back to "less_likely" on an unknown enum value (safe default)', async () => {
    const drifted: ClinicianReport = wellFormedReport({
      impression: [
        { statement: 'first', likelihood: 'extremely_high' as unknown as 'primary' },
      ],
    });
    const r = await runInterpreter(drifted);
    expect(r.report.impression).toHaveLength(1);
    expect(r.report.impression[0]?.likelihood).toBe('less_likely');
  });

  it('caps impression at 4 entries', async () => {
    const overFull = wellFormedReport({
      impression: [
        { statement: 'a', likelihood: 'primary' },
        { statement: 'b', likelihood: 'consider' },
        { statement: 'c', likelihood: 'less_likely' },
        { statement: 'd', likelihood: 'less_likely' },
        { statement: 'e', likelihood: 'less_likely' },
        { statement: 'f', likelihood: 'less_likely' },
      ],
    });
    const r = await runInterpreter(overFull);
    expect(r.report.impression.map((d) => d.statement)).toEqual(['a', 'b', 'c', 'd']);
  });

  it('always injects the two mandatory limitation caveats even when the model omits them', async () => {
    const stripped = wellFormedReport({ limitations: [] });
    const r = await runInterpreter(stripped);
    expect(r.report.limitations.some((l) => /single-view frontal radiograph/i.test(l))).toBe(true);
    expect(
      r.report.limitations.some((l) =>
        /radiographic features are not diagnostic of microbiological status/i.test(l),
      ),
    ).toBe(true);
  });

  it('defaults missing findings sub-sections to neutral radiology-report phrasing', async () => {
    // Strict-mode schema would normally enforce findings shape, but the validator
    // also defends against a model that emits a placeholder with no content.
    const stripped = wellFormedReport({
      findings: {
        lungs_and_airways: '',
        pleura: '',
        cardiomediastinum: '',
        bones_and_soft_tissues: '',
      },
    });
    const r = await runInterpreter(stripped);
    expect(r.report.findings.pleura).toMatch(/pleural spaces are clear/i);
    expect(r.report.findings.cardiomediastinum).toMatch(/cardiomediastinal silhouette is unremarkable/i);
    expect(r.report.findings.bones_and_soft_tissues).toMatch(/no osseous abnormality/i);
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

// ---------------------------------------------------------------------------
// Shared test helper: stubs fetch with the given report payload + invokes
// gptInterpreter with the standard test params.
// ---------------------------------------------------------------------------
async function runInterpreter(report: ClinicianReport): Promise<Awaited<ReturnType<typeof gptInterpreter>>> {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      new Response(JSON.stringify(mockResponsesEnvelope(report)), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    ),
  );
  return gptInterpreter({
    apiKey: 'sk-test',
    primaryModel: 'gpt-5.5',
    fallbackModel: 'gpt-5.5-instant',
    imageDataUrl: 'data:image/png;base64,iVBORw0K',
    localResult: mockLocal(),
  });
}
