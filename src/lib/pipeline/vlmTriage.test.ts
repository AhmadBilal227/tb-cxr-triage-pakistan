/**
 * Unit tests for the VLM PRIMARY perception module (Milestone 21, Task A + B).
 *
 * Covers the pure pieces that don't need the network:
 *   - structured-output payload validation + forced-abstain rails
 *     (image_quality !== 'diagnostic' / projection unknown + positive / low band)
 *   - borderline-band predicate that decides when to fire the consistency check
 *   - prompt + schema hashes are stable across imports (audit trail contract)
 *
 * The actual `vlmTriage()` HTTP call is covered indirectly by the orchestrator
 * tests via the `openaiJSON` seam; here we lock down the schema-contract surface.
 */
import { describe, it, expect } from 'vitest';
import {
  VLM_BORDERLINE_LOW,
  VLM_BORDERLINE_HIGH,
  VLM_PROMPT_HASH_PRIMARY,
  VLM_PROMPT_HASH_VERIFIER,
  VLM_SCHEMA_HASH,
  VLM_TRIAGE_SCHEMA,
  VLM_TRIAGE_SCHEMA_VERSION,
  hasScarMimicFlag,
  isBorderlineForConsistencyCheck,
  validateAndNormalizeSubmission,
  type TriageSubmission,
} from './vlmTriage';

function basePayload(over: Partial<Record<string, unknown>> = {}): Record<string, unknown> {
  return {
    image_quality: 'diagnostic',
    projection: 'pa_ap',
    tb_screen_result: 'screen_negative',
    tb_score_uncalibrated: 0.1,
    confidence_band: 'high',
    scar_shape_score_uncalibrated: 0.1,
    mimic_features_present: [],
    abnormality_localization: [],
    safety_flags: [],
    short_rationale: 'normal lung fields, no concerning pattern visible.',
    refusal_or_limitation: null,
    model_version_seen_by_client: 'gpt-5.5-2026-04-23',
    ...over,
  };
}

describe('validateAndNormalizeSubmission — happy path', () => {
  it('passes through a well-formed diagnostic-quality screen_negative', () => {
    const s = validateAndNormalizeSubmission(basePayload(), 'gpt-5.5-2026-04-23');
    expect(s.tb_screen_result).toBe('screen_negative');
    expect(s.image_quality).toBe('diagnostic');
    expect(s.tb_score_uncalibrated).toBe(0.1);
    expect(s.short_rationale).toContain('normal lung fields');
  });

  it('passes through a well-formed screen_positive without modifying it', () => {
    const s = validateAndNormalizeSubmission(
      basePayload({
        tb_screen_result: 'screen_positive',
        tb_score_uncalibrated: 0.82,
        confidence_band: 'high',
        abnormality_localization: ['right upper lobe cavity'],
      }),
      'gpt-5.5-2026-04-23',
    );
    expect(s.tb_screen_result).toBe('screen_positive');
    expect(s.tb_score_uncalibrated).toBe(0.82);
  });

  it('clamps tb_score_uncalibrated out-of-range values into [0,1]', () => {
    const high = validateAndNormalizeSubmission(
      basePayload({ tb_score_uncalibrated: 1.7 }),
      'gpt-5.5',
    );
    const low = validateAndNormalizeSubmission(
      basePayload({ tb_score_uncalibrated: -0.4 }),
      'gpt-5.5',
    );
    const bad = validateAndNormalizeSubmission(
      basePayload({ tb_score_uncalibrated: 'oops' }),
      'gpt-5.5',
    );
    expect(high.tb_score_uncalibrated).toBe(1);
    expect(low.tb_score_uncalibrated).toBe(0);
    expect(bad.tb_score_uncalibrated).toBe(0);
  });
});

describe('validateAndNormalizeSubmission — forced-abstain rails', () => {
  it('forces ABSTAIN when image_quality is limited (even on a positive read)', () => {
    const s = validateAndNormalizeSubmission(
      basePayload({
        image_quality: 'limited',
        tb_screen_result: 'screen_positive',
        tb_score_uncalibrated: 0.8,
      }),
      'gpt-5.5',
    );
    expect(s.tb_screen_result).toBe('abstain');
    expect(s.short_rationale).toContain('forced abstain');
    expect(s.short_rationale).toContain('image_quality=limited');
  });

  it('forces ABSTAIN when projection is unknown AND the model said screen_positive', () => {
    const s = validateAndNormalizeSubmission(
      basePayload({
        projection: 'unknown',
        tb_screen_result: 'screen_positive',
        tb_score_uncalibrated: 0.6,
      }),
      'gpt-5.5',
    );
    expect(s.tb_screen_result).toBe('abstain');
    expect(s.short_rationale).toContain('projection=unknown');
  });

  it('projection=unknown with high-confidence NEGATIVE passes through (rail only fires on positive)', () => {
    // Rails only force on (proj=unknown AND positive) OR (band=low) OR (quality≠diagnostic).
    // proj=unknown alone with a negative read isn't a safety violation; forcing abstain there
    // would just over-block. Pin the documented behavior so a future "tighten the rail" PR
    // gets flagged on the test rather than silently raising the abstain rate.
    const s = validateAndNormalizeSubmission(
      basePayload({
        projection: 'unknown',
        tb_screen_result: 'screen_negative',
        confidence_band: 'high',
      }),
      'gpt-5.5',
    );
    expect(s.tb_screen_result).toBe('screen_negative');
  });

  it('forces ABSTAIN when confidence_band is low (regardless of screen_result)', () => {
    const s = validateAndNormalizeSubmission(
      basePayload({ confidence_band: 'low', tb_screen_result: 'screen_negative' }),
      'gpt-5.5',
    );
    expect(s.tb_screen_result).toBe('abstain');
    expect(s.short_rationale).toContain('confidence_band=low');
  });

  it('preserves an explicit ABSTAIN without prepending the forced-abstain note', () => {
    const s = validateAndNormalizeSubmission(
      basePayload({
        tb_screen_result: 'abstain',
        confidence_band: 'low',
        short_rationale: 'image rotated, cannot adjudicate.',
      }),
      'gpt-5.5',
    );
    expect(s.tb_screen_result).toBe('abstain');
    expect(s.short_rationale).toBe('image rotated, cannot adjudicate.');
  });
});

describe('validateAndNormalizeSubmission — slop-tolerance', () => {
  it('falls back to safe defaults for unknown enum values', () => {
    const s = validateAndNormalizeSubmission(
      {
        image_quality: 'great', // not in enum
        projection: 'oblique', // not in enum
        tb_screen_result: 'maybe', // not in enum
        confidence_band: 'medium-high', // not in enum
      },
      'gpt-5.5',
    );
    expect(s.image_quality).toBe('limited');
    expect(s.projection).toBe('unknown');
    expect(s.tb_screen_result).toBe('abstain'); // safest default
    expect(s.confidence_band).toBe('low');
  });

  it('non-string-array fields default to []', () => {
    const s = validateAndNormalizeSubmission(
      basePayload({ mimic_features_present: 'fibrosis', safety_flags: null }),
      'gpt-5.5',
    );
    expect(s.mimic_features_present).toEqual([]);
    expect(s.safety_flags).toEqual([]);
  });

  it('records the response-envelope model_id when the model omits the echo field', () => {
    const s = validateAndNormalizeSubmission(
      basePayload({ model_version_seen_by_client: undefined }),
      'gpt-5.5-2026-09-01',
    );
    expect(s.model_version_seen_by_client).toBe('gpt-5.5-2026-09-01');
  });
});

describe('isBorderlineForConsistencyCheck', () => {
  function s(over: Partial<TriageSubmission>): TriageSubmission {
    return {
      image_quality: 'diagnostic',
      projection: 'pa_ap',
      tb_screen_result: 'screen_negative',
      tb_score_uncalibrated: 0.05,
      confidence_band: 'high',
      scar_shape_score_uncalibrated: 0,
      mimic_features_present: [],
      abnormality_localization: [],
      safety_flags: [],
      short_rationale: '',
      refusal_or_limitation: null,
      model_version_seen_by_client: 'gpt-5.5',
      ...over,
    };
  }

  it('confident screen_negative: not borderline', () => {
    expect(isBorderlineForConsistencyCheck(s({ tb_score_uncalibrated: 0.08 }))).toBe(false);
  });

  it('confident screen_positive: not borderline', () => {
    expect(isBorderlineForConsistencyCheck(s({ tb_score_uncalibrated: 0.92 }))).toBe(false);
  });

  it('mid-band: borderline triggers verifier', () => {
    expect(isBorderlineForConsistencyCheck(s({ tb_score_uncalibrated: 0.5 }))).toBe(true);
  });

  it('low confidence band: borderline regardless of score', () => {
    expect(
      isBorderlineForConsistencyCheck(s({ tb_score_uncalibrated: 0.05, confidence_band: 'low' })),
    ).toBe(true);
  });

  it('mimic-feature scar tag: borderline regardless of score', () => {
    expect(
      isBorderlineForConsistencyCheck(
        s({ tb_score_uncalibrated: 0.05, mimic_features_present: ['apical_fibrosis'] }),
      ),
    ).toBe(true);
  });

  it('band edge values respect the documented bounds', () => {
    expect(VLM_BORDERLINE_LOW).toBe(0.35);
    expect(VLM_BORDERLINE_HIGH).toBe(0.65);
    expect(isBorderlineForConsistencyCheck(s({ tb_score_uncalibrated: VLM_BORDERLINE_LOW }))).toBe(true);
    expect(isBorderlineForConsistencyCheck(s({ tb_score_uncalibrated: VLM_BORDERLINE_HIGH }))).toBe(true);
  });
});

describe('hasScarMimicFlag', () => {
  it('matches the documented scar tags (case-insensitive)', () => {
    expect(hasScarMimicFlag(['Fibrosis'])).toBe(true);
    expect(hasScarMimicFlag(['Pleural Thickening'])).toBe(true);
    expect(hasScarMimicFlag(['HEALED_SCAR'])).toBe(true);
    expect(hasScarMimicFlag(['apical_fibrosis'])).toBe(true);
    expect(hasScarMimicFlag(['scar tissue noted'])).toBe(true);
  });

  it('does not match non-scar findings', () => {
    expect(hasScarMimicFlag(['consolidation'])).toBe(false);
    expect(hasScarMimicFlag(['cavity'])).toBe(false);
    expect(hasScarMimicFlag([])).toBe(false);
  });
});

describe('audit trail constants', () => {
  it('schema name + version are stable identifiers', () => {
    expect(VLM_TRIAGE_SCHEMA.name).toBe('submit_triage');
    expect(VLM_TRIAGE_SCHEMA_VERSION).toBe('vlm-triage-v1');
  });

  it('primary and verifier prompt hashes are distinct (the two prompts differ on purpose)', () => {
    expect(VLM_PROMPT_HASH_PRIMARY).not.toBe(VLM_PROMPT_HASH_VERIFIER);
    expect(VLM_PROMPT_HASH_PRIMARY).toMatch(/^[0-9a-f]{8}$/);
    expect(VLM_PROMPT_HASH_VERIFIER).toMatch(/^[0-9a-f]{8}$/);
  });

  it('schema hash is a deterministic 8-hex string', () => {
    expect(VLM_SCHEMA_HASH).toMatch(/^[0-9a-f]{8}$/);
  });
});
