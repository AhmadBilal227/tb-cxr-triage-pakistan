/**
 * classifyHttpFailure unit test (Milestone 20).
 *
 * The HF Inference router signals retired models with HTTP 400 + a body whose
 * .error is "Model not supported by provider hf-inference". This is a CONFIG
 * problem (user must update the model id), not an AUTH problem, so it gets
 * its own stable tag — pinned here so a future refactor doesn't collapse it
 * back into "other-error" and re-silence the production failure mode.
 */
import { describe, it, expect } from 'vitest';
import { classifyHttpFailure } from './providerStatus';

describe('classifyHttpFailure', () => {
  it('tags 401 as unauthorized', () => {
    const r = classifyHttpFailure(401, 'Unauthorized — Please check your credentials');
    expect(r.state).toBe('unauthorized');
    expect(r.humanReason).toContain('401');
  });

  it('tags 403 as unauthorized', () => {
    const r = classifyHttpFailure(403, 'Forbidden');
    expect(r.state).toBe('unauthorized');
  });

  it('tags 429 as rate-limited', () => {
    const r = classifyHttpFailure(429, 'Too many requests');
    expect(r.state).toBe('rate-limited');
  });

  it('tags the hf-inference 400 retirement message as model-unsupported', () => {
    const r = classifyHttpFailure(400, '{"error":"Model not supported by provider hf-inference"}');
    expect(r.state).toBe('model-unsupported');
    expect(r.humanReason).toContain('update the model id');
  });

  it('tags 410 deprecation as model-unsupported', () => {
    const r = classifyHttpFailure(
      410,
      '{"error":"The requested model is deprecated and no longer supported by provider hf-inference"}',
    );
    expect(r.state).toBe('model-unsupported');
  });

  it('tags 404 "Model not found" as model-unsupported', () => {
    const r = classifyHttpFailure(404, '{"error":"Model not found"}');
    expect(r.state).toBe('model-unsupported');
  });

  it('does NOT tag a 400 unrelated to retirement as model-unsupported', () => {
    const r = classifyHttpFailure(400, 'Bad request: missing image field');
    expect(r.state).toBe('other-error');
  });

  it('falls back to other-error for unknown 5xx', () => {
    const r = classifyHttpFailure(500, 'Internal server error');
    expect(r.state).toBe('other-error');
    expect(r.humanReason).toContain('500');
  });
});
