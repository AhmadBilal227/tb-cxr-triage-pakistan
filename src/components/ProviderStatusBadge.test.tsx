/**
 * ProviderStatusBadge per-provider status display test.
 *
 * The badge translates the in-memory provider-status state into a calm,
 * actionable line in the Settings drawer. This is the diagnostic the
 * production user was missing on the 2026-05-24 Netlify failure: they saw
 * 400s in the browser console but had no way to see WHICH key was the
 * problem from inside the app.
 *
 * Milestone 23 removed Hugging Face — the previous HF test rows were
 * rewritten to assert the same diagnostic behavior on the Replicate path
 * (which inherits the same `classifyHttpFailure` dispatcher).
 *
 * Renders via react-dom/server against the standalone ProviderStatusBadge
 * module — no Radix Dialog tree, no localStorage shim required.
 */
import { describe, it, expect } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { ProviderStatusBadge } from './ProviderStatusBadge';

function render(props: Parameters<typeof ProviderStatusBadge>[0]): string {
  return renderToStaticMarkup(<ProviderStatusBadge {...props} />);
}

describe('ProviderStatusBadge', () => {
  it('renders nothing when state=unknown and no key is set', () => {
    const html = render({
      provider: 'replicate',
      hasKey: false,
      status: { state: 'unknown', at: null },
    });
    expect(html).toBe('');
  });

  it('shows "configured · not yet called" when state=unknown and a key is set', () => {
    const html = render({
      provider: 'replicate',
      hasKey: true,
      status: { state: 'unknown', at: null },
    });
    expect(html).toContain('data-testid="provider-status-replicate"');
    expect(html).toContain('configured');
    expect(html).toContain('not yet called');
  });

  it('shows the not-configured tag when the provider has no key', () => {
    const html = render({
      provider: 'replicate',
      hasKey: false,
      status: { state: 'not-configured', note: 'no token', at: Date.now() },
    });
    expect(html).toContain('data-testid="provider-status-replicate"');
    expect(html).toContain('not configured');
  });

  it('shows the ok tag and surfaces the note on success', () => {
    const html = render({
      provider: 'openai',
      hasKey: true,
      status: { state: 'ok', note: 'model gpt-5.5', at: Date.now() },
    });
    expect(html).toContain('data-testid="provider-status-openai"');
    expect(html).toContain('openai: ok');
    expect(html).toContain('model gpt-5.5');
  });

  it('surfaces a 401 unauthorized failure with the human reason', () => {
    const html = render({
      provider: 'replicate',
      hasKey: true,
      status: {
        state: 'unauthorized',
        note: '401 unauthorized — token missing or invalid',
        at: Date.now(),
      },
    });
    expect(html).toContain('data-testid="provider-status-replicate"');
    expect(html).toContain('last call unauthorized');
    expect(html).toContain('401');
    expect(html).toContain('token missing or invalid');
  });

  it('surfaces a model-unsupported failure (retired Replicate version, etc.)', () => {
    const html = render({
      provider: 'replicate',
      hasKey: true,
      status: {
        state: 'model-unsupported',
        note: '400 model not available on this provider — update the model id in Settings',
        at: Date.now(),
      },
    });
    expect(html).toContain('last call model-unsupported');
    expect(html).toContain('model not available on this provider');
    expect(html).toContain('update the model id in Settings');
  });

  it('surfaces a network/CORS failure', () => {
    const html = render({
      provider: 'replicate',
      hasKey: true,
      status: { state: 'network', note: 'network: Failed to fetch', at: Date.now() },
    });
    expect(html).toContain('last call network');
    expect(html).toContain('Failed to fetch');
  });
});
