/**
 * NoKeysBanner BYOK-contract test (Milestone 20).
 *
 * The banner must appear when no provider key is set (the failure mode that
 * caused the silent 400s on the production deploy at tb-triage-research.netlify.app
 * on 2026-05-24) and must disappear the moment ANY key is configured.
 *
 * Renders via react-dom/server.renderToStaticMarkup against the pure
 * presentational `NoKeysBannerView` so we don't need to shim localStorage
 * inside the node env that vitest runs in.
 */
import { describe, it, expect } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { NoKeysBannerView } from './NoKeysBanner';

describe('NoKeysBanner', () => {
  it('renders the BYOK prompt when no keys are set (hasAny=false)', () => {
    const html = renderToStaticMarkup(
      <NoKeysBannerView hasAny={false} onOpenSettings={() => undefined} />,
    );
    expect(html).toContain('data-testid="no-keys-banner"');
    expect(html).toContain('Add at least one API key in');
    expect(html).toContain('BYOK');
    expect(html).toContain('Open Settings');
    // No marketing emoji or exclamation marks — ethos check.
    expect(html).not.toMatch(/[!]/);
  });

  it('renders nothing when at least one key is set (hasAny=true)', () => {
    const html = renderToStaticMarkup(
      <NoKeysBannerView hasAny={true} onOpenSettings={() => undefined} />,
    );
    expect(html).toBe('');
  });
});
