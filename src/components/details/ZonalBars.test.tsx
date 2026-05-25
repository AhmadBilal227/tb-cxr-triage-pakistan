/**
 * ZonalBars — M24 component test.
 *
 * Pins the load-bearing properties: sorted-desc order, all 7 honest zone keys
 * rendered when supplied, "evidence not available" copy when empty, and the
 * per-zone two-decimal probability rendering.
 */
import { describe, it, expect } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { ZonalBars } from './ZonalBars';
import type { ZonalScores } from '@/lib/providers/localTriage';

const FULL: ZonalScores = {
  upper_r: 0.406,
  mid_r: 0.027,
  lower_r: 0.005,
  upper_l: 0.163,
  mid_l: 0.038,
  lower_l: 0.0006,
  hilar: 0.037,
};

describe('ZonalBars', () => {
  it('renders all 7 supplied zone bars with two-decimal probabilities', () => {
    const html = renderToStaticMarkup(<ZonalBars scores={FULL} />);
    for (const key of [
      'upper_r', 'mid_r', 'lower_r', 'upper_l', 'mid_l', 'lower_l', 'hilar',
    ] as const) {
      expect(html).toContain(`data-testid="zonal-bar-${key}"`);
      expect(html).toContain(`data-testid="zonal-value-${key}"`);
    }
    // Headline value uses 0.41 (2-dp)
    expect(html).toContain('>0.41<');
    expect(html).toContain('>0.16<');
    expect(html).toContain('>0.04<');
  });

  it('sorts descending — the leading bar (upper_r) renders before the trailing (lower_l)', () => {
    const html = renderToStaticMarkup(<ZonalBars scores={FULL} />);
    const idxLead = html.indexOf('data-testid="zonal-bar-upper_r"');
    const idxTail = html.indexOf('data-testid="zonal-bar-lower_l"');
    expect(idxLead).toBeGreaterThan(0);
    expect(idxTail).toBeGreaterThan(idxLead);
  });

  it('renders the per-zone label translations (clinical-register names)', () => {
    const html = renderToStaticMarkup(<ZonalBars scores={FULL} />);
    expect(html).toContain('Upper R');
    expect(html).toContain('Hilar / Mediastinum');
  });

  it('renders the empty-state copy when no zones are supplied', () => {
    const html = renderToStaticMarkup(<ZonalBars scores={{}} />);
    expect(html).toContain('Zone evidence not available');
  });

  it('drops zones with non-finite values gracefully', () => {
    const partial: ZonalScores = {
      upper_r: 0.4,
      hilar: Number.NaN as unknown as number,  // simulated drift in the wire
    };
    const html = renderToStaticMarkup(<ZonalBars scores={partial} />);
    expect(html).toContain('data-testid="zonal-bar-upper_r"');
    expect(html).not.toContain('data-testid="zonal-bar-hilar"');
  });
});
