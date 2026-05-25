/**
 * BoxEvidenceHeatmap — M24 component test.
 *
 * Mirrors the project's existing component-test pattern (renderToStaticMarkup
 * in the node env vitest already runs in — no jsdom dep). Asserts the
 * structural invariants the UI carries: 64 cells, the always-on caption with
 * the load-bearing honesty framing, and degradation on a malformed grid.
 */
import { describe, it, expect } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { BoxEvidenceHeatmap } from './BoxEvidenceHeatmap';

function makeGrid(fill: number): number[][] {
  return Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => fill));
}

describe('BoxEvidenceHeatmap', () => {
  it('renders 64 cells with the M24 honesty caption', () => {
    const html = renderToStaticMarkup(<BoxEvidenceHeatmap grid={makeGrid(0.5)} />);
    // 64 cells (8x8). data-testid="box-cell-r-c" patterns
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        expect(html).toContain(`data-testid="box-cell-${r}-${c}"`);
      }
    }
    expect(html).toContain('data-testid="box-evidence-caption"');
    expect(html).toContain('NOT a radiologist annotation');
  });

  it('renders the source image base layer when imageUrl is provided', () => {
    const html = renderToStaticMarkup(
      <BoxEvidenceHeatmap grid={makeGrid(0.1)} imageUrl="data:image/png;base64,xxx" />,
    );
    expect(html).toContain('data-testid="box-evidence-base"');
    expect(html).toContain('src="data:image/png;base64,xxx"');
  });

  it('omits the base layer when imageUrl is absent (orchestrator does not always thread it)', () => {
    const html = renderToStaticMarkup(<BoxEvidenceHeatmap grid={makeGrid(0.1)} />);
    expect(html).not.toContain('data-testid="box-evidence-base"');
  });

  it('degrades silently to an empty grid on a malformed input (defensive)', () => {
    // 7x8 — a real regression we want to render-not-crash.
    const bad = Array.from({ length: 7 }, () => Array(8).fill(0)) as unknown as ReadonlyArray<
      ReadonlyArray<number>
    >;
    const html = renderToStaticMarkup(<BoxEvidenceHeatmap grid={bad} />);
    // The defensive fallback STILL emits 64 cells (all zero).
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        expect(html).toContain(`data-testid="box-cell-${r}-${c}"`);
      }
    }
  });

  it('honors a custom caption override', () => {
    const html = renderToStaticMarkup(
      <BoxEvidenceHeatmap grid={makeGrid(0.2)} caption="custom caption" />,
    );
    expect(html).toContain('custom caption');
    expect(html).not.toContain('NOT a radiologist annotation');
  });
});
