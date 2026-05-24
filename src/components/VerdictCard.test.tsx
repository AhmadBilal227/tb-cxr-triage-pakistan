/**
 * VerdictCard always-on disclosure test (Milestone 19).
 *
 * The card must surface the scar-shape-FPR footnote on EVERY verdict — not just NO_TB —
 * because the NIH stress test (EXPERIMENT_LOG §C "NIH stress test") established a same-
 * provenance scar disaster (TBX11K scar FPR 0.842) that does NOT show up on cross-site
 * NIH Fibrosis (FPR 0.102). Telling the user about the failure mode is the ethos.
 *
 * Renders via react-dom/server to keep the test in the existing node env (no jsdom dep).
 * That's enough to assert the disclosure is in the static markup; interaction tests live
 * elsewhere.
 */
import { describe, it, expect } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { VerdictCard } from './VerdictCard';
import type { Adjudication, EnsembleResult, RagResult, Verdict } from '@/lib/types';

function mockAdjudication(verdict: Verdict): Adjudication {
  return {
    verdict,
    confidence: 72,
    rationale: 'unit-test rationale',
    auto_abstained: false,
    auto_abstain_reasons: [],
  };
}

function mockEnsemble(): EnsembleResult {
  return {
    members: [],
    weightedScore: 0.42,
    std: 0.05,
    disagreement: 0.1,
    replicateFallbackCount: 0,
  };
}

function mockRag(): RagResult {
  return { neighbors: [], embedding_provider: null, skipped: true };
}

const DISCLOSURE_FRAGMENT = 'Higher false-positive rate (~10%) expected on radiographically scar-shaped findings';

describe('VerdictCard scar-FPR disclosure', () => {
  for (const verdict of ['tb', 'no_tb', 'abstain'] as const) {
    it(`renders the scar-shape FPR disclosure when verdict=${verdict}`, () => {
      const html = renderToStaticMarkup(
        <VerdictCard
          adjudication={mockAdjudication(verdict)}
          ensemble={mockEnsemble()}
          rag={mockRag()}
          fallbackRate={0}
          onDisagree={async () => undefined}
        />,
      );
      expect(html).toContain(DISCLOSURE_FRAGMENT);
      // tagged for non-marketing tone — no exclamation marks, no "guarantee" or "comprehensive"
      expect(html).not.toMatch(/guarantee|comprehensive|robust/i);
    });
  }

  it('includes the test id for downstream a11y/visual hooks', () => {
    const html = renderToStaticMarkup(
      <VerdictCard
        adjudication={mockAdjudication('no_tb')}
        ensemble={mockEnsemble()}
        rag={mockRag()}
        fallbackRate={0}
        onDisagree={async () => undefined}
      />,
    );
    expect(html).toContain('data-testid="scar-fpr-disclosure"');
  });
});
