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

/**
 * Perception-unavailable state (Milestone 20).
 *
 * When ALL three perception members error (no HF/Replicate keys, all keys
 * rejected, all configured HF models retired from the hf-inference router),
 * the orchestrator forces an abstain via the safety net AND sets the new
 * adjudication.perception_unavailable flag. VerdictCard must render a
 * distinct honest empty-state card rather than the misleading "UNCERTAIN —
 * REFER" card with a near-zero confidence ring.
 */
describe('VerdictCard perception-unavailable state', () => {
  function perceptionUnavailableAdjudication(): Adjudication {
    return {
      verdict: 'abstain',
      confidence: 0,
      rationale: 'Adjudication call failed; defaulting to abstain.',
      abstain_reason: 'no perception model returned a result',
      auto_abstained: true,
      auto_abstain_reasons: ['no perception model returned a result'],
      perception_unavailable: true,
    };
  }

  it('renders the perception-unavailable card instead of the regular abstain card', () => {
    const html = renderToStaticMarkup(
      <VerdictCard
        adjudication={perceptionUnavailableAdjudication()}
        ensemble={null}
        rag={null}
        fallbackRate={0}
        onDisagree={async () => undefined}
        onOpenSettings={() => undefined}
      />,
    );
    expect(html).toContain('data-testid="verdict-perception-unavailable"');
    expect(html).toContain('Perception unavailable');
    expect(html).toContain('Configure an API key in Settings');
    // The fake "UNCERTAIN — REFER" headline MUST NOT appear when this flag is set.
    expect(html).not.toContain('UNCERTAIN');
    // The misleading scar-FPR disclosure must also not appear: nothing was measured.
    expect(html).not.toContain('data-testid="scar-fpr-disclosure"');
  });

  it('includes the Open Settings button when onOpenSettings is provided', () => {
    const html = renderToStaticMarkup(
      <VerdictCard
        adjudication={perceptionUnavailableAdjudication()}
        ensemble={null}
        rag={null}
        fallbackRate={0}
        onDisagree={async () => undefined}
        onOpenSettings={() => undefined}
      />,
    );
    expect(html).toContain('Open Settings');
  });

  it('still renders without the button when onOpenSettings is omitted', () => {
    const html = renderToStaticMarkup(
      <VerdictCard
        adjudication={perceptionUnavailableAdjudication()}
        ensemble={null}
        rag={null}
        fallbackRate={0}
        onDisagree={async () => undefined}
      />,
    );
    expect(html).toContain('Perception unavailable');
    expect(html).not.toContain('Open Settings');
  });

  it('falls back to the regular verdict card when perception_unavailable is false', () => {
    const adj: Adjudication = {
      verdict: 'abstain',
      confidence: 30,
      rationale: 'genuine uncertainty',
      auto_abstained: false,
      auto_abstain_reasons: [],
      perception_unavailable: false,
    };
    const html = renderToStaticMarkup(
      <VerdictCard
        adjudication={adj}
        ensemble={mockEnsemble()}
        rag={mockRag()}
        fallbackRate={0}
        onDisagree={async () => undefined}
        onOpenSettings={() => undefined}
      />,
    );
    expect(html).not.toContain('data-testid="verdict-perception-unavailable"');
    expect(html).toContain('UNCERTAIN');
  });
});
