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

function mockOnnxAdjudication(verdict: Verdict): Adjudication {
  return { ...mockAdjudication(verdict), perception_path: 'onnx-primary' };
}

function mockVlmAdjudication(verdict: Verdict): Adjudication {
  return {
    ...mockAdjudication(verdict),
    perception_path: 'vlm-primary',
    vlm_audit: {
      prompt_hash: 'deadbeef',
      schema_version: 'vlm-triage-v1',
      schema_hash: 'cafebabe',
      model_id_from_response: 'gpt-5.5-2026-04-23',
      image_preprocessing_version: 'browser-passthrough-v1',
      consistency_check_ran: false,
      consistency_check_disagreed: false,
    },
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

/**
 * M19 scar-FPR disclosure still renders ON THE ONNX PATH (when the validated
 * Rad-DINO + TXRV head ran). On the VLM-primary path (today's default) a
 * DIFFERENT disclosure renders — see the next describe block. Pin both so we
 * can never silently leak the M19 disclosure onto a VLM-derived verdict.
 */
describe('VerdictCard scar-FPR disclosure — ONNX path (Phase B)', () => {
  for (const verdict of ['tb', 'no_tb', 'abstain'] as const) {
    it(`renders the scar-shape FPR disclosure when verdict=${verdict} AND path=onnx-primary`, () => {
      const html = renderToStaticMarkup(
        <VerdictCard
          adjudication={mockOnnxAdjudication(verdict)}
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
        adjudication={mockOnnxAdjudication('no_tb')}
        ensemble={mockEnsemble()}
        rag={mockRag()}
        fallbackRate={0}
        onDisagree={async () => undefined}
      />,
    );
    expect(html).toContain('data-testid="scar-fpr-disclosure"');
    expect(html).toContain('data-testid="perception-path-indicator"');
    expect(html).toContain('local ONNX');
  });
});

/**
 * M21 — VLM-primary path disclosure. The today-default. The VLM is uncalibrated,
 * has not been validated against the project's LODO holdout, and may overreact
 * to scar-shaped findings; every claim in this disclosure must be defensible
 * against a hostile reader.
 */
describe('VerdictCard VLM-primary disclosure (Milestone 21 default path)', () => {
  for (const verdict of ['tb', 'no_tb', 'abstain'] as const) {
    it(`renders the VLM disclosure (NOT the M19 ONNX one) when verdict=${verdict} AND path=vlm-primary`, () => {
      const html = renderToStaticMarkup(
        <VerdictCard
          adjudication={mockVlmAdjudication(verdict)}
          ensemble={mockEnsemble()}
          rag={mockRag()}
          fallbackRate={0}
          onDisagree={async () => undefined}
        />,
      );
      expect(html).toContain('data-testid="vlm-primary-disclosure"');
      expect(html).toContain('general-purpose vision-language model');
      expect(html).toContain('uncalibrated');
      expect(html).toContain('may miss disease');
      // The M19 scar-FPR fragment talks about OUR head's measured FPR — must
      // NOT appear on a VLM verdict (the VLM has no labeled FPR number).
      expect(html).not.toContain(DISCLOSURE_FRAGMENT);
      expect(html).not.toContain('data-testid="scar-fpr-disclosure"');
      // Hostile-reader tone check — no marketing language slipping in.
      expect(html).not.toMatch(/guarantee|comprehensive|robust/i);
    });
  }

  it('also surfaces the deployed-but-inactive ONNX honesty note', () => {
    const html = renderToStaticMarkup(
      <VerdictCard
        adjudication={mockVlmAdjudication('no_tb')}
        ensemble={mockEnsemble()}
        rag={mockRag()}
        fallbackRate={0}
        onDisagree={async () => undefined}
      />,
    );
    expect(html).toContain('data-testid="onnx-deployed-but-inactive-note"');
    expect(html).toContain('Phase B gap');
  });

  it('renders the model id from the Responses API + verifier-state indicator', () => {
    const adj = mockVlmAdjudication('abstain');
    if (adj.vlm_audit) {
      adj.vlm_audit = { ...adj.vlm_audit, consistency_check_ran: true, consistency_check_disagreed: true };
    }
    const html = renderToStaticMarkup(
      <VerdictCard
        adjudication={adj}
        ensemble={mockEnsemble()}
        rag={mockRag()}
        fallbackRate={0}
        onDisagree={async () => undefined}
      />,
    );
    expect(html).toContain('data-testid="perception-path-indicator"');
    expect(html).toContain('gpt-5.5-2026-04-23');
    expect(html).toContain('unvalidated VLM');
    expect(html).toContain('verifier ran');
    expect(html).toContain('verifier disagreed');
  });

  it('default (no perception_path field) falls back to the VLM disclosure (today is VLM-primary)', () => {
    const html = renderToStaticMarkup(
      <VerdictCard
        adjudication={mockAdjudication('no_tb')}
        ensemble={mockEnsemble()}
        rag={mockRag()}
        fallbackRate={0}
        onDisagree={async () => undefined}
      />,
    );
    expect(html).toContain('data-testid="vlm-primary-disclosure"');
    expect(html).not.toContain(DISCLOSURE_FRAGMENT);
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
    // M23 — the copy now points at OpenAI / local mode instead of HF + Replicate.
    expect(html).toContain('Set an OpenAI API key in Settings');
    expect(html).toContain('local FastAPI server');
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
