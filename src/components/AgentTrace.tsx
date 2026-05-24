import type { RunState } from '@/hooks/usePipeline';
import { StageCard } from './StageCard';
import { TbGauge } from './TbGauge';
import { RagStrip } from './RagStrip';

/**
 * The Agent Trace panel — the centerpiece. One card per stage, live status,
 * provider badges, latencies, and expandable raw JSON.
 *
 * Milestone 23 collapsed the perception "ensemble" to a single-path view: the
 * local trained model card (active when local mode is on) OR the gpt-5.5 vision
 * card. The legacy "TB Classifier (HF) w0.5" and "General CXR (HF) w0.2" cards
 * were removed alongside the HF runtime path.
 */
export function AgentTrace({ state }: { state: RunState }): JSX.Element {
  const { stageStatus, members, rag, quality } = state;
  const idle = state.status === 'idle';

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-offwhite">Agent Trace</h2>
        <p className="text-[11px] text-muted">Live multi-stage triage pipeline</p>
      </header>

      {idle ? (
        <div className="flex flex-1 items-center justify-center p-6 text-center text-[11px] text-muted">
          Drop a chest X-ray to begin. Each stage will stream here with its provider and latency.
        </div>
      ) : (
        <div className="flex-1 space-y-3 overflow-y-auto scroll-thin p-4">
          {/* Stage 1 — Quality gate */}
          <StageCard
            title="1 · Quality Gate"
            status={stageStatus.quality}
            provider="openai"
            error={state.errors.quality}
            raw={quality}
          >
            {quality && (
              <dl className="space-y-0.5 text-[11px]">
                <Row k="is_cxr" v={String(quality.is_cxr)} />
                <Row k="quality" v={quality.quality} />
                <Row k="reason" v={quality.reason} />
              </dl>
            )}
          </StageCard>

          {/* Stage 2 — Perception */}
          <SectionLabel>2 · Perception</SectionLabel>

          {/* Local trained-model card — only rendered when local mode actually
              produced a member. Carries the validated 0.922-AUROC tb_prob. */}
          {members.tb && (
            <StageCard
              title="Local validated head"
              status={stageStatus['ensemble.tb']}
              provider={members.tb.provider_used ?? null}
              latencyMs={members.tb.latency_ms ?? null}
              fellBack={state.fallbacks['ensemble.tb']}
              error={members.tb.error ?? state.errors['ensemble.tb']}
              note={state.stageNotes['ensemble.tb']}
              raw={members.tb.raw}
            >
              <TbGauge value={members.tb.tb_prob ?? null} label="tb_prob (calibrated)" />
            </StageCard>
          )}

          <StageCard
            title="GPT-5.5 Vision"
            status={stageStatus['ensemble.vlm']}
            provider={members.vlm?.provider_used ?? null}
            latencyMs={members.vlm?.latency_ms ?? null}
            error={members.vlm?.error ?? state.errors['ensemble.vlm']}
            note={state.stageNotes['ensemble.vlm']}
            raw={members.vlm?.raw}
          >
            <TbGauge value={members.vlm?.tb_prob ?? null} label="tb_score (uncalibrated)" />
            {members.vlm?.samples != null && members.vlm.samples > 1 && (
              <p className="mt-1 font-mono text-[10px] text-muted">
                verifier ran · spread {members.vlm.uncertainty?.toFixed(3) ?? '—'}
              </p>
            )}
            {members.vlm?.findings && members.vlm.findings.length > 0 && (
              <ul className="mt-2 space-y-0.5">
                {members.vlm.findings.map((f, i) => (
                  <li key={i} className="text-[11px] text-muted">• {f}</li>
                ))}
              </ul>
            )}
          </StageCard>

          {/* Stage 3 — RAG */}
          <SectionLabel>3 · Retrieval (kNN)</SectionLabel>
          <StageCard
            title="CXR-Foundation Retrieval"
            status={stageStatus.rag}
            provider={rag?.embedding_provider ?? undefined}
            fellBack={state.fallbacks.rag}
            note={rag?.skipped ? rag.skipReason : state.stageNotes.rag}
            error={state.errors.rag}
          >
            {rag && !rag.skipped && <RagStrip neighbors={rag.neighbors} />}
          </StageCard>

          {/* Stage 4 — Adjudicator */}
          <SectionLabel>4 · Adjudicator (GPT-5.5)</SectionLabel>
          <StageCard
            title="Adjudication"
            status={stageStatus.adjudicate}
            provider="openai"
            error={state.errors.adjudicate}
          >
            {state.adjudicationText && (
              <pre className="mono max-h-40 overflow-auto whitespace-pre-wrap rounded bg-ink p-2 text-[10px] text-muted scroll-thin">
                {state.adjudicationText}
              </pre>
            )}
            {state.adjudication?.auto_abstained && (
              <p className="mt-2 text-[11px] text-verdict-uncertain">
                Guardrail override → abstain: {state.adjudication.auto_abstain_reasons.join('; ')}
              </p>
            )}
          </StageCard>
        </div>
      )}
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }): JSX.Element {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="font-mono text-[10px] text-muted">{k}</span>
      <span className="text-right text-[11px] text-offwhite">{v}</span>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <div className="pt-1 font-mono text-[10px] uppercase tracking-widest text-muted">{children}</div>
  );
}
