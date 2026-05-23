import { useState } from 'react';
import { motion } from 'framer-motion';
import { HelpCircle, ThumbsDown, Check } from 'lucide-react';
import type { Adjudication, EnsembleResult, RagResult, Verdict } from '@/lib/types';
import { Button } from './ui/button';
import { ConfidenceRing } from './ConfidenceRing';
import { cn } from '@/lib/utils';

const VERDICT_META: Record<Verdict, { label: string; color: string }> = {
  tb: { label: 'TB SUSPECTED', color: '#C8102E' },
  no_tb: { label: 'NO TB', color: '#00754A' },
  abstain: { label: 'UNCERTAIN — REFER', color: '#F59E0B' },
};

export function VerdictCard({
  adjudication,
  ensemble,
  rag,
  fallbackRate,
  onDisagree,
}: {
  adjudication: Adjudication;
  ensemble: EnsembleResult | null;
  rag: RagResult | null;
  fallbackRate: number;
  onDisagree: (label: 0 | 1) => Promise<void>;
}): JSX.Element {
  const [showWhy, setShowWhy] = useState(false);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [saved, setSaved] = useState(false);
  const meta = VERDICT_META[adjudication.verdict];

  const handleDisagree = async (label: 0 | 1): Promise<void> => {
    await onDisagree(label);
    setSaved(true);
    setFeedbackOpen(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border bg-surface p-4"
      style={{ borderColor: `${meta.color}66` }}
    >
      <div className="flex items-center gap-4">
        <ConfidenceRing value={adjudication.confidence} color={meta.color} size={92} />
        <div className="min-w-0 flex-1">
          <div className="text-xl font-semibold tracking-tight" style={{ color: meta.color }}>
            {meta.label}
          </div>
          <p className="mt-1 text-sm leading-relaxed text-offwhite/90">{adjudication.rationale}</p>
          {adjudication.abstain_reason && (
            <p className="mt-1 text-[11px] text-verdict-uncertain">{adjudication.abstain_reason}</p>
          )}
          {fallbackRate > 0 && (
            <p className="mt-1 font-mono text-[10px] text-provider-replicate">
              fallback rate {(fallbackRate * 100).toFixed(0)}% — degraded run
            </p>
          )}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" onClick={() => setShowWhy((v) => !v)}>
          <HelpCircle className="h-3.5 w-3.5" /> Why?
        </Button>
        {saved ? (
          <span className="inline-flex items-center gap-1 text-[11px] text-verdict-clear">
            <Check className="h-3.5 w-3.5" /> Added to corpus
          </span>
        ) : feedbackOpen ? (
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-muted">Correct label?</span>
            <Button variant="danger" size="sm" onClick={() => handleDisagree(1)}>
              TB
            </Button>
            <Button variant="outline" size="sm" onClick={() => handleDisagree(0)}>
              No TB
            </Button>
          </div>
        ) : (
          <Button variant="ghost" size="sm" onClick={() => setFeedbackOpen(true)}>
            <ThumbsDown className="h-3.5 w-3.5" /> Disagree?
          </Button>
        )}
      </div>

      {showWhy && (
        <div className="mt-3 space-y-2 rounded-lg border border-border bg-surface-2 p-3 text-[11px]">
          <div className="font-mono uppercase tracking-wide text-muted">Full trace</div>
          {ensemble && (
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              <Detail k="weighted_score" v={ensemble.weightedScore.toFixed(3)} />
              <Detail k="ensemble_std" v={ensemble.std.toFixed(3)} />
              <Detail k="disagreement" v={ensemble.disagreement.toFixed(3)} />
              <Detail k="replicate_fallbacks" v={String(ensemble.replicateFallbackCount)} />
              {ensemble.members.map((m) => (
                <Detail
                  key={m.id}
                  k={`${m.label} (${m.provider_used ?? 'n/a'})`}
                  v={m.tb_prob === null ? 'error' : m.tb_prob.toFixed(3)}
                />
              ))}
            </div>
          )}
          {rag && !rag.skipped && rag.neighbors.length > 0 && (
            <div className="border-t border-border pt-2">
              <span className="text-muted">top retrieval: </span>
              <span className={cn(rag.neighbors[0]?.label === 1 ? 'text-verdict-tb' : 'text-verdict-clear')}>
                {rag.neighbors[0]?.label === 1 ? 'TB' : 'NEG'} @ {(((rag.neighbors[0]?.similarity ?? 0)) * 100).toFixed(0)}%
              </span>
            </div>
          )}
          {adjudication.auto_abstain_reasons.length > 0 && (
            <div className="border-t border-border pt-2 text-verdict-uncertain">
              {adjudication.auto_abstain_reasons.map((r, i) => (
                <div key={i}>• {r}</div>
              ))}
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}

function Detail({ k, v }: { k: string; v: string }): JSX.Element {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="font-mono text-[10px] text-muted">{k}</span>
      <span className="text-offwhite">{v}</span>
    </div>
  );
}
