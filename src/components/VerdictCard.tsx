import { useState } from 'react';
import { motion } from 'framer-motion';
import { HelpCircle, KeyRound, ThumbsDown, Check } from 'lucide-react';
import type { Adjudication, EnsembleResult, RagResult, Verdict } from '@/lib/types';
import { Button } from './ui/button';
import { ConfidenceRing } from './ConfidenceRing';
import { cn } from '@/lib/utils';
import { BoxEvidenceHeatmap } from './details/BoxEvidenceHeatmap';
import { ZonalBars } from './details/ZonalBars';
import { PathologyList } from './details/PathologyList';
import { ClinicianReport } from './details/ClinicianReport';
import { SecondaryObservations } from './details/SecondaryObservations';
import { ImageLightbox } from './details/ImageLightbox';
import type { LocalTriageResult } from '@/lib/providers/localTriage';

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
  onOpenSettings,
  imageDataUrl,
  openaiKey,
  primaryModel,
  fallbackModel,
  lightboxOpen: lightboxOpenProp,
  onLightboxOpenChange,
}: {
  adjudication: Adjudication;
  ensemble: EnsembleResult | null;
  rag: RagResult | null;
  fallbackRate: number;
  onDisagree: (label: 0 | 1) => Promise<void>;
  /** Required for the "perception unavailable" state's CTA. Optional everywhere else. */
  onOpenSettings?: () => void;
  /**
   * M24 — required only for the ClinicianReport CTA in the Details panel. When
   * absent the ClinicianReport section is omitted (graceful degradation; VLM-
   * primary runs and pre-M24 adjudications never set it).
   */
  imageDataUrl?: string;
  openaiKey?: string;
  primaryModel?: string;
  fallbackModel?: string;
  /** Lifted lightbox state — App owns the URL binding. */
  lightboxOpen?: boolean;
  onLightboxOpenChange?: (next: boolean) => void;
}): JSX.Element {
  const [showWhy, setShowWhy] = useState(false);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [saved, setSaved] = useState(false);
  // Lightbox state is LIFTED to the parent (App.tsx) so the URL-bound hook
  // lives where the Router context exists, and existing VerdictCard tests
  // don't need a router wrapper. When props are absent we fall back to local
  // useState — the lightbox still works, just without back-button parity.
  const [localLightbox, setLocalLightbox] = useState(false);
  const lightboxOpen = lightboxOpenProp ?? localLightbox;
  const setLightboxOpen = onLightboxOpenChange ?? setLocalLightbox;
  const meta = VERDICT_META[adjudication.verdict];

  // Honesty contract: when every perception path failed (no OpenAI key, local
  // server unreachable + VLM unavailable), the safety net forces an abstain —
  // but rendering "UNCERTAIN — REFER" implies the model evaluated the image and
  // was uncertain. That is a lie. Surface a distinct state instead.
  if (adjudication.perception_unavailable) {
    return (
      <div
        role="alert"
        data-testid="verdict-perception-unavailable"
        className="rounded-xl border border-provider-replicate/40 bg-provider-replicate/5 p-4"
      >
        <div className="flex items-start gap-3">
          <KeyRound className="mt-0.5 h-5 w-5 shrink-0 text-provider-replicate" />
          <div className="min-w-0 flex-1">
            <div className="text-base font-semibold text-offwhite">Perception unavailable</div>
            <p className="mt-1 text-[12px] leading-relaxed text-offwhite/80">
              No perception model returned a result, so the app has not evaluated this image. This
              is not an "uncertain" reading — there is no reading at all. Set an OpenAI API key in
              Settings (gpt-5.5 vision is the deployed primary), or start the local FastAPI server
              with local mode enabled, and re-drop the image.
            </p>
            {adjudication.auto_abstain_reasons.length > 0 && (
              <ul className="mt-2 space-y-0.5 font-mono text-[10px] text-muted">
                {adjudication.auto_abstain_reasons.map((r, i) => (
                  <li key={i}>· {r}</li>
                ))}
              </ul>
            )}
            {onOpenSettings && (
              <div className="mt-3">
                <Button variant="outline" size="sm" onClick={onOpenSettings}>
                  Open Settings
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  const handleDisagree = async (label: 0 | 1): Promise<void> => {
    await onDisagree(label);
    setSaved(true);
    setFeedbackOpen(false);
  };

  // ----------------------------------------------------------------------
  // M24 — surface the validated-model intermediates when present. The fields
  // populate on the local-onnx-via-server pathway; the VLM-primary path leaves
  // `local_enrichment` undefined, and each sub-field on `local_enrichment` is
  // individually optional. Reach across `ensemble.members` for the FULL
  // LocalTriageResult (kept in member.raw for the local-triage member) — the
  // ClinicianReport's gpt-interpreter needs more than just the enrichment
  // sub-fields (calibration + threshold + s_inactive too).
  // ----------------------------------------------------------------------
  const enrichment = adjudication.local_enrichment;
  const boxGrid = enrichment?.box_evidence_grid;
  const zonalScores = enrichment?.zonal_scores;
  const txrvPathologies = enrichment?.txrv_pathologies;
  const localMember = ensemble?.members.find((m) => m.id === 'tb' && m.provider_used === 'local-triage');
  const localResult =
    localMember && localMember.raw && typeof localMember.raw === 'object'
      ? (localMember.raw as LocalTriageResult)
      : null;
  const clinicianReportReady = Boolean(
    localResult && imageDataUrl && primaryModel && fallbackModel,
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border bg-surface p-4"
      style={{ borderColor: `${meta.color}66` }}
    >
      {/* VERDICT — dominant, up top */}
      <div
        className="rounded-lg px-4 py-3 text-center"
        style={{ background: `${meta.color}14`, border: `1px solid ${meta.color}40` }}
      >
        <div className="text-3xl font-bold leading-none tracking-tight" style={{ color: meta.color }}>
          {meta.label}
        </div>
        <div className="mt-1.5 font-mono text-[10px] uppercase tracking-wider text-muted">
          radiographic TB screen · not a diagnosis · confirm bacteriologically
        </div>
        <PerceptionPathDisclosure adjudication={adjudication} />
      </div>
      {adjudication.verdict === 'no_tb' && (
        <p className="mt-2 rounded-md bg-verdict-uncertain/10 px-3 py-2 text-[11px] leading-snug text-verdict-uncertain">
          A negative screen does <strong>not</strong> rule out subclinical or early TB — chest X-ray misses
          roughly 40–50% of subclinical disease. Test symptomatic or high-risk patients regardless of this result.
        </p>
      )}

      {/* one-line reading + confidence; numbers live in Details below */}
      <div className="mt-3 flex items-center gap-4">
        <ConfidenceRing value={adjudication.confidence} color={meta.color} size={72} />
        <div className="min-w-0 flex-1">
          <p className="text-sm leading-relaxed text-offwhite/90">{adjudication.rationale}</p>
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

      {/* M24 always-on: BoxEvidence heatmap renders when the local pathway emitted a grid.
          Fullscreen affordance opens the URL-bound ImageLightbox below. */}
      {boxGrid && (
        <div className="mt-3 rounded-lg border border-border bg-surface-2 p-3">
          <BoxEvidenceHeatmap
            grid={boxGrid}
            imageUrl={imageDataUrl}
            onOpenLightbox={imageDataUrl ? () => setLightboxOpen(true) : undefined}
          />
        </div>
      )}

      {/* Full-screen image viewer with progressive evidence overlays. Mounts only
          when we have an image to show; chrome auto-hides after 2.5s of inactivity. */}
      {imageDataUrl && (boxGrid || zonalScores || txrvPathologies) && (
        <ImageLightbox
          open={lightboxOpen}
          onOpenChange={setLightboxOpen}
          imageUrl={imageDataUrl}
          boxGrid={boxGrid ?? null}
          zonalScores={zonalScores ?? null}
          txrvPathologies={txrvPathologies ?? null}
          verdictLabel={meta.label}
        />
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" onClick={() => setShowWhy((v) => !v)}>
          <HelpCircle className="h-3.5 w-3.5" /> {showWhy ? 'Hide details' : 'Details & stats'}
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
          <div className="font-mono uppercase tracking-wide text-muted">Details — stats &amp; full trace</div>
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
          {/* M24 — per-zone bar chart, sorted desc by probability. */}
          {zonalScores && Object.keys(zonalScores).length > 0 && (
            <div className="border-t border-border pt-2">
              <ZonalBars scores={zonalScores} />
            </div>
          )}
          {/* M24 — TorchXRayVision named-finding chips (18 pathology scores). */}
          {txrvPathologies && Object.keys(txrvPathologies).length > 0 && (
            <div className="border-t border-border pt-2">
              <PathologyList pathologies={txrvPathologies} />
            </div>
          )}
        </div>
      )}

      {/* M24 — radiology report CTA lives BELOW the body content (action row
          + details panel), per UX direction. Renders only when the local
          pipeline produced a result we can ground a narrative against. */}
      {clinicianReportReady && localResult && imageDataUrl && primaryModel && fallbackModel && (
        <div className="mt-4 space-y-4 border-t border-border pt-3">
          <ClinicianReport
            apiKey={openaiKey ?? ''}
            primaryModel={primaryModel}
            fallbackModel={fallbackModel}
            imageDataUrl={imageDataUrl}
            localResult={localResult}
            adjudication={adjudication}
          />
          {/* Phase B — non-TB side information (image quality, devices,
              cardiomediastinal, incidentals). Opt-in, never influences verdict. */}
          <SecondaryObservations
            apiKey={openaiKey ?? ''}
            primaryModel={primaryModel}
            fallbackModel={fallbackModel}
            imageDataUrl={imageDataUrl}
            localResult={localResult}
          />
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

/**
 * PATH-SPECIFIC DISCLOSURES.
 *
 * The M19 always-on scar-FPR disclosure described our TRAINED ONNX head's
 * mimic FPR (NIH cross-site Fibrosis/Pleural_Thickening ~10%). That number is
 * NOT honest to display when the actual perception was gpt-5.5 vision — a
 * different model with a different failure profile, no labeled validation.
 * Render the correct disclosure based on `adjudication.perception_path`:
 *
 *   - 'vlm-primary'  (today's default): VLM uncalibrated, may miss disease,
 *                                       may hallucinate, may overreact to scar.
 *   - 'onnx-primary' (Phase B, future): the M15 validated head, AUROC 0.925
 *                                       LODO, with the cross-site mimic FPR.
 *   - 'local-onnx-via-server' (M22):    user's own M4 ran the validated head
 *                                       through the FastAPI server.
 *
 * Milestone 23 removed the `'hf-ensemble'` branch — HF is no longer a runtime
 * perception path in the app.
 *
 * Also renders a small path indicator (`data-testid="perception-path-indicator"`)
 * near the verdict so the user can SEE which model produced the result.
 */
function PerceptionPathDisclosure({
  adjudication,
}: {
  adjudication: Adjudication;
}): JSX.Element {
  const path = adjudication.perception_path ?? 'vlm-primary';

  // Milestone 22 — LOCAL-MODE path. The user's M4 actually ran the full validated
  // pipeline (Rad-DINO + TXRV + TBHeadT2 + InactiveSequelaeHead under their
  // calibrated temperatures) and we OWN those numbers. Replace the M21 generic-
  // VLM disclosure with the LODO sensitivity/specificity/AUROC plus the M18
  // NIH-stress mimic FPR caveat (~10% on scar-shaped findings, measured).
  // The "general-purpose VLM" line MUST NOT appear here — that disclosure is
  // wrong for the local-mode pathway and would lie about what produced the
  // verdict.
  if (path === 'local-onnx-via-server') {
    const verifierRan = adjudication.vlm_audit?.consistency_check_ran ?? false;
    const verifierDisagreed = adjudication.vlm_audit?.consistency_check_disagreed ?? false;
    const verifierTag = verifierDisagreed
      ? 'gpt verifier disagreed'
      : verifierRan
        ? 'gpt verifier agreed'
        : 'gpt verifier not fired';
    return (
      <>
        <div
          className="mt-1 px-1 text-[10px] leading-snug text-muted"
          data-testid="local-mode-disclosure"
        >
          This result is produced by the validated Rad-DINO + TorchXRayVision research model
          running on your machine. Reported LODO sensitivity 0.800 / specificity 0.911 /
          AUROC 0.922 on 13,092 held-out predictions; per-site recalibration recommended.
          Higher false-positive rate (~10%) expected on radiographically scar-shaped findings
          (healed fibrosis, pleural thickening — M18 NIH stress). Not a medical device.
        </div>
        <div
          className="mt-1 px-1 font-mono text-[9px] uppercase tracking-wider text-muted"
          data-testid="perception-path-indicator"
        >
          perception path: local trained model (validated) · {verifierTag}
        </div>
      </>
    );
  }

  // ONNX path: the M19 disclosure (our trained head's cross-site mimic FPR).
  if (path === 'onnx-primary') {
    return (
      <>
        <div
          className="mt-1 px-1 text-[10px] leading-snug text-muted"
          data-testid="scar-fpr-disclosure"
        >
          Higher false-positive rate (~10%) expected on radiographically scar-shaped findings
          (healed fibrosis, pleural thickening).
        </div>
        <div
          className="mt-1 px-1 font-mono text-[9px] uppercase tracking-wider text-muted"
          data-testid="perception-path-indicator"
        >
          perception path: local ONNX (rad-dino + txrv, validated head)
        </div>
      </>
    );
  }

  // VLM-primary path (the deployed default today). The disclosure is LONGER
  // because the model is unvalidated and the user is owed every caveat.
  const modelId = adjudication.vlm_audit?.model_id_from_response ?? 'gpt-5.5';
  return (
    <>
      <div
        className="mt-1 px-1 text-[10px] leading-snug text-muted"
        data-testid="vlm-primary-disclosure"
      >
        This result is produced by a general-purpose vision-language model (gpt-5.5 vision), not the
        project's validated Rad-DINO + TorchXRayVision research model. The TB score is uncalibrated
        and has not been validated to the LODO sensitivity/specificity numbers reported in the case
        study. The model may miss disease, hallucinate findings, overreact to scar/fibrosis/pleural-
        thickening, or abstain on image-quality issues. Research triage support only — not diagnosis.
      </div>
      <div
        className="mt-1 px-1 text-[9px] leading-snug text-muted"
        data-testid="onnx-deployed-but-inactive-note"
      >
        Local validated ONNX heads (Rad-DINO + TorchXRayVision, AUROC 0.922 LODO) ship with this
        build but cannot execute without browser-side backbone features (Phase B gap).
      </div>
      <div
        className="mt-1 px-1 font-mono text-[9px] uppercase tracking-wider text-muted"
        data-testid="perception-path-indicator"
      >
        perception path: {modelId} (unvalidated VLM)
        {adjudication.vlm_audit?.consistency_check_ran ? ' · verifier ran' : ''}
        {adjudication.vlm_audit?.consistency_check_disagreed ? ' · verifier disagreed' : ''}
      </div>
    </>
  );
}
