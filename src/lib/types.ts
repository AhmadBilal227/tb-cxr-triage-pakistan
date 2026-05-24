/**
 * Core contract for the TB triage pipeline.
 *
 * Everything downstream — provider clients, the orchestrator, and every UI card —
 * depends on the normalized shapes defined here. Provider-specific payloads are
 * collapsed into these shapes at the `parseOutput` seam so the orchestrator and UI
 * never see raw Replicate/OpenAI structures.
 *
 * Milestone 23 removed Hugging Face from the runtime path entirely (the free
 * hf-inference router dropped every default classifier we relied on through 2024-
 * 2025). The validated local-mode FastAPI server is the primary perception when
 * reachable; gpt-5.5 vision is the deployment-side primary; Replicate stays as an
 * optional configurable fallback. The HF Python library is still used OFFLINE in
 * `training/` to load Rad-DINO + TXRV weights from the local HF cache — that is
 * a file-system read, not a runtime API call, and the frontend has no awareness
 * of it.
 */

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

export type Provider = 'replicate' | 'openai' | 'local-triage';

/** A provider that actually produced a classifier signal (never 'openai' here — GPT cards are tracked separately). */
export type ClassifierProvider = 'replicate';

/**
 * The normalized output of any Replicate classifier stage.
 * Spec says `raw: any`; under strict TS we use `unknown` and narrow at the edges.
 */
export interface ClassifierResult {
  tb_prob: number; // 0..1
  raw: unknown;
  provider_used: ClassifierProvider;
  latency_ms: number;
}

// ---------------------------------------------------------------------------
// Settings (BYOK + per-stage overrides), persisted to localStorage
// ---------------------------------------------------------------------------

export interface ModelOverrides {
  /**
   * Optional Replicate TB classifier (BYO). Empty by default; surfaces as an
   * "not configured" state in the UI. Only ever runs when both the slug AND
   * the version hash AND a replicateToken are populated.
   */
  tbClassifierReplicate: string;
  tbClassifierReplicateVersion: string;

  /** Replicate CLIP-style model used for retrieval embeddings. Empty disables RAG. */
  embeddingReplicate: string;
  embeddingReplicateVersion: string;
}

export interface OrchestrationModels {
  /** Primary adjudicator + vision model. */
  adjudicator: string; // default 'gpt-5.5'
  /** GPT-5.5 has no Replicate fallback; on failure we drop to this. */
  adjudicatorFallback: string; // default 'gpt-5.5-instant'
  /** For any text embeddings. */
  textEmbedding: string; // default 'text-embedding-3-large'
}

export interface Settings {
  openaiKey: string;
  replicateToken: string; // optional — empty disables Replicate fallback
  overrides: ModelOverrides;
  models: OrchestrationModels;
  /** Whether the user accepted the research-only first-use modal. */
  acceptedDisclaimer: boolean;
  /** Fitted calibration params, or null to use the hard-coded SCREENING_POLICY. */
  calibration: CalibrationParams | null;
  /**
   * Milestone 22 — LOCAL-MODE TRIAGE.
   * When `localMode === true` AND the FastAPI server at `localServerUrl` is
   * reachable, the orchestrator uses the local validated pipeline (Rad-DINO +
   * TXRV + TBHeadT2 + InactiveSequelaeHead under their calibrated temperatures)
   * as the PRIMARY perception, and gpt-5.5 vision is reduced to a borderline
   * second-opinion verifier. When `localMode === false` (default) OR the server
   * is unreachable, the M21 VLM-primary flow runs unchanged.
   */
  localMode: boolean;
  localServerUrl: string;
}

// ---------------------------------------------------------------------------
// Calibration types
// ---------------------------------------------------------------------------

export interface MemberCalibration {
  method: 'temperature' | 'platt';
  T: number; // used when method==='temperature'
  A: number;
  B: number; // used when method==='platt'
  nllRaw: number;
  nllCal: number;
}

export interface CalibrationParams {
  version: 1;
  fittedAt: number;
  nSamples: number;
  source: 'fitted' | 'default';
  perModel: Partial<Record<EnsembleMemberId, MemberCalibration>>;
  fusion: { mode: 'fixed' | 'fitted'; weights: Record<EnsembleMemberId, number>; bias: number };
  conformal: {
    tauLow: number;
    tauHigh: number;
    alphaSens: number;
    gammaSpec: number;
    nPos: number;
    nNeg: number;
    incomplete: boolean;
  };
  vlmSafetyThreshold: number;
}

export interface CalibrationSample {
  filename: string;
  label: 0 | 1;
  memberProbs: Partial<Record<EnsembleMemberId, number>>;
  vlmUncertainty: number | null;
}

// ---------------------------------------------------------------------------
// Pipeline stage status + events (drives the live Agent Trace panel)
// ---------------------------------------------------------------------------

export type StageStatus =
  | 'queued'
  | 'running'
  | 'fallback' // primary failed, now attempting Replicate
  | 'done'
  | 'error'
  | 'skipped';

/**
 * StageId — pipeline stage identifiers.
 *
 * M23 cleanup: `ensemble.tb` and `ensemble.general` previously identified the
 * two HF classifier stages. With HF removed, the local-mode perception emits
 * under `ensemble.tb` (so the existing UI trace card keeps working — it carries
 * the local trained model now), `ensemble.general` is retained as an opt-in
 * Replicate-only slot, and `ensemble.vlm` is the gpt-5.5 vision stage.
 */
export type StageId =
  | 'quality'
  | 'ensemble.tb'
  | 'ensemble.general'
  | 'ensemble.vlm'
  | 'rag'
  | 'adjudicate'
  | 'verdict';

// --- Stage 1: Quality gate (GPT-5.5 vision, no fallback) ---
export interface QualityResult {
  is_cxr: boolean;
  quality: 'good' | 'poor' | 'unreadable';
  reason: string;
}

// --- Stage 2: Perception ensemble members ---
export type EnsembleMemberId = 'tb' | 'general' | 'vlm';

export interface EnsembleMember {
  id: EnsembleMemberId;
  label: string;
  weight: number;
  status: StageStatus;
  tb_prob: number | null;
  /** 'openai' for the VLM member, which has no Replicate fallback. */
  provider_used: Provider | null;
  latency_ms: number | null;
  raw: unknown;
  error?: string;
  /** VLM-only structured findings. */
  findings?: string[];
  /** Self-consistency spread across K samples (VLM member); a real uncertainty estimate. */
  uncertainty?: number;
  /** Number of self-consistency samples that succeeded (VLM member). */
  samples?: number;
}

export interface EnsembleResult {
  members: EnsembleMember[];
  /** Weighted mean of member tb_probs (weights renormalized over members that returned). */
  weightedScore: number;
  /** Population std of member tb_probs that returned. */
  std: number;
  /** max(tb_prob) - min(tb_prob) across returning members. */
  disagreement: number;
  /** Count of classifier members that fell back to Replicate. */
  replicateFallbackCount: number;
}

// --- Stage 3: RAG retrieval ---
export interface RagNeighbor {
  filename: string;
  similarity: number; // cosine, 0..1
  label: 0 | 1;
  /** object URL for the stored blob, for thumbnail display */
  thumbUrl: string;
}

export interface RagResult {
  neighbors: RagNeighbor[];
  embedding_provider: ClassifierProvider | null;
  skipped: boolean;
  skipReason?: string;
}

// --- Stage 4: Adjudication ---
export type Verdict = 'tb' | 'no_tb' | 'abstain';

export interface Adjudication {
  verdict: Verdict;
  confidence: number; // 0..100
  rationale: string;
  abstain_reason?: string;
  /** True if our deterministic guardrails / screening policy forced an abstain. */
  auto_abstained: boolean;
  /** Human-readable reasons the guardrail / policy fired. */
  auto_abstain_reasons: string[];
  /**
   * True when NO perception provider succeeded — every ensemble member (TB classifier,
   * general CXR, VLM) errored out, e.g. because the user has not set any API key or
   * every configured provider returned an error. The verdict in this case is forced to
   * abstain by the safety net, but the UI MUST distinguish this from a real "uncertain"
   * reading: the model did not actually evaluate the image at all.
   *
   * Setting this flag is the contract VerdictCard uses to render a dedicated
   * "Perception unavailable — configure an API key in Settings" state rather than a
   * misleading "UNCERTAIN — REFER" card with a near-zero confidence ring.
   */
  perception_unavailable?: boolean;
  /**
   * Which perception pathway produced this verdict.
   *
   *   - 'vlm-primary'        : gpt-5.5 vision is the primary perception (deployed default).
   *   - 'onnx-primary'       : the local Rad-DINO + TXRV head ran (Phase B, not deployed yet).
   *   - 'local-onnx-via-server' : Milestone 22 — the user's M4 ran the full validated
   *                            pipeline (Rad-DINO + TXRV + TBHeadT2 + InactiveSequelaeHead
   *                            under their calibrated temperatures) via the FastAPI server.
   *                            GPT-5.5 vision is the second-opinion verifier on this path.
   *   - 'perception-unavailable': nothing ran — same flag as `perception_unavailable: true`.
   *
   * M23 removed `'hf-ensemble'` from the union — Hugging Face is no longer a runtime
   * perception path in the app.
   */
  perception_path?:
    | 'vlm-primary'
    | 'onnx-primary'
    | 'local-onnx-via-server'
    | 'perception-unavailable';
  /** Audit pins for the VLM path: prompt/schema versions + the model id the API returned. */
  vlm_audit?: {
    prompt_hash: string;
    schema_version: string;
    schema_hash: string;
    model_id_from_response: string;
    image_preprocessing_version: string;
    /** True when the borderline consistency-check verifier was actually called. */
    consistency_check_ran: boolean;
    /** True when primary + verifier disagreed on screen_result → forced ABSTAIN. */
    consistency_check_disagreed: boolean;
  };
  /** Transparency: how the safety-net combine reached the final verdict. */
  screening?: {
    policyVerdict: Verdict;
    modelVerdict: Verdict;
    fusedProb: number;
    vlmProb: number | null;
    vlmUncertainty: number;
  };
  /**
   * M24 LOCAL-MODEL ENRICHMENT.
   * Validated-model intermediates the engine USED to discard: the 8x8 box-evidence
   * grid, the per-zone calibrated TB probabilities (7 keys, see ZoneKey union in
   * providers/localTriage.ts), the 18 TXRV pathology scores, and the seg crop box.
   * Populated ONLY on the local-onnx-via-server path; absent on every VLM-primary
   * adjudication. UI components MUST treat each sub-field as optional.
   */
  local_enrichment?: {
    box_evidence_grid?: number[][];
    zonal_scores?: Partial<Record<
      'upper_l' | 'upper_r' | 'mid_l' | 'mid_r' | 'lower_l' | 'lower_r' | 'hilar',
      number
    >>;
    txrv_pathologies?: Record<string, number>;
    crop_box?: { x: number; y: number; w: number; h: number };
    inversion_detected?: boolean;
  };
}

// ---------------------------------------------------------------------------
// Orchestrator event stream (discriminated union)
// ---------------------------------------------------------------------------

export type PipelineEvent =
  | { type: 'stage_status'; stage: StageId; status: StageStatus; note?: string }
  | { type: 'quality_done'; result: QualityResult }
  | { type: 'ensemble_member'; member: EnsembleMember }
  | { type: 'ensemble_done'; result: EnsembleResult }
  | { type: 'rag_done'; result: RagResult }
  | { type: 'adjudicate_token'; token: string }
  | { type: 'adjudicate_done'; result: Adjudication }
  | { type: 'halted'; reason: string; stage: StageId }
  | { type: 'error'; stage: StageId; message: string }
  | { type: 'fallback_fired'; stage: StageId; from: Provider; to: Provider };

// ---------------------------------------------------------------------------
// Aggregate run result (persisted + exported)
// ---------------------------------------------------------------------------

export interface PipelineRun {
  id: string;
  createdAt: number;
  imageName: string;
  quality: QualityResult | null;
  ensemble: EnsembleResult | null;
  rag: RagResult | null;
  adjudication: Adjudication | null;
  halted: { reason: string; stage: StageId } | null;
  /** Per-stage provider usage for the export audit trail. */
  providerLog: ProviderLogEntry[];
  /** Fraction of classifier stages that used Replicate (degraded-run signal). */
  fallbackRate: number;
  modelVersions: Record<string, string>;
}

export interface ProviderLogEntry {
  stage: StageId;
  provider_used: Provider | null;
  latency_ms: number | null;
  fell_back: boolean;
}

// ---------------------------------------------------------------------------
// DB records (Dexie)
// ---------------------------------------------------------------------------

export interface LabeledCaseRecord {
  id: string;
  filename: string;
  blob: Blob;
  embedding: number[] | null;
  embedding_provider: ClassifierProvider | null;
  label: 0 | 1; // 1 = TB, 0 = not TB
  createdAt: number;
  /** 'import' = from labeled set; 'feedback' = added via Disagree? button. */
  source: 'import' | 'feedback';
}

export interface CaseHistoryRecord {
  id: string;
  imageName: string;
  blob: Blob;
  verdict: Verdict | null;
  confidence: number | null;
  createdAt: number;
  run: PipelineRun;
}
