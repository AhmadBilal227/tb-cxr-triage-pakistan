/**
 * Core contract for the TB triage pipeline.
 *
 * Everything downstream — provider clients, the orchestrator, and every UI card —
 * depends on the normalized shapes defined here. Provider-specific payloads are
 * collapsed into these shapes at the `parseOutput` seam so the orchestrator and UI
 * never see raw HF/Replicate/OpenAI structures.
 */

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

export type Provider = 'hf' | 'replicate' | 'openai';

/** A provider that actually produced a classifier signal (never 'openai' here — GPT cards are tracked separately). */
export type ClassifierProvider = 'hf' | 'replicate';

/**
 * The normalized output of any HF->Replicate classifier stage.
 * Spec says `raw: any`; under strict TS we use `unknown` and narrow at the edges.
 */
export interface ClassifierResult {
  tb_prob: number; // 0..1
  raw: unknown;
  provider_used: ClassifierProvider;
  latency_ms: number;
}

// ---------------------------------------------------------------------------
// Stage configuration (the classifyWithFallback contract)
// ---------------------------------------------------------------------------

export interface PrimarySpec {
  provider: 'hf';
  model: string;
  /** Map an opaque HF response into tb_prob in [0,1]. */
  parseOutput: (raw: unknown) => number;
}

export interface FallbackSpec {
  provider: 'replicate';
  model: string;
  version: string;
  /** Map an opaque Replicate prediction.output into tb_prob in [0,1]. */
  parseOutput: (raw: unknown) => number;
}

export interface StageConfig {
  primary: PrimarySpec;
  /** null => no fallback available for this stage (Replicate token missing or slot unconfigured). */
  fallback: FallbackSpec | null;
}

// ---------------------------------------------------------------------------
// Settings (BYOK + per-stage overrides), persisted to localStorage
// ---------------------------------------------------------------------------

export interface ModelOverrides {
  tbClassifierHf: string;
  tbClassifierReplicate: string;
  tbClassifierReplicateVersion: string;

  generalCxrHf: string;
  generalCxrReplicate: string;
  generalCxrReplicateVersion: string;

  /** HF Inference Endpoint URL for google/cxr-foundation (not on free serverless). */
  embeddingEndpointUrl: string;
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
  hfToken: string;
  replicateToken: string; // optional — empty disables Replicate fallback
  overrides: ModelOverrides;
  models: OrchestrationModels;
  /** Whether the user accepted the research-only first-use modal. */
  acceptedDisclaimer: boolean;
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
  /** Transparency: how the safety-net combine reached the final verdict. */
  screening?: {
    policyVerdict: Verdict;
    modelVerdict: Verdict;
    fusedProb: number;
    vlmProb: number | null;
    vlmUncertainty: number;
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
