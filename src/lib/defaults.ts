import type { Settings } from './types';

/**
 * Sensible defaults per spec. Model slots that the user MUST provide themselves
 * (Replicate fallbacks, the CXR-Foundation Inference Endpoint) are intentionally
 * left empty so the UI can surface a clear "unconfigured" message rather than
 * silently pretending a fallback exists.
 */
export const DEFAULT_SETTINGS: Settings = {
  openaiKey: '',
  hfToken: '',
  replicateToken: '',
  overrides: {
    tbClassifierHf: 'Owos/tb-classifier',
    tbClassifierReplicate: '',
    tbClassifierReplicateVersion: '',

    generalCxrHf: 'keremberke/yolov8m-chest-xray-classification',
    generalCxrReplicate: '',
    generalCxrReplicateVersion: '',

    embeddingEndpointUrl: '',
    // Default to a verified public CLIP model on Replicate so retrieval works out of
    // the box when a Replicate token is present (CXR-Foundation needs a paid HF endpoint).
    // Output shape: { embedding: number[768] }.
    embeddingReplicate: 'krthr/clip-embeddings',
    embeddingReplicateVersion:
      '1c0371070cb827ec3c7f2f28adcdde54b50dcd239aa6faea0bc98b174ef03fb4',
  },
  models: {
    adjudicator: 'gpt-5.5',
    adjudicatorFallback: 'gpt-5.5-instant',
    textEmbedding: 'text-embedding-3-large',
  },
  acceptedDisclaimer: false,
};

export const SETTINGS_STORAGE_KEY = 'tb-triage.settings.v1';

/** Ensemble vote weights (Stage 2). Must sum to 1 across the three members. */
export const ENSEMBLE_WEIGHTS = {
  tb: 0.5, // TB-specific classifier — primary signal
  general: 0.2, // general CXR pathology distribution
  vlm: 0.3, // GPT-5.5 independent vision read
} as const;

/** Deterministic auto-abstain thresholds (Stage 4 guardrails). */
export const ABSTAIN_RULES = {
  minConfidence: 75,
  maxEnsembleStd: 0.2,
  minTop1Similarity: 0.6,
  maxDisagreementForLowSim: 0.3,
  maxReplicateFallbacks: 2, // >=2 stages on Replicate => degraded run, flag it
} as const;

export const KNN_K = 5;

/**
 * Self-consistency: how many times the VLM reads each image. The mean is the
 * calibrated probability; the spread is a real uncertainty estimate (single-prompt
 * verbalized confidence does not fix overconfidence — arXiv:2604.02543).
 */
export const SELF_CONSISTENCY_K = 3;

/**
 * Screening-biased decision policy (sensitivity-first). Asymmetric on purpose:
 * a LOW bar to flag, a HIGH bar to clear. This is the calibration lever that took
 * generic-VLM TB F1 from 0.48 -> 0.77 in arXiv:2510.00411. Fit these on a labeled
 * holdout via /validate for your population/scanner.
 */
export const SCREENING_POLICY = {
  /** fused or VLM probability at/above this => flag TB. */
  tbFlag: 0.3,
  /** only CLEAR a patient when prob <= this AND uncertainty is low. */
  negClear: 0.15,
  maxClearUncertainty: 0.15,
  /** safety net: VLM alone at/above this escalates even when the fused score is low. */
  vlmSafetyThreshold: 0.5,
} as const;
