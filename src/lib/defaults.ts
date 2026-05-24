import type { Settings } from './types';

/**
 * Sensible defaults per spec. Model slots that the user MUST provide themselves
 * (Replicate fallbacks) are intentionally left empty so the UI can surface a
 * clear "unconfigured" message rather than silently pretending a fallback exists.
 *
 * Milestone 23 removed Hugging Face from the runtime path: the free hf-inference
 * router dropped every default classifier the project relied on (`Owos/tb-classifier`
 * and `keremberke/yolov8m-chest-xray-classification` retired in 2024-2025, and the
 * backbones `microsoft/rad-dino` + `torchxrayvision/densenet121-res224-all` are not
 * deployed on the router either). The validated path is the M22 local-mode server
 * running the deployed trained model; the deployed app falls back to gpt-5.5 vision
 * (M21); Replicate remains a configurable BYOK fallback.
 */
export const DEFAULT_SETTINGS: Settings = {
  openaiKey: '',
  replicateToken: '',
  overrides: {
    // Optional Replicate TB classifier — BYO; empty means "not configured".
    tbClassifierReplicate: '',
    tbClassifierReplicateVersion: '',

    // Default to a verified public CLIP model on Replicate so retrieval works
    // out of the box when a Replicate token is present. Output shape:
    // { embedding: number[768] }.
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
  calibration: null,
  // Milestone 22 (revised 2026-05-25): default depends on build mode.
  // - `npm run dev` -> Local mode ON, URL points at :8001 (port 8000 is commonly
  //   held on the dev machine by other tools — verified live). The user's own
  //   trained model should be the default when developing locally; nothing else
  //   makes sense given the validated 0.922-AUROC head is sitting on disk.
  // - `npm run build` (Netlify) -> Local mode OFF; the deployed app can't reach
  //   a user's localhost so it falls through to M21 VLM-primary unchanged.
  // Settings page lets the user override either default at any time.
  localMode: import.meta.env.DEV,
  localServerUrl: import.meta.env.DEV ? 'http://localhost:8001' : 'http://localhost:8000',
};

export const SETTINGS_STORAGE_KEY = 'tb-triage.settings.v1';

/**
 * Ensemble vote weights — fallback for calibration when there aren't enough
 * samples to fit fusion weights. Post-M23 the orchestrator does NOT call this
 * map (the pipeline is single-perception: local OR vlm), but `calibration.ts`
 * still references it as the default-weights anchor for the `/validate` flow
 * across the historical EnsembleMemberId union (`tb`/`general`/`vlm`).
 *
 * Numbers reflect the prior intuition: VLM carries most weight today (it is the
 * deployed primary), with the historical TB-classifier slot kept as a non-zero
 * stub so a future BYO Replicate model can slot back in without re-fitting.
 */
export const ENSEMBLE_WEIGHTS = {
  tb: 0.2,
  general: 0.0,
  vlm: 0.8,
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
