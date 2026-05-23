import type { Settings, StageConfig } from '@/lib/types';
import { parseTbProb, parseGeneralCxrTbProb } from '@/lib/providers/parsers';

/** Stage 2A: TB-specific classifier (HF primary -> Replicate fallback). */
export function buildTbStageConfig(s: Settings): StageConfig {
  const o = s.overrides;
  const hasFallback =
    o.tbClassifierReplicate.trim().length > 0 && o.tbClassifierReplicateVersion.trim().length > 0;
  return {
    primary: { provider: 'hf', model: o.tbClassifierHf, parseOutput: parseTbProb },
    fallback: hasFallback
      ? {
          provider: 'replicate',
          model: o.tbClassifierReplicate,
          version: o.tbClassifierReplicateVersion,
          parseOutput: parseTbProb,
        }
      : null,
  };
}

/** Stage 2B: general CXR classifier (HF primary -> Replicate fallback). */
export function buildGeneralStageConfig(s: Settings): StageConfig {
  const o = s.overrides;
  const hasFallback =
    o.generalCxrReplicate.trim().length > 0 && o.generalCxrReplicateVersion.trim().length > 0;
  return {
    primary: { provider: 'hf', model: o.generalCxrHf, parseOutput: parseGeneralCxrTbProb },
    fallback: hasFallback
      ? {
          provider: 'replicate',
          model: o.generalCxrReplicate,
          version: o.generalCxrReplicateVersion,
          parseOutput: parseGeneralCxrTbProb,
        }
      : null,
  };
}
