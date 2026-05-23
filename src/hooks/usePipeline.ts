import { useCallback, useReducer } from 'react';
import type {
  Adjudication,
  EnsembleMember,
  EnsembleMemberId,
  EnsembleResult,
  PipelineEvent,
  PipelineRun,
  QualityResult,
  RagResult,
  StageId,
  StageStatus,
} from '@/lib/types';
import { runPipeline } from '@/lib/pipeline/orchestrator';
import { settingsStore } from '@/store/settings';
import { addHistory } from '@/lib/db';
import { shortId } from '@/lib/utils';

const STAGE_IDS: StageId[] = [
  'quality',
  'ensemble.tb',
  'ensemble.general',
  'ensemble.vlm',
  'rag',
  'adjudicate',
  'verdict',
];

export interface RunState {
  status: 'idle' | 'running' | 'done' | 'halted';
  stageStatus: Record<StageId, StageStatus>;
  stageNotes: Partial<Record<StageId, string>>;
  fallbacks: Partial<Record<StageId, boolean>>;
  errors: Partial<Record<StageId, string>>;
  quality: QualityResult | null;
  members: Record<EnsembleMemberId, EnsembleMember | null>;
  ensemble: EnsembleResult | null;
  rag: RagResult | null;
  adjudicationText: string;
  adjudication: Adjudication | null;
  halted: { reason: string; stage: StageId } | null;
  run: PipelineRun | null;
}

function emptyStageStatus(): Record<StageId, StageStatus> {
  return STAGE_IDS.reduce(
    (acc, id) => ({ ...acc, [id]: 'queued' }),
    {} as Record<StageId, StageStatus>,
  );
}

function initialState(): RunState {
  return {
    status: 'idle',
    stageStatus: emptyStageStatus(),
    stageNotes: {},
    fallbacks: {},
    errors: {},
    quality: null,
    members: { tb: null, general: null, vlm: null },
    ensemble: null,
    rag: null,
    adjudicationText: '',
    adjudication: null,
    halted: null,
    run: null,
  };
}

type Action =
  | { kind: 'reset' }
  | { kind: 'start' }
  | { kind: 'event'; event: PipelineEvent }
  | { kind: 'finish'; run: PipelineRun };

function reducer(state: RunState, action: Action): RunState {
  switch (action.kind) {
    case 'reset':
      return initialState();
    case 'start':
      return { ...initialState(), status: 'running' };
    case 'finish':
      return { ...state, run: action.run, status: state.status === 'halted' ? 'halted' : 'done' };
    case 'event': {
      const e = action.event;
      switch (e.type) {
        case 'stage_status':
          return {
            ...state,
            stageStatus: { ...state.stageStatus, [e.stage]: e.status },
            stageNotes: e.note ? { ...state.stageNotes, [e.stage]: e.note } : state.stageNotes,
            fallbacks:
              e.status === 'fallback' ? { ...state.fallbacks, [e.stage]: true } : state.fallbacks,
          };
        case 'fallback_fired':
          return { ...state, fallbacks: { ...state.fallbacks, [e.stage]: true } };
        case 'quality_done':
          return { ...state, quality: e.result };
        case 'ensemble_member':
          return { ...state, members: { ...state.members, [e.member.id]: e.member } };
        case 'ensemble_done':
          return { ...state, ensemble: e.result };
        case 'rag_done':
          return { ...state, rag: e.result };
        case 'adjudicate_token':
          return { ...state, adjudicationText: state.adjudicationText + e.token };
        case 'adjudicate_done':
          return { ...state, adjudication: e.result };
        case 'halted':
          return { ...state, halted: { reason: e.reason, stage: e.stage }, status: 'halted' };
        case 'error':
          return { ...state, errors: { ...state.errors, [e.stage]: e.message } };
        default:
          return state;
      }
    }
    default:
      return state;
  }
}

export interface UsePipeline {
  state: RunState;
  analyze: (image: Blob, imageName: string) => Promise<PipelineRun>;
  reset: () => void;
}

export function usePipeline(): UsePipeline {
  const [state, dispatch] = useReducer(reducer, undefined, initialState);

  const analyze = useCallback(async (image: Blob, imageName: string): Promise<PipelineRun> => {
    dispatch({ kind: 'start' });
    const run = await runPipeline(image, imageName, settingsStore.get(), (event) =>
      dispatch({ kind: 'event', event }),
    );
    dispatch({ kind: 'finish', run });

    // Persist to case history (left rail).
    await addHistory({
      id: run.id || shortId(),
      imageName,
      blob: image,
      verdict: run.adjudication?.verdict ?? null,
      confidence: run.adjudication?.confidence ?? null,
      createdAt: run.createdAt,
      run,
    });
    return run;
  }, []);

  const reset = useCallback(() => dispatch({ kind: 'reset' }), []);

  return { state, analyze, reset };
}
