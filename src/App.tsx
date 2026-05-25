import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  Command as CommandIcon,
  History as HistoryIcon,
  Settings as SettingsIcon,
  ShieldAlert,
} from 'lucide-react';
import { usePipeline } from '@/hooks/usePipeline';
import { useUrlBoundOverlay } from '@/hooks/useUrlBoundOverlay';
import { useSettings } from '@/store/settings';
import { embedWithFallback } from '@/lib/providers/classify';
import { blobToDataURL } from '@/lib/utils';
import { addLabeledCase, getHistory, listHistory } from '@/lib/db';
import { importLabeledSet, type ImportProgress } from '@/lib/labeledSet';
import { buildSessionExport, downloadJSON } from '@/lib/export';
import type { CaseHistoryRecord } from '@/lib/types';

import { SafetyBanner } from '@/components/SafetyBanner';
import { NoKeysBanner } from '@/components/NoKeysBanner';
import { FirstUseModal } from '@/components/FirstUseModal';
import { LeftRail } from '@/components/LeftRail';
import { HistoryList } from '@/components/HistoryList';
import { VerdictSummaryBar } from '@/components/VerdictSummaryBar';
import { DropCanvas, type SampleEntry } from '@/components/DropCanvas';
import { SettingsDrawer } from '@/components/SettingsDrawer';
import { Dialog, LeftDrawerContent, DialogTitle } from '@/components/ui/dialog';
import { CommandPalette, buildActions } from '@/components/CommandPalette';
import { makeSampleCXR } from '@/lib/sample';
import { Button } from '@/components/ui/button';

// Code-split the result subtree. VerdictCard pulls in the entire details/
// stack (heatmap, zonal bars, pathology list, clinician report, secondary
// observations, image lightbox); none of it is needed until an analysis
// completes. AgentTrace only renders behind the ?trace toggle. Splitting both
// keeps the first-paint bundle under the 500 kB warning floor.
const VerdictCard = lazy(() =>
  import('@/components/VerdictCard').then((m) => ({ default: m.VerdictCard })),
);
const AgentTrace = lazy(() =>
  import('@/components/AgentTrace').then((m) => ({ default: m.AgentTrace })),
);

interface ActiveImage {
  blob: Blob;
  url: string;
  name: string;
}

export default function App(): JSX.Element {
  const navigate = useNavigate();
  const settings = useSettings();
  const { state, analyze, reset } = usePipeline();

  const [image, setImage] = useState<ActiveImage | null>(null);
  // M24 — eagerly convert the active blob to a data URL once. The ClinicianReport
  // CTA in VerdictCard hands the data URL straight to gpt-5.5 vision via the
  // Responses API; doing the conversion here keeps the component synchronous
  // and the conversion off the click path (~ms latency on a typical CXR).
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!image) {
      setImageDataUrl(null);
      return;
    }
    let cancelled = false;
    blobToDataURL(image.blob)
      .then((url) => {
        if (!cancelled) setImageDataUrl(url);
      })
      .catch(() => {
        if (!cancelled) setImageDataUrl(null);
      });
    return () => {
      cancelled = true;
    };
  }, [image]);
  const [leftOpen, setLeftOpen] = useState(true);
  // Overlays are URL-bound (?settings=1, ?palette=1, ?trace=1) so the
  // browser back button closes them instead of exiting the SPA. FirstUseModal
  // stays unbound on purpose, it's a consent gate.
  const [settingsOpen, setSettingsOpen] = useUrlBoundOverlay('settings');
  const [paletteOpen, setPaletteOpen] = useUrlBoundOverlay('palette');
  const [traceOpen, setTraceOpen] = useUrlBoundOverlay('trace');
  // Mobile-only history sheet (below md, where the desktop rail is hidden).
  const [historyOpen, setHistoryOpen] = useUrlBoundOverlay('history');
  // Lightbox state lifted from VerdictCard so it can be URL-bound here in
  // App (where the Router context lives) without forcing the existing
  // VerdictCard tests to wrap in MemoryRouter.
  const [lightboxOpen, setLightboxOpen] = useUrlBoundOverlay('lightbox');
  // Loaded case is also URL-bound (?case=<historyId>) so cases are
  // refresh-survivable and shareable. The effect below loads from
  // IndexedDB when the param is present; absent means "in-progress
  // case from a drop/sample" which lives in local state only.
  const [searchParams, setSearchParams] = useSearchParams();
  const urlCaseId = searchParams.get('case');
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleEntry[]>([]);
  // Collapse the findings card to a sticky summary bar so the viewer owns
  // the canvas. Resets to expanded whenever a new analysis starts.
  const [verdictCollapsed, setVerdictCollapsed] = useState(false);

  // Load the demo-sample manifest (real public-dataset CXRs in public/samples/).
  useEffect(() => {
    fetch('/samples/manifest.json')
      .then((r) => (r.ok ? r.json() : []))
      .then((data: SampleEntry[]) => setSamples(Array.isArray(data) ? data : []))
      .catch(() => setSamples([]));
  }, []);

  const imageInputRef = useRef<HTMLInputElement>(null);
  const labeledInputRef = useRef<HTMLInputElement>(null);

  // Cmd-K palette
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setPaletteOpen]);

  const startAnalysis = useCallback(
    async (blob: Blob, name: string) => {
      const url = URL.createObjectURL(blob);
      setImage((prev) => {
        if (prev) URL.revokeObjectURL(prev.url);
        return { blob, url, name };
      });
      setStatus(null);
      setVerdictCollapsed(false);
      try {
        await analyze(blob, name);
      } catch (err) {
        setStatus(`Pipeline error: ${(err as Error).message}`);
      }
    },
    [analyze],
  );

  const handleFiles = useCallback(
    (files: FileList | File[] | null) => {
      if (!files) return;
      const arr = Array.from(files);
      const img = arr.find((f) => f.type.startsWith('image/'));
      if (img) void startAnalysis(img, img.name);
    },
    [startAnalysis],
  );

  // Global drag-and-drop (drop anywhere)
  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles],
  );

  const onSample = useCallback(async () => {
    const { blob, name } = await makeSampleCXR();
    void startAnalysis(blob, name);
  }, [startAnalysis]);

  const onPickSample = useCallback(
    async (file: string) => {
      try {
        const res = await fetch(`/samples/${file}`);
        const blob = await res.blob();
        void startAnalysis(blob, file);
      } catch (err) {
        setStatus(`Could not load sample: ${(err as Error).message}`);
      }
    },
    [startAnalysis],
  );

  const onNewCase = useCallback(() => {
    setImage((prev) => {
      if (prev) URL.revokeObjectURL(prev.url);
      return null;
    });
    reset();
    setStatus(null);
    // Clear ?case=<id> so the URL stops claiming a history record is active.
    if (searchParams.get('case')) {
      const next = new URLSearchParams(searchParams);
      next.delete('case');
      setSearchParams(next, { replace: true });
    }
  }, [reset, searchParams, setSearchParams]);

  const onLabeledFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;
      setStatus('Importing labeled set…');
      try {
        const summary = await importLabeledSet(
          Array.from(files),
          settings,
          (p: ImportProgress) => setStatus(`Importing ${p.done}/${p.total}: ${p.current}`),
        );
        setStatus(
          `Imported ${summary.imported} cases (${summary.embedded} embedded, ${summary.skippedNoMatch} unmatched, ${summary.failed} embed-failed).` +
            (summary.embeddingConfigured ? '' : ' No embedding provider — retrieval stays disabled.'),
        );
      } catch (err) {
        setStatus(`Import failed: ${(err as Error).message}`);
      }
    },
    [settings],
  );

  const onExport = useCallback(async () => {
    const history = await listHistory(200);
    if (history.length === 0) {
      setStatus('Nothing to export yet.');
      return;
    }
    downloadJSON(buildSessionExport(history.map((h) => h.run)), `tb-triage-session-${Date.now()}.json`);
    setStatus(`Exported ${history.length} run(s).`);
  }, []);

  const onDisagree = useCallback(
    async (label: 0 | 1) => {
      if (!image) return;
      let embedding: number[] | null = null;
      let provider: 'replicate' | null = null;
      try {
        const e = await embedWithFallback(image.blob, {
          replicateToken: settings.replicateToken,
          replicateModel: settings.overrides.embeddingReplicate,
          replicateVersion: settings.overrides.embeddingReplicateVersion,
        });
        embedding = e.embedding;
        provider = e.provider_used;
      } catch {
        // store without embedding if no provider — still grows the labeled corpus
      }
      await addLabeledCase({
        filename: image.name,
        blob: image.blob,
        embedding,
        embedding_provider: provider,
        label,
        source: 'feedback',
      });
      setStatus(`Added correction to corpus as ${label === 1 ? 'TB' : 'NO TB'}.`);
    },
    [image, settings],
  );

  // History selection is URL-driven: clicking a thumbnail pushes ?case=<id>,
  // and the effect below loads the record from IndexedDB and rehydrates the
  // image. This also handles fresh loads of a shared ?case=<id> link.
  const onSelectHistory = useCallback(
    (rec: CaseHistoryRecord) => {
      if (searchParams.get('case') === rec.id) return;
      const next = new URLSearchParams(searchParams);
      next.set('case', rec.id);
      setSearchParams(next);
    },
    [searchParams, setSearchParams],
  );

  useEffect(() => {
    if (!urlCaseId) return;
    let cancelled = false;
    void getHistory(urlCaseId).then((rec) => {
      if (cancelled || !rec) return;
      const url = URL.createObjectURL(rec.blob);
      setImage((prev) => {
        if (prev) URL.revokeObjectURL(prev.url);
        return { blob: rec.blob, url, name: rec.imageName };
      });
      setStatus(`Loaded ${rec.imageName} from history (read-only — re-drop to re-run).`);
    });
    return () => {
      cancelled = true;
    };
  }, [urlCaseId]);

  const actions = useMemo(
    () =>
      buildActions({
        newCase: onNewCase,
        importLabeled: () => labeledInputRef.current?.click(),
        validate: () => navigate('/validate'),
        settings: () => setSettingsOpen(true),
        exportSession: () => void onExport(),
      }),
    [onNewCase, navigate, onExport],
  );

  // Validated-model evidence that drives the in-canvas viewer overlays
  // (BoxEvidence heatmap + per-zone chips). Populated on the local-mode
  // pathway; undefined on VLM-primary, where the toggles stay disabled.
  const enrichment = state.adjudication?.local_enrichment;
  const overlaysReady = Boolean(state.adjudication && !state.halted && enrichment);

  return (
    <div
      className="flex h-screen flex-col bg-ink"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <NoKeysBanner onOpenSettings={() => setSettingsOpen(true)} />
      <SafetyBanner />
      <FirstUseModal />

      {/* Header */}
      <header className="flex items-center justify-between border-b border-border px-4 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setHistoryOpen(true)}
            aria-label="Open history"
          >
            <HistoryIcon className="h-4 w-4" />
          </Button>
          <ShieldAlert className="h-4 w-4 shrink-0 text-verdict-tb" />
          <span className="text-sm font-semibold tracking-tight">TB Triage</span>
          <span className="hidden font-mono text-[10px] text-muted sm:inline">research preview</span>
        </div>
        <div className="flex items-center gap-1 sm:gap-2">
          {state.run && state.run.fallbackRate > 0 && (
            <span className="hidden font-mono text-[10px] text-provider-replicate sm:inline">
              fallback {(state.run.fallbackRate * 100).toFixed(0)}%
            </span>
          )}
          <Button variant="ghost" size="sm" onClick={() => navigate('/validate')} aria-label="Validate">
            <BarChart3 className="h-3.5 w-3.5" /> <span className="hidden sm:inline">Validate</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setTraceOpen(!traceOpen)}
            aria-pressed={traceOpen}
            aria-label={traceOpen ? 'Hide trace' : 'Show trace'}
            // The trace panel only renders at lg+; hide its toggle below lg
            // so it isn't a dead button on tablet / mobile.
            className={`hidden lg:inline-flex ${traceOpen ? 'text-provider-openai' : ''}`}
          >
            <Activity className="h-3.5 w-3.5" /> <span className="hidden sm:inline">Trace</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setPaletteOpen(true)} aria-label="Command palette">
            <CommandIcon className="h-3.5 w-3.5" /> <span className="hidden sm:inline">K</span>
          </Button>
          <Button variant="ghost" size="icon" onClick={() => setSettingsOpen(true)} aria-label="Settings">
            <SettingsIcon className="h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Main */}
      <div className="flex flex-1 overflow-hidden">
        <LeftRail open={leftOpen} onToggle={() => setLeftOpen((v) => !v)} onSelect={onSelectHistory} />

        <main className="relative flex flex-1 flex-col overflow-y-auto">
          <div className="min-h-[40vh] flex-1 overflow-hidden">
            <DropCanvas
              imageUrl={image?.url ?? null}
              samples={samples}
              onBrowse={() => imageInputRef.current?.click()}
              onSample={() => void onSample()}
              onPickSample={(f) => void onPickSample(f)}
              boxGrid={enrichment?.box_evidence_grid ?? null}
              zonalScores={enrichment?.zonal_scores ?? null}
              cropBox={enrichment?.crop_box ?? null}
              overlaysReady={overlaysReady}
            />
          </div>

          {state.halted && (
            <div className="mx-6 mb-3 rounded-lg border border-verdict-uncertain/40 bg-verdict-uncertain/10 p-3 text-sm text-verdict-uncertain">
              Pipeline halted at <span className="font-mono">{state.halted.stage}</span>: {state.halted.reason}
            </div>
          )}

          {state.adjudication && !state.halted && !verdictCollapsed && (
            <div className="mx-6 mb-4">
              <Suspense
                fallback={<div className="skeleton h-40 w-full rounded-xl" aria-label="Loading verdict" />}
              >
                <VerdictCard
                  adjudication={state.adjudication}
                  ensemble={state.ensemble}
                  rag={state.rag}
                  fallbackRate={state.run?.fallbackRate ?? 0}
                  onDisagree={onDisagree}
                  onOpenSettings={() => setSettingsOpen(true)}
                  imageDataUrl={imageDataUrl ?? undefined}
                  openaiKey={settings.openaiKey}
                  primaryModel={settings.models.adjudicator}
                  fallbackModel={settings.models.adjudicatorFallback}
                  lightboxOpen={lightboxOpen}
                  onLightboxOpenChange={setLightboxOpen}
                  onCollapse={() => setVerdictCollapsed(true)}
                />
              </Suspense>
            </div>
          )}

          {state.adjudication && !state.halted && verdictCollapsed && (
            <VerdictSummaryBar
              adjudication={state.adjudication}
              onExpand={() => setVerdictCollapsed(false)}
            />
          )}

          {status && (
            <div className="absolute bottom-2 left-3 max-w-md rounded-md border border-border bg-surface px-3 py-1.5 text-[11px] text-muted shadow-lg">
              {status}
            </div>
          )}

          {dragging && (
            <div className="pointer-events-none absolute inset-0 z-10 m-3 rounded-xl border-2 border-dashed border-provider-openai bg-provider-openai/5" />
          )}
        </main>

        {traceOpen && (
          <aside className="hidden w-96 shrink-0 border-l border-border bg-surface lg:block">
            <Suspense fallback={<div className="skeleton m-3 h-24 rounded-lg" aria-label="Loading trace" />}>
              <AgentTrace state={state} />
            </Suspense>
          </aside>
        )}
      </div>

      {/* Hidden inputs */}
      <input
        ref={imageInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <input
        ref={labeledInputRef}
        type="file"
        multiple
        // @ts-expect-error non-standard but widely supported folder picker
        webkitdirectory=""
        className="hidden"
        onChange={(e) => void onLabeledFiles(e.target.files)}
      />

      <SettingsDrawer open={settingsOpen} onOpenChange={setSettingsOpen} />
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} actions={actions} />

      {/* Mobile-only history sheet. The desktop LeftRail is hidden below md;
          this off-canvas drawer keeps case history reachable on phones
          (adapt.md: never hide core functionality on mobile). */}
      <Dialog open={historyOpen} onOpenChange={setHistoryOpen}>
        <LeftDrawerContent aria-describedby={undefined}>
          <div className="flex items-center gap-2 border-b border-border px-4 py-3">
            <HistoryIcon className="h-4 w-4 text-muted" />
            <DialogTitle className="text-sm font-semibold tracking-tight">History</DialogTitle>
          </div>
          <HistoryList
            expanded
            onSelect={(rec) => {
              onSelectHistory(rec);
              setHistoryOpen(false);
            }}
          />
        </LeftDrawerContent>
      </Dialog>
    </div>
  );
}
