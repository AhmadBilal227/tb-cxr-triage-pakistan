import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Command as CommandIcon, Settings as SettingsIcon, ShieldAlert } from 'lucide-react';
import { usePipeline } from '@/hooks/usePipeline';
import { useSettings } from '@/store/settings';
import { parseBoxes } from '@/lib/providers/parsers';
import { embedWithFallback } from '@/lib/providers/classify';
import { addLabeledCase, listHistory } from '@/lib/db';
import { importLabeledSet, type ImportProgress } from '@/lib/labeledSet';
import { buildSessionExport, downloadJSON } from '@/lib/export';
import type { CaseHistoryRecord } from '@/lib/types';

import { SafetyBanner } from '@/components/SafetyBanner';
import { FirstUseModal } from '@/components/FirstUseModal';
import { LeftRail } from '@/components/LeftRail';
import { DropCanvas, type SampleEntry } from '@/components/DropCanvas';
import { AgentTrace } from '@/components/AgentTrace';
import { VerdictCard } from '@/components/VerdictCard';
import { SettingsDrawer } from '@/components/SettingsDrawer';
import { CommandPalette, buildActions } from '@/components/CommandPalette';
import { makeSampleCXR } from '@/lib/sample';
import { Button } from '@/components/ui/button';

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
  const [leftOpen, setLeftOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleEntry[]>([]);

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
  }, []);

  const startAnalysis = useCallback(
    async (blob: Blob, name: string) => {
      const url = URL.createObjectURL(blob);
      setImage((prev) => {
        if (prev) URL.revokeObjectURL(prev.url);
        return { blob, url, name };
      });
      setStatus(null);
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
  }, [reset]);

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
      let provider: 'hf' | 'replicate' | null = null;
      try {
        const e = await embedWithFallback(image.blob, {
          hfToken: settings.hfToken,
          replicateToken: settings.replicateToken,
          endpointUrl: settings.overrides.embeddingEndpointUrl,
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

  const onSelectHistory = useCallback((rec: CaseHistoryRecord) => {
    const url = URL.createObjectURL(rec.blob);
    setImage({ blob: rec.blob, url, name: rec.imageName });
    setStatus(`Loaded ${rec.imageName} from history (read-only — re-drop to re-run).`);
  }, []);

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

  const boxes = useMemo(
    () => (state.members.general?.raw != null ? parseBoxes(state.members.general.raw) : []),
    [state.members.general],
  );

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
      <SafetyBanner />
      <FirstUseModal />

      {/* Header */}
      <header className="flex items-center justify-between border-b border-border px-4 py-2">
        <div className="flex items-center gap-2">
          <ShieldAlert className="h-4 w-4 text-verdict-tb" />
          <span className="text-sm font-semibold tracking-tight">TB Triage</span>
          <span className="font-mono text-[10px] text-muted">research preview</span>
        </div>
        <div className="flex items-center gap-2">
          {state.run && state.run.fallbackRate > 0 && (
            <span className="font-mono text-[10px] text-provider-replicate">
              fallback {(state.run.fallbackRate * 100).toFixed(0)}%
            </span>
          )}
          <Button variant="ghost" size="sm" onClick={() => setPaletteOpen(true)}>
            <CommandIcon className="h-3.5 w-3.5" /> K
          </Button>
          <Button variant="ghost" size="icon" onClick={() => setSettingsOpen(true)} aria-label="Settings">
            <SettingsIcon className="h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Main */}
      <div className="flex flex-1 overflow-hidden">
        <LeftRail open={leftOpen} onToggle={() => setLeftOpen((v) => !v)} onSelect={onSelectHistory} />

        <main className="relative flex flex-1 flex-col overflow-hidden">
          <div className="flex-1 overflow-hidden">
            <DropCanvas
              imageUrl={image?.url ?? null}
              boxes={boxes}
              samples={samples}
              onBrowse={() => imageInputRef.current?.click()}
              onSample={() => void onSample()}
              onPickSample={(f) => void onPickSample(f)}
            />
          </div>

          {state.halted && (
            <div className="mx-6 mb-3 rounded-lg border border-verdict-uncertain/40 bg-verdict-uncertain/10 p-3 text-sm text-verdict-uncertain">
              Pipeline halted at <span className="font-mono">{state.halted.stage}</span>: {state.halted.reason}
            </div>
          )}

          {state.adjudication && !state.halted && (
            <div className="mx-6 mb-4">
              <VerdictCard
                adjudication={state.adjudication}
                ensemble={state.ensemble}
                rag={state.rag}
                fallbackRate={state.run?.fallbackRate ?? 0}
                onDisagree={onDisagree}
              />
            </div>
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

        <aside className="w-96 shrink-0 border-l border-border bg-surface">
          <AgentTrace state={state} />
        </aside>
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
    </div>
  );
}
