import { useCallback, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { jsPDF } from 'jspdf';
import { ArrowLeft, Play, Download, FileText } from 'lucide-react';
import { runPipeline } from '@/lib/pipeline/orchestrator';
import { parseLabelCsv } from '@/lib/labeledSet';
import { computeMetrics, type Metrics, type ValItem } from '@/lib/metrics';
import { useSettings } from '@/store/settings';
import { settingsStore } from '@/store/settings';
import { DISCLAIMER, downloadJSON } from '@/lib/export';
import { SafetyBanner } from '@/components/SafetyBanner';
import { Button } from '@/components/ui/button';
import { fitCalibration } from '@/lib/calibration';
import type { CalibrationSample } from '@/lib/types';

interface HoldoutImage {
  filename: string;
  blob: File;
  label: 0 | 1;
}

export default function Validate(): JSX.Element {
  const settings = useSettings();
  const inputRef = useRef<HTMLInputElement>(null);

  const [holdout, setHoldout] = useState<HoldoutImage[]>([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [items, setItems] = useState<ValItem[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [samples, setSamples] = useState<CalibrationSample[]>([]);

  const onFiles = useCallback(async (files: FileList | null) => {
    if (!files) return;
    const arr = Array.from(files);
    const csv = arr.find((f) => f.name.toLowerCase().endsWith('.csv'));
    if (!csv) {
      setMsg('No CSV found. Include a filename,label CSV with the images.');
      return;
    }
    const labels = parseLabelCsv(await csv.text());
    const imgs = arr.filter((f) => /\.(png|jpe?g|webp|bmp)$/i.test(f.name));
    const byName = new Map(imgs.map((f) => [f.name.split(/[\\/]/).pop() ?? f.name, f]));
    const built: HoldoutImage[] = [];
    for (const [name, label] of labels) {
      const file = byName.get(name);
      if (file) built.push({ filename: name, blob: file, label });
    }
    setHoldout(built);
    setItems([]);
    setMetrics(null);
    setMsg(`Loaded ${built.length} labeled images (${labels.size - built.length} unmatched).`);
  }, []);

  const run = useCallback(async () => {
    if (holdout.length === 0) return;
    setRunning(true);
    setProgress({ done: 0, total: holdout.length });
    const collected: ValItem[] = [];
    const sampleAcc: CalibrationSample[] = [];
    for (let i = 0; i < holdout.length; i++) {
      const h = holdout[i]!;
      try {
        const r = await runPipeline(h.blob, h.filename, settings, () => undefined);
        collected.push({
          filename: h.filename,
          trueLabel: h.label,
          verdict: r.adjudication?.verdict ?? null,
          score: r.ensemble?.weightedScore ?? null,
          abstained: r.adjudication?.verdict === 'abstain',
          halted: !!r.halted,
        });
        // Harvest per-member raw tb_prob for calibration.
        const mp: CalibrationSample['memberProbs'] = {};
        for (const m of r.ensemble?.members ?? []) {
          if (m.tb_prob !== null) mp[m.id] = m.tb_prob;
        }
        sampleAcc.push({
          filename: h.filename,
          label: h.label,
          memberProbs: mp,
          vlmUncertainty: r.ensemble?.members.find((m) => m.id === 'vlm')?.uncertainty ?? null,
        });
      } catch (err) {
        collected.push({
          filename: h.filename,
          trueLabel: h.label,
          verdict: null,
          score: null,
          abstained: false,
          halted: true,
          error: (err as Error).message,
        });
      }
      setProgress({ done: i + 1, total: holdout.length });
      setItems([...collected]);
    }
    setMetrics(computeMetrics(collected));
    setSamples([...sampleAcc]);
    setRunning(false);
  }, [holdout, settings]);

  const exportJSON = useCallback(() => {
    if (!metrics) return;
    downloadJSON(
      {
        disclaimer: DISCLAIMER,
        generatedAt: new Date().toISOString(),
        model_versions: {
          tb_classifier_replicate: settings.overrides.tbClassifierReplicate || '(none)',
          adjudicator: settings.models.adjudicator,
          embedding_replicate: settings.overrides.embeddingReplicate || '(none)',
        },
        metrics,
        items,
      },
      `tb-triage-validation-${Date.now()}.json`,
    );
  }, [metrics, items, settings]);

  const calibrate = useCallback(() => {
    if (samples.length === 0) return;
    const params = fitCalibration(samples);
    settingsStore.setCalibration(params);
    setMsg(
      `Calibrated on ${params.nSamples} cases. τ_low=${params.conformal.tauLow.toFixed(3)} τ_high=${params.conformal.tauHigh.toFixed(3)}${params.conformal.incomplete ? ' (insufficient per-class samples — band is conservative)' : ''}`,
    );
  }, [samples]);

  const exportPDF = useCallback(() => {
    if (!metrics) return;
    const doc = new jsPDF();
    let y = 16;
    const line = (t: string, size = 11): void => {
      doc.setFontSize(size);
      doc.text(t, 14, y);
      y += size * 0.6;
    };
    line('TB Triage — Validation Report', 16);
    y += 2;
    doc.setTextColor(150);
    line(DISCLAIMER, 8);
    doc.setTextColor(0);
    y += 4;
    line(`Generated: ${new Date().toISOString()}`, 9);
    line(`Adjudicator: ${settings.models.adjudicator}`, 9);
    line(`TB classifier (Replicate): ${settings.overrides.tbClassifierReplicate || '(none)'}`, 9);
    y += 4;
    line(`Total: ${metrics.total}   Decided: ${metrics.nDecided}   Abstain: ${metrics.nAbstain}   Halted: ${metrics.nHalted}`, 10);
    line(`Accuracy: ${fmt(metrics.accuracy)}   Sensitivity: ${fmt(metrics.sensitivity)}   Specificity: ${fmt(metrics.specificity)}`, 10);
    line(`AUC: ${Number.isNaN(metrics.auc) ? 'n/a' : metrics.auc.toFixed(3)}`, 10);
    y += 4;
    line('Confusion matrix (decided cases):', 11);
    line(`  TP ${metrics.confusion.tp}    FP ${metrics.confusion.fp}`, 10);
    line(`  FN ${metrics.confusion.fn}    TN ${metrics.confusion.tn}`, 10);
    doc.save(`tb-triage-validation-${Date.now()}.pdf`);
  }, [metrics, settings]);

  return (
    <div className="flex h-screen flex-col bg-ink text-offwhite">
      <SafetyBanner />
      <header className="flex items-center justify-between border-b border-border px-4 py-2">
        <Link to="/" className="flex items-center gap-2 text-sm text-muted hover:text-offwhite">
          <ArrowLeft className="h-4 w-4" /> Back to triage
        </Link>
        <span className="text-sm font-semibold">Validation</span>
      </header>

      <div className="flex-1 overflow-y-auto scroll-thin p-6">
        <div className="mx-auto max-w-3xl space-y-6">
          <section className="space-y-3">
            <p className="text-sm text-muted">
              Load a holdout labeled set (a <span className="font-mono">filename,label</span> CSV plus
              the images). Each image runs through the full pipeline; metrics are computed from the
              verdicts. Abstains and halted images are excluded from the binary confusion matrix but
              counted separately. AUC uses the ensemble weighted score.
            </p>
            <div className="flex items-center gap-3">
              <Button variant="outline" onClick={() => inputRef.current?.click()} disabled={running}>
                Choose holdout folder
              </Button>
              <Button onClick={() => void run()} disabled={running || holdout.length === 0}>
                <Play className="h-3.5 w-3.5" /> {running ? 'Running…' : `Run (${holdout.length})`}
              </Button>
              {metrics && (
                <>
                  <Button variant="ghost" onClick={exportJSON}>
                    <Download className="h-3.5 w-3.5" /> JSON
                  </Button>
                  <Button variant="ghost" onClick={exportPDF}>
                    <FileText className="h-3.5 w-3.5" /> PDF
                  </Button>
                </>
              )}
              <Button variant="outline" onClick={calibrate} disabled={samples.length === 0}>
                Calibrate from this set
              </Button>
              <Button
                variant="ghost"
                onClick={() => {
                  settingsStore.setCalibration(null);
                  setMsg('Calibration cleared (using default policy).');
                }}
              >
                Clear calibration
              </Button>
            </div>
            {msg && <p className="text-[11px] text-muted">{msg}</p>}
            <input
              ref={inputRef}
              type="file"
              multiple
              // @ts-expect-error folder picker
              webkitdirectory=""
              className="hidden"
              onChange={(e) => void onFiles(e.target.files)}
            />
          </section>

          {running || progress.total > 0 ? (
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] text-muted">
                <span>Progress</span>
                <span className="font-mono">{progress.done}/{progress.total}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-surface-3">
                <div
                  className="h-full bg-provider-openai transition-all"
                  style={{ width: `${progress.total ? (progress.done / progress.total) * 100 : 0}%` }}
                />
              </div>
            </div>
          ) : null}

          {metrics && (
            <section className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <Stat label="Accuracy" value={fmt(metrics.accuracy)} />
              <Stat label="Sensitivity" value={fmt(metrics.sensitivity)} />
              <Stat label="Specificity" value={fmt(metrics.specificity)} />
              <Stat label="AUC" value={Number.isNaN(metrics.auc) ? 'n/a' : metrics.auc.toFixed(3)} />
            </section>
          )}

          {metrics && (
            <section className="space-y-2">
              <h2 className="text-sm font-semibold">Confusion matrix (decided)</h2>
              <div className="grid w-64 grid-cols-3 gap-px overflow-hidden rounded-lg border border-border bg-border text-center text-sm">
                <Cell className="bg-surface text-muted" />
                <Cell className="bg-surface text-muted">Pred TB</Cell>
                <Cell className="bg-surface text-muted">Pred NEG</Cell>
                <Cell className="bg-surface text-muted">True TB</Cell>
                <Cell className="bg-surface-2 text-verdict-clear">{metrics.confusion.tp}</Cell>
                <Cell className="bg-surface-2 text-verdict-tb">{metrics.confusion.fn}</Cell>
                <Cell className="bg-surface text-muted">True NEG</Cell>
                <Cell className="bg-surface-2 text-verdict-tb">{metrics.confusion.fp}</Cell>
                <Cell className="bg-surface-2 text-verdict-clear">{metrics.confusion.tn}</Cell>
              </div>
              <p className="text-[11px] text-muted">
                {metrics.nAbstain} abstained · {metrics.nHalted} halted (excluded from matrix)
              </p>
            </section>
          )}

          {items.length > 0 && (
            <section className="space-y-1">
              <h2 className="text-sm font-semibold">Per-image</h2>
              <div className="overflow-hidden rounded-lg border border-border">
                <table className="w-full text-left text-[11px]">
                  <thead className="bg-surface-2 text-muted">
                    <tr>
                      <th className="px-2 py-1">file</th>
                      <th className="px-2 py-1">true</th>
                      <th className="px-2 py-1">verdict</th>
                      <th className="px-2 py-1">score</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    {items.map((it) => (
                      <tr key={it.filename} className="border-t border-border">
                        <td className="px-2 py-1">{it.filename}</td>
                        <td className="px-2 py-1">{it.trueLabel === 1 ? 'TB' : 'NEG'}</td>
                        <td className="px-2 py-1">{it.halted ? 'halted' : (it.verdict ?? '—')}</td>
                        <td className="px-2 py-1">{it.score === null ? '—' : it.score.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

function fmt(n: number): string {
  return Number.isNaN(n) ? 'n/a' : `${(n * 100).toFixed(1)}%`;
}

function Stat({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div className="rounded-lg border border-border bg-surface-2 p-3">
      <div className="text-[10px] uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 font-mono text-lg text-offwhite">{value}</div>
    </div>
  );
}

function Cell({ children, className = '' }: { children?: React.ReactNode; className?: string }): JSX.Element {
  return <div className={`px-2 py-2 ${className}`}>{children}</div>;
}
