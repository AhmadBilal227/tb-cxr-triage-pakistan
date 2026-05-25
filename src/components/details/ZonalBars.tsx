/**
 * ZonalBars — M24 evidence panel #2.
 *
 * Horizontal bar chart of the 7 per-zone calibrated TB probabilities
 * (sigmoid(zone_logit / T)). Sorted descending so the dominant zone is first.
 * Bar color tracks the same viridis palette the BoxEvidenceHeatmap uses
 * (consistent semantic mapping across the evidence panels).
 *
 * Honest framing: zones are upper/mid/lower per hemithorax + hilar, NOT 8
 * zones. The trained ZonalSoftOR has N_ZONES=7 (`ZONE_NAMES = (RUZ, RMZ, RLZ,
 * LUZ, LMZ, LLZ, HILAR)` in extract_features.py); the user-facing labels here
 * translate that convention into clinical-register names. The hilar channel
 * is NOT split into L/R because the trained model does not compute that split.
 */
import type { ZoneKey, ZonalScores } from '@/lib/providers/localTriage';
import { viridis } from './colorScale';

const ZONE_LABELS: Record<ZoneKey, string> = {
  upper_l: 'Upper L',
  upper_r: 'Upper R',
  mid_l: 'Mid L',
  mid_r: 'Mid R',
  lower_l: 'Lower L',
  lower_r: 'Lower R',
  hilar: 'Hilar / Mediastinum',
};

export interface ZonalBarsProps {
  scores: ZonalScores;
}

export function ZonalBars({ scores }: ZonalBarsProps): JSX.Element {
  // Materialize an ordered list — desc by probability so the leading zone is at the top.
  const rows = (Object.entries(scores) as Array<[ZoneKey, number | undefined]>)
    .filter((kv): kv is [ZoneKey, number] => typeof kv[1] === 'number' && Number.isFinite(kv[1]))
    .sort((a, b) => b[1] - a[1]);

  return (
    <div data-testid="zonal-bars" className="space-y-1.5">
      <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
        per-zone calibrated TB probability (sigmoid(zone_logit / T))
      </div>
      {rows.length === 0 ? (
        <p className="text-[11px] text-muted/80">
          Zone evidence not available for this run (the lung segmenter returned an empty mask, or
          the engine ran with zero-zone supervision).
        </p>
      ) : (
        rows.map(([key, value]) => {
          const pct = Math.max(0, Math.min(1, value));
          const widthPct = Math.max(2, pct * 100); // min-width so the smallest bars stay visible
          const bg = viridis(pct);
          return (
            <div key={key} className="flex items-center gap-2">
              <div className="w-32 truncate text-[11px] text-offwhite/90">{ZONE_LABELS[key]}</div>
              <div
                className="relative h-3 flex-1 overflow-hidden rounded-sm border border-border bg-surface-2"
                data-testid={`zonal-bar-${key}`}
              >
                <div
                  className="h-full"
                  style={{ width: `${widthPct}%`, background: bg }}
                  title={`${ZONE_LABELS[key]}: ${value.toFixed(4)}`}
                />
              </div>
              <div
                className="w-14 text-right font-mono text-[10px] text-muted"
                data-testid={`zonal-value-${key}`}
              >
                {value.toFixed(2)}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
