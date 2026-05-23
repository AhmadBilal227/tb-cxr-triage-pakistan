import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

/** Horizontal probability gauge for a single tb_prob value (0..1). */
export function TbGauge({ value, label }: { value: number | null; label?: string }): JSX.Element {
  const pct = value === null ? 0 : Math.round(value * 100);
  const color =
    value === null
      ? 'bg-muted'
      : value >= 0.66
        ? 'bg-verdict-tb'
        : value >= 0.4
          ? 'bg-verdict-uncertain'
          : 'bg-verdict-clear';
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-muted">{label ?? 'tb_prob'}</span>
        <span className="font-mono tabular-nums text-offwhite">
          {value === null ? '—' : value.toFixed(3)}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3" aria-hidden>
        <motion.div
          className={cn('h-full rounded-full', color)}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}
