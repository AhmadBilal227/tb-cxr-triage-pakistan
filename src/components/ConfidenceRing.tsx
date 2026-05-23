import { motion } from 'framer-motion';

/** Animated SVG confidence ring for the verdict card. */
export function ConfidenceRing({
  value,
  color,
  size = 120,
}: {
  value: number; // 0..100
  color: string; // hex
  size?: number;
}): JSX.Element {
  const stroke = 8;
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - value / 100);

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Confidence ${value} percent`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#242424" strokeWidth={stroke} />
      <motion.circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={circumference}
        initial={{ strokeDashoffset: circumference }}
        animate={{ strokeDashoffset: offset }}
        transition={{ duration: 0.9, ease: 'easeOut' }}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text
        x="50%"
        y="50%"
        textAnchor="middle"
        dominantBaseline="central"
        className="fill-offwhite font-mono"
        style={{ fontSize: size * 0.24 }}
      >
        {Math.round(value)}
      </text>
    </svg>
  );
}
