/** @type {import('tailwindcss').Config} */
export default {
  // Dark-only by intent. `color-scheme: dark` is pinned in src/index.css.
  // If a light theme is ever added, switch to `darkMode: 'class'` here and
  // introduce `dark:` variants throughout — but ship light mode as a real
  // feature, not as a config-without-implementation artifact.
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Surfaces (dark-mode default per spec)
        ink: '#0A0A0A',
        surface: '#141414',
        'surface-2': '#1C1C1C',
        'surface-3': '#242424',
        offwhite: '#FAFAF7',
        border: '#2A2A2A',
        muted: '#8A8A85',

        // Verdict colors (exact per spec)
        'verdict-tb': '#C8102E', // TB SUSPECTED
        'verdict-clear': '#00754A', // NO TB
        'verdict-uncertain': '#F59E0B', // UNCERTAIN — REFER

        // Provider badges
        'provider-hf': '#00754A', // HF green
        'provider-replicate': '#F59E0B', // Replicate amber
        'provider-openai': '#6366F1', // orchestration

        // Stage status
        'status-queued': '#8A8A85',
        'status-running': '#6366F1',
        'status-fallback': '#F59E0B',
        'status-done': '#00754A',
        'status-error': '#C8102E',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        // Indeterminate progress: a segment slides across the track. Uses
        // transform only (composited, no layout thrash) per the design law
        // against animating layout properties.
        indeterminate: {
          '0%': { transform: 'translateX(-120%)' },
          '100%': { transform: 'translateX(320%)' },
        },
      },
      animation: {
        shimmer: 'shimmer 1.6s ease-in-out infinite',
        'fade-in': 'fade-in 0.3s ease-out',
        indeterminate: 'indeterminate 1.4s cubic-bezier(0.4, 0, 0.2, 1) infinite',
      },
    },
  },
  plugins: [],
};
