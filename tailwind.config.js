/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
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
      },
      animation: {
        shimmer: 'shimmer 1.6s ease-in-out infinite',
        'fade-in': 'fade-in 0.3s ease-out',
      },
    },
  },
  plugins: [],
};
