import { AlertTriangle } from 'lucide-react';

/** Persistent, non-dismissable research-use banner. */
export function SafetyBanner(): JSX.Element {
  return (
    <div
      role="alert"
      className="flex items-center justify-center gap-2 border-b border-verdict-uncertain/30 bg-verdict-uncertain/10 px-4 py-1.5 text-center text-xs text-verdict-uncertain"
    >
      <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
      <span>Research preview. Not a medical device. Not for diagnostic use.</span>
    </div>
  );
}
