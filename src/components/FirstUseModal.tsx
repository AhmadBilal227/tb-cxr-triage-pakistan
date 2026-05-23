import { useState } from 'react';
import { Dialog, DialogContent, DialogTitle, DialogDescription } from './ui/dialog';
import { Button } from './ui/button';
import { settingsStore, useSettings } from '@/store/settings';

/** Blocking first-use modal. Cannot be dismissed without explicit confirmation. */
export function FirstUseModal(): JSX.Element {
  const settings = useSettings();
  const [checked, setChecked] = useState(false);

  return (
    <Dialog open={!settings.acceptedDisclaimer}>
      <DialogContent hideClose onEscapeKeyDown={(e) => e.preventDefault()} onPointerDownOutside={(e) => e.preventDefault()}>
        <DialogTitle>Research-only confirmation</DialogTitle>
        <DialogDescription className="mt-2">
          This tool is a research preview that calls third-party AI models directly from your
          browser. It is <strong className="text-offwhite">not a medical device</strong> and must not
          be used to make clinical decisions.
        </DialogDescription>

        <label className="mt-4 flex items-start gap-2 text-sm text-offwhite">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => setChecked(e.target.checked)}
            className="mt-0.5 h-4 w-4 accent-provider-openai"
          />
          <span>
            I confirm research-only use, and that no clinical decisions will be made from this
            output.
          </span>
        </label>

        <div className="mt-5 flex justify-end">
          <Button
            disabled={!checked}
            onClick={() => settingsStore.set({ acceptedDisclaimer: true })}
          >
            Continue
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
