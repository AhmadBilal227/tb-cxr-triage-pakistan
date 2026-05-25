import { useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';

type SetOpen = (next: boolean | ((prev: boolean) => boolean)) => void;

/**
 * Bind a boolean overlay state to a URL search param so the browser back
 * button closes the overlay instead of exiting the SPA. Drop-in for
 * `useState<boolean>(false)`.
 *
 * Opening pushes a new history entry (back closes the overlay). Closing
 * replaces the current entry (no junk entry left in the stack).
 */
export function useUrlBoundOverlay(
  key: string,
  value = '1',
): readonly [boolean, SetOpen] {
  const [params, setParams] = useSearchParams();
  const open = params.get(key) === value;
  const setOpen: SetOpen = useCallback(
    (next) => {
      const nextOpen =
        typeof next === 'function' ? next(params.get(key) === value) : next;
      const p = new URLSearchParams(params);
      if (nextOpen) p.set(key, value);
      else p.delete(key);
      setParams(p, { replace: !nextOpen });
    },
    [params, setParams, key, value],
  );
  return [open, setOpen] as const;
}
