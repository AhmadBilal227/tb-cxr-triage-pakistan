import { useCallback, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';

type SetOpen = (next: boolean | ((prev: boolean) => boolean)) => void;

/**
 * Bind a boolean overlay state to a URL search param so the browser back
 * button closes the overlay instead of exiting the SPA. Drop-in for
 * `useState<boolean>(false)`.
 *
 * Opening pushes a new history entry (back closes the overlay). Closing
 * replaces the current entry (no junk entry left in the stack).
 *
 * IMPLEMENTATION NOTE — setOpen reads `params` via a ref instead of a
 * closure, so the callback identity is stable across every URL change.
 * Without this, consumers like `App.tsx`'s Cmd-K keydown effect would have
 * to re-register their listener on every params mutation. The ref pattern
 * keeps `setOpen` referentially stable per (key, value, setParams) pair.
 */
export function useUrlBoundOverlay(
  key: string,
  value = '1',
): readonly [boolean, SetOpen] {
  const [params, setParams] = useSearchParams();
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const open = params.get(key) === value;
  const setOpen: SetOpen = useCallback(
    (next) => {
      const current = paramsRef.current;
      const nextOpen =
        typeof next === 'function' ? next(current.get(key) === value) : next;
      const p = new URLSearchParams(current);
      if (nextOpen) p.set(key, value);
      else p.delete(key);
      setParams(p, { replace: !nextOpen });
    },
    [setParams, key, value],
  );
  return [open, setOpen] as const;
}
