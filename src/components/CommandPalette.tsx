import { Command } from 'cmdk';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { FilePlus2, FolderUp, BarChart3, Settings as SettingsIcon, Download } from 'lucide-react';
import type { ComponentType } from 'react';

export interface PaletteAction {
  id: string;
  label: string;
  hint?: string;
  icon: ComponentType<{ className?: string }>;
  onSelect: () => void;
}

export function buildActions(handlers: {
  newCase: () => void;
  importLabeled: () => void;
  validate: () => void;
  settings: () => void;
  exportSession: () => void;
}): PaletteAction[] {
  return [
    { id: 'new', label: 'New case', hint: 'Clear and drop a new X-ray', icon: FilePlus2, onSelect: handlers.newCase },
    { id: 'import', label: 'Import labeled set', hint: 'CSV + images for RAG corpus', icon: FolderUp, onSelect: handlers.importLabeled },
    { id: 'validate', label: 'Validate (holdout)', hint: 'Run metrics on a labeled set', icon: BarChart3, onSelect: handlers.validate },
    { id: 'settings', label: 'Settings', hint: 'BYOK + model overrides', icon: SettingsIcon, onSelect: handlers.settings },
    { id: 'export', label: 'Export session', hint: 'JSON with full audit trail', icon: Download, onSelect: handlers.exportSession },
  ];
}

export function CommandPalette({
  open,
  onOpenChange,
  actions,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  actions: PaletteAction[];
}): JSX.Element {
  return (
    <Command.Dialog
      open={open}
      onOpenChange={onOpenChange}
      label="Command palette"
      className="fixed left-1/2 top-[20%] z-50 w-full max-w-md -translate-x-1/2 overflow-hidden rounded-lg border border-border bg-surface shadow-2xl"
    >
      {/* Radix Dialog (rendered by cmdk) requires a Title + Description for screen readers. */}
      <DialogPrimitive.Title className="sr-only">Command palette</DialogPrimitive.Title>
      <DialogPrimitive.Description className="sr-only">
        Search and run an action: new case, import labeled set, validate, settings, export.
      </DialogPrimitive.Description>
      <Command.Input
        placeholder="Type a command…"
        className="w-full border-b border-border bg-transparent px-4 py-3 text-sm text-offwhite outline-none placeholder:text-muted"
      />
      <Command.List className="max-h-72 overflow-y-auto scroll-thin p-2">
        <Command.Empty className="px-3 py-4 text-center text-sm text-muted">No commands.</Command.Empty>
        {actions.map((a) => (
          <Command.Item
            key={a.id}
            value={a.label}
            onSelect={() => {
              onOpenChange(false);
              a.onSelect();
            }}
            className="flex cursor-pointer items-center gap-3 rounded-md px-3 py-2 text-sm text-offwhite aria-selected:bg-surface-2"
          >
            <a.icon className="h-4 w-4 text-muted" />
            <span className="flex-1">{a.label}</span>
            {a.hint && <span className="text-[10px] text-muted">{a.hint}</span>}
          </Command.Item>
        ))}
      </Command.List>
    </Command.Dialog>
  );
}
