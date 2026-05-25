---
name: TB Triage
description: Honest chest X-ray TB triage; engineering as evidence, not marketing.
colors:
  ink: "#0A0A0A"
  surface: "#141414"
  surface-2: "#1C1C1C"
  surface-3: "#242424"
  offwhite: "#FAFAF7"
  muted: "#8A8A85"
  border: "#2A2A2A"
  verdict-tb: "#C8102E"
  verdict-clear: "#00754A"
  verdict-uncertain: "#F59E0B"
  provider-openai: "#6366F1"
  provider-replicate: "#F59E0B"
typography:
  display:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "1.875rem"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "0.625rem"
    fontWeight: 500
    letterSpacing: "0.06em"
  mono:
    fontFamily: "\"IBM Plex Mono\", ui-monospace, monospace"
    fontSize: "0.625rem"
    fontWeight: 400
    lineHeight: 1.5
rounded:
  sm: "4px"
  md: "6px"
  lg: "8px"
  xl: "12px"
  full: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
components:
  button-primary:
    backgroundColor: "{colors.offwhite}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "8px 16px"
  button-outline:
    backgroundColor: "transparent"
    textColor: "{colors.offwhite}"
    rounded: "{rounded.md}"
    padding: "8px 16px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.offwhite}"
    rounded: "{rounded.md}"
    padding: "8px 16px"
  button-danger:
    backgroundColor: "{colors.verdict-tb}"
    textColor: "#FFFFFF"
    rounded: "{rounded.md}"
    padding: "8px 16px"
  surface-card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.xl}"
    padding: "16px"
  audit-row:
    typography: "{typography.mono}"
    textColor: "{colors.muted}"
---

# Design System: TB Triage

## 1. Overview

**Creative North Star: "The Reading Room After Hours."**

The aesthetic is a clinical second-read seat at the end of a long shift. Dim, focused, the screen is the brightest surface in the room. There is no marketing energy. No "welcome aboard". No illustrations of doctors smiling. The film is the subject; everything else is chrome that should disappear when the radiologist starts reading.

The visual language is built from three commitments. **Calm at rest, alarming only when meaningful** — the surfaces are neutral until a verdict lands, at which point the verdict color carries the message. **Audit data is typographically first-class** — the calibration constants, the model SHA, the threshold are rendered in IBM Plex Mono because they are the evidence the product wants to be judged on. **Disclosure beats decoration** — every safety-net rule, every fallback, every prompt hash gets surfaced in the UI; nothing is hidden behind a chevron or a developer console.

What this system explicitly rejects: the medical-AI marketing dashboard (gradient hero metric, glowing accuracy %, three-up feature cards), the consumer wellness aesthetic (pastel cards, smiling illustrations), the AI-copilot family (purple-to-indigo gradients, Sparkles-icon-everywhere, "Ask me anything"), and dark-mode neon maximalism (glowing borders, glassmorphism floating in space). Dark mode here is a working posture, not a costume.

**Key Characteristics:**
- Dark by default, near-black canvas (`#0A0A0A`), tuned to long reading sessions under moderate ambient light.
- Restrained color strategy. One accent (indigo) for interactivity and focus; the verdict triad (red / green / amber) reserved exclusively for verdicts and provider status; no decorative color anywhere else.
- Inter for UI, IBM Plex Mono for audit data — every monospace string carries semantic weight (calibration, IDs, timestamps, hashes).
- Flat by default. Cards have borders, not shadows. Shadows appear only on floating overlays where physical separation is genuinely required.
- Verdict severity is the only loud signal. The rest of the surface is quiet until the model has something to say.

## 2. Colors

A near-black canvas with one accent and a three-way verdict triad. The verdict colors are meaning-bearing, never decorative — they only appear on verdict-related surfaces.

### Primary

- **Surface Ink** (`#0A0A0A`): The page canvas. Nearly true black, but tinted very slightly cool (eye-rest at low ambient light without the over-pure black that strains contrast detection).
- **Off White** (`#FAFAF7`): Primary text. Tinted warm to soften the eye-strain reading sessions impose; never `#FFFFFF`.
- **Muted Slate** (`#8A8A85`): Secondary text, metadata, captions. Sits at ~4.55:1 against `surface` — the floor for AA on body-size text. Do NOT drop opacity below `muted/80` for any text intended to be readable.

### Verdict Triad (semantic, not decorative)

- **TB Suspected Red** (`#C8102E`): The TB verdict color. Also serves as danger / error. Used only on TB verdict surfaces, the danger button variant, and `status-error`.
- **Clear Green** (`#00754A`): The NO_TB verdict color. Also serves as `status-done` / `provider-hf`. Used only on negative verdict surfaces and successful-completion states.
- **Uncertain Amber** (`#F59E0B`): The ABSTAIN verdict color. Also serves as `provider-replicate` / `status-fallback`. Used on abstain verdicts and any "degraded but usable" state.

### Accent

- **Provider OpenAI Indigo** (`#6366F1`): The interactivity accent. Focus rings, primary CTA hover hints, the GPT model badge, the in-progress stage indicator. Used at ≤10% of any given screen, in line with the Restrained Color Strategy.

### Surfaces (tonal layering, not shadows)

- **Surface** (`#141414`): Elevated panels, the verdict card.
- **Surface 2** (`#1C1C1C`): Nested content within a panel (heatmap container, code blocks).
- **Surface 3** (`#242424`): Hover state, scroll-track thumb.
- **Border** (`#2A2A2A`): The default divider. Sits one step lighter than `surface-3` so panels read separated without ever needing a shadow.

### Named Rules

**The Verdict Color Reservation Rule.** The red / green / amber triad belongs exclusively to verdicts, danger / error states, and provider status. Never use them as decorative accents, gradient stops, chart colors, or "this thing is important" highlights. If you need to draw attention to a non-verdict element, use weight or scale, not color.

**The One Accent Rule.** Indigo (`#6366F1`) is the only interactivity color. It signals focus, hover hints, in-progress states, and the GPT model badge. It never exceeds ~10% of any screen. If you find yourself wanting a second accent for "another important thing", that thing is not as important as you think.

**The Tinted Neutral Rule.** No raw `#000` or `#FFF` anywhere. Every neutral has a tiny tint (`#0A0A0A` instead of `#000`, `#FAFAF7` instead of `#FFF`). True black on true white is for spec sheets, not interfaces.

## 3. Typography

**Display Font:** Inter (variable, with system-ui fallback).
**Body Font:** Inter.
**Label/Mono Font:** IBM Plex Mono (with ui-monospace fallback).

**Character:** Two voices, sharply separated by role. Inter handles every word a clinician reads. IBM Plex Mono handles every datum the engineering wants to be judged on. The mono is not decorative; if a string is in monospace, it is auditable.

### Hierarchy

- **Verdict Display** (700, `1.875rem` / `30px`, `leading-none`, `-0.01em`): The verdict label inside the verdict card. The single largest type on any screen.
- **Section Title** (600, `0.875rem` / `14px`, `leading-tight`, `-0.005em`): Header titles, modal titles, panel headers.
- **Body** (400, `0.875rem` / `14px`, `leading-relaxed`): The disclosure copy, the rationale paragraph, the findings prose inside the report.
- **Body Small** (400, `0.75rem` / `12px`, `leading-snug`): Captions under heatmaps, panel descriptions, inline help text.
- **Caption** (400, `0.6875rem` / `11px`, `leading-snug`): Secondary metadata, sub-points, list captions. Floor for body-text readability.
- **Label** (500, `0.625rem` / `10px`, `uppercase`, `tracking-wide`): Section-label tags. Always uppercase, always wide-tracked. Sets section context without competing with the content.
- **Mono Audit** (400, `0.625rem` / `10px`, mono, `text-muted`): Calibration constants, model SHA, prompt hash, timestamps, perception path. The evidence font.

### Named Rules

**The Monospace Reservation Rule.** IBM Plex Mono appears only on strings that have audit value: calibration constants (`T`, `T_seq`, `thr_at_95sens`), identifiers (model SHA, git SHA, prompt hash, schema version), perception path enum, timestamps, latency values, zone keys. Do not use mono for decorative effect. If a string is mono, a reader should be able to copy-paste it into a CSV and have it mean something.

**The Uppercase Label Rule.** Section labels (`technique`, `findings`, `impression`, `radiology report`, `clinician report ready`) are always uppercase, mono, `0.625rem`, `tracking-wide`, `text-muted`. They establish context without claiming the eye. Body type stays in mixed case; label type stays in uppercase.

**The Line-Length Cap.** Body copy never exceeds ~75 characters per line on any viewport. The radiology report modal is `max-w-2xl` for exactly this reason — comfortable reading width, not maximal width.

## 4. Elevation

This system is **flat by default**. Depth is conveyed through tonal layering (`ink` → `surface` → `surface-2` → `surface-3`), not shadows.

Shadows appear only on floating overlays where physical separation from the underlying canvas is required for sense-of-place: the Settings drawer, the Command Palette, the radiology-report modal, the image lightbox, the first-use disclaimer. These are the only places `shadow-2xl` is applied. Every other panel (verdict card, evidence panels, history rail, agent trace aside) uses a border on a tonal layer, not a shadow.

### Shadow Vocabulary

- **Overlay shadow** (Tailwind `shadow-2xl`): Settings drawer, Command Palette, full-screen modal content, FullscreenContent (lightbox + report). The only shadow class allowed.
- **No card shadows.** Cards have borders on `surface-2` or `surface-3` to read as elevated, not shadows.

### Named Rules

**The Flat-By-Default Rule.** Cards, panels, banners, and inline disclosures are flat at rest. They use a border on a tonal step. If you find yourself reaching for `shadow-md` to make a card "feel like a card", switch to a one-step tonal lift (`surface` → `surface-2`) instead.

**The Overlay-Only Shadow Rule.** Shadow is a state, not a style. Shadows appear only when content has visibly broken out of the document flow — drawers, modals, palettes. Nothing in the document body casts a shadow.

## 5. Components

### Buttons

- **Shape:** Subtly rounded corners (`rounded-md` = `6px`). Never pill-shaped, never fully square, never variable per variant.
- **Variants:**
  - **Default** (filled, light on dark): `bg-offwhite text-ink hover:bg-offwhite/90`. The primary action on a surface; used sparingly (the verdict card has one or two at most).
  - **Outline:** Transparent background, `border-border`, `text-offwhite`. The most common variant. Use for almost everything.
  - **Ghost:** Transparent, no border, hover reveals `bg-surface-2`. Use for secondary actions that should not draw attention until the user hovers (header icons, modal close).
  - **Danger:** `bg-verdict-tb text-white`. Used for destructive feedback actions (Disagree → TB), never for navigation.
- **Sizes:** `default` (h-9, px-4), `sm` (h-8, px-3, text-xs), `icon` (h-9 w-9 square).
- **Focus:** 2px solid `provider-openai` outline, 2px offset (global, via `:focus-visible` in `src/index.css:23`).
- **Disabled:** `pointer-events-none opacity-50`. The label sometimes swaps to a longer explanatory string ("Generate (set OpenAI key in Settings first)") to make the disabled state self-documenting.

### Verdict Card

The signature component. A bordered panel on `bg-surface` with a tinted header band carrying the verdict label. The header band uses the verdict color at `14` opacity for background and `40` for border (e.g. `${color}14`, `${color}40` for the verdict-uncertain amber).

- **Header:** Verdict label at 30px / 700, tinted-background panel, three-line caveat below in `0.625rem` mono uppercase ("radiographic TB screen · not a diagnosis · confirm bacteriologically"), perception-path disclosure block.
- **Body:** Confidence ring + rationale, optional BoxEvidence heatmap, action row (Details, Disagree), optional Details panel.
- **Footer (M24-v3):** ClinicianReport CTA section, SecondaryObservations CTA section. Both have the same idle / loading-with-progress / ready / error machine.

### Heatmap (BoxEvidenceHeatmap)

- **Container:** `aspect-square max-w-[256px]`, `rounded-md`, `border-border`, `bg-black` (the base layer for the X-ray composite).
- **Cells:** 8×8 grid, viridis palette, near-zero cells render at 0 alpha to avoid visual noise.
- **Caption:** Always-on honesty framing ("Box-evidence overlay — where the trained model sees TB-suggestive patterns. NOT a radiologist annotation; not a region of interest in the clinical sense.").
- **Fullscreen trigger:** Maximize2 icon top-right when `onOpenLightbox` is wired. Opens the URL-bound image lightbox.

### Image Lightbox

A full-screen `FullscreenContent` overlay with progressive evidence disclosure. Three toggles (Heatmap, Zones, Findings) hide / show overlays without obscuring the X-ray. Chrome auto-hides after 2.5s of mouse inactivity — the image is the subject, the controls are accessories.

### Dialog Family

Built on Radix `@radix-ui/react-dialog`. Three variants in `src/components/ui/dialog.tsx`:

- **DialogContent**: Centered modal, max-w-lg. Used for FirstUseModal.
- **DrawerContent**: Right-side drawer, `max-w-md`. Used for SettingsDrawer.
- **FullscreenContent**: `fixed inset-2`, the entire viewport (minus an 8px frame). Used for the report viewer and image lightbox.

All three carry a Close X in the top-right by default.

### Audit Row

A horizontal row of label/value pairs rendered in mono `0.625rem` `text-muted`. Lines up calibration constants, model identifiers, timestamps, perception path. Always rendered with `font-mono` so values align cleanly. This is the project's signature element — it appears in the verdict card, the report disclosure, the PDF footer, and the secondary-observations disclosure.

### Named Rules

**The Audit Row Rule.** Audit metadata is rendered as a single horizontal mono row at the bottom of the surface it belongs to. The Format: `[key]: [value] · [key]: [value] · ...`. Centre dot (`·`) is the field separator, never a comma. Mono, `0.625rem`, `text-muted`. If a value is long, wrap to a new audit row rather than breaking the format.

**The CTA Below Body Rule.** Action CTAs that fire an asynchronous side effect (ClinicianReport, SecondaryObservations) live BELOW the primary content, never inline within it. The primary content is the verdict and the evidence; the CTAs are optional follow-ups.

## 6. Do's and Don'ts

### Do:

- **Do** reserve the verdict triad (red / green / amber) for verdicts, status, and danger. Never decorate with them.
- **Do** use IBM Plex Mono only for auditable strings (constants, IDs, hashes, timestamps, paths, perception-path enum).
- **Do** disclose every fallback. If the orchestrator dropped from local-mode to VLM-primary, the perception-path indicator says so. If the verifier disagreed, the disclosure says so.
- **Do** render audit data inline (model SHA, calibration `T`, threshold `0.6105`, prompt hash) directly under the verdict, in mono, in muted. Not in a developer console.
- **Do** use borders + tonal lifts to convey depth. `bg-surface` → `bg-surface-2` is a step; that step is depth.
- **Do** keep body line-length ≤ 75 characters. The report modal is `max-w-2xl` for this reason.
- **Do** make CTAs that fire async GPT calls (Generate report, Run secondary observations) opt-in. They are advisory, never auto-fired.
- **Do** label every GPT-derived narrative with the load-bearing honesty sentence ("Narrative interpretation only — does not change the verdict." / "VLM observation — not validated, advisory only.").

### Don't:

- **Don't** use `border-left` or `border-right` greater than 1px as a colored stripe accent. If you need to flag a status, use a full border or a tinted background. (Absolute ban.)
- **Don't** use `background-clip: text` with a gradient on any heading. Use a single solid color. Emphasis via weight or size. (Absolute ban.)
- **Don't** apply glassmorphism (`backdrop-blur` + translucent panel) decoratively. Acceptable only on overlays that need to convey "I am floating above the canvas" — the report modal, the image lightbox header. Not on cards in the document body.
- **Don't** ship the medical-AI hero-metric template (huge accuracy %, gradient accent, supporting stats). The project's honesty contract specifically rejects this pattern (`CLAUDE.md`: "No ≥90% sensitivity claim without ~150+ held-out TB positives").
- **Don't** use Sparkles ✨ or other "magic AI" iconography as the primary affordance for GPT features. The Sparkles icon currently used on `Generate radiology report` is a known polish item — replace with a neutral document/quill glyph in the next pass.
- **Don't** introduce a second accent color. Indigo (`#6366F1`) is the only interactivity color. If a new feature wants to "stand out", make it stand out by position or by weight, not by adding teal or magenta.
- **Don't** use `#000` or `#FFF`. Tint every neutral toward the project palette (we use `#0A0A0A` and `#FAFAF7`).
- **Don't** wrap every UI element in a card. Inline disclosures, the audit row, the perception-path indicator should all sit directly on the surface they belong to, not inside a nested container.
- **Don't** animate `width`, `height`, `top`, `left`, or any layout property in a way that triggers reflow. Progress bar `transition-[width]` on a known small element is the only acceptable exception, and that bar caps animation at 200ms.
- **Don't** add a "modal as first thought" affordance. Inline disclosure, expansion in place, and URL-bound side panels are preferred. Modals only when the content genuinely warrants the full canvas (the report viewer; the image lightbox; the first-use consent gate).
- **Don't** confuse mono with audit. If a string is in IBM Plex Mono, it must be auditable. Decorative monospace (e.g. "settings" in a button label) is forbidden.
