# Product

## Register

product

## Users

**Primary: the radiologist or clinical reviewer at the second-read seat.** A pulmonologist, radiologist, or trained TB-screen reader on a 14-inch laptop in a clinic, hospital reading room, or NGO field site, often under moderate lighting and time pressure. They are not the first reader; the AI pipeline is. Their job is to confirm screen-positives, dispose of clear negatives, and decide what to do with the abstains. They distrust unexplained model outputs and want the system's reasoning visible without ML jargon.

**Secondary: the data scientist or ML engineer evaluating model behavior.** Someone running the local FastAPI server with the validated head, scrubbing through historical cases, watching the agent trace, calibrating against a holdout set. They want the raw numbers (`tb_prob`, `s_inactive`, `T`, threshold), the per-zone breakdown, the prompt hash, the model SHA. They are looking at the same screen as the radiologist but reading a different layer.

**Tertiary: the portfolio reviewer.** Hiring managers, PMs, and engineers who arrive at this preview to evaluate the *engineering and product judgment* of the maker, not to triage their own films. They read both the UI and the case study together; the UI is the proof.

## Product Purpose

A chest X-ray TB triage research preview. Returns one of three verdicts (`tb`, `no_tb`, `abstain`) with the validated trained model on the local-mode path and gpt-5.5 vision on the deployed fallback. Surfaces the reasoning that produced the verdict: per-zone calibrated probabilities, the BoxEvidence heatmap, the top TXRV findings, the safety-net escalators that fired or did not.

It is explicitly **not a medical device**. The UI says so on every page. It is a research preview whose purpose is to make a specific argument: that a TB triage tool built around intellectual honesty (escalate caution only, anchor every threshold to data, never overclaim) is more useful than one that hides its uncertainty behind a confident percentage. Success is a reviewer who trusts the abstain decisions and treats the no-TB decisions as a real screen, not a black-box yes.

## Brand Personality

Three words: **honest, calibrated, precise.**

Voice: declarative, evidence-led, conservative. Never enthusiastic. Sentences end with periods, not exclamation points. When unsure, "abstain" is louder than "uncertain" and "uncertain" is louder than "low confidence". Numbers always have units and contexts. Quotes the case study's own language back to itself when it would otherwise drift.

Emotional goals: the tool should feel like the kind of senior colleague who picks up a film at 11pm on a Tuesday and says "I don't know, get a second look." It should never feel like a wellness app, a SaaS dashboard, or a magic AI assistant.

## Anti-references

- **Medical-AI marketing dashboards.** Hero metric "99.7% accuracy", glowing gradient cards, three-up "Faster / Smarter / Better" panels. The dishonesty starts in the layout.
- **Consumer wellness aesthetics.** Soft pastels, friendly illustrations, lifestyle photography. This is not your wellness journey.
- **The "AI co-pilot" interface family.** Purple-to-indigo gradients, ✨ Sparkles icons everywhere, "Ask me anything" command bars as the default mode of interaction. The model is not a friend.
- **Generic SaaS admin shells.** Coloured sidebar, emoji icons in nav, oversized hero cards with stat counters. We are not a CRM.
- **Marketing pages pretending to be products.** Hero-image-with-headline, scroll-revealed feature grids, testimonials. This is a tool; it should look like one.
- **Dark mode tech-bro maximalism.** Neon gradients on black, glowing borders, glassmorphism cards floating in space. Dark is a posture, not a costume.

## Design Principles

1. **Honesty is the product, not just the policy.** Every threshold has a derivation visible in the comment. Every fallback is disclosed to the user. The audit trail (model SHA, calibration constants, prompt hash) is reachable from the verdict card, not hidden in a developer console. If the engineering is honest, the UI should look like it.
2. **Caution is monotonic.** Verdict severity moves only upward (`no_tb` < `abstain` < `tb`). Every safety-net rule, every escalator, every disagreement signal can push the verdict toward more caution, never less. The design should make escalation feel safe and demotion feel impossible.
3. **The model advises; the guardrails decide.** The LLM is a tool inside the pipeline, not the conclusion. UI affordances for GPT calls (generate report, secondary observations) are always opt-in, always labelled as advisory, never auto-trigger on a verdict change.
4. **Visible degradation beats invisible competence.** When a path fails (no key, server down, fallback fired), surface it. A user who is being silently helped by a worse model deserves to know it. Banners over silence.
5. **Audit data is first-class UI, not a footer.** Calibration `T`, threshold `0.6105`, model SHA, perception path, prompt hash — these are not boilerplate. They are the evidence that the engineering is honest. They get monospace, they get their own row, they get rendered into the PDF report.

## Accessibility & Inclusion

- **Target: WCAG 2.2 AA** on every shipped screen. Lighthouse a11y ≥95 per the project's own discipline (`CLAUDE.md`).
- **Keyboard navigable end-to-end.** Every action that can be taken with a click can be taken with the keyboard. Focus rings are visible (`:focus-visible` is explicit in `src/index.css:23`).
- **Color is never the only channel.** The verdict triad uses red/green/amber, but the verdict text label is always present and is the primary signal. Heatmap cells have `<title>` for screen readers.
- **Respect `prefers-reduced-motion`.** Pulsing skeletons, shimmer animations, and the progress bar should hold still for users who request it.
- **Color-blind safe verdict colors.** The red/green pair is the worst case for protanopia/deuteranopia; we mitigate by always pairing color with the verdict text label and pattern (color tint of background, plus literal label string). Audit recommendation: verify with a simulator on the verdict triad before any future logo work.
- **Text contrast minimums.** Body text ≥4.5:1. Muted text used only for non-load-bearing audit-trail data; the audit-trail metadata is mono and tested at small sizes. Decorative `muted/70` and `muted/80` opacity values are a known audit gap — see DESIGN.md guardrails.
