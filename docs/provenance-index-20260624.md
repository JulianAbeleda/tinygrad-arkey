# Docs Provenance Index (2026-06-24)

After the 2026-06-24 declutter, the fork's historical `docs/*.md` (the full 06-16→06-22 probe log,
superseded results/handoffs, and completed scopes) were **moved into `docs/archive/`** (git mv, nothing
deleted) and references were rewritten. `docs/` root now holds only current/canonical + active docs.

## Authority order (read these; ignore docs/archive/ unless tracing provenance)
1. `docs/current-project-state-handoff-20260624.md` — canonical current state
2. `docs/README.md` — curated navigation map
3. `bench/README.md` — bench/evaluator map
4. `docs/decode-campaign-final-synthesis-20260623.md` — how decode reached llama parity
5. `docs/prefill-decode-next-workstreams-codex-scope-20260624.md` — next-work map

## Provenance
`docs/archive/` holds 797 superseded docs — the chronological probe log; their verdicts are folded into
the canonical docs above. Kept for history, **not authority**. (Refs among archived docs stay bare-filename;
they resolve as siblings within `docs/archive/`.)
