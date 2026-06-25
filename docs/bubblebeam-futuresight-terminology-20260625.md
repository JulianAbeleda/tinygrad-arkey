# BubbleBeam FutureSight terminology

Date: 2026-06-25

## Names

- **BubbleBeam**: this fork's machine-search system. It is not the inherited tinygrad timing beam.
- **FutureSight**: BubbleBeam's static pre-timing selector that predicts the coalesced/lane-partition winner.
- **Inherited beam**: tinygrad's timing-based scheduler beam in `tinygrad/codegen/opt/search.py`.

## Current flag

Preferred:

```bash
BUBBLEBEAM_FUTURESIGHT=1
```

Deprecated temporary alias:

```bash
BEAM_COALESCE=1
```

The alias remains only so older docs/scripts do not fail immediately. New docs, tests, and harnesses should use
`BUBBLEBEAM_FUTURESIGHT`.

## Compiler terms that remain

`OptOps.COALESCE`, `coalesced`, `unit_stride`, and `vector_width` remain as low-level compiler vocabulary. They describe
what FutureSight predicts, not the search system's user-facing name.
