# BubbleBeam FutureSight terminology

Date: 2026-06-25

## Names

- **BubbleBeam**: this fork's machine-search system. It is not the inherited tinygrad timing beam.
- **FutureSight**: BubbleBeam's static pre-timing selector that predicts the coalesced/lane-partition winner.
- **Inherited beam**: tinygrad's timing-based scheduler beam in `tinygrad/codegen/opt/search.py`.

## Current flag

```bash
BUBBLEBEAM_FUTURESIGHT=1
```

Old beam-named compatibility wiring has been removed. New docs, tests, and harnesses should use
`BUBBLEBEAM_FUTURESIGHT`.

## Compiler terms that remain

`OptOps.COALESCE`, `coalesced`, `unit_stride`, and `vector_width` remain as low-level compiler vocabulary. They describe
what FutureSight predicts, not the search system's user-facing name.
