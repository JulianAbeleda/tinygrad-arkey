# Q4 tiled multi-wave validation

`q4k_wmma_tiled_multiwave_validation.py` is a validation-only harness. It
uses the existing generated Q4_K/Q8_1 tiled lifecycle and does not change
dispatch, route selection, emitters, or lowering. The default expansion is
`M=32,N=128,K=256`: two row waves and the full 128-column ownership span.

Run with:

```sh
PYTHONPATH=. python3 extra/qk/q4k_wmma_tiled_multiwave_validation.py
```

The JSON artifact is written to
`bench/q4k-wmma-tiled-multiwave/latest.json`. It records correctness,
relative RMSE, kernel count, compile/runtime timing, safe output-elements/s,
WMMA evidence, and whether final code-object resource metadata was available.
Resource values are never inferred from source or timing.

## Run record (2026-07-15)

The pre-existing bounded gate passed first at `32x32x256`: rel-RMSE
`1.39e-7` (RTOL `0.006`), 59 kernels, WMMA surface evidence present, and
`0.618 ms` runtime (`~51,784` output-elements/s). The expansion passed at
`32x128x256`: rel-RMSE `3.00e-7`, 131 kernels, `4,583.1 ms` compile evidence,
`1.513 ms` runtime, and `~21,155` output-elements/s. WMMA evidence was
present via the selected `iu8` surface.

The expansion is the smallest tested shape that retains the two-row-wave
coverage of the bounded case while spanning all eight 16-column fragments of
the current 128-column ownership plan. Final VGPR/LDS, spill, occupancy, and
code-object resource evidence was unavailable from this generated runtime
debug path; the harness records that explicitly and does not claim a resource
pass. No compiler/runtime failure occurred in this expansion.
