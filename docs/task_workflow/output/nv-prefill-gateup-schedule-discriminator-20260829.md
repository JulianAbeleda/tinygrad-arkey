# NVIDIA prefill gate/up schedule discriminator (2026-08-29)

## Decision

**STOP.** B0 does not authorize B1. The fragment load-to-use reorder is a
primitive-only signal, but matched physical counters and the required exact
72-role lifecycle were not obtained. The other declared mechanisms do not
qualify.

## Fixed protocol

Control and candidates use the retained compiler-owned K64 gate body, real
`blk.0.ffn_gate.weight`, deterministic legal compact-Q8 input, grid
`(96,4,1)`, block `(32,2,4)`, CUDA `sm_120a`, 10 warmups, and 9 timed bridge
rounds. The control is the retained unroll-4 cubin. No tile, Stream-K,
cp.async, TMA, fusion, or queue placement was tested.

## G0 and primitive results

| arm | min us | median us | output hash | readonly | resources/counters | result |
|---|---:|---:|---|---|---|---|
| retained unroll-4 control | 350.016 | 350.848 | `26ae24b2...e7c57` | PASS | control resources retained; matched NCU counters unavailable in this run | PASS |
| fragment load-to-use reorder | 345.504 | 346.944 | `26ae24b2...e7c57` | PASS | no matched physical NCU counter capture | primitive signal only |
| metadata-load reorder | 351.424 | 352.672 | `26ae24b2...e7c57` | PASS | no matched physical NCU counter capture | NO-SIGNAL / slower |
| register-safe double-buffered fragment schedule | not run | not run | n/a | n/a | no register-safe source spelling exists in retained lifecycle; constructing one would be B1 implementation work | BLOCKED |

The fragment arm improves the isolated bridge median by `3.904 us` (1.11%),
but this is not a B0 PASS: it lacks the required matched tensor-duty,
eligible-warp, and long-scoreboard counters and was not exercised through the
exact 72-role lifecycle. No claim is booked against the model wall.

## Evidence

- `docs/task_workflow/evidence/nv-prefill-gateup-schedule-discriminator-20260829/control-unroll4.json`
- `docs/task_workflow/evidence/nv-prefill-gateup-schedule-discriminator-20260829/fragment_load_to_use_reorder.json`
- `docs/task_workflow/evidence/nv-prefill-gateup-schedule-discriminator-20260829/metadata_load_reorder.json`
- corresponding `.cu` and `.cubin` files in the same directory
- source transform helper: `docs/task_workflow/evidence/nv-prefill-gateup-schedule-discriminator-20260829/make_variants.py`

All three completed bridge G0 checks where executed: finite and nonzero full
output, zero unwritten elements, unchanged packed weights and activation
record, and bit-exact output hash. The missing physical-counter and 72-role
requirements are decisive, so no B1 mechanism is named.
