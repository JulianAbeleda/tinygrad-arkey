# NV boundary-free ordinary-UOp gate v4 re-open (REDUCE_OUTPUT arm)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731`
Status: **construction gate re-opened and green for the norms population on
realized/production identity inputs. No implementation or GPU arm was run in
this step.**

## Why this re-open

The v3 gate predates the landed `Ops.REDUCE_OUTPUT` primitive.  v3 reported
`norms: CONSTRUCTION_GAP` because ordinary `nn.RMSNorm` lowers as a scalar
reduce plus a dependent epilogue.  The reduce-output body now lowers that pair
as one ordinary `CALL`/`SINK` program for identity inputs, so the v3 blocker no
longer represents the available construction.

## What changed

- `extra/llm_research/decode/nv_boundary_free_ordinary_uop_gate.py` now emits
  schema `tinygrad.nv_boundary_free_ordinary_uop_gate.v4`.
- A `reduce_output` arm runs for `norms` using the production call-site helper
  (`_decode_reduce_output_rmsnorm`) with the load-time fp16 identity weight.
- The `lazy_add` row is recorded for that arm but is not a pass condition: the
  primitive deliberately rejects non-identity inputs and keeps the ordinary
  fallback, which is the honest scope of the primitive.
- The old `run` behavior is preserved as `run_v3()` for any closed pre-primitive
  norms/residual A/B harness that still needs the v3 shape.

## v4 verdict table

| population | verdict | meaning |
| --- | --- | --- |
| norms | `REDUCE_OUTPUT_PASS` | one ordinary program for realized identity inputs |
| flash | `CONSTRUCTION_GAP` | score+softmax+combine is not expressible by the current recipes |
| residual_cast_contiguous | `OPAQUE_PRODUCER_GAP` | ordinary stand-in folds, real block_output producer does not |
| vocab_feedback | `ORDINARY_PASS` | ordinary arms only |
| rope_kv | `ORDINARY_PASS` | ordinary arms only |
| quant_core | `ORDINARY_PASS` | dense fp16 stand-in, not the Q4_K packed body |
| llama_q8_pack | `ORDINARY_PASS` | attribution-only |
| other | `NO_CONSTRUCTION` | unclassified fallback |

The key re-open is norms: `REDUCE_OUTPUT_PASS` replaces the v3
`CONSTRUCTION_GAP`.  It is a lower bound, not an upper bound: it proves the
identity-input construction is one ordinary program, not that the production
multi-consumer CALL-input census is body-free.  Body-free removal is still
gated separately by the M1 per-site admission scope and its CPU decode census.

## Validation

- `DEV=CPU pytest test/unit/test_nv_boundary_free_ordinary_uop_gate.py`: 9 passed.
- Gate artifact:
  `docs/task_workflow/output/nv-boundary-free-ordinary-uop-gate-20260813.json`.

## Next

The norms REDUCE_OUTPUT arm is construction-passing, so the remaining question
is no longer "can the primitive express the population" but "can the production
decode census show body-free removal".  That is the CPU-first admission step in
`nv-m1-norm-epilogue-generic-primitive-scope-20260812.md`, before any NV arm.
