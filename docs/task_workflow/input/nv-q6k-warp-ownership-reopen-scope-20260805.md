# Native NV Q6_K warp-ownership reopen

Date: 2026-08-05
Status: **research-only, default-off; native Gate 1 NO-GO (+0.18505 us)**

## Finding that changes the candidate class

The d512 Q6 path in the pinned local llama.cpp source is **MMVQ**
(`ggml-cuda/mmvq.cu`), not the tiled multi-vector MMQ path.  For one output
column on a modern NVIDIA target, its generic parameter table selects four
warps per output row (`calc_nwarps(..., ncols_dst=1)=4`).  Each lane owns a
small Q6/Q8 fragment and the four warp partials are reduced before one f32
write.  The relevant Q6 vector routine uses packed signed int8x4 dots
(`vec_dot_q6_K_q8_1_impl_mmvq` in `vecdotq.cuh`).

This is materially different from the failed native Q8+DP4A construction:

| property | failed `q6k_q8_dp4a_rt2` | new ownership candidate |
|---|---:|---:|
| physical lanes per output row | 4 | 128 (4 warps) |
| Q6 blocks owned per warp | n/a | 4 |
| int8x4 chunks per lane per Q6 block | 16 serial groups across four lanes | 2 |
| output ABI | contiguous f32 | existing `float32[1024,4]` partials |
| downstream consumer | replaced by a scalar sum in probe | unchanged-equivalent partial sum in probe |

Thus this is a lane ownership / memory-level-parallelism test, not a cosmetic
vector-load rewrite.  It addresses the binding mechanism named by the prior
Q6 static audit: four lanes leave the packed-weight latency exposed even
though `dp4a.s32.s32` exists.

## Candidate and invariant

`extra/llm_research/decode/q6k_q8_warp_partial_microgate.py` emits one block
per output row with local axes `(warp=4, lane=32)`.  Warp `w` owns Q6
superblocks `4*w .. 4*w+3`; each lane owns two contiguous int8x4 chunks for
each such block.  It writes exactly one partial `out[row,w]`, preserving the
installed four-partial consumer boundary.

The pure mapping witness proves all 1,024 int8x4 chunks / all 4,096 input
elements are covered exactly once, and each warp owns exactly its four Q6
blocks.  `test/unit/test_q6k_warp_partial_mapping.py`: **2 passed**.

The first native compile attempt established the intended body is expressible
up to final code generation: the renderer produced a `4 x 32` workgroup, five
warp shuffles, the expected dynamic packed-Q6 loads, and two `__dp4a` calls
per lane per Q6 block.  Its final store failed because the two-LOCAL-axis
spelling rendered an undefined inactive pointer index (`... : Invalid`).  The
candidate is now rewritten as one flat 128-thread LOCAL range with
`warp=lid//32`, `lane=lid%32`; a third CPU test proves that this spelling has
exactly identical ownership.  The flat spelling then compiled and executed
after using `UOp.special` for the physical grid/local IDs and `UOp.eq` for the
static Q6-group selection.

The included-cost native A/B/A gate is correctness-passing but closes the
candidate:

| arm | median us / graph replay |
|---|---:|
| installed partial4 + sum control midpoint | 66.74259 |
| four-warp Q8 + DP4A partial4 candidate | 66.92764 |
| candidate - control | **+0.18505** |

All five candidate samples were `66.73147--67.32338 us`; the two five-sample
control brackets were `66.47879--67.02843 us`.  The candidate passes the
predeclared lossy-Q8 reference contract (`0.1730957` max absolute error <=
`0.4677861` bound); the installed baseline error is `2.95639e-05`.  Therefore
the result is a performance NO-GO, not a numerical or compiler failure.

Compact artifact:
`docs/task_workflow/output/nv-q6k-q8-warp-partial-microgate-20260805.json`.
Raw `/tmp` SHA256:
`a19dc93362b2de9ee2aa7094d132fef50f35809ba18dbb4117809d3dc3ee887a`.

## Gates

1. **Static / body gate: PASS.** The flat 128-thread spelling compiles,
   executes, preserves the four-partial ABI, and passes correctness.
2. **Included-cost microgate: NO-GO.** Same Q8 producer, packed-Q6 bytes, and
   `partials.sum(axis=1)` tail as the installed baseline; collect native
   A/B/A after clock stabilization.  Continue only on repeatable positive
   delta.  A loss closes this exact ownership construction.
3. **Not reached.** Live Q/K/V topology, full-logit,
   cache-state, and reverse token-wall qualification.  This document does not
   integration, selection, and parity credit are not authorized.

## Hard stops

- Do not call it an implementation of llama or copy llama source/cubin.
- Do not combine this result with the prior four-lane Q8+DP4A loss or the
  synthetic shared-Q8 result.
- Do not change the production emitter, route selector, or partial consumer.
- If the native renderer cannot realize a 4x32 workgroup with warp-local
  shuffles, record a compiler/substrate block rather than changing geometry.
