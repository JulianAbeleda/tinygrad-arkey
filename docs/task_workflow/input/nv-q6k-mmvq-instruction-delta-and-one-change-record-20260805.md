# NV Q6_K MMVQ instruction delta and one-change gates — 2026-08-05

> **2026-08-05 amendment:** the structured-control blocker described below is
> resolved by the generic typed `PostBarrierRegion` implementation. NV SASS now
> emits a real predicated producer-warp `EXIT` immediately after the barrier.
> The completed implementation and wall verdict are in
> `nv-q6k-post-barrier-region-implementation-record-20260805.md`. The original
> sections remain as the pre-implementation hypothesis and evidence trail.

## Verdict

The observed llama Q6_K decode route is MMVQ, not MMQ. Its smallest material
physical difference is now accounted for: four warps calculate lane partials,
warps 1–3 publish their unreduced lane partials to 384 bytes of shared memory,
all warps cross one barrier, the producer warps return, and warp 0 alone loads
the three corresponding lane values and performs the five-shuffle reduction.

The one-change tinygrad reproduction matches the 384-byte stage, three LDS,
one barrier, and five shuffles, but it is **+0.204664 us** slower than the flat
control. It cannot reproduce the early return: ordinary UOp graphs reject
pre-existing `Ops.IF`/`Ops.ENDIF`, so all four warps remain live through the
post-barrier loads and shuffle ladder. This is a precise emitter/control-flow
blocker, not a GPU concurrency verdict.

There is also a positive result that changes the ledger. On current commit
`a1a51c349`, the supposedly failed flat four-warp Q8+DP4A control beats the
installed Q6 route in all three full sessions by **1.403–2.481 us**. The old
`+0.18505 us` result at `5b70ac` is no longer reproducible. The primitive is
therefore **REOPEN / measured win**, but not promotion-qualified: it has not yet
been integrated into the model's mixed Q4/Q4/Q6 shared-Q8 group or passed the
full-logit semantic contract.

## What was compared

The canonical llama evidence is cubin function 355:

```text
mul_mat_vec_q<GGML_TYPE_Q6_K, 1, false, false>
block = (32, 4, 1), registers = 48, explicit shared = 384 B
295 SASS, 6 IDP.4A, 5 SHFL.BFLY, 3 LDS
```

The implementation is visible in
`/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`: lines 594–610 publish
three warps of lane partials, line 611 is the barrier, lines 612–614 return
producer warps, lines 624–625 perform the three lane-corresponding LDS, and
line 632 performs the only warp reduction.

The llama absolute time and the microgate absolute time are not compared: the
launches do not have the same included graph. The llama SASS is structural
evidence. Every wall verdict below is a same-session tinygrad A/B/A with the
same included producer and output boundary.

## Instruction-by-instruction delta

| Physical count | llama MMVQ | flat static control | dynamic-address gate | 384 B lane-stage gate |
|---|---:|---:|---:|---:|
| SASS instructions | 295 | 727 | 191 | 726 |
| registers | 48 | 44 | 39 | 38 |
| `IDP.4A.S8.S8` | 6 | 2 | 2 | 2 |
| `SHFL.BFLY` | 5 | 5 | 5 | 5 |
| `LDS` | 3 | 0 | 0 | 3 |
| explicit shared bytes | 384 | 0 | 0 | 384 |
| `LDG.E.U16` | 21 | 50 | 10 | 50 |
| `LOP3.LUT` | 47 | 230 | 41 | 229 |
| `PRMT` | 12 | 109 | 0 | 102 |
| `SHF.R.U32.HI` | 7 | 142 | 29 | 141 |

Two different mismatches are visible:

1. The static Q6 group selector explodes into unpack/address instructions.
2. The cross-warp reduction topology is physically different from llama.

The first is visually larger, but the dynamic-address microgate proves that
static instruction count is not the wall limiter by itself. Removing 536 SASS
instructions made the graph slower. The smallest remaining physical mismatch
that could be changed independently was therefore the lane staging and
producer-warp lifetime.

## One-change microgates

All GPU arms ran under `/tmp/gpu-bench.lock` on the same current baseline. The
Q8 producer, 128-thread ownership, four blocks per warp, Q6/Q8 dot algebra, and
independent Q8 oracle remained fixed unless explicitly named.

### G1 — direct 16-byte cross-warp scalar merge

Change only four global partials plus external sum into four shared scalar
stores, a barrier, and one direct output.

| Metric | Result |
|---|---:|
| control midpoint | 64.653055 us |
| candidate median | 64.684490 us |
| candidate minus control | **+0.031435 us** |
| candidate minus installed | -1.371528 us |
| correctness | bitwise equal to control; Q8 oracle 0.01093483 <= 0.02 |
| gate | **FAIL** |

This closes the hypothesis that the external scalar sum tail explains the old
flat-route loss. Its included-cost effect is wall-neutral to slightly worse.

### G2 — direct dynamic Q6 byte addressing

Change only the sixteen-arm static group selector into lane-derived byte
offsets and shifts.

| Metric | Result |
|---|---:|
| control midpoint | 64.101683 us |
| candidate median | 64.439628 us |
| candidate minus control | **+0.337945 us** |
| candidate minus installed | -1.799673 us |
| correctness | bitwise equal to control; Q8 oracle 0.01093483 <= 0.02 |
| gate | **FAIL** |

The target kernel shrank from 727 to 191 SASS, `LDG.E.U16` from 50 to 10,
`LOP3` from 230 to 41, `PRMT` from 109 to zero, and high shifts from 142 to
29, yet lost 0.338 us. The falsified belief is "shorter generated code is the
next wall lever." The direct runtime address pattern gives the compiler less
static information and produces worse physical memory scheduling/coalescing.

### G3 — llama-shaped 384-byte lane stage

Change only the reduction topology: warps 1–3 publish all 32 lane partials,
warp 0 adds three lane counterparts, then one five-shuffle ladder runs.

| Metric | Result |
|---|---:|
| control midpoint | 62.570314 us |
| candidate median | 62.774978 us |
| candidate minus control | **+0.204664 us** |
| candidate minus installed | -2.275860 us |
| correctness | Q8 oracle 0.01093578 <= 0.02; association delta 0.000003815 |
| resources | 38 registers; 384 B explicit shared; 3 LDS; 5 SHFL; 1 barrier |
| gate | **FAIL / substrate-limited reproduction** |

This candidate reproduces llama's staging resources but not its lifetime. In
llama, 96 producer lanes return immediately after the barrier. In this UOp
candidate, those lanes execute the loads and shuffle computation even though
only lane 0 of warp 0 stores. Predicating the final store is not equivalent to
retiring three warps.

## Precise substrate blocker

`tinygrad/codegen/__init__.py:327` rejects `Ops.IF` and `Ops.ENDIF` already in a
kernel graph with `if not allowed in graph`. Lines 328–330 synthesize control
flow only as a lowering of a gated store. The custom kernel API therefore has
no ordinary primitive for this sequence:

```text
if warp > 0: publish lane partial
workgroup_barrier()
if warp > 0: return
warp0_load_three_lane_partials()
warp0_reduce_and_store()
```

The blocker is falsifiable. Add a structured post-barrier early-exit primitive
or an equivalently safe divergent region, generate an actual post-barrier
branch/exit in SASS, and rerun G3. Do not call a gated store or dead output a
substitute; the SASS must show the producer warps skip the three LDS and five
shuffles.

## Current-baseline reopen

The flat static control beat the installed Q6 route independently in every
full run:

| Session | flat control minus installed |
|---|---:|
| G1 | -1.402963 us |
| G2 | -2.137618 us |
| G3 | -2.480524 us |

This is primitive evidence, not a model parity claim. The route changes the
activation contract from fp16 to shared Q8_1 and currently exists as a
standalone Q6 consumer. The model route is a mixed Q4/Q4/Q6 group. A previous
shared-Q8 model experiment passed the declared relative-L2 contract through
g12 and failed at g18; that does not automatically qualify this consumer.

## Next decisive work, in order

1. Add the flat four-warp Q6 consumer behind a closed-default flag in the
   existing mixed Q4/Q4/Q6 shared-Q8 group. Do not alter the Q4 consumers in
   the same patch.
2. At g1, g4, g8, and g12 require exact generated tokens, equal argmax, equal
   top-10 token set, relative L2 <= `1e-3`, and accumulated perturbation below
   the minimum observed logit margin. Stop at the first semantic failure.
3. Only after semantics pass, run settled same-session wall A/B/A. The
   primitive win predicts direction; it does not predict the model magnitude.
4. Separately, treat post-barrier early exit as an emitter capability task.
   Rerun the 384-byte stage only if generated SASS proves producer-warp
   retirement. Do not tune the currently incomplete emulation.

The completion condition for this record is met: there is a repeatable
measured win worth integrating and a precise emitter blocker for the remaining
llama-shaped reduction experiment. No production route was changed or
promoted.

## Artifacts

- Machine summary:
  `docs/task_workflow/output/nv-q6k-mmvq-instruction-delta-and-one-change-gates-20260805.json`
- Direct and lane-stage gate:
  `extra/llm_research/decode/q6k_q8_warp_direct_microgate.py`
- Dynamic-address gate:
  `extra/llm_research/decode/q6k_q8_warp_dynamic_microgate.py`
- Raw G1: `/tmp/q6k_q8_warp_direct_full.json`, SHA256
  `fdbfda4461eac874faaf5ff6ad20ba9cc3120b52e5cb63c7cebb97dbd3e8d529`
- Raw G2: `/tmp/q6k_q8_warp_dynamic_full.json`, SHA256
  `eb37e5d28a87af3f44cd07558a548c94ac38afa8331eee5bc120f02b0644d93c`
- Raw G3: `/tmp/q6k_q8_warp_lane_stage_full.json`, SHA256
  `fa2307790ee4e66362074af917e72c37390401687143c165158afd7c76b032f0`
