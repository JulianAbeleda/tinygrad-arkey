# Decode Attention A3.2b Scoped Lane-Map Scope

## Goal

Replace the failed global cross-lane attempt with a scoped attention lane-map plan.

A3.2 proved:

- cross-lane primitives exist in the repo
- global `WARP_REDUCE_LOWERING=1` is too broad
- full decode capture fails before W==D

A3.2b must make cross-lane explicit and local to the generated attention candidate.

## Current blocker

Artifact:

- `bench/qk-decode-attention-a3-2-cross-lane/latest.json`

Verdict:

- `A3_2_BLOCKED_BY_CODEGEN_GLOBAL_WARP_REDUCE`

Failure:

```text
RuntimeError: UOp verification failed ... Ops.UNROLL dtypes.float ... ((4, 4),)
```

Interpretation:

```text
WARP_REDUCE_LOWERING is not safely scoped.
The attention route needs explicit lane mapping, not a global rewrite pass.
```

## Working precedent from GEMV

The generated GEMV G3 path works because it is explicit:

- `UOp.special(WARP, "lidx0")`
- `LanePartition(lane)`
- lane-local work ownership
- `lane_partition_reduce_sum(...)`
- stable named generated program

Relevant files:

- `extra/qk_lane_partition_reduce.py`
- `extra/qk_gemv_g3_codegen_lowering.py`

A3.2b should follow that style for attention instead of relying on global `WARP_REDUCE_LOWERING`.

## Attention target

Initial target:

- `flash_score_whole_cache_32_128`

Current shape:

- one generated program computes score for each `(head, token)`
- reduction over `Hd=128` is scalar/generated
- no explicit wave lane ownership

Desired scoped shape:

- program name: `flash_score_whole_cache_xlane_32_128`
- one wave cooperates on a score or score tile
- each lane owns a subset of `Hd`
- lane partials are combined with `lane_partition_reduce_sum`
- output score buffer contract stays identical

## Non-goals

- Do not enable global `WARP_REDUCE_LOWERING`.
- Do not rewrite unrelated model kernels.
- Do not hand-write a full attention kernel.
- Do not reintroduce sliced KV or `E_49152`.
- Do not promote without W==D.

## Required implementation phases

### B0: Probe current attention lane-map readiness

Tool:

- `extra/qk_decode_attention_a3_2b_lane_map_probe.py`

Checks:

- A2 route is clean.
- A2 generated attention programs are present.
- `flash_score_whole_cache_32_128` exists.
- no explicit `lidx0`/lane-map score program exists yet.
- cross-lane primitive files exist.
- global lowering failure artifact exists.

Expected verdict:

- `A3_2B_ATTENTION_LANE_MAP_NOT_WIRED`

This is not a failure. It records the current wall before modifying score code.

### B1: Add explicit attention score lane-map kernel

Flag:

```text
DECODE_ATTN_SCORE_XLANE=1
```

Program:

```text
flash_score_whole_cache_xlane_32_128
```

Required properties:

- uses `UOp.special(32, "lidx0")`
- maps `Hd=128` across wave32 lanes, likely 4 elements per lane
- uses `lane_partition_reduce_sum` or equivalent staged `ds_bpermute`
- stores one score per `(head, token)`
- reads whole `assigned_kv`
- keeps the same score buffer output layout

### B2: Route and correctness gate

Compare A2 vs A3.2b:

- generated route fires
- owned flash does not fire
- `E_49152` absent
- tokens match
- `flash_score_whole_cache_xlane_32_128` captured
- cross-lane evidence appears in source/ISA

Passing verdict:

- `A3_2B_ATTENTION_LANE_MAP_ROUTE_CLEAN`

### B3: W==D transfer gate

Only after B2 passes:

Compare:

- owned
- A2
- A3.2b

Contexts:

- `512`
- `1024`
- `2048`
- `4096`

Verdicts:

- `A3_2B_CROSS_LANE_TRANSFERS`
- `A3_2B_CROSS_LANE_NO_TRANSFER`
- `A3_2B_CROSS_LANE_REGRESSES`

## Kill conditions

Stop and classify if:

- explicit lane-map score kernel cannot compile
- tokens diverge
- `E_49152` returns
- owned flash fires
- `ds_bpermute` only appears through hand-owned whole-kernel code
- W==D does not transfer beyond spread

## Next executable command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_2b_lane_map_probe.py
```

Expected artifact:

- `bench/qk-decode-attention-a3-2b-lane-map/latest.json`
