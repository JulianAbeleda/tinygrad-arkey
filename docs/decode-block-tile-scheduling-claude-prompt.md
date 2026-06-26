# Claude prompt: resolve the generated block-tile decode scheduling gap

You are auditing `/home/ubuntu/tinygrad-arkey` after the generated decode block-tile milestone.

## Goal

Resolve the remaining decode performance gap after the generated UOp tile became structurally expressible.

The important reframe: the problem is no longer "can tinygrad generate the owned-style tile?" The ordered gates now prove that it can express the core topology. The remaining problem is that the structurally-correct generated tile does not transfer to owned-kernel throughput. Treat this as a scheduling/economics/codegen-quality problem, not another attention-layout problem.

## What just landed

Implemented files:

- `extra/qk_decode_isa_vectorization_gate.py`
  - Authoritative vectorization gate.
  - Counts RDNA3 `global_load_b128|b96|b64|b32` plus existing `d16/dword*` markers.
  - Captures generated kernel ISA and records numeric correctness, route cleanliness, marker dict, LDS bytes, and disassembly.

- `extra/qk_decode_attention_block_tile_microgate.py`
  - Proves the generated 128-thread / 4-warp / TK=16 block tile numerically.
  - Oracle uses fp16-staged K/V semantics because the generated tile stages K/V through half LDS like the owned path.

- `extra/qk_flash_decode.py`
  - Adds `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`.
  - Default-off behind `DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1` plus `DECODE_ATTN_BLOCK_TILE=1`.

- `extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py`
  - Recognizes `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` when `DECODE_ATTN_BLOCK_TILE=1`.

- `docs/decode-block-tile-generated-result.md`
  - Result summary and interpretation.

## Ordered gate results

All structural/correctness gates passed:

| Gate | Result | Evidence |
|---|---:|---|
| Block-tile microgate | `BLOCK_TILE_MICROGATE_PASS` | max_abs <= 2.29e-05, rel_rmse <= 1.34e-07 across Tc 32/128/130/256 |
| Route cleanliness | `FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT` | token match, no materialization, owned absent, generated block tile present |
| ISA vectorization | `ISA_VEC_AUTHORITATIVE_PASS` | LDS 8192B, wide_load_count 34, scratch 0 |
| W==D | partial transfer only | generated improves vs previous generated route but remains far below owned |

## Current W==D numbers

| ctx | Previous generated tok/s | Block-tile generated tok/s | Owned/baseline tok/s |
|---:|---:|---:|---:|
| 128 | 82.7 | 82.4 | 82.4 |
| 512 | 7.2 | 19.0 | 103.5 |
| 1024 | 4.1 | 11.8 | 101.8 |
| 4096 | 1.1 | 3.5 | 94.6 |

Runtime conclusion from `extra/qk_decode_runtime_overhead.py`: GPU-bound, host-sync median 2.6% of wall. Do not chase Python/runtime overhead first.

## ISA marker movement

Previous one-warp generated tile vs new generated block tile:

| Marker | Previous generated xlane | Generated block tile | Interpretation |
|---|---:|---:|---|
| LDS bytes | 256 | 8192 | block staging is now present |
| wide load count | 10 | 34 | wide loads are present |
| `global_load_d16` | 0 | 32 | K/V load form improved |
| `global_load_b64` | 10 | 2 | still present |
| cross-lane ops | 20 | 10 | improved but not owned-level |
| scratch | 0 | 0 | no spill blocker |
| VGPR | 80 | 56 | better resource footprint |
| `s_barrier` | 0 | 1 | expected for TK staging |

This means the previous blocker label is resolved:

`SEARCH_BLOCKED_BY_CODEGEN__BLOCK_TILED_MULTI_WARP_TILE_NOT_EXPRESSED` is no longer accurate.

Use the new narrow blocker label:

`SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING`

## Problem statement

The generated UOp tile now has the owned tile's high-level structure:

- 4 warps per workgroup.
- One warp per GQA query head.
- TK=16 K/V block staged in LDS.
- 8192B LDS resource in ISA.
- Wide global loads in ISA.
- fdot2 path.
- Clean route, no materialization, no owned kernel fallback.

Despite this, W==D remains far below owned at ctx512+.

Therefore the remaining issue is likely one or more of:

- poor instruction scheduling around LDS writes, LDS reads, `s_waitcnt`, and fdot2 use,
- bad loop structure or insufficient unrolling around the TK inner loop,
- address-arithmetic/control-flow overhead in the hot loop,
- generated barrier/wait placement differing from owned HIP/comgr output,
- DS read/write pressure or ordering that prevents latency hiding,
- occupancy/economics mismatch despite similar LDS/VGPR resources,
- missing renderer/codegen hints that the hand-authored HIP naturally gives to comgr.

## What not to do

Do not write another attention route first.

Do not revive score-broadcast.

Do not claim "machine search solved decode" because the generated route is still much slower than owned at long context.

Do not treat vectorization as missing unless the new ISA gate says so. The new gate says vectorization and LDS staging are present.

## Commands to reproduce

```bash
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 PYTHONPATH=. python3 extra/qk_decode_isa_vectorization_gate.py
DEV=AMD JIT=1 DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 DECODE_ATTN_BLOCK_TILE=1 V_DOT2_LOWERING=1 PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py
```

## Requested next scope

Produce a concrete resolution plan for `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING`.

The plan should compare `owned_flash_tile_gqa_whole` against `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` at the ISA and generated-source/codegen levels, then identify the smallest next experiment that can explain the long-context gap.

Required outputs:

1. A table of owned vs generated differences that could plausibly explain 19 tok/s vs 103.5 tok/s at ctx512 and 3.5 tok/s vs 94.6 tok/s at ctx4096.
2. A ranked list of hypotheses by expected performance impact.
3. For each hypothesis, the exact gate or measurement that would prove/refute it.
4. A recommended first code change, if any, default-off and correctness-gated.
5. A stop condition: when to conclude this is a codegen scheduler wall rather than an attention-kernel-authoring issue.

The strongest candidate first audit is an owned-vs-generated static/dynamic scheduling diff focused on:

- waitcnt placement,
- DS read/write ordering,
- block-loop and TK-loop structure,
- fdot2 count and placement,
- instruction mix in the hot loop,
- address arithmetic in the hot loop,
- occupancy from LDS/VGPR/SGPR,
- whether generated code reloads/recomputes values owned keeps in registers.

Use `docs/decode-block-tile-generated-result.md` as the starting artifact.
