## Summary

Packed-WMMA is now the default 14B prefill route (`6ca798568`). It hits ~1829 tok/s at pp512 — ~5.2× over the old direct-packed baseline (~354 tok/s), and at parity with llama.cpp (~1837 tok/s).

There's a remaining ~6% gap to the theoretical ceiling (~1940 tok/s, projected from the hand kernel `build_gemm_lds2_q4k` scaled to 14B). This doc records what was tried to close it and why none of it worked.

## Current state

| | 8B fp16 overlay | 14B packed-WMMA |
|---|---|---|
| pp512 | 3448 tok/s | 1829 tok/s |
| pp1024 | 3209 | 1726 |
| pp2048 | 2792 | 1510 |
| pp4096 | 2234 | 1255 |
| Warmstart | TC + UPCAST + UNROLL(0,8) | TC only |
| Weight format | Contiguous fp16 (`_pf16_w`) | View chain from Q4_K packed bytes |
| Activated by | `FULL_RESIDENT_OVERLAY` policy | `TINYGRAD_PREFILL_PACKED_WMMA=1` (now default) |

## Why the warmstart can't take UPCAST/UNROLL

The 8B path feeds a pre-materialized contiguous `_pf16_w` into the matmul. The scheduler sees a clean fp16 × fp16 GEMM and can apply UPCAST/UNROLL aggressively.

The 14B path constructs the weight through a view chain:

```
packed_weight.bitcast(uint16).reshape(blocks, halfwords).pad(…)
  .reshape(blocks, 128, 1).expand(blocks, 128, 2).reshape(n, k).bitcast(half)
```

The `expand(…, 2)` is load-bearing for three reasons:

1. **Element-count arithmetic.** Pad goes to 128 halfwords per block. Expand doubles to 256 to match `block_elems`. Without it the reshape(n, k) fails.
2. **Range analysis in postrange.** The broadcast from expand routes postrange's reduce-dimension propagation correctly. Removing it causes a matmul with K=20 (attention score) to be misidentified as K=5120 and fail the packed-weight validation.
3. **PackedPrecontractOperandTemplate validation.** `original_axes[0].vmax+1` must equal `packed_weight.rows`. The expand creates the right axis range. The row-axis check is dead code for codegen (packed dequant reads directly from the PARAM), but can't be removed because of reason 2.

Adding UPCAST/UNROLL crashes `devectorize_symbolic` — the expand's GEP indices don't survive loop restructuring.

## What was tried

1. **UPCAST/UNROLL on original view chain.** Crash in `devectorize_symbolic` at `symbolic.py:207` — GEP indices out of range after UNROLL restructures K.
2. **`.contiguous()` on the view chain.** Breaks packed PARAM detection — scheduler can't trace through the materialized tensor to find the original packed bytes. `PackedPrecontractOperandTemplate` rejects it.
3. **Drop expand, pad-to-256, direct reshape.** Range analysis breaks — postrange routes the wrong reduce dimension through the TC analysis.
4. **Relax row-axis validation in kernel_lds.py.** Dead end — the validation isn't the blocker, the range analysis is.

## The real ceiling

The hand kernel `build_gemm_lds2_q4k` (`extra/qk/prefill/wmma.py:501-654`) does Q4_K dequant in registers with hand-tuned tile geometry and hits ~3400 tok/s on 8B. Scaled to 14B: ~1940 tok/s. Current packed-WMMA: 1829 tok/s. Gap: ~6%.

The packed-WMMA route uses the scheduler's generic WMMA lowering for the view-chain matmul. The hand kernel picks optimal geometry directly. Closing the gap means either:

- **Make the scheduler Q4_K-aware.** Add a packed-quant UOp or lowering path that the scheduler recognizes natively, so it can pick tile sizes, wave counts, and LDS layouts that match what the hand kernel does. Compiler project.
- **Accept the ceiling.** 1829 tok/s beats llama.cpp, and the 6% gap may be irreducible without a first-class packed-quant primitive in the scheduler.

## Key files

- `extra/qk/prefill/packed_wmma_prefill_candidates.py` — packed-WMMA candidate dispatch, view chain, warmstart entry
- `tinygrad/llm/prefill_routes.py` — route dispatch, `packed_wmma_prefill_enabled()` (now defaults True)
- `tinygrad/codegen/opt/kernel_lds.py:175-215` — `PackedPrecontractOperandTemplate` validation
- `tinygrad/codegen/opt/postrange.py:530-595` — warmstart key computation, `apply_opts`
- `extra/qk/prefill/wmma.py:501-654` — hand kernel `build_gemm_lds2_q4k` (reference for optimal geometry)
- `tinygrad/llm/model.py:264-276` — `_prefill_v2_opts` (the richer warmstart the 8B path uses)

## The packed-WMMA default change

Commit `6ca798568`: flipped `TINYGRAD_PREFILL_PACKED_WMMA` default from `0` to `1`. The route is fail-closed — any ungated (quant, role, shape) combo silently declines and falls through to direct-packed. Verified at 1829 tok/s pp512 on 14B.
