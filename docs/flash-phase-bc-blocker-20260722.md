# Phase B/C Blocker Analysis

Date: 2026-07-22

## Phase C: WMMA on both attention matmuls

NOT fundamentally blocked. With fp32 scale and fp16 PV inputs, both matmuls
get WMMA:

| Configuration | WMMA kernels | Notes |
|-------------|-------------|-------|
| `a@b` fp16, TC_OPT=2 | 1 | Baseline works |
| `Q@K^T` no scale | 1 | Bare matmul works |
| `Q@K^T * fp32 scale` | 1 | Scale in separate kernel, matmul stays clean |
| `Q@K^T * fp16 scale` | 0 | Scale fuses with matmul, breaks TC MUL pattern |
| `P @ V` fp16 | 1 | PV matmul works |
| Full attn (fp32 scale, fp16 softmax→PV) | **2** | Both matmuls WMMA! |

The recipe: fp32 scale (separates from QK^T) + softmax in fp32 + cast probs to
fp16 before PV. This is a 3-kernel solution where both matmuls are WMMA.

## Phase B: Composite REDUCE in attention

BLOCKED. The composite REDUCE requires NOOPT=1 (preserves loop structure for
the online-softmax correction). But NOOPT=1 breaks the rest of the attention
graph (mismatched dtype WHERE ops at spec_program verification).

Without NOOPT, the expander collapses the REDUCE loop to a horizontal
vector reduction, which the composite lowering can't handle (online-softmax
needs sequential state updates, not horizontal reduction).

## Root cause

The expander (`fix_reduce_unroll` in expander.py) converts RANGE-based REDUCE
loops to UNROLL-based horizontal reductions. For composite reduces, this
destroys the loop structure needed for the correction combine.

The fix: `fix_reduce_unroll` should skip composite REDUCEs. This was tested
and works at the unit level, but the lowered combine then hits shape mismatches
between scalar accumulators and vector inputs from the UPCAST/UNROLL machinery.

## What's needed

1. `fix_reduce_unroll` skips composite REDUCEs (1-line check)
2. The composite lowering handles vector inputs from the scheduler's UPCAST
   (or the scheduler doesn't UPCAST composite reduces)
3. Rangeify emits composite REDUCE from the attention pattern (Phase B proper)

Items 1-2 are the minimum to unblock — they let the composite survive the
expander with correct shapes. Item 3 is the rangeify pattern match.
