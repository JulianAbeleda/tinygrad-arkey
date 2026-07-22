# Full Build Plan: Flash-Fusion Scheduler Primitive (Option B)

**Author:** deepseek (plan), Claude (scope) · **Date:** 2026-07-21
**Parent:** docs/flash-prefill-wmma-mvp-scope-20260721.md §7 Option B

## Summary

Teach the rangeify scheduler to fuse attention — keeping the score tile in
LDS/registers (never materialized to HBM) — by rewriting the UOp graph at the
reduce-boundary between QK^T and PV. WMMA is free via the existing TC opt.
No hand kernel ships.

## Key Changes

### B.1 — Insertion point (line ~657 in rangeify.py:_get_kernel_graph)

After run_rangeify(), both QK^T and PV share the same K-dim LOOP range but
are separated by an intermediate BUFFER (the score spill). The rewrite must run
BEFORE pm_limit_bufs / split_kernels so the fused graph stays as one range nest.

### B.2 — Rewrite design

Pattern-match: REDUCE(K: softmax(REDUCE(K: Q x K^T)) x V)
Replace with:  REDUCE(K_blocked: online_softmax_merge over blocks)
Reuses online-softmax math from flash_kernels.py + additive mask.

### B.3 — WMMA (free)
postrange._apply_tc_opt + tc.py:amd_rdna3 lower fp16 matmuls to WMMA automatically.

## Milestones

| Milestone | What | Gate |
|---|---|---|
| B-M0 | Trace + confirm insertion point with Claude | Claude confirms |
| B-M1 | Correct single-kernel no-spill fusion | Output == SDPA reference |
| B-M2 | WMMA on for QK^T and PV | TC opt fires |
| B-M3 | Occupancy/geometry tune | BubbleBeam sweep |
| B-M4 | Gate report | Hard number replaces projection |

Hard stop after B-M0 until Claude confirms.

## Deferred: GQA, multi-KV, routing, 8B, autotuner.

## Fallback: bank precise blocker if rangeify cannot express recurrence without buffer. No hand kernel.
