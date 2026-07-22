# Shared Flash Attention — Status Report

## Phase 0: Baselines — COMPLETE (de7d4d356)
- 8B/14B shapes documented, ceilings measured, baselines recorded

## Phase 1: Generic Combine — PARTIAL (ba9c247f8)
- R1-R3 refactor: combine functions separated from reduce_to_acc
- No-range handler made generic (COMBINE_STEP_REGISTRY)
- Remove optimizer carve-outs
- Composite survives expander generically
- TODO: add second unrelated combine, packed element support, partial/full UNROLL

## Phase 2: Multi-Output — PARTIAL
- REDUCE_SLOT in Ops enum, spec rules, dispatch fixes
- Structural tests pass (one REDUCE, two REDUCE_SLOTs)
- TODO: full pipeline realization of both slots from one reduce

## Phase 3: Rangeify Rewrite — BLOCKED
This is the fundamental unsolved problem. Rangeify must recognize the
canonical attention graph and emit a composite REDUCE that replaces
the full softmax+matmul chain. This requires:
- Pattern matching the attention subgraph in rangeify IR
- Emitting a score-resident composite REDUCE structure
- Keeping QK and PV as visible contractions inside the composite

The composite REDUCE lowering (Phase 1), multi-output (Phase 2), and
WMMA attachment (Phase 4) all depend on rangeify emitting the structure.
Until rangeify can produce the composite REDUCE from the attention graph,
the remaining phases cannot proceed.

Current workaround: manual construction via UOp.composite_reduce().
This proves the lowering works but doesn't satisfy the requirement that
rangeify automatically recognizes and rewrites the attention graph.

## Phases 4-6: Blocked on Phase 3
