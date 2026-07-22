# Composite-Accumulator REDUCE — Progress Report

Date: 2026-07-22
Commits: Phase 0 (111c4bf1a), Phase 1 (e4e01a74e)

## Phase 0: Design — COMPLETE

Design doc at `docs/flash-composite-reduce-design-20260722.md`. Specifies:
- CompositeReduce + AccumulatorSlot types extending REDUCE arg[0]
- UOp sub-graph for combine function (online-softmax monoid)
- Lowering in reduce_to_acc: multiple DEFINE_ACCs with custom combine
- WMMA attachment: TC opt finds inner QKᵀ/PV contractions inside composite reduce
- Attention pattern matching: rangeify/graph_rewrite restructures softmax(matmul)@v

## Phase 1: Toy Composite Reduce — COMPLETE

### What was built
- `AccumulatorSlot` and `CompositeReduce` NamedTuples in `tinygrad/uop/ops.py`
- Composite-aware lowering in `tinygrad/codegen/late/devectorizer.py:reduce_to_acc`
- Spec updated in `tinygrad/uop/spec.py` to accept CompositeReduce args

### Verification
- Unit test: a REDUCE with (ADD, MAX) composite slots creates 2 DEFINE_ACCs
  (ctx.acc_num = 2), each with independent init/body/end
- Test suite unregressed: 36 passed, 4 xfailed (test_amd_isa_wmma.py)
- Normal reduces (sum, max) and matmuls continue to work

### What this proves
The composite accumulator primitive IS expressible in the current architecture.
The devectorizer lowering is straightforward and backward-compatible.

## Phase 2-4: NOT STARTED — requires multi-week rangeify integration

### What's needed
1. **Rangeify integration:** the scheduler must emit REDUCE with CompositeReduce
   arg when it recognizes the attention pattern (QKᵀ → softmax → PV). This
   requires modifying rangeify's pattern matching and kernel formation.

2. **Online-softmax combine as UOp sub-graph:** the combine function must be
   encoded as a UOp tree that `reduce_to_acc` can lower. The combine does:
   `m_new = max(m, score); corr = exp(m - m_new); l_new = l*corr + exp(score-m_new); acc_new = acc*corr + exp(score-m_new)*v`

3. **TC opt multi-contraction support:** postrange.py's `_apply_tc_opt` currently
   assumes one TC-tagged REDUCE (`get_single_element` at line 391). For composite,
   it must find and WMMA both the QKᵀ (over Hd) and PV (over KV_block) contractions
   inside the composite reduce body.

4. **Pipeline plumbing:** SINK/KernelInfo construction for composite reduces,
   proper candidate_context propagation, range expansion through rangeify.

### Risk assessment (unchanged from design doc)
- **Medium:** Full pipeline integration touches rangeify, scheduler, and codegen
- **Medium:** TC opt multi-contraction support requires careful tag management
- **Low:** Devectorizer lowering is proven (Phase 1)

### Recommendation
Pause here for design review. The Phase 1 result confirms the composite
accumulator is expressible at the devectorizer level. Phase 2 (rangeify
integration) is the hard part — it's genuinely multi-week scheduler work
that should proceed after the design and toy proofs are reviewed.
