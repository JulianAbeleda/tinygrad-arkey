# Flash Attention Fusion Route — Implementation Scope

## Goal

Extend the composite REDUCE to accept a second input (the V tensor), so the `online_softmax` (3-slot: m, l, acc) combine can consume both `(score, v)` at each KV iteration. This enables a single score-resident kernel that fuses softmax + PV accumulation without materializing the full score/probability tensor.

## Architecture principle

One centralized path: the composite REDUCE is the single mechanism for score-resident attention. No hand kernels, no second WMMA path, no per-route duplication.

## Task: extend composite REDUCE with a second input

### Step 1: extend CompositeReduce to carry V input

File: `tinygrad/uop/ops.py`

- Add an optional `v_uop` field to `CompositeReduce` (a UOp reference to the V tensor)
- Extend `UOp.composite_reduce()` signature: add `v_uop: UOp = None` parameter
- Store `v_uop` in the CompositeReduce instance

### Step 2: index V at the current reduce position during lowering

File: `tinygrad/codegen/late/devectorizer.py` — `reduce_to_acc`

When lowering a composite REDUCE with `composite.v_uop is not None`:
- Create a LOAD/INDEX UOp that reads from `v_uop` at the current reduce_range position
- The V tensor is indexed to match the reduce loop iteration
- Pass the loaded V value to the combine function

The V indexing: if scores reduce over axis=3 (KV), and V has shape (B, H, KV, Hd), the V tensor needs to be indexed at axis=2 (KV). The combine function receives the score (from `inp`, already indexed by reduce_range) and the V value (loaded at the same reduce_range position).

Simpler approach: reshape V before the composite reduce so its KV axis aligns with the reduce axis. For attention:
- scores shape: (B, H, T, KV) — axis=3 is KV
- V reshaped: (B, H, 1, KV, Hd) — axis=3 is also KV
- The reduce loop indexes both at the same position

### Step 3: update combine functions

File: `tinygrad/codegen/late/composite_combines.py`

Update `online_softmax` combine (in COMBINE_REGISTRY) to:
- Accept `v_inp` parameter (the V value at current iteration, loaded by the devectorizer)
- Compute: `acc_new = acc_old * corr + exp_score * v_inp`
- Return `acc / l` as the final output

The devectorizer passes `v_inp` to the combine function.

### Step 4: update rangeify rewrite to emit full (m,l,acc) composite

File: `tinygrad/schedule/rangeify.py` — `_flash_attn_match`

Currently matches ADD REDUCE and emits `online_softmax_l` (2-slot). Update to:
- Recognize the full attention pattern: `(Q@K.T * scale).softmax(-1) @ V`
- Trace from the ADD REDUCE backward to find the QK contraction, and forward to find the PV matmul
- Identify the V tensor from the PV matmul's inputs
- Emit `online_softmax` (3-slot) composite REDUCE with `v_uop=V`

The rewrite replaces BOTH the softmax and the PV matmul with ONE composite REDUCE.

Matching strategy:
1. Start from the ADD REDUCE (softmax denominator — already matched)
2. Walk forward: the ADD REDUCE's output feeds into a DIV (normalization) and then a MUL (PV)
3. The MUL feeds into another REDUCE (PV accumulation)
4. Identify the V tensor as the non-score input to the MUL
5. Replace the PV REDUCE + softmax with one composite REDUCE

### Step 5: correctness gate

```python
# Verify: max_rel_err ≤ 1e-2 vs fp32 reference
# Verify: no full T×KV score buffer in generated code (check BUFFER sizes)
# Verify: acc/l output matches standard attention
```

### Step 6: Phase 4 WMMA

With TC_OPT=2, verify that the QK contraction (separate kernel) uses WMMA. The composite REDUCE kernel contains the softmax+PV fusion — WMMA for PV depends on whether the scheduler can apply TC to the PV accumulation inside the composite. If not, that's OK for now — document it.

### Step 7: Phase 5+6 model wiring

- Enable `prefill_flash_attn` flag in model config
- Benchmark 8B and 14B routes
- Compare attention `tm` vs materialized baseline
- Verify end-to-end prefill tok/s

## Non-negotiable rules

1. No hand kernels, custom_kernel, flash_kernels.py
2. No optimizer carve-outs for composite REDUCE
3. No combine_fn string dispatch in lowering (use registry)
4. Keep tests green (45 passed, 4 xfailed baseline)
5. Commit each step separately on master, push
6. Remove debug prints before committing
7. No Co-Authored-By trailers

## Starting point

Repo: `/home/ubuntu/tinygrad-arkey`, branch `master`, HEAD `fa46b48a5`
Test suite: `DEV=AMD .venv/bin/python -m pytest test/unit/test_composite_reduce.py test/unit/test_amd_isa_wmma.py -q` — 45 passed, 4 xfailed

## Key files

- `tinygrad/uop/ops.py` — CompositeReduce class, UOp.composite_reduce
- `tinygrad/codegen/late/composite_combines.py` — combine functions, COMBINE_REGISTRY
- `tinygrad/codegen/late/devectorizer.py` — reduce_to_acc, pm_reduce
- `tinygrad/schedule/rangeify.py` — _flash_attn_match, _get_kernel_graph
- `tinygrad/codegen/late/flash_attn.py` — flash_attention() public API
- `tinygrad/llm/model.py` — model attention wiring (lines 560-630)
