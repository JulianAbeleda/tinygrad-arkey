
# Complete Plan: Flash-Fusion Scheduler Primitive (Option B)

**Status:** B-M0 confirmed. B-M1 through B-M4 to go.

---

## B-M0: Insertion Point ✓ (DONE)

**Site:** Tensor.realize(), between linear_with_vars() and run_linear().

The detection pass in flash_fusion.py sees all 4 attention CALLs in the LINEAR
graph (QK^T REDUCE → softmax-max REDUCE → softmax-exp REDUCE → PV REDUCE),
connected by shared BUFFERs. The old insertion in _get_kernel_graph was wrong
— the graph was already split by then.

---

## B-M1: Build the Fused UOp Graph

### What: Construct a new SINK that replaces the 4-CALL chain.

Inputs: Q, K, V BUFFERs (from the LINEAR args). Mask is folded into the online-softmax
as additive -inf on causal blocks.

Output: one CALL with a SINK that does:



### How: Construct the UOp graph programmatically.

1. Extract Q, K, V shapes from the CALL args
2. Build the REDUCE over K_blocks (outer loop)
3. Inside: LDS staging subgraph for K_block, V_block
4. Inside: QK^T REDUCE (this auto-lowers to WMMA via TC opt in B-M2)
5. Inside: online-softmax elementwise ops
6. Inside: PV REDUCE
7. WAR barrier between iterations

### Validation:
- Run fusion at T=KV=512, single head (Hkv=1, G=1)
- Diff output vs materialized SDPA reference
- Must pass: max relative error < 1e-2 (fp16 tolerance)

### Key risks:
- UOp graph construction is complex and error-prone
- LDS staging requires correct UOp annotations (DEFINE_LOCAL, barrier)
- The recurrence (m, l, acc across K_blocks) must preserve UOp ordering

### Fallback if too complex:
- Build a simplified version: no LDS staging, just block the K-range without
  explicit LDS (rely on cache). This is easier to construct and still proves
  the pattern. LDS optimization is B-M3.

---

## B-M2: WMMA Verification

### What: Confirm TC opt fires on the fused matmuls.

The QK^T and PV REDUCEs in the fused graph are plain fp16 matmuls. The existing
postrange._apply_tc_opt recognizes REDUCE patterns on fp16 inputs and lowers
them to WMMA. Verify via DEBUG=4 trace that __WMMA_16_16_16_half_float appears
in the compiled kernel.

If TC opt does NOT fire (REDUCE pattern differs after fusion):
- Debug: add a minor pattern extension to tc.py:amd_rdna3 to recognize
  the fused REDUCE shape
- This is a small pattern fix, not a new emitter

---

## B-M3: Geometry / Occupancy Tune

### What: Find optimal tile sizes.

Two parameters exposed via env vars:
- TINYGRAD_FLASH_TK — K/V block size (default 128)
- TINYGRAD_FLASH_M_TILE — query rows per workgroup (default from existing)

Sweep via BubbleBeam (extra/qk/bubblebeam_futuresight.py) pattern:
- TK ∈ {64, 128, 256}
- Stop when tm improvement < 5%

Select best config. Commit as defaults.

---

## B-M4: Gate Report

### What: Hard numbers replacing the theoretical projection.

Measure at 14B, T=KV=4096, all heads:
1. Fused kernel: tm, compute_frac, mem_frac
2. SDPA baseline: same
3. Correctness diff vs reference

### Gate (from MVP scope §5):
- correctness held (hard prerequisite)
- compute_frac higher + mem_frac lower than SDPA
- Score HBM deleted ≥ 80% of SDPA spill
- Faster absolute tm

### Deliverable:
Two-ceiling table, deleted HBM bytes, absolute tm, correctness diff, GO/NO-GO.

---

## Files

| File | State | Notes |
|---|---|---|
| tinygrad/schedule/flash_fusion.py | Detection phase | B-M1 adds graph construction |
| tinygrad/tensor.py | 1-line intercept | Done |
| tinygrad/schedule/rangeify.py | Reverted | Old insertion removed |

## Deferred (post-gate)
- GQA head-sharding
- Multi-KV-size coverage
- Routing integration
- 8B validation
- Autotuner
