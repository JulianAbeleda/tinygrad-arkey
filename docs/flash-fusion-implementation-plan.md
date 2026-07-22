# Implementation Plan: Flash-Fusion Scheduler Primitive

**Author:** deepseek · **Date:** 2026-07-21
**Parent:** `docs/flash-fusion-full-build-plan.md`

---

## B-M0: Insertion Point Trace ✓ (COMPLETED)

**Site:** `_get_kernel_graph()` in `tinygrad/schedule/rangeify.py:657`, after
`run_rangeify(tsink)` and before `pm_limit_bufs`.

```
655:  tsink = graph_rewrite(tsink, pm_syntactic_sugar+pm_mops+earliest_rewrites, ...)
656:  tsink, rctx = run_rangeify(tsink, ...)
657:→ tsink = graph_rewrite(tsink, pm_flash_fusion, name="flash-fusion")  ← INSERT HERE
658:  tsink = graph_rewrite(tsink, symbolic+pm_reduce_simplify+..., ...)
```

**Confirmed:** The K-dim LOOP range exists for both QK^T and PV at this point.
The intermediate buffer is present but unsplit.

---

## B-M1: Correct Single-Kernel No-Spill Fusion

**Goal:** The rewrite produces a single ranged kernel whose output matches
the materialized SDPA reference bit-for-bit (fp16 tolerance).

### Step 1: Write the attention recognition pattern

File: `tinygrad/schedule/flash_fusion.py`

The pattern must recognize this UOp subgraph after rangeify:

```
STORE(out)
  └── REDUCE(K_loop:
        (REDUCE(K_loop: Q_vals × K_vals, init=0) → softmax elementwise) × V_vals,
        init=0)
```

Where `K_loop` is the same range for both inner and outer REDUCE. After
rangeify, the softmax is decomposed into elementwise ops (max, sub, exp, sum, div)
with the K_loop already applied.

Signature of the matcher:

```python
# Match: a STORE of a REDUCE over K whose inner src is a REDUCE over the same K range
# The inner REDUCE produces scores; elementwise ops between them are softmax
def match_flash_attention_single_head(store: UOp) -> UOp | None:
    """
    store: UOp(Ops.STORE) or UOp(Ops.END) containing Ops.STORE
    Returns the fused replacement UOp, or None if pattern doesn't match.
    """
```

### Step 2: Build the online-softmax recurrence

Given matched Q, K, V tensors and the K range, construct:

```
acc = zeros(M, Hd)
m = -inf × M
l = zeros(M)

for k_block in K_blocks:
    k_tile = K[k_block : k_block + TK]
    v_tile = V[k_block : k_block + TK]
    scores = Q × k_tile^T        # M × TK
    scores = scores * scale + mask[k_block]
    m_new = max(m, row_max(scores))
    alpha = exp(m - m_new)
    l_new = alpha * l + row_sum(exp(scores - m_new))
    acc = (l / l_new) * alpha * acc + (scores - m_new).exp() × v_tile / l_new
    m, l = m_new, l_new
    barrier()  # WAR: k_tile/v_tile in LDS must be consumed before next load
```

This is a new REDUCE over K-blocks (not K-elements). The K-blocks are derived
by tiling the original K range: `K_blocks = K_range // TK`.

### Step 3: LDS staging for K/V blocks

Each K-block iteration:
1. Cooperative load of TK×Hd elements of K and V into LDS (DEFINE_LOCAL)
2. Barrier after staging
3. Compute scores against staged K (Q × LDS_K^T)
4. Online-softmax merge
5. PV accumulation against staged V
6. Barrier before next iteration (WAR: stage writes vs current reads)

The LDS pattern is identical to `flash_kernels.py` staging but expressed as
a rewrite of the range structure, not as a hand-authored kernel.

### Step 4: Validation

```python
# Test config: 14B, single head (Hkv=1, G=1), T=KV=2048, fp16, causal
# Reference: model.py SDPA path (materialized scores)
# Check: max relative error < 1e-2 (fp16 tolerance)
```

Commit with correctness report. If correctness fails, debug pattern matching
before proceeding.

---

## B-M2: WMMA On

**Goal:** Confirm the fused kernel's matmuls lower to WMMA automatically.

### Step 1: Verify TC opt fires

Run with `DEBUG=4` and confirm the kernel trace shows `__WMMA_16_16_16_half_float`
in the generated assembly.

### Step 2: Confirm no hand emission

The QK^T and PV matmuls are plain fp16×fp16 ops in the UOp graph. The TC opt
in `postrange._apply_tc_opt` recognizes them and inserts the WMMA lowering.
No new WMMA code.

If TC opt does NOT fire on the fused graph, debug why the matmul nodes aren't
being recognized (likely: the REDUCE structure changed enough that the
existing pattern doesn't match; may need a minor extension to the TC opt's
pattern).

---

## B-M3: Occupancy / Geometry Tune

**Goal:** Find the optimal tile sizes (TK, M-tile) for the fused kernel.

### Step 1: Parameterize

Expose two geometry parameters via env vars:
- `TINYGRAD_FLASH_TK` — KV block size (default 128)
- `TINYGRAD_FLASH_M_TILE` — query rows per workgroup (default 4)

### Step 2: BubbleBeam sweep

Use `extra/qk/bubblebeam_futuresight.py` pattern to sweep over these parameters
and measure `tm` for each. The sweep space is small (2 params, ~10 candidates).

### Step 3: Select best

Pick the (TK, M_tile) with lowest `tm`. Commit as defaults.

---

## B-M4: Gate Report

**Goal:** Replace the 2.45× theoretical projection with a measured number.

### Measurements

Run at 14B, T=KV=4096 (strongest signal), all heads:
1. Fused kernel: measure `tm`, `compute_frac`, `mem_frac`
2. SDPA baseline: measure same
3. Llama reference: measure same (for calibration, not gate)

### Gate conditions (from MVP scope §5)

With correctness held:

**GO** if:
1. compute_frac higher than SDPA's AND mem_frac lower than SDPA's
2. Score HBM deleted ≥ 80% of SDPA's score spill
3. Faster absolute `tm` than SDPA

### Deliverable

Two-ceiling table (SDPA vs fused), deleted HBM bytes, absolute tm,
correctness diff, GO/NO-GO recommendation.

---

## Files Changed

| File | Change |
|---|---|
| `tinygrad/schedule/flash_fusion.py` | **NEW** — pattern matcher + rewrite |
| `tinygrad/schedule/rangeify.py` | 1-line insertion at line 657 |
| `test/unit/test_flash_fusion.py` | **NEW** — correctness + WMMA validation |

## Breadth Explicitly Deferred

- GQA head-sharding (single head in M1-M2; GQA comes after gate passes)
- Multi-KV-size coverage  
- Routing integration (`prefill_routes.py`)
- 8B validation sweep
- Static→dynamic autotuner flip
- quant-KV / concrete-KV integration

---

## Fallback

If rangeify genuinely cannot express the cross-block `(m, l, acc)` recurrence
without a buffer (the three running values must live across K-block iterations),
bank the precise blocker and stop. Document why. Do not regress to a hand kernel.
