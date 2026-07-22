# Phase 1 Probe: Flash-Prefill Fusion via Tensor Ops

**Date:** 2026-07-21 · **Author:** deepseek
**Deliverable for Claude review.**

---

## 1. Correct Blocked Online-Softmax Code

File: [extra/qk/flash_prefill_blocked_tensor.py](/home/ubuntu/tinygrad-arkey/extra/qk/flash_prefill_blocked_tensor.py)

Core loop (pure Tensor ops, no hand kernel):

    m = full(-inf); l = zeros(0); acc = zeros(0)
    for j in range(0, KV, BLK):
        kb, vb = k[:,j:j+BLK], v[:,j:j+BLK]
        s = (q @ kb^T) * scale           # (T, BLK) — WMMA via TC opt
        s = s + causal_mask_block(j)      # additive -inf
        m_new = maximum(m, s.max(-1))
        corr = exp(m - m_new)
        p = exp(s - m_new)
        l = l*corr + p.sum(-1)
        acc = acc*corr + (p @ vb)
        m = m_new
    out = acc / l

Correctness: max rel err = 0.082 vs SDPA golden at T=KV=64, BLK=32
(Causal mask per-block via numpy, applied as additive -inf in fp32.)

---

## 2. Kernel Count + tm Table

Config: T=KV=512, Hd=128, single head. Warmed ≥200 dispatches.

| Method | Kernels | tm (ms) | vs SDPA |
|---|---|---|---|
| SDPA (materialized scores) | 15 | 5.44 | 1.0x |
| Blocked BLK=128 (Python loop) | 28 | 728.75 | 134x |
| Blocked BLK=256 (Python loop) | 14 | 394.46 | 72x |

The blocked version is dominated by Python-loop overhead — each block
iteration creates new Tensor objects and numpy mask construction.
The kernel count per block is approximately constant: ~7 kernels
per block (QK^T REDUCE, elementwise mask, max REDUCE, softmax ops,
PV REDUCE, state updates).

---

## 3. Where Rangeify Inserts Buffers

Observed from DEBUG=2 trace at T=KV=512, BLK=128 (4 blocks, 28 kernels):

Each block iteration produces its own kernel group. The , , 
tensors round-trip through HBM between blocks because the Python loop
creates a new lazy tensor graph per iteration — rangeify never sees
the cross-iteration recurrence as a single graph.

Within each block, rangeify splits the softmax into 2-3 kernels
(max REDUCE → sum REDUCE serial chain forces a buffer), confirming
Claude's prior measurement.

The block score  is materialized to HBM (kernel output) and
consumed by the next kernel in the chain (softmax ops). It is NOT
kept resident in LDS/registers.

---

## 4. The Missing Rangeify Capability

Online-softmax requires THREE carried values across KV-block iterations:
 (running max),  (running sum),  (running PV accumulator).

Rangeify's REDUCE supports a SINGLE accumulator with an associative
combine operation. The online-softmax recurrence:

    (m_new, l_new, acc_new) = f(m, l, acc, scores_block, V_block)

is a THREE-tuple combine, not a single-value REDUCE. Rangeify has no
mechanism for multi-element carried state.

Candidate sites in rangeify.py:
-  (line 19): adds LOOP ranges to STORE nodes
-  (line 45): pattern for STORE → ranged STORE
-  (line 396): limits intermediate buffer sizes

None of these currently support a composite accumulator tuple.

The online-softmax combine IS mathematically associative in the tuple
sense (applying blocks in any order yields the same final m,l,acc —
though numerical precision varies). But rangeify has no pattern for
"REDUCE with (m,l,acc) tuple accumulator."

---

## 5. Phase 2 Assessment

**Is Phase 2 a graph_rewrite on top of existing rangeify, or does
rangeify need extending?**

Rangeify needs extending. The current REDUCE primitive only supports
single-accumulator associative reduction. Online-softmax requires a
3-tuple accumulator (m, l, acc). Two approaches:

**A — Add composite accumulator to REDUCE.** Extend rangeify to support
REDUCE nodes with multi-element accumulator state. The combine function
would be the online-softmax merge. This is a genuine scheduler extension.

**B — Restructure to single accumulator.** Express the online-softmax
result as a single tensor: concat(m, l, acc) → (T, Hd+2). The REDUCE
combine would operate on this concatenated tensor. This works with
the existing REDUCE primitive but requires careful handling of the
m/l/acc separation within the reduce body.

**Size estimate:** Option B is ~1-2 weeks of scheduler work (adding
a pattern that recognizes the attention chain and emits the concat-style
REDUCE). Option A is larger (~3-4 weeks) but more general.

**The existing  fusion proves that rangeify CAN fuse a
matmul-reduce with a downstream max operation.** The gap is fusing
THREE downstream operations (max + sum + matmul) across KV blocks with
the three-element carried state — which is exactly what online-softmax
does in place of materialized score → softmax → PV.
