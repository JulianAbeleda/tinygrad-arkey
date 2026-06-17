# AMD flash-attention reference — extraction audit (Phase A, 2026-06-17)

Audit of `extra/gemm/amd_flash_attention.py` as a base for a Qwen3-8B prefill flash kernel. **Verdict: the
STRUCTURE is exactly what Phase 5 lacked (high-occupancy WMMA + LDS tiling + register-resident online softmax
via warp shuffles, already `custom_kernel`-shaped) — but it does NOT run under this fork's tinygrad** (shapeless
`Ops.CUSTOM` warp-shuffles trip shape inference), and it lacks GQA + causal. It is **not a drop-in base**;
adopting it is a dedicated kernel-engineering arc (revive + adapt), not an extraction.

## The 12 audit questions

1. **Shapes:** inputs `(B*H, N, D)`; requires `N % 64 == 0` and `D % 16 == 0`. Defaults B=1,H=32,N=1024,D=64.
2. **WMMA/tensor cores:** YES — `Ops.SHAPED_WMMA`, 16×16×16, `arg=((16,16,16),'AMD',32)`; two WMMA matmuls
   (S=Q@Kᵀ at line 99, acc+=P@V at line 163).
3. **Threads/workgroup:** **128** (`WARP_SIZE 32 × WAVES_M 4 × WAVES_N 1`) — high occupancy (vs Phase-5's 16–32).
4. **LDS:** `QP_lds[64, D+4]` (slot 0, Q→P reuse) + `KV_lds[64, D+4]` (slot 1, K→V reuse), fp16; one
   `BLOCK_N=64` KV tile per iteration → ~34 KB at D=128, within the 64 KB ceiling.
5. **Softmax:** register-resident **online softmax** (`m_i`/`l_i`/`acc` in REG) with **warp-shuffle tree
   reductions** (`ds_bpermute` across 16 lanes) for row max/sum, and online `alpha`/`beta` correction across KV
   tiles. NOTE: this proves online softmax IS expressible — via REG state + LOOP ranges + warp shuffles, NOT the
   single-REDUCE-axis accumulator the linearizer rejected in our earlier Attempt A.
6. **Memory map:** contiguous `(B*H, N, D)`, tiled `[N//BLOCK, BLOCK, D]`, indexed by `block_bh`, `block_m`/`n_tile`.
7. **Causal:** **NO** — full self-attention, no mask.
8. **GQA/MQA:** **NO** — `(B*H,N,D)` assumes `Hq == Hkv` (K/V per query head). No `kv_head = h//G` mapping.
9. **Fixed sizes:** `BLOCK_M=BLOCK_N=64`, WMMA 16, `WAVES_M=4,WAVES_N=1`, 128 threads — module-level constants;
   `D%16`, `N%64`. Changing shape = edit constants (and re-derive the WMMA fragment reshapes).
10. **custom_kernel-compatible:** SHAPED as one — `Tensor.custom_kernel(o,q,k,v, fxn=amd_flash_attention)` in
    `__main__`, timed via `GlobalCounters.time_sum_s`@DEBUG=2 — **but it does NOT compile** here (see below).
11. **Smallest extractable:** it's a single function + 3 warp helpers + constants; not meaningfully
    sub-extractable. Smallest *run* = small B/H/N/D — but it runs at NONE (broken at every shape tried).
12. **Gap to Qwen3-8B prefill:** (a) **BROKEN** under current tinygrad — must fix first; (b) **no GQA**
    (Hkv=8<Hq=32 needs `kv_head=h//4`); (c) **no causal**; (d) **D=128** also breaks (same error; only D=64
    exercised upstream); (e) assumes **KV length == query length** (square N×N) — a prefill chunk at start_pos
    attends `start_pos+T` keys (KV≠N) except the first chunk (sp=0, KV=T=512, square — maps directly).

## The blocker (empirical)

At line 116 (`m_ij.reshape(TM,1)…`) shape inference fails: `AssertionError: None input shape not supported for
Ops.MAX`. Root cause: `warp_reduce_max`/`warp_reduce_sum` build `Ops.MAX`/`+` over `warp_shfl_xor(...)`, whose
`Ops.CUSTOM` (`ds_bpermute`) result is **shapeless** in current tinygrad. Feeding a shapeless UOp into
`Ops.MAX` (and then `reshape`) trips the `all(x is not None for x in input_shapes)` assert. Confirmed at **D=64
AND D=128**; a quick "reshape the CUSTOM to val.shape" patch does NOT fix it (val is scalar `shape ()`, and the
CUSTOM op has no shape to begin with). So the reference is **stale against this fork's tinygrad** — its
warp-shuffle reductions predate the current shape-inference rules for `Ops.CUSTOM`.

## Decision — STOP (Phase B/C not run; reference doesn't compile)

Per the decision table, this is outcome **(1)+(4)**: the reference is `custom_kernel`-shaped but **cannot be
called/compiled as-is** (shapeless-CUSTOM warp-shuffle vs current shape inference), and it assumes
incompatible shapes (no GQA, no causal, square N×N). The missing bridge is **shape-carrying `Ops.CUSTOM`
(warp-shuffle) ops, or a warp-reduce restructured to not feed shapeless UOps into shape-requiring ops** — a
codegen/kernel fix, deliberately NOT attempted here.

**This is the same wall-class as the other gated levers** (2nd compute ring; BEAM-hang; per-d recompute): the
abstraction can express the *math/locality* but the *production kernel path* (here: warp-shuffle reductions
under current shape rules) needs surgery. Flash-prefill remains banked.

**Reopening flash-prefill is a dedicated arc**, roughly: (i) revive the warp-shuffle reductions (shape-aware
CUSTOM, or replace ds_bpermute reductions with LDS-based reductions that ARE shape-clean — we proved LDS works
in Phases 2–4); (ii) add GQA (`kv_head=h//G`, no repeat_interleave); (iii) add causal; (iv) target D=128, T=512,
KV-tiled; (v) prove it beats SDPA on DEBUG=2 GPU time. That is `[codegen]`/`[test]` kernel engineering, not an
extraction — and a real decision to fund, not an automatic next rung.

## What this arc DID establish (kept)
- The high-occupancy WMMA+LDS+warp-reduce **online-softmax** flash structure exists in-repo and is the right
  target (correcting "online softmax is unexpressible" — it's expressible via REG state + warp shuffles).
- The precise blocker to reusing it: shapeless `Ops.CUSTOM` warp-shuffles vs current shape inference.
- LDS reductions (Phases 2–4) are a shape-clean alternative to ds_bpermute warp reductions if the kernel is rebuilt.
