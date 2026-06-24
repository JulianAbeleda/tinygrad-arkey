# Warp-reduction + WMMA revival arc (WR1–WR4) — 2026-06-17

The "rebuild the missing shape-safe primitives" arc, chosen after the flash-attention reference audit found
`extra/gemm/amd_flash_attention.py` is the right structure but stale against current tinygrad. Goal: revive the
sub-primitives the reference needs, smallest-first, so the high-performance flash path becomes live.

## WR1–WR3: DONE (warp-reduction primitive revived — kernel-authoring fixes only, no codegen surgery)

`extra/amd_warp_reduce.py`, `test/external/test_amd_warp_reduce.py` (all pass, AMD-gated, gfx1100 wave32).

- **WR1 shape-safe lane shuffle** (ds_bpermute): use **`Ops.CUSTOMI`** (carries `src[0]._shape`, ops.py:306),
  NOT `Ops.CUSTOM` (shapeless, ops.py:229 — the reference's break); and tie the lane to a **real `lidx`
  thread dim**, not a bare `AxisType.WARP` range (which renders a serial for-loop in a 1-thread workgroup → no
  wave → garbage). Correct xor-shuffle, all offsets.
- **WR2 warp_reduce_max/sum**: tree reduce over the wave. `reduce_sum` is fine inline; **`reduce_max` must
  STAGE the cross-lane read in a register** before `.maximum()` — `.maximum()` lowers to a ternary, and an
  inlined CUSTOMI puts the wave-level ds_bpermute inside a data-dependent conditional → lane divergence →
  garbage. (Also: binary max = `.maximum()`, NOT `UOp(Ops.MAX,(a,b))` which is a REDUCE.)
- **WR3 online-softmax row state** (max + denominator, register-resident): correct (m_err=l_err=0). The
  single-pass **online** recurrence (coupled m/l in one group store) hits a store-ordering hazard — codegen
  stores `m=m_new` then re-reads the m buffer for `alpha=exp(m_old-m_new)` → reads m_new → `alpha≡1` (rescale
  never applies). So WR3 uses **two single-accumulator passes** (max-reduce, then sum-reduce-using-max) — the
  shape-clean equivalent. (Same lesson as Attempt A / Phase 2: sequential single-accumulator beats coupled.)

**Net WR1–3:** the warp reductions the stale flash reference lacked now exist, shape-clean and tested — all via
kernel-authoring (no shape-system/renderer changes). This is real progress toward reviving flash.

## WR4: WALL — the manual `SHAPED_WMMA` custom-kernel idiom is stale (codegen-spec revival needed)

Goal was one WMMA Q@Kᵀ tile in a custom kernel with LDS. Both in-repo WMMA examples
(`amd_flash_attention.py`, `amd_copy_matmul.py` WMMA path) **fail verification** under current tinygrad:
1. `acc_frag.after(reduce)` → `AFTER[INDEX, RANGE]` (index-then-after). **Fixable** by reordering to
   `acc.after(reduce)[idx]` (after-then-index, matching the working FMA matmul `c_regs.after(k)[*rngs]`).
2. After that, `Ops.SHAPED_WMMA` with three `INDEX(ptr)` srcs (the q/k/acc fragments as indexes into LDS/reg
   pointers) fails verification — **the WMMA op's fragment-passing convention itself changed** and is NOT a
   one-line fix. This is codegen/UOp-spec work.

**Important contrast:** WMMA *works* via the optimizer/`OptOps.TC` path — `test/opt/test_tensor_cores.py`
PASSES, and it's the same warmstart-TC lever that made the prefill-v2 FFN fast. But that path applies TC to a
whole matmul kernel; it does NOT give the in-custom-kernel fragment control needed to fuse WMMA with
LDS-resident K/V tiles + online softmax in ONE kernel (the flash structure).

## Status / decision

- **WR1–3 banked** (warp-reduce primitive revived, tested) — a genuine asset.
- **WR4 is the wall**: manual `SHAPED_WMMA` in a custom kernel needs a codegen/UOp-spec revival (the fragment
  convention), deeper than WR1–3's authoring fixes. Same wall-class as the overlap 2nd-ring and BEAM-hang.
- **Flash-prefill remains banked.** The two live paths to revisit it, both substantial dedicated arcs:
  (a) revive the `SHAPED_WMMA` custom-kernel fragment convention (codegen), then assemble WR4→WR5→GQA/causal;
  (b) build flash from separate optimizer-TC matmuls (Q@Kᵀ, P@V) + the WR1–3 warp-reduce softmax, accepting
  score materialization between the matmuls (loses some of the fused-LDS benefit — measure vs SDPA first).

Anchors: `amd-flash-attention-reference-extraction-20260617.md` (the audit), `amd-lds-tiling-primitive-arc-20260617.md`
(LDS Phases 0–5), `extra/amd_warp_reduce.py` (the revived primitive).
