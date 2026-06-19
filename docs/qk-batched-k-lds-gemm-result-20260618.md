# Arc A Phase 1 — Q4_K batched-K weight-reuse: RESULT → **hypothesis REDIRECTED** (the Q4_K GEMM is already only 2.58×; the verify cost is elsewhere)

The first decisive experiment ("can one kernel compute T=K+1 outputs while reusing each Q4_K weight block?") was
run isolated, in-model-faithful config (LOCAL:0:64, real ffn_gate weights, fp-exact). `extra/qk_batched_k_lds_gemm.py`.
**Outcome: the premise that the Q4_K batched GEMM scales ~linearly with K is FALSE in isolation — it's already
2.58× one pass (not 5×), so the in-model 4.53× verify cost is dominated by OTHER roles, not the Q4_K GEMM.**

## Phase 1 measurement (ffn_gate 12288×4096, LOCAL:0:64, DEBUG=2 device time, exact)

| variant | device ms | × one T==1 pass | argmax exact |
|---|---:|---:|:--:|
| one T==1 pass (anchor) | 0.069 | 1.00 | — |
| **baseline batched (UPCAST:1:T)** | **0.178** | **2.58×** | ✓ (max_abs 5.7e-3) |
| reuse_unroll (Python-unrolled, shared weight) | — | — | UOp `GROUP` verification wall (see below) |

(Earlier `--measure-verify` numbers — 4.04/4.53/9.10× at K=2/4/8 — were the **whole-forward** verify; this is the
**isolated ffn_gate kernel**.)

## The three findings

1. **`UPCAST:1:T` already achieves PARTIAL weight reuse.** No-reuse would be ~5× one pass for T=5; the kernel is
   **2.58×**. tinygrad's UPCAST opt already hoists the bb-independent dequant across the unrolled columns to a
   significant degree. Forcing the column axis to `AxisType.UNROLL` or `UPCAST` (vs the opt) was **byte- and
   time-identical** — the axis-type is not a lever.
2. **REDIRECT — the Q4_K batched GEMM is NOT the verify bottleneck.** Isolated ffn_gate T=5 = 2.58× one pass, but
   the in-model verify is 4.53×. Summing the isolated batched cost across all Q4_K roles (≈72 ffn_gate/up @0.178ms
   + 72 attn_q/o + Q6_K ffn_down/lm_head) lands near ~22ms, vs the measured 59ms whole-verify — so **~⅔ of the
   verify cost is elsewhere**: the **attention over K+1 positions at ctx512** (SDPA cost scales with K×KV), Q6_K
   batched roles (incl. lm_head 151936×4096 over K+1), and per-pass overhead. A Q4_K-ffn reuse kernel would address
   only ~20% of the verify and (per Amdahl from 2.58×→1.5×) move the whole verify by low single digits.
3. **The register-blocked reuse kernel needs the `c_regs` idiom, not a naive store-group.** The minimal Python-
   unrolled T-accumulator construction (`UOp.group(*stores)` of T `.set(...,end=pos)` AFTER-stores) fails UOp
   verification (`Ops.GROUP` over multiple `Ops.AFTER` ptrs). A valid register-blocked reuse needs the
   `DEFINE_REG` placeholder + accumulate-over-reduce + epilogue-copy idiom from `extra/gemm/amd_uop_matmul.py:95-114`
   — a Phase-2-scale build, not a "minimal" prototype.

## Gate disposition

- Phase 1 success gate ("T=5 ≤ 2.5× in first prototype"): the **baseline is already 2.58×** — there is little
  headroom on the Q4_K GEMM itself, and the reuse prototype didn't construct. **Do not promote a Q4_K-ffn reuse
  kernel build** on this evidence.
- The arc's premise (fix Q4_K weight reuse → fast verify) is **not supported**: the Q4_K GEMM already reuses
  partially and isn't the dominant verify cost.

## Redirected next step (before any kernel build)

**Phase 1.5 — per-component batched-verify breakdown.** Decompose the T=K+1 verify GPU time by component
(Q4_K GEMM / Q6_K GEMM incl. lm_head / attention-over-K+1 / norms+overhead) via the decode-block-map method
(`extra/qk_decode_block_map.py`) extended to T=K+1. Only then is it known which component is the linear-in-K cost
worth a weight-reuse / batched primitive. **Most likely target: the batched attention (K+1 queries × KV) and/or
the Q6_K batched roles — not the Q4_K ffn GEMM.** If the breakdown confirms a single linear-in-K dominant
component with reuse headroom, build the `c_regs` register-blocked kernel for *that* role; otherwise the spec-decode
verify is bounded by attention/overhead and the route stays closed (bank precisely which layer).

## Cross-arc note (prefill still benefits)

The weight-reuse primitive's *prefill* payoff (T≫K, the documented LDS=0 re-read at ~27% peak) is **unchanged** by
this finding — at large T the reuse matters much more than at T=5. The double-leverage argument holds for prefill;
it's the **spec-verify** half that this Phase 1 weakens (T=5 is too small for Q4_K reuse to dominate, and the Q4_K
GEMM already reuses partially). Prefill-class reuse (Arc A Phase 5) remains the stronger motivation.

## Files
`extra/qk_batched_k_lds_gemm.py` (probe: T==1 anchor + baseline + reuse-unroll attempt). Oracle: `q4k_gemm_kernel`
(`extra/q4_k_gemv_primitive.py:337`). Register-block idiom: `extra/gemm/amd_uop_matmul.py:95-114`. No kernel routed,
no defaults changed. fp-exact throughout (no q8).
