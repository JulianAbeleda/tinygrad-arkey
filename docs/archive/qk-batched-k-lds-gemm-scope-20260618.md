# Arc A — LDS-tiled batched-K Q4_K GEMM (weight-reuse primitive): SCOPE + Phase-1 baseline → **GO to Phase 2**

Research primitive: `qk_batched_verify_gemm_with_LDS_weight_reuse`. The higher-leverage arc: a weight-reuse batched-K
GEMM makes **both** prefill-class matmul **and** spec-decode T=K+1 verify cheap (it pays off twice). Hypothesis: the
current batched verify costs ~K× one pass because it does **not** reuse the packed weight across the K token columns;
a correct LDS/register-tiled primitive reads+dequantizes each weight block **once** and applies it across all K+1
columns. gfx1100, Qwen3-8B-Q4_K_M. **Nothing routed; this is scope + the smallest proof.**

## Phase 1 — baseline capture (DONE) — current batched GEMM is ~linear in K (no reuse)

`extra/qk_spec_decode_lowsync.py --measure-verify --k {2,4,8} --measure-ctx 512` (W==D, decode_enabled toggled):

| K | T=K+1 | single T==1 pass | **batched GEMM verify** | ×one pass | dense fallback | argmax exact |
|---|---|---:|---:|---:|---:|:--:|
| 2 | 3 | 12.79ms | 51.71ms | **4.04×** | 86.19ms (6.74×) | ✓ |
| 4 | 5 | 13.07ms | 59.21ms | **4.53×** | 144.06ms (11.0×) | ✓ |
| 8 | 9 | 12.92ms | 117.63ms | **9.10×** | 270.29ms (20.9×) | ✓ |

**Reading:** the batched GEMM (`q4k/q6k_gemm_kernel`, already wired for 2≤K≤32) is **exact** (argmax identical to
the dense path) and **2.4× faster than the dense fallback** the spec harnesses accidentally used — but it scales
**~linearly with T** (≈1× one-pass per column), i.e. **no weight amortization.** A weight-reuse primitive should
make T=K+1 ≈ **1–1.5× one pass regardless of K** (weight read once, dequant once, K cheap dots). **Target: T=5
verify ≤ 1.5× one pass (~20ms)** = a ~3× kernel improvement over the current 59ms.

## Root cause (code-level) — the K-column axis re-reads the weight

`q4k_gemm_kernel` (`extra/q4_k_gemv_primitive.py:337`): the output `row` (axis 0) and the **batch column `bb`**
(axis 1) are both **parallel ranges**, while the weight index `base = (row·k_blocks + blk)·Q4K_WORDS_PER_BLOCK` is
**independent of `bb`**. So each `(row, bb)` work-item independently streams the same packed weight block from HBM →
the weight is read **~K times**, not once. The model adds `UPCAST:1:min(K,16)` (`model.py:219`, intended to hoist
the dequant across columns) but the measurement shows it does **not** achieve weight-read reuse in practice (still
~K). The same class as the prefill GEMM (`amd-decode-prefill-plan.md`: tinygrad matmul LDS=0, re-reads operands,
~27% peak vs llama rocBLAS LDS-tile ~80%). Batch-1 decode escapes it (one column, irreducible 1× read).

## The primitive design (Phases 2–4)

Stage each packed Q4_K weight block (or a tile of rows) so it is read from HBM **once** and reused across the K+1
activation columns. Two candidate dataflows (Phase 3 picks by LDS-pressure vs instruction-reuse):
- **register-block the K axis** (Simon Boehm step 3): one work-item holds the dequantized weight block in registers
  and loops K+1 columns — simplest; no LDS; reuse factor = K+1. Likely sufficient at K≤16 (weight block is small
  per-thread). **Try first.**
- **LDS-tile the weight** (Simon Boehm step 2): stage a tile of packed/dequantized weight rows into LDS, a
  workgroup of K columns reads from LDS. Needed only if register pressure caps the K-reuse.
- **Stage format (Phase 3):** packed-q4 in LDS then dequant per column (lowest LDS, more ALU), vs dequantized-fp
  in LDS (more LDS — 64KB/wg cap: at fp16 a 256-weight block = 0.5KB, fine), vs unpacked-int lanes. Start with
  **dequant-once-into-registers** (the register-block path), measure, then escalate to LDS only if needed.

**Expressibility is already proven** (the LDS-tiling primitive arc, Phases 2–4, `amd-lds-tiling-primitive-arc-20260617.md`):
`AddrSpace.LOCAL → Ops.DEFINE_LOCAL` renders `__attribute__((shared))`, `UOp.barrier → Ops.BARRIER` (`s_barrier`),
REG accumulators work, and **LDS reuse beat redundant HBM reads up to ~3–4× at W=129** in-repo — directly the
weight-reuse regime. Templates: `extra/gemm/amd_uop_matmul.py`, `extra/gemm/amd_flash_attention.py`. The existing
`q4k_gemm_kernel` is the correctness oracle (argmax-exact).

## Phases & gates (remaining)

2. **LDS/register feasibility probe** — one role (ffn_gate 12288×4096) one kernel: stage weight once, reuse across
   K columns. Gate: exact vs `q4k_gemm_kernel`; device-time sublinear in K.
3. **Dequant-reuse format probe** — packed-in-LDS vs dequant-fp vs int-lanes; pick by LDS pressure vs reuse.
4. **K-axis tiling sweep** — K∈{2,4,8,16}; **success = sublinear cost vs K**, not just "faster than dense."
5. **Verify-only integration** — route spec verify through the primitive behind an explicit flag (no decode default).

**Success gates:** correctness exact argmax vs current batched verify; **T=K+1=5 verify ≤ 1.5× one T==1 pass**;
spec loop projected >1.2× production before any route.
**Kill gates:** LDS/register staging can't beat the current batched GEMM by ≥1.5×; or T=5 stays ≥3× one pass; or
correctness needs a lossy format change. (Q4_K dequant is fp-exact, so no lossiness is expected — the reuse is a
scheduling change, not a format change. This is the key advantage over the int-dot arc.)

## Verdict: **GO to Phase 2**

Phase 1 + the code-level root cause establish a clean, bounded target: the current batched GEMM re-reads the weight
~K times (measured ~linear in K; root-caused to the bb-parallel axis), the LDS/register weight-reuse primitive that
fixes it is **already proven expressible and effective in-repo** (≤4× reuse demonstrated), and the fix is a
**byte-identical scheduling change** (Q4_K dequant stays fp-exact — unlike the int-dot arc, no q8 pack, no
lossiness, no dNLL gate). It pays off **twice** (prefill + spec verify) — the unique double-leverage arc. The deep
part is the kernel build itself (Phase 2, a UOp custom_kernel with weight-reuse across the K axis) + the
correctness gate; it is the recommended next focused step. **Nothing routed; no defaults changed.**

## Files
Baseline probe: `extra/qk_spec_decode_lowsync.py --measure-verify` (committed). Oracle: `q4k_gemm_kernel`
(`extra/q4_k_gemv_primitive.py:337`). Templates: `extra/gemm/amd_uop_matmul.py`, `amd_flash_attention.py`,
`amd-lds-tiling-primitive-arc-20260617.md`. Prefill-class precedent: `amd-decode-prefill-plan.md`.
