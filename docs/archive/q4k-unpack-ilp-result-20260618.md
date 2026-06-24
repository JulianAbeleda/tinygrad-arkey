# Q4_K ffn_gate/up llama-style MMVQ rewrite — RESULT: codegen-deep, STOP (2026-06-18)

The single bounded attempt authorized by the audit (`llama-q4k-mmvq-inner-loop-audit-20260618.md`). Implemented
llama's exact portable inner loop in tinygrad `custom_kernel`. **Verdict: the audit's tricks were correctly
identified and DID help (39%→52% peak), but the kernel still falls short of llama's 70% and the +1.3%
whole-linear gate. The last mile (52→70%) is a tinygrad codegen/scheduling limitation, not a primitive
formulation. STOP per the kill condition.** RX 7900 XTX, Qwen3-8B; no kernels retained, no defaults changed.

## What was built (then removed)

A signed-dp4a coop kernel replicating llama's `vec_dot_q4_K_q8_1_impl_vmmq`:
- **packed nibble extract** `(qword >> (grp%2)*4) & 0x0F0F0F0F` (4 nibbles in 1 shift+1 AND — the trick Family
  A missed).
- **signed `v_dot4_i32_i8`** inline asm (NOT `__builtin_amdgcn_sdot4`, which needs target feature `dot1-insts`
  not enabled for gfx1100), so NO bias correction (the ~16 scalar ops/dp4a that crippled the unsigned Family A).
- **`v_dot4(0x01010101, q8)` q8-sum** on the dp4a unit (llama's `dp4a(0x01010101,u)`).
- **per-group scale** (`sc`,`m`), block `d/dmin` applied once.

## Measured (isolated, Q4_K ffn_gate/up 12288×4096, real weights)

| variant | µs | GB/s | % HBM peak | note |
|---|---|---|---|---|
| base fp default | 77.6 | 361 | 40% | |
| fp coop | 65.8 | 425 | 48% | |
| Family A (unsigned dp4a + scalar pack) | 80.0 | 354 | 39% | bias-correction overhead |
| **llama-style (signed dp4a + 0x0F0F0F0F + dp4a-qsum)** | **60.7** | **466** | **52%** | the audit's tricks |
| llama MMVQ / READRAW roofline | — | 626/632 | 70% | target |

The audit's tricks **moved the kernel 39%→52%** (and past fp coop's 48%) — the diagnosis was right and the tricks
are portable + expressible in tinygrad. But:
- **Whole-linear gate FAILS:** pair (1 q8 pack 27.6µs + 2 kernels) = 1.04× over base; even with a **free** q8 pack,
  2 kernels = **1.28× < the 1.3× gate**.
- **52% << llama 70%.** The remaining 18 points are codegen, not math.
- (A correctness bug remained in the fp epilogue — rel 5.4; not resolved because the kernel fails the speed gate
  regardless, so perfecting it was moot per the kill condition.)

## Why it stalls at 52% (the codegen-deep diagnosis)

The audit proved the op-count gap is portable (packed extract + dp4a + per-group scale), and implementing it
did help. But tinygrad's `custom_kernel` lowering of the dp4a inner loop cannot match llama's hand-unrolled
register-tight `v[2]/u[4]` loop:
- the dp4a is an **inline-asm CUSTOMI** (signed builtin unavailable for gfx1100), which the optimizer treats as
  opaque — no fusion/vectorization across the 8 per-group dp4a calls;
- **per-lane redundant scale decode** (`_q4k_group_params` recomputed by all 8 lanes of a row — 8× redundant;
  llama decodes scales once per group/warp);
- no register-tight unroll of the QR4_K accumulator chain.
These are linearizer/scheduler/register-allocation properties, not expressible by reshaping the UOp graph.

## Verdict / decision

**STOP (codegen-deep), per the kill condition.** The portable primitive was implemented and validated (39→52%),
but the 52→70% mile is a tinygrad codegen limitation (inline-asm-dp4a opacity + redundant scale decode +
register scheduling), which is a deep linearizer arc — out of scope and not earned by a +28%-kernel/below-gate
result. The decode/prefill bounded-kernel program is at its ceiling on 8B (~68% decode, ~81% prefill of llama).

## What remains
The recurring wall across this whole campaign: tinygrad **custom_kernel** can express llama's *math* but not its
*instruction scheduling/register allocation* — isolated wins (dp4a 1.77×, TC attn 2.5×) and now this 52% all
stall below llama. Closing it needs either (1) a **tinygrad codegen/linearizer** investment (auto-dp4a lowering
from int-mul-reduce + register-tight scheduling — a framework arc, very high risk), or (2) a **different target**
(14B/32B, more GPU-bound, where these primitives amortize better and the host/codegen overhead is a smaller
fraction). **Recommendation: 14B/32B.**

## Files
This doc; the experimental kernels (`_sdot4`, `q4k_coop_sdot4_partial_kernel`, `q4k_coop_mmvq_partial_kernel`)
were added, measured, and removed (refuted). `extra/q4_k_gemv_primitive.py` back to clean. No `[nn]`, no defaults.
