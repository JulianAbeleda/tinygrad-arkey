# Q4_K unpack/ILP — safe rewrite plan (2026-06-18)

Follows `llama-q4k-mmvq-inner-loop-audit-20260618.md`. The audit proved the 48→70% gap is **portable** (packed-
integer nibble/scale extraction + dp4a + per-group scale; not asm-only). This plan scopes the single earned
attempt and its kill condition.

## Recommendation: **Option 2/4 hybrid — ONE scoped custom_kernel attempt, hard-gated.**

NOT Option 1 (stop) — the gap is portable, so it's worth one try. NOT Option 5 (measurement) — the gap is real
(both READRAW and llama hit 70%; tinygrad 48%). NOT Option 3 (codegen lowering change) first — try the
custom_kernel UOp expression before touching the linearizer.

## The change

Replicate llama's `vec_dot_q4_K_q8_1_impl_vmmq` **exactly**, in tinygrad `custom_kernel` UOps, inside the
coalesced lane structure:

| piece | llama | tinygrad now | minimal change |
|---|---|---|---|
| nibble extract | `(v>>4i)&0x0F0F0F0F` (4/op) | `_q4k_quant`: scalar 1/op | read packed weight **ints** (`words[...]`), mask `0x0F0F0F0F`, shift 4*i — 4 dp4a-aligned nibbles/op |
| dot | `dp4a(v0i,u,..)` | per-weight fp madd | `__builtin_amdgcn_sdot4` (have `_vdot4_q4_q8_accum`); feed the packed nibble int + q8 int |
| q8 sum (min term) | `dp4a(0x01010101,u,..)` | n/a | dp4a with a const-ones int |
| scale decode | `scales&0x3f3f` (2/op) | per-byte gymnastics | uint16 packed mask |
| scale apply | per-group `sc[i]`,`m[i]`; `dm` once | per-weight affine | accumulate int dp4a per group, apply `d8*sc`/`dm` per group, not per weight |
| activation | q8_1 ints (`quantize_q8_1`, once) | fp16 | reuse `q8_1_quantize` + `q8_1_bias_pack_u32_kernel` (exist); amortize across gate+up |

Key correctness note vs Family A: Family A built the q4 u32 with 4 scalar shift+mask+or (expensive). This plan
uses `(v>>4i)&0x0F0F0F0F` directly on the loaded packed int — **the specific portable trick that was missed.**
This requires reading the Q4_K `qs` bytes as ints in llama's dp4a-aligned order (the GGUF bytes are the same;
only the access expression changes).

## Gates

- **isolated whole-linear ≥1.3×** over the current Q4_K ffn_gate/up default (q8 pack cost included, fresh input,
  BW < HBM peak, no less-work artifact). This is the real test of whether tinygrad codegen reaches llama's
  tightness with the packed-int UOps.
- correctness: fp/int reassoc tol (q8 quant ~rel 0.02-0.04, as Family A); **must keep greedy byte-identical
  in-model** before any ship.
- in-model (only if isolated passes): W==D pp/decode ≥+5%, byte-identical, no regression.

## Kill condition (single-attempt budget)

If the isolated whole-linear < 1.3× even with the exact `0x0F0F0F0F` + dp4a + per-group structure, **conclude
tinygrad's custom_kernel codegen cannot schedule the packed-int/dp4a inner loop as tightly as llama's hand-
unrolled `v[2]/u[4]` (register allocation + instruction ordering), and STOP** — the remaining gap is a
linearizer/scheduler limitation (Option 3, a deep codegen arc), not a custom_kernel-authoring win. Do not
open-endedly iterate.

## Risk assessment

- **Portable:** yes (plain uint ops + the `sdot4` builtin already in the repo). No CUDA/HIP-only idiom.
- **Precedent caution:** every prior tinygrad dp4a attempt (Q4K_VDOT +1%, Q6_K split-K, Family A +0.6%) cashed
  out far below its isolated/standalone promise — tinygrad codegen has not matched llama's int-loop tightness so
  far. BUT none used the `0x0F0F0F0F` packed extract; this is genuinely the untried variable. Medium-high risk.
- **Effort:** one custom_kernel (a few dozen UOp lines), bounded.
- **Scope:** Q4_K ffn_gate/up only; do not touch decode, Q6_K, attention, defaults.

## Proceed?

**Yes — one scoped attempt is justified** (portable gap, the specific missed trick identified). Bound it: build
the llama-exact packed-int kernel, run the isolated whole-linear gate, ship only on ≥1.3× + byte-identical
in-model, else conclude codegen-limited and stop. If the user prefers to avoid even one more dp4a attempt given
the precedent, Option 1 (stop, pivot to 14B) is defensible — the audit has de-risked the *decision*, not
guaranteed the *win*.
