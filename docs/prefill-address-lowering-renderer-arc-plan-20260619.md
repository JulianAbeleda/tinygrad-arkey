# Implementation plan — prefill address-lowering (the dependency-free path to llama-class prefill)

Formal plan for the one open, highest-leverage, dependency-free prefill lever (per
`primitive-level-open-questions-20260619.md`): tinygrad's WMMA matmul hot loop spends ~160 integer-ALU ops (120
`v_mov` + 40 `v_add`) vs 16 `v_wmma` — **the address ALU exceeds the WMMA compute** — because the global→LDS copy's
strided `half` gather materializes a separate 64-bit address per element instead of base + `offset:` immediates.
Closing it → ~halve the loop overhead → toward llama-class prefill (1.41× ceiling already proven via the Tensile
route), **no dependency, general (all AMD matmuls)**.

## ⚠️ VALIDATION GAP (must close before trusting this plan) — added 2026-06-19
The CG-W diagnosis below was measured on `extra/gemm/amd_copy_matmul.py`, a **proxy with `opts_to_apply=()` (no
UPCAST/vectorize opts)**. An attempt to capture the **actual in-model PREFILL_V2 matmul** (via `m.logits` with
`_prefill_v2=True` + warmstart) found kernels that look **nothing like the proxy**: `v_wmma`=0 (scalar `v_fma`/`v_dot`,
not tensor cores) and addressing dominated by `offset:` immediates (80) not per-load `off` (24). That capture may be
the **non-warmstarted fallback** (not the `prefill_v2_jit` path the measure harness uses for 2709 tok/s), OR the
in-model matmul genuinely differs. **Conclusion: the "address-lowering is the in-model lever" premise is confirmed
only for the amd_copy proxy, NOT for the in-model kernel.** Per *in-model authority*, this plan is **NOT YET
validated** and must start with:

- **CG-W1.5 (gating, do FIRST):** capture the **dominant warmstarted in-model ffn matmul kernel** on the real
  `prefill_v2_jit` path (clone inputs to satisfy the JIT), disassemble it, and confirm (a) does it use WMMA? (b) is it
  address-ALU-bound like the proxy, or `offset:`-clean? If the in-model kernel is *not* address-bound (or not WMMA),
  the entire CG-W/CG-R/CG-W plan retargets — the in-model 80%-of-llama gap would then be a *different* primitive
  constraint (e.g. TC not firing, or a non-WMMA scalar matmul), and this address-lowering plan does not apply.

This caveat supersedes the confidence below until CG-W1.5 resolves it. (Lesson, against *in-model authority* +
*isolated kernels are diagnostic only until they transfer*: the amd_copy proxy was a convenient isolated kernel; the
in-model path uses warmstart opts that change the kernel — the proxy did not transfer, and I should have validated on
the in-model kernel before scoping the renderer arc.)

## Confirmed diagnosis (ISA + emitted HIP source) — ON THE amd_copy PROXY (see validation gap above)
- ISA: `global_load_d16_b16 vN, v[156:157], off` — per-load 64-bit address, **no `offset:`**; 120 `v_mov`/iter
  broadcasting bases; the LDS loads (contrast) use one base + `offset:N`.
- HIP source (the matmul kernel): `half val = *(data1 + (… + (Ridx100<<4) + Lidx1009))` — a **per-element 16-bit
  strided gather**; `Ridx100<<4` is the k-loop term (recomputed per k-tile), `Lidx1009` the per-load unroll term.
  The 16 per-thread loads differ by a small constant over a common k-dependent base ⇒ *should* be base + immediate
  offsets, but are emitted as separate `half` loads with full addresses.

## The fork the plan must resolve first (cheap, kernel-level — CG-W2)
Two candidate root causes; resolve before any renderer change:
1. **tinygrad isn't vectorizing the per-thread strided `half` loads** — loading 16 individual halves instead of wide
   loads (e.g. `float4` = 8 halves) means 16 addresses instead of 2. Fix lives in the copy access pattern / a
   vectorize-the-gather opt (kernel- or renderer-level, but testable in the kernel).
2. **clang won't fold the strided d16 loads to `offset:` immediates** even though tinygrad emits a base+offset-shaped
   index — a clang addressing-mode/codegen limit for half flat-loads. Fix = restructure the emitted access so clang
   uses offsets (or load as `uint`/`float4` and bitcast), or accept it as a clang limit.

**CG-W2 experiment (do first):** in `extra/gemm/amd_copy_matmul.py`, change the global→LDS copy to read **coalesced,
vectorized** (each thread reads a contiguous run as `float4`/`uint4`, transpose in LDS) instead of the strided
`[:, tid]` half gather. Disassemble + measure:
- if `v_mov`/iter drops and TFLOPS climbs toward 62 → the lever is **vectorization of the copy** (kernel/opt-level,
  not deep renderer) → promote to a tinygrad opt (CG-W3a);
- if the addresses persist (clang still per-load) → it's the d16/offset clang issue (CG-W3b, harder).

## Phases
- **CG-W2 — kernel coalesced/vectorized copy probe** (cheap, days): the experiment above. Gate: does it cut the
  address ALU? Decides CG-W3a vs CG-W3b. **Reuses** `amd_copy_matmul.py`; no tinygrad change yet.
- **CG-W3a — vectorize-strided-gather as a tinygrad opt/lowering** (if fork 1): make tinygrad emit wide loads for
  contiguous-per-thread copies + strength-reduce the k-loop base. Lives in the load/index lowering
  (`renderer/cstyle.py` load rendering + the symbolic-index simplification / UPCAST-on-copy). Bounded-in-intent.
- **CG-W3b — offset-immediate emission for strided global loads** (if fork 2): emit constant-stride loads as
  base + `offset:` (and/or load as wider int + bitcast so clang folds). Renderer load-rendering change.
- **CG-W4 — isolated gate:** the WMMA matmul (ffn_gate/up + ffn_down + attn_q/o shapes) must reach **≥62 TFLOPS
  isolated**, ALU/WMMA ratio down, correct (mse < 1e-6). KILL if ≤50 (the fix isn't the lever after all).
- **CG-W5 — in-model + regression:** behind no flag (it's a general codegen improvement, not external): warm pp512 +
  pp1024 vs PREFILL_V2 and vs llama (3394), dNLL ≤0.01, decode **W==D unchanged** (the change touches shared codegen
  → must not regress decode), and the tinygrad test suite (`test/test_ops.py`, `test/test_linearizer*.py`,
  `test/test_schedule.py`) green. This is the highest-risk part — a shared-codegen change.

## Exact surfaces to touch (grounded)
- **`tinygrad/renderer/cstyle.py`** — how `Ops.LOAD`/`Ops.INDEX` render to C++ (the `*(buf + idx)` form); whether
  contiguous-per-thread loads become `*(float4*)(buf+base)` (vectorized) vs per-element half loads.
- **the symbolic-index simplification** (UOp `graph_rewrite` / index lowering) — strength-reduce the k-loop-variant
  term `(Ridx100<<4)` into a loop-carried base increment instead of recompute.
- **the load vectorization opt** — ensure contiguous per-thread copy loads coalesce to the widest aligned type.

## Gates
| gate | threshold |
|---|---|
| isolated matmul | ≥62 TFLOPS (ffn shapes), ALU/WMMA ratio ↓, mse < 1e-6 |
| in-model | warm pp512 ≥1.25× (research)/≥1.35× (strong) PREFILL_V2, dNLL ≤0.01 |
| **no decode regression** | decode W==D ctx-sweep unchanged (shared-codegen change) |
| test suite | tinygrad ops/linearizer/schedule tests green |
| no dependency | pure tinygrad; no external artifact, no BEAM (gfx1100 hangs) |

KILL: CG-W2 shows the copy vectorization doesn't cut the address ALU AND clang won't fold offsets → the gap is a
clang half-flat-load limit not addressable from tinygrad's C++ → close the pure-tinygrad prefill path (rest at
PREFILL_V2; the Tensile route remains the only ≥llama option, with its dependency).

## Against the principles
- *audit before build*: CG-W2 (cheap kernel probe) gates the renderer work — don't touch shared codegen until the
  fork is resolved. (This plan's whole structure.)
- *in-model authority*: CG-W4 isolated TFLOPS is diagnostic only; CG-W5 pp512/dNLL + decode-W==D is the real gate.
- *test at the boundary / contain dangerous power*: a shared-codegen change must pass the full suite + not regress
  decode — the biggest risk, gated explicitly.
- *reduce knowledge duplication*: a renderer-level fix is the right altitude (one change, all AMD matmuls) vs
  per-kernel hacks — but only if CG-W2 proves it's a general lowering issue, not this kernel's structure.
- *label state*: this is OPEN/project-level; the fix is bounded-in-intent (vectorize + strength-reduce) but lives in
  shared codegen with broad blast radius.

## Effort / risk
CG-W2: days (kernel probe, low risk). CG-W3a/b: 1–3 weeks (renderer + index lowering, **high blast radius** — all AMD
codegen). CG-W5 regression is the gate that makes or breaks landing. Realistic outcome distribution: CG-W2 likely
shows partial gain (vectorization helps some), full ≥62 may need both W3a+W3b, and the decode-regression bar is the
true constraint on shipping.

## Deliverables
CG-W2 probe `extra/qk_wmma_coalesced_copy.py` + `bench/qk-codegen-wmma/coalesced.json`; result appended to
`prefill-codegen-wmma-issue-result-20260619.md`; if it passes, the CG-W3 renderer patch + the CG-W5 in-model/regression
measurement. No model/default change until CG-W4 ≥62 TFLOPS isolated.
