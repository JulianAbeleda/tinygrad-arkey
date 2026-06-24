# MMVQ lowering scaffold — VERDICT: B/C, STOP (deep arc not minimally scaffoldable) — 2026-06-18

Per `qk-mmvq-lowering-scaffold-20260618.md` + harness `extra/qk_mmvq_lowering_scaffold.py`.

## The four scaffold questions, answered

1. **Can tinygrad represent an MMVQ block as a single *schedulable* unit?** Only as scalarized int MACs (no
   dot4) — 2.2% peak. The dot4 forms exist only inside opaque CUSTOM/asm bodies, which the linearizer does not
   schedule into.
2. **Can generated source preserve packed extract + dot4 + qsum + per-group scale as linearizer-visible ops?**
   No. The packed extract is expressible, but **dot4 has no UOp** — the int-reduce that *should* become dot4
   scalarizes (`native_dot4_emitted: 0`).
3. **Can it beat the existing custom_kernel packed implementation?** No. Linearizer-visible = 2.2% (scalarized);
   the only ≥45% paths (udot4 46%, asm 52%) are the opaque CUSTOM kernels themselves. Nothing minimal beats 52%.
4. **Is a full deep-linearizer arc justified?** It is the *only* path (no minimal version exists), but its EV is
   low/uncertain — see below.

## Microkernel results (% HBM peak, Q4_K ffn_gate/up)

| representation | % peak | verdict |
|---|---|---|
| base fp | 41 | — |
| fp coop | 48 | shipped-class |
| pure-UOp int reduce (linearizer-visible) | **2.2** | scalarized, no dot4 |
| `_dp4a` udot4 (schedulable, in CUSTOM) | 46 | opaque loop |
| opaque asm signed dot4 | 52 | the prior custom_kernel ceiling |
| llama / READRAW | 70 | target |

## Verdict: **B (matches-not-beats) + C-flavored — STOP**

A minimal scaffold **cannot** beat the opaque custom_kernel. The gap is structural in tinygrad's codegen:
- **No first-class dot4 UOp and no int-reduce→dot4 lowering** (the int reduce scalarizes to 2%). dot4 only via
  CUSTOM/asm → opaque to the linearizer → 46-52% ceiling.
- To make a dot4 inner loop schedulable, tinygrad needs **(1) a first-class `Ops.DP4A`/`SDOT4` UOp + renderer
  lowering** (like `Ops.WMMA`), **(2) register-aware scheduling of the accumulator chain**, and **(3)
  scale-decode hoisting across the workgroup**. That is the full deep-linearizer arc — multi-week framework work.
- Additional blocker: the signed dot4 builtin `__builtin_amdgcn_sdot4` needs target feature `dot1-insts` **not
  present on gfx1100/RDNA3** (only inline-asm `v_dot4_i32_i8` works), so a portable schedulable signed dot4 would
  also need a correct RDNA3 builtin/intrinsic mapping.

## Full-linear feasibility estimate

Even if the deep arc landed a register-tight scheduled dot4 loop reaching llama's ~70% peak, the full Q4_K
ffn_gate/up would be ~70/41 = ~1.7× over base → comfortably clears the +1.3× gate → ~+8-12% decode e2e (decode
~68%→~73-75% of llama). So the *prize* is real. But the *probability* is low: the prior custom_kernel with all
the right math (packed extract + signed dot4 + per-group scale) already stalled at 52%, so closing 52→70%
depends entirely on the linearizer's register allocation / scheduling / scale-hoist — exactly the part a UOp
alone doesn't fix, and which is a deep, uncertain framework investment.

## Recommendation

**Do not start the full deep-linearizer rewrite.** The scaffold proved the only path is the full framework arc
(no minimal version exists), and the EV is low-confidence (prize +8-12% decode; risk: multi-week framework work
that may still stall at ~52% like the custom_kernel, plus the RDNA3 signed-dot4 builtin gap). **Higher-EV
direction: 14B/32B** — the shipped MMVQ_COOP + flash-decode primitives amortize better on larger, more
GPU-bound models, likely a higher % of llama with no framework risk.

The 8B quantized-matvec ceiling is now fully characterized: tinygrad's `custom_kernel` expresses llama's *math*
but not its *scheduling/register allocation*, and — proven here — there is no minimal lowering that changes this;
closing it is a first-class-dot4-UOp + register-aware-linearizer arc, deferred as low-confidence.

## Files / commits
`extra/qk_mmvq_lowering_scaffold.py` (`[test]`), `bench/qk-mmvq-lowering-scaffold/` (baseline/source_check/perf/
generated_source), `qk-mmvq-lowering-scaffold-20260618.md` + this verdict (`[docs]`). No `[codegen]` (Option A
needed none and was refuted; Options B/C are the deferred deep arc), no `[nn]`, no defaults.
