# MMVQ codegen/linearizer arc — Phase 0/1: premise REFUTED by measurement, STOP (2026-06-18)

Scoping the framework arc ("make packed-int MMVQ visible to the optimizer so it generates llama-class kernels").
**Phase 0 measurement refutes the arc's core premise:** tinygrad *already* has a schedulable, optimizer-visible
`_dp4a`, and it reaches the same ~46-52% as the opaque inline-asm path — nowhere near llama's 70%. So dp4a
visibility (strategy A/B) is **not** the binding constraint; the gap is deep-linearizer scheduling, which is a
very-high-risk/low-confidence investment. **Recommend STOP and pivot to 14B.** No codegen built. RX 7900 XTX.

## Phase 0 — codegen baseline + the decisive measurement

Q4_K ffn_gate/up (12288×4096), isolated % HBM peak (peak ~900 GB/s):

| path | % peak | dp4a? | optimizer-visible dot? | coalesced? |
|---|---|---|---|---|
| base fp default | 40 | no | n/a | no |
| fp coop | 48 | no | n/a | yes |
| **schedulable `_dp4a` builtin (udot4)** | **46** | **yes** | **YES (device helper)** | no (one-row/thread) |
| inline-asm dp4a (the 52% custom_kernel) | 52 | yes | no (opaque asm) | yes |
| llama MMVQ / READRAW | 70 | yes | (hand-tuned) | yes |

**tinygrad already exposes a schedulable dp4a** — `_dp4a(a,b,c){ return __builtin_amdgcn_udot4(a,b,c,false); }`
emitted on demand (`renderer/cstyle.py:393`), used by `q4k_q8_1_vdot_builtin_partial_kernel` (the Q4K_VDOT
"builtin" path). It reaches **46%** (uncoalesced); coalesced it would be ~the inline-asm 52%. Either way the dp4a
path (visible OR opaque) lands at **~46-52%, ≈ fp coop's 48%, and far below llama's 70%.**

### Existing infra
- **WMMA: first-class UOp** (`Ops.WMMA`, `renderer/cstyle.py:74`, `codegen/opt/tc.py`) — the precedent a new
  int-dot op would follow.
- **dp4a: a schedulable `_dp4a` device helper already exists** (not first-class, but optimizer-visible as a
  function call, not inline asm). Used, gave +1% e2e historically.
- signed `__builtin_amdgcn_sdot4` needs target feature `dot1-insts` **not enabled for gfx1100** — only inline-asm
  `v_dot4_i32_i8`/`v_dot4_u32_u8` work.

## Phase 1 — representation decision (evidence-driven)

| strategy | expected payoff | risk | reviewability | verdict |
|---|---|---|---|---|
| A. pattern-recognize dp4a | **~0** | med | low | **REFUTED** — a schedulable `_dp4a` already exists and gives 46% |
| B. explicit `SDOT4` UOp | **~0** | high | med | **REFUTED** — same as A; visibility is not the constraint (the visible `_dp4a` already doesn't reach 70%) |
| C. fused MMVQ op (hand-lowered inner loop) | maybe, but ≈ the custom_kernel that stalled at 52%; hand-lowering = a renderer special-case ("un-tinygrad") | very high | low | low-EV (≈ what already failed) |
| D. scheduling controls on custom_kernel | likely exhausted | low | high | exhausted (52% is the custom_kernel ceiling) |
| **E. deep linearizer** (auto scale-decode hoist across workgroup + register-tight QR4_K unroll + occupancy) | *could* reach ~70% **if** it works | **VERY high** (multi-week framework arc) | low | high-risk, low-confidence |

**The binding constraint (52→70%) is NOT dp4a visibility** (already solved) — it's:
1. **redundant per-lane scale decode** (each of 8 lanes recomputes the group's 6-bit scales; llama decodes once
   per group/warp — needs cross-lane hoisting / LDS sharing the linearizer doesn't do here);
2. **no register-tight unrolled accumulator chain** (llama's hand-unrolled `v[2]/u[4]`);
3. **occupancy + the partials + stage-2-`.sum` reduction overhead**.

These are exactly the deep-linearizer/scheduler/regalloc properties — strategy E — which the prompt itself flags
as "very high risk." And A/B (the *narrow* version the arc hoped would work) is refuted because the visible
`_dp4a` already exists and doesn't help.

## Verdict / recommendation: STOP the codegen arc; pivot to 14B

The arc's hypothesis ("dp4a is opaque to the optimizer → make it first-class → close the gap") is **false by
measurement**: tinygrad's optimizer-visible `_dp4a` already reaches ~46-52%, the same as the opaque path, and
neither approaches 70%. The remaining gap is the deep-linearizer scheduling (E), a very-high-risk multi-week
framework investment with **low confidence of reaching 70%** (the custom_kernel with all the right math already
stalled at 52%). Per the campaign's discipline ("stop before wasting a week if llama relies on scheduling
tinygrad can't express cleanly"), this is that stop.

**Highest-EV next direction: 14B/32B** — more GPU-bound, where the shipped MMVQ_COOP + flash-decode wins amortize
better and the host/codegen-tightness overhead is a smaller fraction; likely a *higher* % of llama than 8B with
no new framework risk.

(If a framework arc is still desired despite the evidence, the only non-refuted option is **E** — a focused,
hard-early-killed attempt at cross-workgroup scale-decode hoisting for the Q4_K kernel — but the EV is low and
it is explicitly a deep linearizer change, not the narrow "make dp4a visible" the arc proposed.)

## Files
This doc; `bench/qk-mmvq-codegen/baseline.json`. No codegen/model changes (Phase 0/1 scoping only; gate stopped
the arc before building).
