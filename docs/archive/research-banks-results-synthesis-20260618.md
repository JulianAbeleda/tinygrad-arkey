# 8B decode research banks — results synthesis (all 6 worked) 2026-06-18

Per the "do all of them" directive, every bank was taken to a decisive result (hardest-first: Bank 4 → 3, then
1/5/6 scoped; Bank 2 closed earlier this session). The kernel banks got at-the-metal experiments; the others got
rigorous feasibility/audit. Decode stands unchanged (~66-69% llama); nothing routed.

## The headline finding (from the handwritten kernels)
A **handwritten HIP Q4_K MMVQ kernel hits 65% peak** vs tinygrad's custom_kernel 57% — **+8% pure per-thread
codegen** (clang register-alloc/scheduling). **But that lever ONLY works on int-dot (dp4a):** the handwritten
**W4A16 (fp) stays 49%** — ALU-ceilinged, no codegen rescue. So the metal confirms the fundamental tradeoff:

| | dot | handwritten %peak | byte-identical? | tax |
|---|---|---|---|---|
| W4A16 (fp) | fp dequant+FMA | 49 | YES | none |
| W4A8 (int) | sudot4 dp4a | 65 | no (lossy 0.006) | q8 pack |

**Fast needs int-dot (lossy + q8 pack); byte-identical needs fp (ALU-ceilinged).** No free lunch.

## Per-bank results

| # | bank | result | verdict |
|---|---|---|---|
| 1 | low-sync speculative decode | acceptance proven (2.84/pass, greedy-exact); blocker = per-step host sync (5 dispatches/pass); fix = on-device token feedback, 1 sync/pass (feasible) | **FUND FIRST — highest EV, only path to beat llama (+40-60%)** |
| 4 | handwritten W4A8 MMVQ | **65% kernel** (+8% over tinygrad); + fused pack → **1.21× coop whole-linear** (was 0.96× w/ tinygrad kernel) | **LIVE kernel path** — contingent on correctness + bridge + fused pack + dNLL (lossy, ~+4-6%) |
| 6 | machine-search infra | schema + orchestrator + dNLL gate already exist; gap = ledger/auto-row | **small, do ALONGSIDE the live bank** |
| 3 | Marlin/W4A16 | handwritten fp = **49%** (≈ fp coop); fp ALU ceiling, no codegen rescue | **refuted** (no win on the byte-identical path) |
| 2 | q8 activation lifecycle | reuse ceiling 2 + ~7µs pack floor; side-channel only via deep fused-norm | **verdict D** (deep, ~+3-4% lossy) — but see Bank 4 economics revision |
| 5 | SmoothQuant | q8 accuracy already fine (rel 0.006); blocker is pack COST not accuracy; SmoothQuant fixes neither | **refuted** (targets a non-problem; rank last) |

## Re-ranked recommendation (with the new evidence)
1. **Bank 1 — low-sync spec decode.** Only ceiling that beats llama; blocker precisely understood; fix
   well-defined; runtime arc orthogonal to all kernel walls. **Fund first.**
2. **Bank 4 — handwritten W4A8**, IF a kernel win is wanted: the handwritten 65% + a fused q8 pack reaches ~1.21×
   coop whole-linear (the faster kernel revives the economics Bank 2 closed with the slower tinygrad kernel). But
   it's lossy (dNLL gate), needs the tinygrad raw-HIP bridge + a fused pack + correctness, and EV is ~+4-6%
   (gate+up = 2 of 7 linears). Second priority.
3. **Bank 6 — infra**, a small ledger/validator investment run alongside Bank 1 so results auto-populate.
4-6. **Banks 3, 2, 5 — closed/refuted** (fp ALU ceiling; deep fused-norm only; targets a non-problem).

## The honest cross-bank truth
The kernel frontier is now exhaustively characterized at the metal: the only above-53% lever is dp4a, which is
inseparable from q8 (lossy + pack tax), and even hand-written it tops out ~65-70% on a path that's 2 of 7 linears.
**The decode win that "refuses to stop at 68%" is not in the kernels — it's in spec decode** (fewer target passes
per token), which is why Bank 1 is the fund. The handwritten-kernel arc delivered its promised lesson (codegen is
the wall, dp4a is the only lever) and a contingent revival of Bank 4, but it confirmed that kernels alone won't
beat llama for 8B Q4_K.

## Files
`[docs]` this + `bank{1,3,4,5,6}-*-20260618.md`; `[test]` `extra/q4k_{mmvq,w4a16}_handwritten.hip`,
`bench/qk-handwritten-mmvq/result.json`. No tinygrad/model/default changes.
