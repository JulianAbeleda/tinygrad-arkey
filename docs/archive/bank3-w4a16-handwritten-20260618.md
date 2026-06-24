# Bank 3 — Marlin/W4A16 handwritten + refined cross-bank economics 2026-06-18

Handwritten fp16×int4 (W4A16, no q8 pack) Q4_K MMVQ, standalone hipcc, same structure as the Bank 4 kernel.

## Result: W4A16 = 49% peak (63.6µs / 445 GB/s) — barely above tinygrad fp coop (48%)
The handwritten codegen lever (+8% that we saw on int-dot, 57→65) **does NOT materialize on the fp path**.
Reason: the fp path is **dequant-ALU-bound** (per weight: nibble extract + 2 fp mul/sub + fp FMA), and clang's
register/scheduling can't move a compute-bound ceiling. This matches the earlier ~53% coalesced-fp work-ceiling.
**Bank 3 (W4A16) refuted for a big win: the fp path is ALU-ceilinged ~49-53% regardless of codegen quality.**

## The fundamental tradeoff, confirmed at the metal
| path | dot | % peak (handwritten) | byte-identical? | q8 tax? |
|---|---|---|---|---|
| W4A16 | fp16 dequant + FMA | 49 | YES | none |
| W4A8 | sudot4 dp4a (4 MAC/instr) | 65 | no (lossy 0.006) | q8 pack |

Fast needs int-dot (dp4a packs 4 MACs/instr) → lossy + q8 pack. Byte-identical needs fp → ALU-ceilinged. No free
lunch. The handwritten kernel proves dp4a is the *only* lever above ~53%, and it's inseparable from q8.

## BUT — refined whole-linear economics (the handwritten W4A8 is faster than assumed)
Bank 2's verdict used tinygrad's sudot4 (55µs/57%). The **handwritten W4A8 is 48.6µs/65%** — faster. Recomputing
paired gate+up:
- handwritten 65% + current 29.7µs pack → 63.5µs/linear → **1.04× coop** (was 0.96× with tinygrad kernel)
- handwritten 65% + fused 12µs pack → 54.6µs/linear → **1.21× coop** ← clears the 1.15× gate

**So the handwritten W4A8 kernel + a fused q8 pack COULD beat coop whole-linear (~1.21×)** — reopening Bank 4 for
ffn_gate/up, contingent on: (1) correctness verified, (2) tinygrad bridge, (3) a fused ~12µs pack built, (4)
dNLL ≤0.01 (lossy). In-model EV still ~+4-6% (gate+up = 2 of 7 linears), lossy.

## Verdict
- **Bank 3 (W4A16): refuted** — fp ALU ceiling ~49%, no codegen rescue.
- **Bank 4 (handwritten W4A8): the live kernel path** — 65% + fused pack → ~1.21× coop whole-linear (vs the prior
  0.96× with tinygrad's kernel). The handwritten codegen lever is real and only pays on int-dot. Funding Bank 4
  to completion (correctness + bridge + fused pack + dNLL) is the one kernel arc that could still ship a decode
  win — but it's lossy + deep + ~+4-6%.

## Files
`[test]` `extra/q4k_w4a16_handwritten.hip`; `[docs]` this. No tinygrad/model changes.
