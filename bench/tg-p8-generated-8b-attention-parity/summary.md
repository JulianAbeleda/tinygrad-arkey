# TG-P8 Terminal: Generated 8B Attention Parity

Verdict: **TG_P8_BLOCKED_ATTENTION_PARITY**

Owned HIP stays the 8B decode-attention default. No generated candidate reaches >=98% of owned at BOTH ctx512 and ctx4096 within the guardrails.

## Phases
- TG-P8.0 PASS: baseline pinned (per-kernel wall split, token-identical, route-bound).
- TG-P8.1 PASS: delta classified SPLIT_GEOMETRY_MISMATCH (generated tile ctx-flat 1.05x vs owned ctx-proportional 3.67x).
- TG-P8.2 REFUTE: geometry search over L — L=128 optimal (87.7%/95.9%); larger L monotonically worse (occupancy-starved).

## Dual blocker (both must clear 98%; neither can within guardrails)
| ctx | best %own | class | binding | fix requires |
|---|---|---|---|---|
| 512 | 87.7% | SPLIT_GEOMETRY | needs occupancy splits | owned's runtime per-split length (fixed S, len=ceildiv(Tc,S)) — codegen capability |
| 4096 | 95.9% | COMBINE_OVERHEAD | **yes** | new combine primitive (collapse is refuted, guardrail #3) |

ctx4096 is the binding cap: to reach 98% the generated wall must drop 228us/tok, but a perfect tile saves only 112us — the 556us combine lifecycle (83% of the delta) dominates, and collapsing it is refuted.

## Outcome
Owned remains default. TINYGRAD_DEFAULT_PURITY_PASS stays blocked on 8B attention, now with a precise mechanistic blocker (geometry-optimal + combine-capped) rather than TG-P5's coarse 'slower'. Reopen only with a symbolic-per-split-length generated tile AND a genuinely new (non-collapse) combine primitive.
