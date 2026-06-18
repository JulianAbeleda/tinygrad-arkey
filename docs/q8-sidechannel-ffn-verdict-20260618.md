# q8 side-channel — VERDICT: D (feasible only as a deep fused-norm build; scope separately) 2026-06-18

The narrow feasibility test for the only credible Q4_K ffn_gate/up int-dot reopening. **The side-channel CAN
plausibly hit the ≤4.8µs cost target — but only by hand-writing a fused custom RMSNorm+q8 kernel that replaces a
hot, shared op and handles decode+prefill+residual+fp-fallback+multi-output. That is a deep producer change for a
lossy ~+3-4% decode gain. Not C ("not viable") and not a small flag-gated build (A); it's D: feasible-but-deep,
scope separately.** Audit + design + producer artifact committed; no kernel/route/default changes.

## Findings recap
- **Producer:** `ffn_norm(h)` (nn.RMSNorm), gate+up share the expression. 19.4µs / 2 kernels (decode).
- **Reduction mismatch:** RMSNorm = per-row mean(x²); q8 = per-32 max. q8 cannot piggyback the existing
  reduction → needs a fresh per-32 reduction over normalized values.
- **Pure-graph side-channel does NOT fuse** (4 separate pack kernels, 29.7µs; proven). Only a hand-written fused
  custom norm kernel folds q8 into the norm's data pass.
- **Break-even:** ≤4.8µs effective for 1.15× coop at reuse=2. A separate fused pack floors ~12µs (fails). A
  folded custom norm could add ~0-5µs effective (plausibly passes) — but unproven (probe not built; the
  4096-unroll attempt was compile-bound, and a proper reduce-range multi-output fused-norm kernel is the deep
  build itself).

## Phase 4 — break-even / in-model estimate
If the side-channel costs X over the producer:
| X | paired gate+up | vs coop (131.9µs) | in-model decode (gate+up = 2 of 7 linears) |
|---|---|---|---|
| 0 (perfect fold) | 110.0µs | 1.20× | ~+3.6% |
| 4.8 (break-even) | 114.8µs | 1.15× | ~+3% |
| 8 | 118.0µs | 1.12× | ~+2.5% |
| 29.7 (current) | 139.8µs | 0.94× | regress |

Even the **perfect-fold** ceiling is ~**+3-4% decode** (gate+up are only 2 of 7 linears/layer), and the path is
**q8-lossy** (dNLL ≤0.01 required, untested).

## Verdict: D — feasible but deep; scope separately
- **Not C:** the cost target is reachable (a folded custom norm kernel could add ≤4.8µs effective).
- **Not A:** the build is not small/safe — it replaces `nn.RMSNorm` (a hot op also used by attn_norm) with a
  custom multi-output kernel handling decode+prefill+residual+fp-fallback; multi-output/two-granularity-reduction
  custom kernels have repeatedly fought tinygrad's plumbing.
- **Decision:** the EV (~+3-4% decode, lossy, deep hot-path change + dNLL gate) is low against the risk. **Bank
  the feasibility (it's reopenable), do not build now.** Reopen only with a higher-EV motivation (e.g. bundled
  with a broader fused-norm refactor, or if q8 becomes reusable across ≥3 linears via a model change).

## Another build earned? No.
The Q4_K ffn_gate/up int-dot path stays closed in practice. The side-channel is the *only* technical reopening
and it's a deep, low-EV arc. Routed nowhere.

## Files / commits
`[docs]` producer-audit + design-options + this verdict; `[test]` `bench/qk-q8-sidechannel/producer.json`.
Feeds Bank 2 of `8b-decode-research-banks-roadmap-20260618.md`. No kernel/route/default changes.
