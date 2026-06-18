# Q8 activation lifecycle — VERDICT: C (not enough; int-dot FFN refuted, bank it) 2026-06-18

Scoped whether q8 activation can be cheap/reusable enough to make the sudot4 kernel win whole-linear. **It
cannot, via any path short of a deep RMSNorm-epilogue side-channel — and even that is low-EV and lossy. int-dot
Q4_K FFN remains refuted; bank it.** Audit/probe only; no kernel, no routing, no defaults.

## The two independent walls

**Wall 1 — reuse ceiling = 2.** The only Q4_K linears sharing an activation are gate+up (post-attn FFN input).
k/v are Q6_K, so the attn input has just 1 Q4_K consumer (q); o and down consume unique activations. No Q4_K
activation is shared by ≥3 linears. n=2 loses (0.95× coop @29.7µs pack; 1.13× coop even @7µs fused pack).

**Wall 2 — per-kernel floor ~7µs.** Each pack kernel floors at ~7µs (launch/ramp on 16KB; the standalone pack
kernel is 6.9µs). The break-even needs ≤5.0µs (1.15× coop) / ≤5.2µs (1.05× opaque). No *separate* kernel can get
there. A fused single kernel inherits the same ~7µs floor → 1.13× coop / 1.03× opaque → **fails**.

## Break-even summary
| reuse n | fused-pack 7µs paired | vs coop | vs opaque | vs base |
|---|---|---|---|---|
| 2 (real ceiling) | 117µs | 1.13× ✗ | 1.03× ✗ | 1.32× ✓ |

Clears the base gate only; fails the gates that matter (coop is shipped + byte-identical; opaque is the prior
best). 

## The only theoretical reopen — and why it's not pursued now
**q8 as a zero-extra-kernel epilogue of the prior RMSNorm** (fused norm kernel emits fp + qpacked + scales) →
pack cost ≈ 0 → paired gate+up ≈ **1.20× coop / 1.10× opaque / 1.41× base** (all pass). But:
- **Deep lifecycle change:** the norm op must produce a q8 side-channel, threaded to gate/up — touches the norm
  custom kernel + block wiring (verdict D territory: "promising but deep").
- **Still q8-lossy** (rel 0.006) vs the byte-identical fp coop → requires a dNLL quality pass.
- **Low EV:** gate+up are 2 of 7 linears/layer; best-case decode gain ≈ **+3-4%** (and lossy). The shipped
  byte-identical coop wins are worth more per unit risk.

## Verdict: C (with D noted, not recommended now)
- **C — q8 lifecycle is not enough:** int-dot Q4_K FFN remains refuted. The sudot4 kernel (57%, correct) is real
  at the kernel level but the mandatory activation quant structurally cancels it whole-linear, and no
  reuse/fusion within reach closes the gap. **Bank it.**
- **D — q8-as-RMSNorm-epilogue** is the only path that clears the gate, but it's a deep, lossy, ~+3-4% change.
  Scope separately only if a byte-identical or clearly higher-EV motivation appears.

**No build earned.** Durable: the activation reuse map, the ~7µs per-kernel floor, the n=2 ceiling, and the
break-even (pack must be ≤5µs / zero-kernel to win). Combined with the prior sudot4 helper fix + 57% kernel
(banked), the Q4_K int-dot FFN investigation is **complete and closed**.

## Files
`[docs]` this + `qk-q8-activation-lifecycle-arc-20260618.md` + `qk-q8-activation-lifecycle-graph-audit-20260618.md`;
`bench/qk-q8-activation-lifecycle/baseline.json`. No `[codegen]`/`[nn]`, no routing, no defaults.
