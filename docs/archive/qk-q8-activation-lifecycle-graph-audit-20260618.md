# Q8 activation lifecycle — graph audit (Phase 3) 2026-06-18

Can q8 activation be a first-class graph value computed once and reused, so the pack stops canceling the sudot4
win? Audit only.

## Findings

1. **fp + q8 side-by-side on one Tensor:** not as one Tensor, but the model can hold two Tensors — `xf` (fp
   activation) and `xq8 = pack(xf)` — both derived from the same RMSNorm output. Legal.
2. **q8 cached per block input during decode:** yes, in principle — compute `xq8` once per activation and pass it
   to every consuming linear. Nothing forces recompute.
3. **computed once, consumed by multiple linears:** yes IF the model code computes `xq8` once and threads it to
   gate and up (don't call `pack()` inside each linear).
4. **TinyJit CSE of the pack:** if gate and up reference the **same** `xq8` UOp expression, the scheduler
   realizes it once (common-subexpression) — so "1 pack for gate+up" is achievable without manual `.realize()`.
   If each linear independently calls `pack(input)`, the expressions are structurally identical and *should* CSE,
   but the safe pattern is to compute `xq8` once explicitly.
5. **`.realize()` once before two linears:** works (forces one pack), but is unnecessary if (4) holds; and
   realizing mid-graph can break jit fusion — prefer leaving it lazy and shared.
6. **does model code duplicate activation expressions:** gate/up currently each take the fp activation; an int
   path would add one shared `xq8` node feeding both.
7. **memory:** q8 is ~1/4 the fp size (int8 vs fp16/fp32) + small scales — negligible.
8. **recompute cadence:** once per (activation, token, layer) — i.e. once per RMSNorm output. That's the minimum;
   it cannot be hoisted across layers (each layer's FFN input differs).

## The decisive lifecycle fact
Reuse-over-2 (gate+up) **is** expressible/CSE-able in the graph — but Phase 2 shows n=2 still loses (0.95×) with
the 29.7µs pack and only reaches 1.13× coop even with a fused ~7µs pack. **Graph reuse does not move the needle
because the ceiling is 2 and the per-kernel floor is ~7µs.**

The only lifecycle that wins is **q8 as an epilogue of the prior RMSNorm kernel** (the norm already reads/writes
the activation; emitting the q8-packed side-output there adds ~0 extra kernel → pack cost ≈ 0 → paired gate+up
≈ 1.20× coop). But that requires:
- the RMSNorm op to emit a **q8 side-channel** (fused custom norm kernel producing fp + qpacked + scales),
- threading that side-channel to gate/up in the model,
- and it remains **q8-lossy** (rel 0.006) vs the byte-identical fp coop (needs a dNLL pass).

This is a deep activation-lifecycle change touching the norm op and the block wiring — out of scope for a probe.
See the verdict.
