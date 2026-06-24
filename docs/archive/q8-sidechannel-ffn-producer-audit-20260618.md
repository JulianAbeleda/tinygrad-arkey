# q8 side-channel — FFN activation producer audit (Phase 0/1) 2026-06-18

The only credible reopening for Q4_K ffn_gate/up sudot4: produce the q8 activation as a side-channel of the op
that already produces the gate/up input, so there's no standalone q8 pack. Audit only. Artifact:
`bench/qk-q8-sidechannel/producer.json`.

## Phase 0 — the producer
`tinygrad/llm/model.py:747`: `return (h + self._feed_forward(self.ffn_norm(h))).contiguous()`.
- **Producer = `self.ffn_norm(h)`** — an `nn.RMSNorm(dim, norm_eps)`; `h` = post-attention residual.
- `_feed_forward` receives the norm output **once** and applies gate + up → **gate and up consume the exact same
  Tensor expression** (q8 commoning across them is free; confirmed in the lifecycle probe).
- Wrapped in the jitted `@function(precompile=True)` block forward.
- **decode vs prefill:** same op; decode T=1 (4096-wide), prefill T>1 ([T,4096]). A custom side-channel must
  handle both shapes.
- An orthogonal q8 amort already exists (`Q4K_VDOT_AMORT`, model.py:149) — it caches/commons the SAME 4-kernel
  pack per token; it does not make the pack cheaper.

## Phase 1 — producer cost + reduction structure (measured, decode 4096)

| stage | kernels | time | existing reduction? | q8 side-channel opportunity |
|---|---|---|---|---|
| RMSNorm only | 2 | 19.4µs | yes — **per-row mean(x²) over 4096** | reduction is wrong granularity for q8 |
| RMSNorm + current q8 pack | 6 | 48.6µs | — | q8 adds **29.2µs** (4 kernels) |

**The decisive structural fact:** RMSNorm reduces **per-row mean(x²)** (one scalar / 4096-row). q8_1 needs
**per-32-block max(|·|)** (128 scales / row). Different axis granularity **and** different op (sum-of-squares vs
max). **So q8's block-max CANNOT piggyback the existing RMSNorm reduction** — it needs a fresh per-32 reduction
over the *normalized* values.

But the normalized values are produced by the norm's apply pass, which already touches all 4096 elements. A
**custom fused norm kernel** could, in one data pass: reduce mean → produce normalized (write fp) → accumulate
per-32 max inline → quant+pack (write qpacked+scales). No extra global read of the activation, no extra launch.
A **pure-graph** side-channel (q8 ops on the norm output) does **not** fuse — proven: it schedules as 4 separate
pack kernels. So the cheap path requires a hand-written custom norm kernel (deep), not graph expression.

## Implication for break-even
q8 effective cost must be ≤4.8µs (for 1.15× coop at reuse=2). A *separate* fused pack floors at ~12µs (lifecycle
probe) > 4.8µs. Only folding q8 into the norm's own pass (a custom multi-output norm kernel) can plausibly add
≤~3-5µs. Feasibility of *that build* is assessed in `q8-sidechannel-design-options-20260618.md` and the verdict.
