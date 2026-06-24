# q8/MMVQ lifecycle deep scope — RESULT: Q8L-0/1 pass, **Q8L-2 KILL (expressibility wall) → route DEFERRED behind codegen capability**

Executed `q8-mmvq-lifecycle-deep-scope-20260618.md` (`decode_q4k_ffn_q8_sidechannel`), the only remaining decode
reopening. Phases are gated; reached the decisive expressibility spike and hit the documented worst case. No kernel
routed, no default changed. Probe: `extra/qk_q8_sidechannel_producer.py`.

## Q8L-0 — producer contract audit → **PASS** [M, from code]

- Producer: `ffn_norm(h)` = `nn.RMSNorm` at `tinygrad/llm/model.py:747` (`h + _feed_forward(self.ffn_norm(h))`).
- Consumers: the default decode `_feed_forward` (model.py:732) is `ffn_down(ffn_gate(x).silu()*ffn_up(x))` with
  `x = ffn_norm(h)` → feeds **exactly `ffn_gate` + `ffn_up`** (reuse = 2; no hidden third consumer).
- `attn_norm` (model.py:678) is a separate RMSNorm, untouched. Decode shape [1,1,4096] fp32.
- Consumer q8 layout (oracle `extra/qk_layout.py:172 q8_1_quantize`): per-32 `scale=max(|x|)/127`, `qs=round(x/scale)`
  int8 — the existing sudot4 consumer reads `xq[i]`/`xscales[i//32]` directly.
- **Gate met:** one producer → exactly the gate/up pair; no q8 needed outside it.

## Q8L-1 — cost model / lower bound → **PASS (plausible ≤4.8µs, conditional on single-kernel)** [I]

- Current `ffn_norm` decode kernel: reads [4096] fp32 (16KB), writes [4096] (16KB); launch/overhead-bound (~few µs;
  block-map rmsnorm ~2.1% of decode across ~73 kernels).
- q8 side-channel **incremental** work: 4096 `|y|` + 128 per-32 max-reductions + 4096 quantize (mul/round/clip) +
  write 4096 int8 (4KB) + 128 fp scales (0.5KB). Real compute ≈ sub-µs; extra traffic ≈ 4.5KB ≈ 0.005µs.
- **If folded onto the apply pass** (values already resident, no HBM re-read, no extra launch), incremental ≈ the
  fused kernel's occupancy/overhead delta — **plausibly ≤4.8µs**. Best-case decode EV ~+3-4% (gate/up = 2 of 7
  linears; reuse ceiling 2 because k/v are Q6_K). **Gate met in principle** — but its hard sub-clause "no extra
  kernel launches" is decided by Q8L-2.

## Q8L-2 — custom-kernel expressibility spike → **KILL: not cleanly expressible as one kernel** [M]

`extra/qk_q8_sidechannel_producer.py`: a single `Tensor.custom_kernel(out_fp, out_scale, out_qs, x, w, …)`
attempting fp-norm + per-32 scale + q8 packed, 3 outputs.

**Multi-output stores ARE mechanically available** (precedent: `custom_kernel(...)[2]` in the asm matmuls writes one
of several passed buffers; `UOp.group(*stores)` in `q4k/q6k_unpack`). **But the producer cannot use them**, because:

1. The store-group idiom is `UOp.group(*stores).end(SHARED ranges)` — it requires all grouped stores to share **one
   range nest** (the unpack kernels group stores over the same `(row,blk,pos)`). Grouping the producer's three
   stage stores fails verification: `UOp verification failed … on Ops.GROUP … [(END),(END),(END)]` (the three
   stages are each ended over **different** ranges — `j`, `b/p`, `j2`).
2. The producer is **serially dependent across two reduction granularities**: per-row `ss=Σx²` reduce → broadcast
   `rinv` → per-32 `max(|y|)` reduce → broadcast scale → quantize. Only the **per-row ss reduce** breaks single-range
   fusion: the q8 scale + qs + fp outputs *could* share `(block, pos)` ranges **if rinv were precomputed**, but rinv
   needs the full-row reduce first. tinygrad's custom_kernel range/store model treats these as **separate kernels**.

So a one-kernel fused producer is **not expressible via the store-group idiom**. The only single-kernel route is to
stage `rinv` in **LDS + a barrier**, then do the per-block pass in the same workgroup — i.e. the **flash-style
LDS-reduction kernel** (the `amd_warp_reduce` WR1-3 + LDS-tiling machinery), which is a deep `[codegen]` build, not
an expressibility spike. The alternative — accept the per-row reduce as its own kernel — **is the separate-pack that
is already refuted** (each stage a ~7µs launch; misses ≤4.8µs).

**Gate FAILED** ("one kernel, no dense fallback"): the fused multi-granularity producer is not one kernel under the
available idiom. → **Q8L-3 through Q8L-6 not run** (the scope gates them behind Q8L-2).

## Verdict — `decode_q4k_ffn_q8_sidechannel` = **DEFERRED behind codegen capability** (scope's "worst case")

The cost target (≤4.8µs) is reachable *in principle* (Q8L-1) and the contract is clean (Q8L-0), but the **blocker is
a custom-kernel capability, not an MMVQ/dataflow question** (the scope's explicit worst case, line 280-282): a
single fused custom kernel that does a **per-row reduce → broadcast → per-32 reduce → multi-output store** is not
expressible via the store-group idiom; it requires an LDS-reduction flash-style kernel. This **confirms with concrete
expressibility evidence** the earlier Track-1 verdict D (`q8-sidechannel-ffn-verdict-20260618.md`: "feasible only as
a deep fused-norm multi-output kernel; no single→multi-output precedent").

This **closes the last bounded decode research question**: every decode lever is now shipped, refuted, sub-gate, or
deferred-behind-codegen. The q8 side-channel does not reopen as a buildable arc without first funding the LDS
multi-output reduction-fusion codegen capability — and even then EV is ~+3-4%, lossy (dNLL-gated), for gate/up only.

## Reopen criteria (precise)
- A custom-kernel capability lands that fuses **two reduction granularities + multi-output** in one kernel (LDS
  staging of the per-row reduce + barrier + per-32 pass), OR
- the project funds the LDS/WR flash-style producer as a deep `[codegen]` build accepting ~+3-4% lossy EV.
Until then this is **deferred**, not open. The higher-EV deep arc remains prefill fp16 WMMA LDS-tiling / BLAS
boundary (`qk-prefill-weight-reuse-result-20260618.md`).

## Files
`extra/qk_q8_sidechannel_producer.py` (Q8L-2 spike), this doc. Provenance: `q8-mmvq-lifecycle-deep-scope-20260618.md`,
`q8-sidechannel-ffn-verdict-20260618.md`, `q4k-ffn-q8-lifecycle-verdict-20260618.md`. No kernel/model/default changes.
