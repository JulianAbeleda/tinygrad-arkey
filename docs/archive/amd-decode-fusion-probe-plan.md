# B1 — horizontal-fusion probe: does reducing launches speed up e2e decode? (scope)

Date: 2026-06-15. The make-or-break that gates the whole "megakernel ladder" toward machine-search-for-decode.

## The question it decides (and why it's cheap)
D1 proved a 1.77× faster GEMV moves e2e tok/s by ~0% → decode is latency-bound across ~252 launches, not
GEMV-throughput-bound. We INFERRED "fewer/fatter launches would help" but never measured it. B1 measures it
with the ONE fusion that is expressible in tinygrad today — horizontal sibling fusion — before committing to
the hard vertical (cross-reduction) megakernel work.

Two kinds of fusion, only one is cheap:
- **Horizontal (siblings):** q/k/v are 3 independent GEMVs reading the SAME input x, writing 3 outputs — NO
  reduction barrier between them. They fuse into ONE GEMV by concatenating their weight rows. Same for
  gate/up. Expressible TODAY (one iteration space over concatenated output rows). **This is B1.**
- **Vertical (chains across a reduction):** layer N → layer N+1 — the hard one tinygrad can't express
  (needs grid-sync/megakernel). Out of scope here; B1 gates whether it's worth attempting.

## Mechanism (expressible now — just concatenate compressed Q4_K rows)
Q4_K quantizes each row independently into 256-element blocks; rows are contiguous in the `words` array, all
sharing K = in_features. So concatenating the `words` arrays of sibling weights along the row axis yields a
valid fused Q4_K weight with rows = Σ rows_i. One `q4k_gemv_partial_kernel` over the fused rows, then split
the output vector back. Bit-identical to the separate GEMVs (same weights, same x, same per-row kernel).

Targets (Qwen3-8B dense, divisibility checked for LOCAL:0:64):
- **qkv:** q_out=4096, k_out=v_out=1024 (GQA) → fused 6144 rows (=96·64 ✓). Split [4096,1024,1024].
- **gate/up:** each 12288 → fused 24576 rows (=384·64 ✓). Split [12288,12288].
- Per layer: 7 Q4_K linears (q,k,v,o,gate,up,down) → **4** (qkv, o, gateup, down). o-proj and down-proj are
  single-consumer / different-input → not horizontally fusable. ~43% fewer Q4_K launches/token.

## Implementation (gated, reversible, behind `Q4K_FUSE`)
1. **Fused linear**: a thin `Q4KFusedLinear` wrapping a `Q4KPrimitiveLinear` over the concatenated words
   (out_features = Σ, in_features shared, parts=1, opts=("LOCAL:0:64",)); `__call__(x)` returns the
   concatenated output; caller splits. Falls back to the dense graph for prefill/batched (same guard as
   Q4KPrimitiveLinear: x.shape[0]==1, x.shape[-1]==in_features).
2. **Install** (`_install_q4k_primitives`, post-pass): when `getenv("Q4K_FUSE")`, for each block whose
   attn_q/k/v are all Q4KPrimitiveLinear, build `block.attn_qkv` = fused(cat words of q,k,v); record split
   sizes. Same for ffn_gate/ffn_up → `block.ffn_gateup`. Concatenate via `Tensor.cat` on the realized
   `.words`; reuse each linear's byte/shape metadata.
3. **Forward** (gated by `hasattr`):
   - `_attention` (line 525): `if hasattr(self,'attn_qkv'): qkv=self.attn_qkv(x); q,k,v = qkv.split([q,k,v],-1)`
     else the current 3 calls.
   - `_feed_forward` (line 492): `if hasattr(self,'ffn_gateup'): gu=self.ffn_gateup(x); gate,up=gu.split([h,h],-1)`
     else the current 2 calls.
   Default (no flag) path is byte-for-byte unchanged.

## Measurement (e2e A/B, same machine, same run-session)
- `Q4K_PRIMITIVE=1` (baseline) vs `Q4K_PRIMITIVE=1 Q4K_FUSE=1`, `cli.py --benchmark 30`, Qwen3-8B.
- Report: decode tok/s, GB/s, **kernels/token** (confirm the launch-count drop), and correctness
  (generated text identical between baseline and fused).
- Also a standalone sanity: fused qkv GEMV correctness vs 3 separate (a unit check, not just e2e).

## Pre-registered gate + the decisive fork
- **≥5% e2e tok/s gain** → launch count / kernel fatness is a real e2e lever → **climb the megakernel
  ladder** (B2: parameterized fused-layer template; B3: point the loop at it = machine search for decode on
  the fusion substrate). Pure single-stream path is alive.
- **2–5%** → partial; horizontal fusion helps modestly; vertical fusion may add more — worth a scoped B2.
- **<2%** → removing ~100 launches/token does ~nothing → the cost is INTRINSIC per-kernel (the batch-1
  reduction's memory latency), which megakernel fusion won't fix either → **pivot to speculation/batching**
  (Strategy A), where our validated loop applies and the draft model is the "fine-tuning" lever. This is the
  strongest possible result for the strategy decision: it would rule OUT the megakernel ladder for
  single-stream and point decisively at batching.

Either outcome decisively forks the path forward — that's the value.

## Risks / honesty
- **TinyJit (HCQ graph replay) may already hide host launch overhead** → a null is plausible and meaningful
  (it would mean the residual cost is GPU-side per-kernel memory latency, not host dispatch). Pre-registered.
- Correctness: the fused GEMV must equal the 3/2 separate ones (it is, by construction) — verify by identical
  generated text + a standalone unit check, don't assume.
- Divisibility / GQA split must be exact (checked above for 8B; assert at build).
- Machine ran 30 tok/s this session (vs historical 58) after GPU faults; the A/B is same-session so relative
  Δ holds — but rerun on a fresh GPU if the Δ is marginal.
- Scope: dense Qwen3-8B path only (TransformerBlock._attention, dense _feed_forward). MoE (ffn_gate_exps) and
  MLA (q_lora) paths and o/down (non-fusable) are out of scope.
