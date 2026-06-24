# AMD Decode -- the reframe finding + the amortized-quant plan (Phase A)

Date: 2026-06-15

## THE FINDING (record this): the kernel was never the wall -- activation-quant PLACEMENT is.

After every codegen/search lever came back negative, the cooperative-fused build (Q0a coop fix)
overturned the working conclusion:

- `q4k_q8_1_coop_fused_kernel` (hand-managed threads: workgroup = block_m rows; the block_m threads
  do BOTH cooperative quant-into-LDS AND their own row's int-dot) is **correct (rel_err 0.004) and
  409 Q4-GB/s standalone** -- vs fp 173, separate-int-dot 242, and APPROACHING llama.cpp's ~470.
  **tinygrad CAN express a near-llama.cpp fused decode GEMV.** "The kernel is the wall" is FALSE.
- But it regresses END-TO-END to 24 tok/s (vs fp 58), and block_m doesn't fix it. WHY: the fused
  kernel re-quantizes x PER WORKGROUP, which forces LDS (~4.5KB, caps occupancy) + a BARRIER
  (phase1->phase2 sync). Small-GEMV decode is occupancy/latency-bound and lives on INTER-KERNEL
  PIPELINING: the barrier-free fp path overlaps consecutive GEMVs (e2e 278 GB/s > per-kernel 173); the
  LDS+barrier coop kernel is the opposite (e2e 117 << standalone 409). **LDS+barrier is the wrong
  structure for pipelined small-GEMV decode.**

So the decode gap is NOT a kernel-expressiveness wall and NOT (as I argued earlier) a hand-asm
necessity. It is **where the activation quantization lives**. The fused kernel re-quantizes per
workgroup (LDS+barrier, bad). The RIGHT structure -- llama.cpp's -- quantizes the activation ONCE per
token (a cheap global pass) and runs BARRIER-FREE int-dot GEMVs that read global q8 and PIPELINE.
That structure is fully expressible in tinygrad today and has never been measured end-to-end (D0 only
tested per-LINEAR quant, whose 7x/layer launches capped it at 28).

## THE PLAN -- Phase A: amortized global quant + barrier-free int-dot

Goal: capture the int-dot win END-TO-END by quantizing each shared activation ONCE per token and
feeding the barrier-free `q4k_q8_1_intdot_partial_kernel` -- the llama.cpp shape, all expressible now.

### Why it should work (and the honest ceiling)
- The barrier-free int-dot is 1.4x faster per-kernel than fp (242 vs 173 GB/s, D0) AND barrier-free,
  so it pipelines like fp. D0's 28 e2e was ONLY the per-linear quant overhead (7 quant launches/layer).
- Per layer there are 4 DISTINCT activations feeding 7 Q4_K linears: attn-input -> q,k,v (3 share);
  attn-output -> o (1); ffn-input -> gate,up (2 share); ffn-intermediate -> down (1). Quantizing each
  ONCE = 4 quant passes/layer, not 7 -- and the q/k/v and gate/up sharing is where the amortization
  pays.
- Pre-registered ceiling: if the int-dots pipeline like fp, e2e ~= fp x (242/173) minus the (now
  amortized) quant ~ **75-81 tok/s** (56% -> ~75% of llama.cpp). A real win over fp, but NOT parity
  (104) -- the residual is the cross-layer / microarchitectural rungs, out of scope. Capturing 58->~78
  is the prize.
- Accuracy: X0 showed int8 activation is broadly viable (0.51-1.07% per-layer error, weakly
  heterogeneous), so uniform int8 is acceptable -- no per-layer precision needed.

### A0 -- make-or-break: amortize the quant, measure e2e (cheapest, FIRST)
Implement quant-once via **caching keyed by the input activation** (cleanest -- no model-forward
surgery): in `Q4KPrimitiveLinear.__call__`, cache `(xq, xscales) = q8_1_quantize(x_vec)` keyed by the
input activation's UOp identity; q/k/v (same attn-input) and gate/up (same ffn-input) hit the cache,
so the JIT graph has ONE quant feeding the shared int-dots. Dispatch the shared linears to the
barrier-free `q4k_q8_1_intdot_partial_kernel(words, xq, xscales)`. Behind a flag; correctness-gated
(rel_err < 2e-2, the ~0.7% int8 error). Measure e2e decode tok/s vs fp 58.
- Gate: e2e > ~70 -> the barrier-free int-dot pipelines and the amortized quant pays -> WIN, proceed.
- e2e ~58-65 -> the quant (even amortized, 4x/layer) still offsets; try Option B (quantize in the
  attn/ffn block forward, fewer/cleaner passes) or accept fp.
- e2e < 58 -> the int-dot doesn't pipeline e2e for a reason we haven't seen; diagnose from the source.

Implementation notes (concrete):
- Cache: a per-forward dict on the registry, keyed by `x_vec.uop.key` (or `id(x)`); q/k/v compute the
  SAME x_vec (CSE'd to one UOp) -> cache hits. Clear per step (non-JIT) / populate-once (JIT trace).
- The o-proj and down-proj activations are NOT shared (1 consumer each) -> amortization doesn't help
  them; they can stay int-dot (still 1.4x per-kernel, barrier-free) or fp -- A0 measures both.
- `q4k_q8_1_intdot_partial_kernel` already exists and is barrier-free (reads global xq/xscales); the
  only new code is the quant-cache + dispatch in `model.py`.

### A1 -- productionize (if A0 wins)
Per-shape: confirm the int-dot beats fp for each linear type (attn vs ffn vs down) end-to-end; build
the policy (which linears use int-dot vs fp). Correctness + an end-to-end accuracy check (perplexity
drift on a calibration set, since int8 activation is lossy -- X0 says it is fine, verify). Wire as the
default decode path when it wins.

### A2 -- measure vs llama.cpp
Report e2e decode tok/s + accuracy vs fp (58) and llama.cpp (104). The honest result is the captured
fraction (e.g. 58 -> ~78 = 56% -> ~75% of llama.cpp), with the residual to parity attributed to the
cross-layer / hand-asm rungs (Mirage direction / Writer), which Phase A does not address.

### Touch points
- `tinygrad/llm/model.py` (`Q4KPrimitiveLinear.__call__` + the registry quant-cache + dispatch).
- `extra/q4_k_gemv_primitive.py` (`q4k_q8_1_intdot_partial_kernel` -- exists; `q4k_q8_1_coop_fused_kernel`
  kept as the standalone-409 reference).
- `extra/qk_layout.py` (`q8_1_quantize`).
- accuracy harness (reuse the X0 capture; add a small perplexity/calibration check for A1).

### Risks / pre-registered honesty
- The quant-cache must actually CSE/hit across the shared linears (q/k/v, gate/up); if x_vec isn't the
  same UOp, the cache misses and it degrades to D0 (per-linear quant). Verify the cache hit-rate.
- The int-dot must pipeline e2e like fp (the make-or-break unknown). The coop result proved the
  COMPUTE is fast; A0 tests whether the barrier-free structure pipelines.
- Honest ceiling ~75-81 (not parity); a null (int-dot doesn't pipeline / quant doesn't amortize) is a
  real result. But this is the FIRST decode lead that is not a dead end, and it is cheap to test (A0
  is a cache + a dispatch swap, no new kernel).

## A0 RESULT (2026-06-15): premise REFUTED -- the int-dot kernel (occupancy), not the quant, is the wall.
`bench/.../q0a/A0_RESULT.md`. Amortized the quant via a cache keyed by x_vec UOp; cache HIT (36 hits,
22% -- q/k/v + gate/up share one quant), so amortization worked. But e2e decode = 28 tok/s / 136 GB/s
-- IDENTICAL to D0 (per-linear quant) and HALF of fp's 278. Amortizing the quant changed NOTHING.

Corrected diagnosis: D0 mis-blamed its 28 on the quant launches. With the quant amortized it is STILL
28 -> the quant was never the bottleneck. The barrier-free int-dot KERNEL is, at 136 GB/s e2e: it does
not pipeline (standalone 242 -> e2e 136, the OPPOSITE of fp's 173 -> 278) because its int accumulators
(~16 REGs) raise register pressure -> low occupancy. BOTH int-dot structures now lose e2e to fp for the
SAME reason -- occupancy, not compute: fused-LDS coop (LDS+barrier -> e2e 24), barrier-free global
(register pressure -> e2e 28). fp wins e2e because its simple accumulator = low registers = high
occupancy = it PIPELINES. In occupancy-bound small-GEMV decode, kernel SIMPLICITY beats compute
efficiency; the int-dot's standalone win is an e2e mirage in every structure.

FINAL decode end-state: fp 58 tok/s (56% of llama.cpp) is the tinygrad/AMD ceiling. llama.cpp's int-dot
wins e2e only because its hand-asm mmvq is OCCUPANCY-efficient (DP4A-packed, minimal accumulator regs) --
hand-written AMD asm (the Writer) that tinygrad's primitives + codegen do not produce. The reframe
finding STANDS (tinygrad CAN express a 409 GB/s fused GEMV standalone), but it does not translate to an
e2e win because of occupancy. This is the multiply-confirmed honest conclusion.
