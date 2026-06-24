# Prefill-focused weight reuse scope — Q4_K/Q6_K batched-forward primitive

Goal: rescope the surviving Arc A work after spec-verify closed. This is **not** a
spec-decode route. It is a prefill / batched-forward research arc: can tinygrad
make the quantized weight path reuse packed/dequantized weights across many token
columns, with enough locality/tiling to move prefill throughput?

Principles status:
- diagnostic/candidate only; no route/default.
- full primitive boundary: packed weights + activation dtype + dequant + tiling +
  attention interaction + in-model pp throughput.
- final authority is warm prefill pp throughput and token/dNLL gate, not isolated
  matmul TFLOPS.
- do not re-open refuted flag paths (`REALIZE=1`, `PREFILL_FP16`,
  `Q4K_UNFUSE`, `Q4K_BATCHED`) except as controls.

## Current landing

Spec-decode no longer justifies a Q4_K batched-K kernel:
- isolated Q4_K ffn_gate T=5 is already 2.58x one pass, not 5x.
- T=K+1 verify is distributed across attention + Q4_K + Q6_K, all T-scaling.
- no single kernel can move verify from 4.66x to <=1.5x one pass.

The surviving motivation is prefill, where T is large and weight reuse is
structurally valuable. This is the same first-principles bucket as the prefill
plan: locality/reuse/tiling, not host overhead.

## Research primitive

`qk_prefill_quant_weight_reuse_tiled_forward`

Boundary:
```
Q4_K/Q6_K packed GGUF weights
+ fp16/fp32 prefill activation stream
+ dequant/unpack
+ reuse across T token columns
+ tiled row/K reduction strategy
+ attention interaction at large T
+ warm in-model prefill throughput
```

This arc is byte/quality sensitive but not necessarily byte-identical. If it
changes accumulation or activation dtype, it needs a dNLL/token-parity gate
before any route.

## What is already refuted

Do not spend more time on these as candidate paths:
- Linear-level `PREFILL_FP16`: worse in-model.
- `REALIZE=1`: worse, materialized fp16 weights read too many bytes / bad
  integration.
- `Q4K_UNFUSE`, `TC=2`, `Q4K_BATCHED`: no meaningful in-model prefill win.
- reuse-free flash-prefill custom kernel: correct/expressible but massively
  slower; missing LDS/register-resident reuse.
- spec-verify Q4_K-only weight reuse: not enough Amdahl.

These can remain measurement controls only.

## Core hypothesis

Prefill currently loses because the model forward does not produce a
rocBLAS/Tensile-class tiled GEMM for quantized weights. Existing isolated tests
show the silicon can run fast when the problem is presented as a clean tiled
matmul, but the real forward loses locality/tiling and/or falls into fused
dequant forms that do not reuse enough.

Hypothesis to test:

```
A prefill-specific quantized-linear primitive that stages/reuses weights across
many T columns can beat the current in-model prefill path, but only if measured
as a warm prefill component and then as full pp throughput.
```

## Phase PWR-0 — authoritative prefill baseline refresh

Run one clean baseline on HEAD before building:
- model: Qwen3-8B-Q4_K_M
- hardware: RX 7900 XTX / gfx1100
- contexts/chunks: pp512 first, then pp1024 if useful
- modes: current default, `PREFILL_V2` if still relevant, llama.cpp reference

Record:
- warm pp tok/s
- first-token/prefill wall and device time if available
- chunk/ubatch size
- attention vs FFN share if measurable
- token parity or dNLL baseline artifact

Artifact:
`bench/qk-prefill-weight-reuse-20260618/baseline.json`

Proceed only if the current gap and bottleneck still match the prior docs.

## Phase PWR-1 — component target selection

Before building a kernel, identify which prefill component can actually move pp:
- Q4_K FFN gate/up
- Q4_K/Q6_K ffn_down
- Q6_K/lm_head if present in prefill path
- attention / SDPA
- residual/norm/SwiGLU overhead

Use the same discipline as the spec-verify breakdown: eager per-kernel shares are
directional; warm in-model pp totals are authoritative.

Decision gate:
- a component or shared primitive family must be >=30% of warm prefill time, or
  a pair of components must share the same tiled-weight primitive.
- expected 2x component win must imply >=1.2x full prefill.

If attention dominates before the weight primitive can pay, do not build a
weight kernel yet; rescope to flash-prefill/LDS attention.

## Phase PWR-2 — isolated quantized-linear reuse probe

Build the smallest quantized-linear probe only after PWR-1 selects a component.

Start with a single role and large T:
- `blk.0.ffn_gate.weight` Q4_K, shape 12288x4096
- T in `{32, 128, 512}`; T=512 is the target regime
- compare current fused/dequant path, existing batched GEMM, and the new reuse
  candidate

Candidate order:
1. register-block T reuse if feasible for small tile sizes.
2. LDS packed-weight tile with dequant per T tile.
3. LDS dequantized fp tile if LDS pressure allows.
4. only then consider WMMA/TC ownership issues.

Measure:
- device ms
- effective TFLOPS / quant-GB/s
- scaling with T
- VGPR/LDS/workgroup where available
- correctness vs existing prefill linear oracle

Isolated gate:
- T=128 and T=512 must scale sublinearly vs T.
- candidate must beat the existing prefill-shaped quantized linear by >=2x.
- if it cannot beat by >=1.5x, stop and bank the failing layer.

## Phase PWR-3 — one-block / one-layer prefill integration

Do not jump straight to full model. Integrate the winning primitive into a
single transformer block or a one-layer harness that includes:
- norm
- Q/K/V or FFN linears as selected
- activation dtype exactly as the model would supply it
- realistic T and contiguous/layout boundaries

Gate:
- one-block warm forward improves >=1.5x for the selected block share.
- no hidden dense fallback.
- no compilation/recompile per chunk.

If isolated wins but one-block fails, classify the transfer failure before
building more kernels.

## Phase PWR-4 — full prefill route behind a flag

Only after PWR-3 passes, add a model route behind an explicit flag:

`PREFILL_WEIGHT_REUSE=1`

No default flip.

Run:
- pp512 warm
- pp1024 warm if memory allows
- decode sanity: no decode regression
- token parity / dNLL gate if accumulation or dtype changes

Ship gate for a candidate route:
- >=1.5x full prefill over current default as a candidate threshold.
- >=3x full prefill for a strong route.
- no decode regression.
- quality accepted.

Anything below 1.2x full prefill is diagnostic only.

## Phase PWR-5 — decide relation to llama and rocBLAS

If custom tiling works but remains far below llama, compare against the explicit
alternative:
- call rocBLAS/hipBLASLt for dequantized fp16 tiles, or
- hand-write a raw HIP/Tensile-like kernel.

This is a separate decision because it changes the authority boundary: tinygrad
codegen vs external BLAS/raw HIP.

## Stop rules

Stop without building deeper if:
- PWR-1 shows attention dominates and weight reuse cannot move full pp.
- isolated weight reuse does not beat current prefill-shaped linear by >=1.5x.
- isolated win does not transfer to one-block.
- the only viable route requires broad forward restructuring before any local
  component clears a gate.

Bank the result as one of:
- refuted: no component-level win.
- deferred: viable but blocked by codegen/runtime capability.
- candidate: one-block/full-prefill gate passed behind a flag.

## Expected outcomes

Best case:
- a quantized prefill linear primitive gives a clear full-pp win and becomes a
  prefill route candidate.

Medium case:
- isolated reuse wins but does not transfer in-model; the result becomes a
  scheduler/codegen-internals diagnosis.

Most likely risk:
- prefill remains a multi-component forward problem: weight tiling, activation
  dtype, attention, and graph scheduling must all be fixed together. In that
  case this arc should stop as a documented D, not become another unbounded
  probe series.

## Next action

Start with PWR-0/PWR-1 only. Do not write the tiled kernel until the refreshed
prefill component breakdown proves the weight primitive has enough Amdahl room.
