# Decode Optimization — Hypothesis & Thesis Statement (for independent audit)

Self-contained statement of the bet for tinygrad AMD decode optimization on
gfx1100 (RX 7900 XTX). Written for an independent reviewer. Every claim is
tagged [MEASURED] (our bench logs, in-repo), [EXTERNAL] (cited URL), or
[ASSUMED] (inference, not yet tested). Plan: `docs/amd-decode-optimization-plan.md`.

## THESIS (the high-level position)

On gfx1100, tinygrad's large gap vs llama.cpp/ROCm in quantized LLM *decode*
(single-stream token generation) is dominated by **memory-traffic
inefficiency** — specifically, dequantization that is emitted as a generic
float32 tensor expression and not fused into the matvec — rather than by a
missing hardware instruction. Decode is memory-bandwidth bound, so the lever is
bytes-moved, and reducing it (fp16 dequant + fusing dequant into the GEMV) is a
compiler/scheduler/search problem expressible in tinygrad's existing ops. The
realistic objective is **~50-70% of llama.cpp decode throughput** via
machine-side optimization (fusion + BEAM), with hand-written fused kernels as a
fallback, not the primary path.

## HYPOTHESIS (specific, falsifiable)

The measured ~7x decode gap (8B Q4_K_M) decomposes multiplicatively as:
- **~4.4x excess bytes-moved** [ASSUMED — derived from one DEBUG=2 reading,
  not independently confirmed]: dequant outputs float32 and is not fused into
  the matvec, so far more bytes cross VRAM than the Q4 weights require.
- **~1.6x scheduling** [ASSUMED — this is the *residual* 7.0/4.4, not a direct
  measurement of BEAM's effect].

Prediction: BEAM + fp16 dequant + dequant->GEMV fusion recovers the majority,
reaching **>=50% of llama.cpp effective bandwidth on 8B decode** (i.e. 8B
>= ~50 tok/s, eff >= ~240 GB/s), **without hand-writing a kernel**.

## Measured facts (our bench logs, in-repo, 2026-06-10/11)

[MEASURED] llama.cpp ROCm 7.2.4, llama-bench, Qwen3 Q4_K_M, single 7900 XTX
(bench/rocm-baseline-20260610.log): 8B 101.2 tok/s, 32B 30.8; large-model
effective bandwidth asymptote ~567 GB/s = ~59% of 960 GB/s peak.

[MEASURED] tinygrad native, DEV=AMD, local PCIe, BEAM=0, same GGUFs
(bench/tg-native-llm-20260611.log): 8B 15.77 tok/s, 32B 4.41; effective
bandwidth asymptote ~81 GB/s = ~8.5% of peak. Gap ~7x.

[MEASURED] DEBUG=2 on 8B (bench/tg-native-8b-debug2.log): individual graph
kernels run ~300-410 GB/s, but the model achieves only ~74-81 GB/s effective
on the Q4 weight size — i.e. bytes-moved >> Q4 size. (This is the sole basis
for the 4.4x figure; treat as one data point.)

## Code facts (verifiable by the auditor in this repo)

[MEASURED] `tinygrad/llm/gguf.py:59-67` — Q4_K dequant (ggml_type 12) is a
multi-op tensor expression (bitcast/stack/cat/reshape/bitwise) returning
**float32**: `return (d * sc.unsqueeze(-1) * q - dmin * mn.unsqueeze(-1)).flatten(-2)`
with `.cast(dtypes.float32)` on the scales. There is a `.contiguous()` on the
quantized blocks at line 45.

[MEASURED] `tinygrad/llm/model.py:131-136` — the forward is jitted with comment
"we unpack the GGUF on the fly"; the dequant lives inside the forward graph
(lazy), intended to fuse into the matmul. Whether it actually fuses or
materializes an fp32 weight is **[ASSUMED] not verified** — a primary audit
question.

## External evidence (cite these)

[EXTERNAL] llama.cpp ROCm performance on AMD, including RDNA3:
https://github.com/ggml-org/llama.cpp/discussions/15021
[EXTERNAL] RX 7900 XTX llama-bench reference numbers:
https://github.com/1337hero/rx7900xtx-llama-bench-rocm
[EXTERNAL] tinygrad's own llama speed tracking (older, LLaMA, multi-GPU; not
directly comparable): https://github.com/tinygrad/tinygrad/issues/5244
[EXTERNAL] WMMA neutral-to-harmful on RDNA3 for llama.cpp inference; standard
TILE/VEC path as fast or faster — supports "decode is not won by matrix units":
https://github.com/ggml-org/llama.cpp/discussions/15021 and
https://github.com/ggml-org/llama.cpp/discussions/21526
[EXTERNAL] GGUF is storage-only quant; weights must be upcast before compute —
fusion is what keeps the upcast off VRAM:
https://dasroot.net/posts/2026/02/gguf-quantization-quality-speed-consumer-gpus/
[EXTERNAL] Fused dequant+matmul is the field-standard fix; "kernel fusion is
essential to beat FP16 throughput" (SplitK W4A16 fused kernel):
https://arxiv.org/pdf/2402.00025
[EXTERNAL] tinygrad BEAM makes it "competitive with PyTorch" (not llama.cpp) —
basis for the 50-70% ceiling, not parity:
https://docs.tinygrad.org/developer/speed/
[EXTERNAL] Templated-autotuning (human primitive + search) is the mainstream
ML-compiler method (AutoTVM template-guided, Ansor template-free; BOLT on
auto-tuners vs hardware-native): https://arxiv.org/pdf/2110.15238

## Known weaknesses (flagged for the auditor to attack)

1. The 4.4x/1.6x split is **one measurement + a residual**, not two independent
   measurements. T0 (BEAM sweep) is what tests it.
2. The "machine fuses it" path is **optimistic**: every fast quantized decode in
   the wild (llama.cpp MMVQ, SplitK) is a *hand-written* fused kernel. No public
   demonstration of compiler-auto-fusion matching hand-fusion for this workload.
   So fusion-via-tinygrad (no hand kernel) is the hopeful case, not the safe one.
3. Part of the 7x could be **CPU/Python dispatch overhead**, not GPU kernel
   bandwidth — not yet ruled out. Fusion would not fix that.
4. The parity ceiling is likely optimistic; ~50-70% of llama.cpp is the honest
   target.
5. This session's author has made repeated optimistic numerical errors; weight
   the [ASSUMED] tags accordingly.

## What we want the audit to check

- Is the bytes-moved decomposition sound, or is the gap better explained by
  occupancy/dispatch/something else?
- Does tinygrad actually fuse the Q4_K dequant into the matvec, or materialize
  fp32? (schedule inspection)
- Is >=50% of llama.cpp realistic for tinygrad fusion+BEAM on gfx1100, or is the
  honest ceiling lower?
- Is the test ordering (T0-T6 in the plan) the right risk-minimizing sequence?

## FALSIFIED by independent audit (Codex, 2026-06-11)

The central thesis claim — "generic float32 expression and not fused into the
matvec" — is MATERIALLY WRONG and is retracted. Independent audit built a
minimal Q4_K GEMV from ggml_data_to_tensor(ggml_type=12) and inspected the
generated AMD kernel:
- tinygrad DOES fuse Q4_K dequant into the GEMV. The kernel loads packed uint8
  Q4_K blocks, unpacks nibbles/scales, converts to half, multiplies the
  activation, writes only the output. NO fp32 dequant buffer is materialized.
- Weights are fp16 by default (model.py:329, HALF=1); decoded weights are only
  materialized under REALIZE=1 (model.py:386). Default REALIZE=0 fuses.
- The 4.4x-bytes / 1.6x-scheduling decomposition is REJECTED. Decode time is
  mostly real graphed GPU work, not CPU dispatch and not an fp32 spill.

### Corrected thesis

tinygrad already fuses Q4_K dequant into a fp16 GEMV. The ~7x gap (measured
~15-16% of llama.cpp on 8B, NOT the proposed 50-70%) is the QUALITY of that
fused kernel: it is generic/scalarized with poor memory-access pattern,
vectorization, and occupancy, versus llama.cpp's tuned packed MMVQ-style GEMV.

### Corrected lever

NOT "make dequant fuse" (already done) and NOT primarily BEAM (the important
fusion already happens). The lever is SPECIALIZING the fused GEMV kernel —
vectorized/packed Q4_K loads and dot, better occupancy — which is a specialized
lowering or hand-written kernel. This is layer-2 (expand the representation),
the part the field consistently hand-writes. The "machine takes most" hope is
weakened: the machine already did the fusion; the residue is the hard part.

### Corrected ceiling

50% of llama.cpp = 3.2x over current. Audit lowers confidence sharply that
BEAM+fusion reaches it, since fusion is not the missing piece. A specialized
packed Q4_K GEMV lowering is likely required; ceiling and effort are now open
questions, not a 50-70% estimate.

### What this vindicates

The layer-1/layer-2 framework predicted exactly this: the machine completed
layer-1 (fusion is reachable, and it reached it); the wall sits at the
representation boundary (vectorized packed-dot GEMV is the primitive the
scheduler's move set does not contain). The human-shaped hole is the
specialized GEMV lowering. The wall is where the theory said; this session's
error was mislocating it one step too early (claiming fusion was the gap).

## Results: BEAM + profile (2026-06-11, commit 1c065d7a8) — diagnosis CONFIRMED

[MEASURED] BEAM produced NO improvement: BEAM=2 caused an AMD HW fault
(memory_lost=1), BEAM=4 not cleanly reached; BEAM=0 stays best (8B ~15.6-15.8,
14B ~5.8 tok/s). The layer-1 search lever is exhausted AND unstable on this
path — it cannot close the gap.

[MEASURED] REALIZE=1 (materialize fp16 weights once) is WORSE: ~13.6-14.2 vs
~15.6 tok/s and ~15.3GB vs ~4.9GB VRAM. Per-token dequant recompute (default)
is correct; materializing fp16 is not a win.

[MEASURED] Generated Q4_K GEMV source: the fused kernel is SCALAR on the quant
loads — many scalar uint8 loads + scalar nibble/dequant, with half4 activation
loads/stores. Kernel shape r_32_32_4_16_4_2_32, 32-thread workgroup. Best
per-kernel BW ~420 GB/s (inside batched-142), but end-to-end only ~75-79 GB/s
(~14% of llama's 567). Dominant cost: batched-256 ~25ms, batched-142 ~16ms,
batched-128 ~12ms per decode step.

### Confirmed diagnosis (evidence, not inference)

The gap is the SCALARIZED QUANT WEIGHT LOAD in the already-fused GEMV. Scalar
uint8 loads cannot coalesce/saturate memory bandwidth; llama.cpp loads Q4
blocks as vectorized words and unpacks in-register. This is the entire residual
and it is squarely layer-2 (the renderer/lowering's move set does not vectorize
this quant-gather pattern; BEAM searching schedules cannot add it).

### Honest verdict on "machine takes most"

For THIS gap, the machine has already given everything it can: it produced the
fusion (the medium-hard part) for free, but BEAM adds nothing (and crashes), so
the remaining ~6.5x is NOT machine-reachable by search. It requires the
specialized vectorized-packed Q4_K GEMV — the layer-2 human-shaped work.

### The one machine-ish shot left (next scoping question)

Is vectorized quant-load reachable by RESTRUCTURING the gguf.py dequant
expression (bitcast Q4 blocks to wider dtype, unpack with vector bit-ops) so
tinygrad's codegen emits vector loads instead of scalar uint8 — vs needing
renderer/hand-kernel work? If the former, it stays "garden the expression, let
codegen vectorize" (machine emits it). If the scalar load is forced by how
tinygrad lowers the gather/index, it needs deeper lowering work. This is
T-SPECIALIZE's first question and the next concrete step.
