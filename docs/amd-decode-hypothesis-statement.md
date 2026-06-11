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
