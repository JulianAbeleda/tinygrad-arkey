# Dense quantized conversion: first-principles research plan

Status: research plan only. No production route is changed and no token-rate
recovery is booked.

## Why this exists

The fused U4Z8 gate/up kernel is materially faster than the installed Q4_K
kernel, but converting the already-quantized Q4_K weights into the smaller
packet damaged recurrent logits. The missing work is therefore weight-artifact
construction, not another CUDA spelling.

The correct pipeline is:

```text
BF16/FP16 source -> representative activations -> calibrated quantizer
                 -> compact packet -> semantic qualification -> production
```

The high-precision source is not the serving format. It is the information
needed to manufacture a compact serving artifact without stacking quantization
error on top of Q4_K error.

## First-principles objective

For a linear projection with weights `W`, representative inputs `X`, and
candidate packet weights `W_hat`, optimize the computation error:

```text
min ||W X - W_hat X||^2
```

Weight-only mean-square error, `||W-W_hat||^2`, is only a proxy and failed to
predict our recurrent behavior. The calibration objective must be measured at
the projection output and, later, at the transformer-block/logit output.

## Evidence already collected

### Kernel substrate

- One 12288x4096 U4Z8 projection: `1.078 us` continuous rotated-cold recovery.
- Conservative gate/up pair: `1.299 us/layer` recovery despite two projection
  launches plus a separate finish.
- Fused U4Z8 pair: `3.625 us/layer` recovery at R9 with independent rings.
- Nonzero fused composition is finite and agrees with the two independently
  qualified projections within the FP16 comparison tolerance.

This exposes about `130.5 us/token` over 36 layers, or an isolated ceiling near
254 tok/s. It is an exposure estimate, not a production claim.

### Quality wall

Using the installed Q4_K model as the source failed every post-hoc compact
contract at the minimum one-layer dose. Direct BF16 source testing separated the
remaining causes:

| Contract | Result against raw-BF16 block baseline |
|---|---|
| symmetric U4Z8, 136 B/block | severe recurrent divergence |
| local MSE U4Z8, 136 B/block | severe recurrent divergence |
| affine U4, 140 B/block | about 0.38% stacked logit error |
| one-vector activation-weighted affine | about 0.43% stacked logit error |

The internal admission limit is stacked relative L2 <= 0.1%, finite logits,
and stable token decisions on held-out positions. The single activation vector
was a calibration substrate check, not a valid corpus.

## Conversion methods to investigate

### AWQ: activation-aware equivalent scaling

AWQ identifies important input channels from activation magnitude and applies an
equivalent transformation: scale columns of `W` up and scale the input channels
down. The computation is unchanged before quantization, but salient channels
become easier to represent. Use one shared channel-scale vector for gate and up,
because they consume the same normalized input. Fold the inverse scale into the
FFN RMSNorm weights so inference adds no separate scale kernel.

Reference: [AWQ paper](https://arxiv.org/abs/2306.00978) and [official implementation](https://github.com/mit-han-lab/llm-awq).

### GPTQ: second-order error feedback

Collect `X` and form a damped approximate Hessian:

```text
H = 2 X X^T + lambda I
```

Quantize weights in small column blocks. When one column is rounded, propagate
its error through `H^-1` to later columns before rounding them. This makes the
final packet minimize output damage rather than independently rounding every
weight. The method supports grouped quantization grids.

Reference: [GPTQ paper](https://arxiv.org/abs/2210.17323), [official implementation](https://github.com/ist-daslab/gptq), and [Qwen quantization notes](https://github.com/QwenLM/Qwen3/blob/main/docs/source/quantization/gptq.md).

### OmniQuant: learned clipping and equivalent transforms

If AWQ plus GPTQ remains outside the quality gate, optimize clipping thresholds
and equivalent transformations jointly using block-output reconstruction. This
is more expensive but still post-training and can run layer-by-layer.

Reference: [OmniQuant paper](https://arxiv.org/abs/2308.13137) and [official implementation](https://github.com/OpenGVLab/OmniQuant).

## Exact future experiment

1. Capture at least 128 calibration sequences and multiple token positions per
   sequence. Keep separate held-out sequences for qualification.
2. Capture the normalized FFN input `X` at each gate/up layer through the
   already-materialized program buffer, not by realizing tensors inside a JIT
   trace.
3. Load BF16 gate/up weights one layer at a time from the streamed safetensor
   shards.
4. Search shared AWQ channel scales for the gate/up pair. Fold inverse scales
   into the corresponding FFN RMSNorm parameters.
5. Run GPTQ-style grouped error-feedback rounding into the exact packet grammar.
6. Compare candidate gate/up outputs with the BF16 gate/up baseline on held-out
   activations. Reject before running the full model if this local test fails.
7. Run recurrent full-logit qualification at doses 1, 4, 18, and 36 layers.
8. Implement the packed fused kernel only after the artifact passes quality.
9. Run predecessor-conditioned timing, then a strict batch-1 endpoint bracket.

## Stop conditions

- A candidate that passes a microgate but fails held-out recurrent quality is
  not promotable.
- A candidate that improves the installed Q4_K model but is not compared with a
  raw-BF16 local baseline has ambiguous evidence and is not promotable.
- Post-hoc conversion from Q4_K is closed for the 136--142 byte contracts
  already tested.
- The current installed endpoint remains 245.948 tok/s until a full strict
  bracket improves it.

## Current machine state

The official Qwen3-8B BF16 index and shard 1 are staged under
`/home/ubuntu/models/qwen3-8b-bf16-stream/`. The 14B GGUF was removed with user
authorization to provide disk headroom. The remaining shards should be fetched
only when the multi-sample calibration collector is ready; shard-by-shard
processing avoids holding the full source and output copies simultaneously.
