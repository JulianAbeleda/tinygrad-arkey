# NV decode parity P6-D — FFN-down construction and preliminary CUDA A/B

Date: 2026-08-04
Route: `DEV=CUDA` diagnostic only
Status: **exact primitive construction closed; wall attribution not yet closed**

## Live llama contract

An observational `cudaStreamEndCapture` tap selected the shape-unique down
population (`grid.x=4096`, `ncols_x=12288`). It captured exactly 36 nodes.
Every node had the same typed fusion contract:

- `x_bias` non-null; `gate` and `gate_bias` null; `glu_op=0`;
- grid `(4096,1,1)`, block `(32,4,1)`, dynamic shared bytes `0`;
- row/q8/dst strides `48 / 384 / 4096`;
- channel and sample fast-div triples `(1,0,1)`;
- channel and sample strides `196608 / 384 / 4096`.

Thus the down epilogue is residual addition, `matvec(x) + x_bias`. This is
direct live evidence that all 36 shape-unique down calls use the fused ABI,
independent of the earlier global Q4 fusion-count ambiguity. Raw capture:
`/tmp/llama_ffn_down_graph_params.jsonl`, SHA-256
`1110896782980f32a7ef2b7acf522dd3836d781ecbb1ba1d230c9b8134ffc064`.

## Exact full-shape numerical oracles

`scratchpad/llama_cuda_ffn_down_oracle.py` launches the extracted llama
`has_fusion=true` entry on independently packed tinygrad-owned buffers. The
reference uses llama's pinned CPU Q4/Q6 quantizer and dequantizer, an
independent Python Q8 decoder, and explicit fp32 residual addition.

| quant | shape | max absolute error | verdict | raw payload SHA-256 |
| --- | --- | ---: | --- | --- |
| Q4_K | 4096x12288 | 1.907e-6 | PASS (`atol=1e-3`) | `7221976f933f73fd0b8d687ec9f40ddc91e51704e4f8b70c6db43289860d4c51` |
| Q6_K | 4096x12288 | 1.937e-6 | PASS (`atol=1e-3`) | `15d49870fb3c413f17a7b5131bb350b27a11e576d35101f9e0b9ba94ef03d0f3` |

The raw payloads are `/tmp/llama_ffn_down_q4_full.json` and
`/tmp/llama_ffn_down_q6_full.json`.

## Diagnostic graph splice

`scratchpad/cuda_decode_ffn_down_llama_graph_ab.py` replaces only the matching
tinygrad down core in a freshly constructed CUDA graph. It retains the
original dependency and consumer edges and the existing tinygrad semantic
consumer. The replacement is fp16-to-fp32, exact llama Q8, exact non-fused
llama MMVQ, plus (for Q6) a row-major 16-partial scatter so tinygrad's existing
reduction remains unchanged. The substrate arm deliberately uses non-fused
MMVQ because absorbing residual addition while retaining the existing consumer
would add the residual twice. A separate Q4 full-semantic arm uses the fused
entry and removes the exact residual consumer while preserving its output
buffer identity. Both are attribution machinery, not candidate routes.

The semantic manifest pins two disjoint populations of 18 Q4_K and 18 Q6_K
down calls. Family construction mapped exactly 18 in each arm. One-role Q4 and
Q6 probes preserved the five-token control sequence.

## Family measurements

Each arm used 31 generated tokens and the first timing sample was discarded.

### Q4_K — causal CUDA diagnostic pass

A clean shared-control bracket measured substrate-only and full-semantic arms.
All four arms preserved the exact 31-token sequence. Control A/B medians were
`5.609160` and `5.615703` ms/token (midpoint `5.612432`), with MADs `0.004419`
and `0.005837`.

| arm | mapped calls | median ms/token | delta from control midpoint |
| --- | ---: | ---: | ---: |
| non-fused llama MMVQ + retained tinygrad residual consumer | 18 | 5.546323 | **-0.066109 ms** |
| fused llama MMVQ + exact residual-consumer graph cut | 18 | 5.546670 | **-0.065761 ms** |

The full-semantic cut identifies the consumer as fp32 three-buffer
`out = residual + down`. It points llama's `x_bias` at the residual input,
writes into the consumer's output buffer, removes both the old down core and
residual consumer, and reconnects the fused node to every downstream consumer.
A one-call safety arm and the 18-call arm both preserve tokens.

The incremental fused-epilogue effect is wall-neutral (`+0.000348` ms versus
the substrate arm). Q4 down therefore contributes about **66 us/token** of the
CUDA-route gap, essentially all MMVQ substrate rather than residual fusion.
Raw SHA-256 values for control A / substrate / fused / control B are
`5cba019302d197179b430380ee0654389adaa5e2b105b7ded2fb73fc21cc53d3`,
`0fd68905b71e49418fed7335c22fe2ca65d6310ae00b3ede4106f54929866ca2`,
`9e879f6786622c02d225314a2cab9ee4f4d3158dc5098e7daf12696305fa6531`,
and `543453357cfcb08a09e7d6de8c8663ec8a15e47cc034dc834d2c1e0bf02c239f`.

### Q6_K

The first control median was `5.616502` and treatment median `5.323767`
ms/token, but the treatment token stream diverged at its second token. The
second control was additionally invalid under contention (`10.164767` ms
median, `3.919414` ms MAD). Therefore Q6 is a hard **NO-BOOKING** result. The
most likely immediate explanation is accumulated numerical drift from
replacing all 18 fp16-direct tinygrad GEMVs with llama's Q8 activation path;
the exact isolated MMVQ and partial-layout construction are already proven.
This explanation must be checked with a layer-count sweep and prompt-final
logit deltas before any wall claim.

The replacement-count sweep now localizes the generation threshold:

| Q6 replacements | mapped calls | first divergent generated-token index |
| ---: | ---: | ---: |
| 1 | 1 | none in 31 |
| 2 | 2 | none in 31 |
| 4 | 4 | 1 |
| 8 | 8 | 1 |
| 18 | 18 | 1 |

The raw sweep SHA-256 values for counts 1/2/4/8 are respectively
`109f8e9452f9002095e1855d79bed34518c3e81f65653d4e5cb924a905a9a30d`,
`5d33f88aee65ab82aa6e89d58bd93127a500c1da88c4626eac5bd26d18ff693e`,
`32f3cc82f2c0f884b92124d164eecbf0d14a7eb53d8e0a09fcda38c9fcfd9502`,
and `3efcf66d711d92769004961af5b4bd7d37b07c4c87110d07eaae2102d0891edd`.
This proves the family failure is cumulative rather than a single-call wiring
failure: the same construction passes a 31-token gate at one and two calls.
A direct post-prefill `model.logits` attempt did not enter the TinyJit CUDA
graph path and mapped zero calls, so it was discarded rather than mislabeled
as replacement evidence. Full-logit comparison still requires an explicit
graph output tap or a logits-returning decode TinyJit.

The largest correctness-valid Q6 arm (two calls) was cleanly bracketed. All 31
tokens match; control A/B medians are `5.613947` / `5.603535` (midpoint
`5.608741`) and treatment is `5.561940`, a **-0.046801 ms** local signal for
two calls. This is causal direction, but not a valid 18-call debit:
extrapolation crosses the observed semantic threshold. Raw SHA-256 values for
A / treatment / B are
`d064616aaf9dc18acead2ed2caf5e0bf874756445ffb2f5883352a82c91dea79`,
`7d1b079704e6ca401f47f7cf5fca4c612505d01f9c1b2148df081093597a13e5`,
and `65c308a3d82c1e9ecb60b31b13080df99e86b6f6af6a1fee129771e925fe79ac`.

## Next gates

1. Add an explicit graph output/logit tap and measure the Q6 delta at the
   now-localized `2 -> 4` replacement threshold. Do not interpret its family
   wall until token/logit correctness passes.
2. Keep all values CUDA-route diagnostic. Native NV has different promotion
   policy (including Q6 epilogue fusion), so no native residual may be debited
   without a native-owned same-session A/B.

No production/default route was edited. Six hermetic oracle/manifest tests
pass in `test_llama_cuda_ffn_down_oracle.py` plus the shared live-oracle tests.
