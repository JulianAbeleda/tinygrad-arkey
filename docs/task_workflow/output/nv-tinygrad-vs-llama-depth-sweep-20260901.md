# NVIDIA tinygrad versus llama depth sweep

This compares Qwen3-8B Q4_K_M batch-1 decode at fixed KV depths on the RTX
5090. It follows the phase separation used by the AMD authority: decode depth
is measured independently from whole-prefill length.

## Decode result

| Depth | tinygrad tok/s | llama tok/s | tinygrad margin | tinygrad decay vs 512 | llama decay vs 512 |
|---:|---:|---:|---:|---:|---:|
| 512 | 242.106 | 247.784 | -2.29% | baseline | baseline |
| 1024 | 232.615 | 244.343 | -4.80% | -3.92% | -1.39% |
| 2048 | 226.442 | 234.713 | -3.52% | -6.47% | -5.28% |
| 4096 | 206.629 | 226.022 | -8.58% | -14.65% | -8.78% |

All tinygrad points selected the production Flash decode route. The tinygrad
drop is not an immediate post-512 cliff. Its largest interval loss is
2048-to-4096: 8.75%, versus llama's 3.70%. The problem is therefore deep-context
Flash/KV service slope, especially the final doubling, rather than a route
transition at 512.

The fresh depth-authority result does not reproduce the retained strict
depth-512 endpoint win. The retained comparison was tinygrad 247.80 versus
llama 246.405 tok/s; this fresh curve starts at tinygrad 242.106 versus llama
247.784. The protocols and thermal sessions differ, so the fresh curve is the
authority for slope and the retained bracket remains the authority only for
its exact endpoint claim.

## Tinygrad protocol qualifications

- 40 decode tokens, three repetitions, genuine prompt prefill to exact depth.
- Production wall path `W`; dispatch diagnostic omitted.
- Depth 512 used the default packed pp512 bootstrap after repairing its missing
  Q6/Q4-down binding construction.
- Depths 1024/2048/4096 disabled only `NV_LLAMA_FULL_PACKED_PP512` during KV
  bootstrap and used exact `max_context=4160`. The production Flash decode
  route was unchanged. This was necessary because retained packed-prefill graph
  workspace exhausted the 32 GB card before decode measurement.

## Prefill blocker

The production tinygrad whole-prefill authority did not produce a curve. At
`max_context=4160`, the packed pp512 route reached 30.22 GB used and failed a
2.02 MB allocation. This is an OOM/workspace-lifetime blocker, not a measured
throughput regression. Llama prefill remains 14,413 / 14,238 / 13,752 tok/s at
the reliable pp1024 / pp2048 / pp4096 points.

Do not substitute the ordinary fallback prefill route and label it as the
current production comparison. The next prefill task is to reduce or release
packed-route records/workspace sufficiently to run the authority curve.

## Next localization

The next decode test should decompose Flash attention and KV-cache work at
depths 2048 and 4096 under one frozen clock/session. The target is the extra
5.95 percentage points of 512-to-4096 decay relative to llama, with attention
score/PV, cache reads, launch count, and non-attention body held separately.

Raw evidence:

- `docs/task_workflow/evidence/nv-depth-sweep-20260901/llama/`
- `docs/task_workflow/evidence/nv-depth-sweep-20260901/tinygrad/`
