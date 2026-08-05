# NV decode non-quantized role partition

Date: 2026-08-05. Workload: Qwen3-8B-Q4_K_M, d512, RTX 5090 / driver
595.84. Status: **PASS — all 731 native non-quantized nodes and the complete
300.736 us llama exposed non-MMQ union are partitioned disjointly.**

## Result

The former `non_quantized` bucket is no longer an undifferentiated 1.409 ms.
The additive comparison is:

| disjoint semantic family | native serialized us | llama total ownership us | llama hidden us | fusion/dataflow/body attribution us | critical-path delta us |
| --- | ---: | ---: | ---: | ---: | ---: |
| norms, including Q/K norms | 673.211 | 177.881 | 79.324 | **+495.330** | **+574.654** |
| flash score and combine | 305.581 | 142.552 | 84.959 | **+163.029** | **+247.989** |
| residuals, casts, activation epilogues, contiguous | 240.762 | 0.656 | 0.213 | **+240.106** | **+240.319** |
| vocab sampler and token feedback | 72.474 | 1.259 | 0.000 | **+71.215** | **+71.215** |
| Q/K RoPE and fused KV store | 116.790 | 98.825 | 15.579 | **+17.965** | **+33.543** |
| llama Q8_1 projection-input quantization | 0.000 | 325.517 | 265.879 | **-325.517** | **-59.639** |
| **total** | **1408.818** | **746.690** | **445.954** | **+662.128** | **+1108.082** |

This separates two mechanisms that the former exposed-only table combined:
**662.128 us** is fusion/dataflow/body attribution after assigning llama's
complete non-MMQ union disjointly, and **445.954 us** comes from work
llama hides behind MMQ. Their sum is the exact 1108.082 us support-work gap.
The split is accounting-safe, but the first term is not a matched raw-kernel
comparison; only a controlled A/B can distinguish body, fusion, and dataflow
choices within a family.

The JSON rows are rounded to 0.001 us, so summing the displayed deltas gives
1108.081 us. The unrounded equation is exact; the native partition differs
from its 1408.818 us authority by 0.001 us and the llama partition by 0.

The causal priority is therefore narrower than “non-quant work”: norms are
the largest support-work deficit, then the two flash kernels, then the six
families of residual/cast/contiguous epilogues. Those first three populations
account for 1062.962 us of positive deficit before llama's Q8 quantization
credit. RoPE/KV storage is comparatively small.

## Native exact ownership

The selected native capture has an exact topology:

- seven prefix nodes;
- 36 repetitions of a 26-node layer template, each anchored by six exact
  quantized identities (Q/K/V/O, fused gate/up, down);
- one quantized vocab node and four sampler-tail nodes.

The parser checks all 36 layer starts, every semantic quant identity, every
unhashed non-quant kernel signature at its exact relative ordinal, the Q4/Q6
KV-store variant selected by the exact V tensor fact, and exact prefix/suffix
signatures. Any mismatch fails closed. It assigns exactly 731 distinct nodes:

| exact role | calibrated us |
| --- | ---: |
| flash score | 204.697 |
| FFN RMSNorm | 187.815 |
| Q norm | 173.106 |
| next-layer/final RMSNorm | 170.901 |
| K norm | 135.625 |
| flash combine | 100.884 |
| vocab sampler | 67.088 |
| Q RoPE | 59.466 |
| FFN residual add | 43.151 |
| attention residual add | 40.725 |
| FFN-down cast | 40.662 |
| FFN activation cast | 40.253 |
| attention cast | 38.458 |
| block-output contiguous | 37.513 |
| fused Q4 KV store/K-RoPE/cast | 29.922 |
| fused Q6 KV store/K-RoPE/cast/partial-reduce | 27.402 |
| initial RMSNorm | 5.764 |
| token feedback | 5.386 |

The Q6 row remains a compound role intentionally. The promoted KV-store ABI
performs the V four-partial reduction, K RoPE, K/V casts, and cache store in
one kernel, so timestamps cannot split those operations without inventing
ownership. Its entire 18-kernel population is only 27.402 us; no GPU A/B is
needed to resolve the ranking.

## Llama overlap-safe ownership

Individual llama class exposure rows cannot be added because non-MMQ classes
overlap each other. The parser instead divides every elementary timeline
interval equally among the active non-MMQ classes. This is the symmetric
Shapley value of the interval-coverage game and is invariant to an arbitrary
class priority order. Median replay shares are normalized to the settled
aggregate authorities.

| llama class | exposed Shapley us | hidden Shapley us |
| --- | ---: | ---: |
| RMSNorm | 98.557 | 79.324 |
| Q8_1 quantization | 59.639 | 265.879 |
| RoPE | 42.138 | 15.579 |
| KV set rows | 41.108 | 0.000 |
| flash score | 36.430 | 32.853 |
| flash combine | 21.162 | 52.106 |
| get rows | 1.259 | 0.000 |
| elementwise | 0.443 | 0.213 |
| **total** | **300.736** | **445.954** |

The compact JSON also records priority-order marginal bounds for every class:
unique coverage is the lower bound and full class coverage is the upper bound.
Shapley ownership is accounting, not a claim that an A/B will recover exactly
that number. A real family A/B remains the causal gate for implementation.

## Artifacts and validation

- Parser: `extra/llm_research/decode/nonquant_role_partition.py`
- Payload: `docs/task_workflow/output/nv-decode-nonquant-role-partition-20260805.json`
- Test: `test/unit/test_nonquant_role_partition.py`
- Inputs and SHA256 values are embedded in the payload.

Validation:

```text
PYTHONPATH=. .venv/bin/pytest -q test/unit/test_nonquant_role_partition.py
2 passed
```

No production route, default, graph schedule, or promoted kernel changed. No
GPU run was necessary; the existing native profile/DAG and llama CUPTI trace
settled the attribution.
