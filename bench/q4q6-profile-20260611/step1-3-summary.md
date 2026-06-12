# Q4+Q6 Profile And Sweep Summary

Scope: Qwen3 8B/14B Q4_K_M, native AMD, `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1`.

## Step 1 Profile

| model | mode | tok/s | AMD ms/tok | residual | decision |
|---|---:|---:|---:|---:|---|
| 8B | batched | 58.77 | 16.32 | 4.28% | runtime residual low |
| 8B | named | 5.42 | 33.94 | 81.63% | attribution only |
| 14B | batched | 28.79 | 34.06 | 1.98% | runtime residual low |
| 14B | named | 4.01 | 78.11 | 68.71% | attribution only |

Named AMD-kernel ownership after Q4+Q6:

| model | Q4 primitive | Q6 primitive | Q4 reductions | fallback quant |
|---|---:|---:|---:|---:|
| 8B | 33.02% | 17.50% | 2.52% | 14.34% |
| 14B | 26.53% | 12.73% | 17.80% | 24.13% |

Verdict: primitive tuning is Amdahl-relevant, but the target is batched
throughput. Named rows are only used for attribution.

## Step 2 Automated Sweep

Q4 sweep:

| model | tensor | current policy | sweep best | carried? |
|---|---|---|---|---|
| 8B | `ffn_gate/up` | `parts=1 LOCAL:0:64` | same | yes, already current |
| 8B | `attn_q/output` | `parts=1 LOCAL:0:64` | same | yes, already current |
| 8B | `ffn_down` | `parts=4 LOCAL:0:32` | same | yes, already current |
| 14B | `ffn_gate/up` | `parts=1 LOCAL:0:64` | `parts=1 LOCAL:0:32` | no |
| 14B | `attn_q/output` | `parts=1 LOCAL:0:64` | same | yes, already current |
| 14B | `ffn_down` | `parts=4 LOCAL:0:32` | `parts=2 LOCAL:0:32` | no |

Q6 sweep:

| model | tensor | current policy | sweep best | microbench ratio | carried? |
|---|---|---|---|---:|---|
| 8B | `ffn_down` | `parts=1 LOCAL:0:64` | `parts=2 LOCAL:0:32` | 1.56x vs current primitive | no |
| 14B | `ffn_down` | `parts=1 LOCAL:0:64` | `parts=2 LOCAL:0:32` | 1.41x vs current primitive | no |
| 8B | `output.weight` | fallback | `parts=1 LOCAL:0:16` | 1.10x vs fused | no |
| 14B | `output.weight` | fallback | `parts=1 LOCAL:0:16` | 1.09x vs fused | no |

The primitive kernels remain far below a full bandwidth roof on several shapes
despite wide loads: Q6 `ffn_down` best reaches only about 202-218 quant-GB/s
and roughly 0.49-0.53 dot TFLOP/s. The immediate limiter is therefore not just
raw DRAM traffic; arithmetic/unpack shape, occupancy, and reduction structure
still matter.

## Step 3 Output And Policy Verdict

The output projection microbench win did not translate to full decode:

| variant | 8B avg | 8B last16 | 14B avg | 14B last16 | verdict |
|---|---:|---:|---:|---:|---|
| previous Q4+Q6 policy | 58.17 | 55.98 | 28.27 | 27.80 | stable baseline |
| output enabled + sweep policies | 53.85 | 25.21 | 28.87 | 28.39 | reject, 8B collapse |
| no output + sweep policies | 59.98 then 15.12 rerun | 54.39 then 14.43 | 27.13 then 28.77 rerun | 17.43 then 28.22 | reject, unstable |
| reverted policy rerun | 57.45 | 54.89 | not rerun | not rerun | stable range restored |

Final decision: no runtime policy change is accepted from this sweep. Keep the
last correctness-verified production policy (`Q6_K ffn_down parts=1
LOCAL:0:64`, Q6 output fallback). The useful carry-forward is the automated Q6
sweep harness and the no-go evidence for output/parts=2 candidates.
