# Prefill Graph GEMM Experimental Promotion Result - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_EXPERIMENTAL_OPT_IN`

`PREFILL_GRAPH_GEMM=1` is approved as an explicit experimental fast path. It remains default-off.

## What It Means

The observed NLL drift is distributed fp16 numeric drift from using a different GEMM implementation, not evidence
of a broken matmul role.

Evidence:

| gate | result |
|---|---:|
| full in-model speed | `4895.9 tok/s`, `104.6ms / 512` |
| speedup vs same-session `PREFILL_V2` | `1.89x` |
| one-role numeric correctness | rel RMSE `0.0002077` |
| sampled quality smoke | `max_abs_dNLL = 0.0` |
| corpus quality degradation | `max_positive_dNLL = 0.009443` |
| greedy generation coverage | `0 / 32` token mismatches |
| role attribution | no role caused argmax mismatch |

The absolute parity report is still visible:

| parity metric | value |
|---|---:|
| corpus `max_abs_dNLL` | `0.017593` |
| focused attribution full-route `max_abs_dNLL` | `0.017593` |

That fails strict near-parity, but the failing side is mostly benign or favorable movement. The worst absolute row
made the true token more likely (`dNLL = -0.017593`), and the worst harmful row stayed under the degradation gate
(`dNLL = +0.009443`).

## Decision

Use this route when the user explicitly opts in:

```bash
DEV=AMD PREFILL_V2=1 PREFILL_GRAPH_GEMM=1 PYTHONPATH=. python3 extra/qk_prefill_v2_measure.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
```

Do not make it default-on yet.

## Default-On Blockers

| blocker | why |
|---|---|
| repeated performance run | confirm no sync/clock artifact across repeated sessions |
| larger generation coverage | current generation pass is 4 prompts x 8 tokens |
| fallback/OOM audit | prove unsupported shapes fall back and no new OOM mode beyond `PREFILL_V2` |
| parity report stays above `0.01` | not a quality blocker, but should remain reported for default-on review |

## Next Step

The next engineering step is not to chase role attribution. It is to run a default-on readiness audit:

1. repeat same-session perf,
2. expand generation coverage,
3. audit unsupported-shape fallback,
4. audit OOM behavior.

If those pass, the remaining decision is policy: whether default-on accepts degradation/generation gates while
reporting absolute parity drift.
