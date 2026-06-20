# Decode q8 Producer Context Isolation Scope - 2026-06-20

Verdict target: `PASS_DECODE_Q8_PRODUCER_CONTEXT_ISOLATION_LIFECYCLE_READY`

The latest decode lifecycle records say the consumer is near expected and the owned q8 producer is the blocker:

| row | value |
|---|---:|
| mixed lifecycle | `123.64us` |
| target lifecycle | `115.24us` |
| producer in lifecycle | `30.54us` |
| consumer in lifecycle | `93.10us` |
| controlled lifecycle attribution | `124.12us` |

Do not reopen Q4_K addressing, q8 addressing, scale/min extraction, dot4 selection, gate/up correctness, or native
schedule search from static similarity. The next executable gate is producer context isolation.

## Tool

`extra/qk_decode_owned_q8_producer_context_isolation.py`

## Rows

| row | purpose |
|---|---|
| producer-only | producer buffers/program only |
| producer after q4 buffers | checks whether resident gate/up buffers perturb producer timing |
| producer after gate/up program load | checks whether loading the hipcc/LLD consumer perturbs timing |
| gate/up after producer | confirms consumer timing/correctness in same context |
| producer after gate/up execution | checks whether prior consumer execution perturbs producer timing |
| producer after second gate/up execution | checks repeatability after a second consumer pass |

## Gates

| gate | threshold |
|---|---:|
| producer correctness in every context | pass |
| gate/up correctness | pass |
| producer after gate/up load <= cross-apply producer row | `<= 30.54us` |
| controlled lifecycle <= artifact target | `<= 115.24us` |

If lifecycle still fails while producer is context-sensitive, the next work is HCQ/cache/resource-state attribution,
not consumer schedule rewriting. If producer is clean but lifecycle still fails, inspect consumer variance or the
sum/harness.
