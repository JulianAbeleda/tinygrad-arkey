# Decode q8 Producer Context Isolation Result - 2026-06-20

Verdict: `BLOCKED_DECODE_Q8_PRODUCER_CONTEXT_ISOLATION_PRODUCER_CONTEXT_SLOW`

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_producer_context_isolation.py --warmups 8 --iters 20
```

## Result

| row | median us |
|---|---:|
| producer-only | `21.60` |
| producer after q4 gate/up buffers | `30.94` |
| producer after gate/up program load | `30.94` |
| gate/up after producer | `93.06` |
| producer after gate/up execution | `30.46` |
| producer after second gate/up execution | `23.32` |
| controlled lifecycle | `124.00` |
| target lifecycle | `115.24` |

Correctness passed for producer and gate/up in all measured contexts.

## Deltas

| delta | us |
|---|---:|
| q4 buffer residency delta | `+9.34` |
| gate/up program load delta | `+0.00` |
| post gate/up execution delta | `-0.48` |
| controlled lifecycle gap | `+8.76` |

## Interpretation

The lifecycle miss is not caused by hipcc/LLD gate/up program load. The producer slowdown appears as soon as the
large q4 gate/up buffers are resident:

```text
21.60us producer-only -> 30.94us after q4 buffers
```

That accounts for the lifecycle gap almost exactly:

```text
124.00us controlled lifecycle - 115.24us target = 8.76us
```

So the next decode step is not consumer schedule rewriting and not more static native schedule search. It is
producer context/resource-state attribution: why does resident q4 storage slow the q8 producer launch/body by
~9us?

## Next Scope

Build `decode-q8-producer-resource-context-audit`:

- vary q4 buffer count and size,
- distinguish allocation/residency from copy-in,
- test producer with dummy same-size buffers vs real q4 buffers,
- check whether the slowdown follows VRAM pressure, cache/TLB effects, queue ordering, or allocator placement,
- attach clock/provenance if available.
