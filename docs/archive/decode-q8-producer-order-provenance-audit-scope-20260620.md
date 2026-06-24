# Decode q8 Producer Order Provenance Audit Scope - 2026-06-20

Verdict target: `PASS_DECODE_Q8_PRODUCER_ORDER_PROVENANCE_NO_CONTEXT_SLOWDOWN`

The resource-context audit did not reproduce the q4-residency slowdown. The remaining question is whether producer
timing follows execution order/session state instead of buffer residency.

## Tool

`extra/qk_decode_owned_q8_producer_order_provenance_audit.py`

## Interleaved Rows

Each round randomizes these rows while keeping all context buffers alive:

| row | meaning |
|---|---|
| `producer_only` | producer timing with no immediately preceding gate/up dispatch |
| `producer_with_real_q4_resident` | same producer arguments, real q4 buffers resident |
| `producer_with_dummy_resident` | same producer arguments, dummy same-size buffers resident |
| `producer_after_gateup_dispatch` | gate/up dispatch immediately before producer |

The producer arguments are identical in all rows. Differences therefore come from process/device state, previous
dispatches, or residency, not producer inputs.

## Gates

| gate | threshold |
|---|---:|
| producer correctness | all rows pass |
| rows per label | `rounds` |
| post-gateup median delta | `< 5us` for pass |
| real-q4 resident median delta | `< 5us` for pass |

## Decision

- If all deltas are small, rerun the full lifecycle gate because the prior slowdown was not stable.
- If `producer_after_gateup_dispatch` is slow, inspect queue/cache ordering.
- If resident rows are slow, inspect allocator/VRAM placement.
