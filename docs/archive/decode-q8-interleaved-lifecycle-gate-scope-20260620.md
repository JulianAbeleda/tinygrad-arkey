# Decode q8 Interleaved Lifecycle Gate Scope - 2026-06-20

Verdict target: `PASS_DECODE_Q8_INTERLEAVED_LIFECYCLE_GATE`

Producer context audits showed that the producer slowdown does not reliably follow q4 residency or previous gate/up
dispatch under interleaving. The next gate is therefore the actual lifecycle under paired/interleaved ordering.

## Tool

`extra/qk_decode_owned_q8_interleaved_lifecycle_gate.py`

## Method

Each round randomizes:

| row | measured |
|---|---|
| `producer_only` | producer only |
| `lifecycle` | producer immediately followed by gate/up consumer, total = producer + consumer |

All q4, q8, output, and producer buffers remain resident. Correctness is checked for producer and gate/up.

## Gates

| gate | threshold |
|---|---:|
| producer correctness | pass |
| consumer correctness | pass |
| rows per label | `rounds` |
| lifecycle total median | `<= 115.24us` |

If this passes, the previous lifecycle miss was harness/order sensitive. If it fails, the producer/consumer split
from this interleaved run becomes the next blocker.
