# Decode Owned q8 Producer Target Reconcile Result - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_PRODUCER_TARGET_RECONCILED`

The prior lowering result used the wrong target for the immediate parity question.

There are two different producer targets:

| row | runtime | producer us | meaning |
|---|---|---:|---|
| HIP-runtime modeled oracle | HIP runtime events | `7.501` | upper oracle target |
| HCQ hipcc/LLD artifact | tinygrad HCQ / AMDProgram | `~21.3-21.7` | actual artifact parity row |
| owned COMGR candidate | tinygrad HCQ / COMGR | `15.70` | owned producer row |

So the owned COMGR producer is:

- correct;
- no in-process HIP runtime;
- faster than the HCQ-loaded hipcc/LLD artifact producer;
- still slower than the HIP-runtime modeled producer oracle.

## Corrected Decision

The owned producer/cache path is **not blocked for HCQ artifact parity**. It is blocked only if the requirement is
HIP-oracle producer parity.

Next: promote the owned producer/cache row as the HCQ-parity candidate, then separately decide whether chasing the
remaining HIP-oracle producer delta is worth codegen work.
