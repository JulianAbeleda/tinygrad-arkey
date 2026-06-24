# Decode q8 NT1024 Lifecycle Gate Scope - 2026-06-20

Verdict target: `PASS_DECODE_Q8_NT1024_LIFECYCLE_GATE`

The producer variant probe showed that changing the owned COMGR q8 producer from `NT=256` to `NT=1024` closes the
producer-only delta against hipcc/LLD. This gate validates the fix in the actual mixed lifecycle:

| component | route |
|---|---|
| producer | tinygrad COMGR q8 producer, `NT=1024` |
| consumer | hipcc/LLD fused q8 gate/up |

## Tool

`extra/qk_decode_q8_nt1024_lifecycle_gate.py`

## Gates

| gate | threshold |
|---|---:|
| producer correctness | pass |
| consumer correctness | pass |
| lifecycle total | `<= 115.24us` |
| producer recovered | producer median within `2us` of `21.72us` producer variant row |

If lifecycle still misses while producer is recovered, the remaining debt is assigned to shared consumer/session timing.
