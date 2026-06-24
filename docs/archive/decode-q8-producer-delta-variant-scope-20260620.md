# Decode q8 Producer Delta Variant Scope - 2026-06-20

Verdict target: `PASS_DECODE_Q8_PRODUCER_DELTA_VARIANT_ATTRIBUTED`

The oracle repeated comparator attributed `~9.16us` of decode lifecycle debt to the tinygrad-owned q8 producer:

| route | producer us |
|---|---:|
| tinygrad COMGR producer | 30.18 |
| hipcc/LLD producer | 21.06 |

The prior static codegen delta captured COMGR vs hipcc/LLD differences, but did not test whether the main issue is the
producer's workgroup shape. The owned producer uses `256` threads and `1024` bytes of LDS reduction storage; the hipcc
oracle uses `1024` threads and `4096` bytes.

## Tool

`extra/qk_decode_q8_producer_delta_variants.py`

## Variants

| variant | compiler | threads |
|---|---|---:|
| `comgr_nt256` | tinygrad COMGR | 256 |
| `comgr_nt512` | tinygrad COMGR | 512 |
| `comgr_nt1024` | tinygrad COMGR | 1024 |
| `hipcc_lld_nt1024` | hipcc/LLD | 1024 |

All variants run the same RMSNorm + q8_1 producer contract over one 4096-wide activation.

## Gates

| gate | threshold |
|---|---:|
| correctness | fp max abs <= `1e-5`, q8 dequant max abs <= `0.02` |
| artifacts load | pass |
| timing rows per variant | `rounds` |
| attribution | best COMGR variant compared to hipcc/LLD |

## Decision Policy

| classification | meaning |
|---|---|
| `THREAD_SHAPE_CLOSES_DELTA` | COMGR `NT=1024` approaches hipcc/LLD within `2us` |
| `THREAD_SHAPE_PARTIAL` | larger NT improves COMGR but leaves material gap |
| `CODEGEN_SHAPE_DEBT` | larger NT does not materially improve COMGR |

This decides whether the producer fix is a bounded launch/source-shape change or a deeper compiler/codegen issue.
