# Decode q8 Producer Delta Variant Result - 2026-06-20

Verdict: `PASS_DECODE_Q8_PRODUCER_DELTA_VARIANT_ATTRIBUTED`

Classification: `THREAD_SHAPE_CLOSES_DELTA`

Command:

```sh
PYTHONPATH=. python3 extra/qk_decode_q8_producer_delta_variants.py --rounds 40
```

## Result

| variant | compiler | threads | median us |
|---|---|---:|---:|
| `comgr_nt256` | tinygrad COMGR | 256 | 30.76 |
| `comgr_nt512` | tinygrad COMGR | 512 | 24.56 |
| `comgr_nt1024` | tinygrad COMGR | 1024 | 21.00 |
| `hipcc_lld_nt1024` | hipcc/LLD | 1024 | 21.72 |

Correctness passed for all variants:

| check | value |
|---|---:|
| producer fp max abs | 0 |
| q8 dequant max abs | 0.010657 |

## Static Shape

| variant | instructions | LDS bytes | private bytes | VALU | branches |
|---|---:|---:|---:|---:|---:|
| `comgr_nt256` | 1394 | 1024 | 0 | 879 | 23 |
| `comgr_nt512` | 1421 | 2048 | 0 | 879 | 24 |
| `comgr_nt1024` | 1427 | 4096 | 0 | 888 | 25 |
| `hipcc_lld_nt1024` | 1207 | 4096 | 0 | 653 | 13 |

The static COMGR-vs-hipcc codegen delta remains visible, but it is not the producer timing blocker. Matching the
workgroup shape closes the measured producer gap anyway.

## Interpretation

The `~9us` owned-producer debt from the repeated comparator was caused by the producer workgroup shape, not by an
unbounded native compiler issue:

| comparison | delta |
|---|---:|
| `comgr_nt256 - comgr_nt1024` | +9.76us |
| `comgr_nt1024 - hipcc_lld_nt1024` | -0.72us |

So the bounded native producer fix is clear: use the COMGR producer at `NT=1024` / 4096-byte LDS reduction storage for
the q8 decode research route. This preserves correctness and avoids importing the hipcc/LLD producer artifact.

## Next Step

Rerun the mixed q8 lifecycle with the `comgr_nt1024` producer. Expected outcome: recover the `~9us` producer gap and
move total lifecycle from `~131.8us` toward the oracle route's `~122.7us`. The remaining miss to `115.24us` should then
belong to the shared consumer/session path.
