# Decode Owned q8 Producer Codegen Delta Result - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_PRODUCER_CODEGEN_DELTA_CAPTURED`

This probe compares the producer-only hipcc/LLD artifact with the owned COMGR producer:

```text
hipcc/LLD producer: extra.q8_ffn_fast_artifact_probe.hip_norm_source(1024)
COMGR producer:     extra.q8_ffn_hcq_artifact.NORM_SOURCE
```

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_producer_codegen_delta_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_owned_q8_producer_codegen_delta_result.json
```

The result records instruction counts, grouped op counts, runtime descriptor fields, and COMGR-minus-hipcc deltas.
Use it only to scope a future producer optimization; the owned COMGR producer is already accepted as the HCQ-parity row.

## Captured Delta

| metric | hipcc/LLD | COMGR | COMGR - hipcc |
|---|---:|---:|---:|
| instruction count | `1207` | `1394` | `+187` |
| VALU | `653` | `879` | `+226` |
| SALU | `509` | `474` | `-35` |
| branches | `13` | `23` | `+10` |
| waitcnt | `88` | `37` | `-51` |
| group segment bytes | `4096` | `1024` | `-3072` |
| private segment bytes | `0` | `0` | `0` |

The obvious static issue is code shape: COMGR emits many more VALU ops and branches. That is the next producer
optimization target if we choose to chase HIP-oracle producer parity.
