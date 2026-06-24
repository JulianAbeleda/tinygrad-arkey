# Decode Owned q8 Producer/Cache Reference Result - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_PRODUCER_CACHE_REFERENCE_SEMANTICS`

The producer/cache byte semantics are now frozen in a reference probe. This is still not a lowering or runtime route.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_producer_cache_reference_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_reference_result.json
```

## Reference Semantics

For each 32-value block:

```text
scale = max(abs(vals)) / 127
d = fp16(scale)
s = fp16(0)
qs = round(vals / scale), clipped to int8
```

Layout:

| field | value |
|---|---:|
| elements | `4096` |
| blocks | `128` |
| block bytes | `36` |
| total bytes | `4608` |

The probe validates q8 byte length, block fields, int8 range, dequant error, and reuse count `2`.

## Next

The next step would be an owned producer/cache lowering candidate. It must match this reference and clear the lifecycle
target before W==D. Search is still blocked until such a candidate exists and measures.
