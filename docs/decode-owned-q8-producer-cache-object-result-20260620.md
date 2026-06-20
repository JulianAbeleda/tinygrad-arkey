# Decode Owned q8 Producer/Cache Object Result - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_PRODUCER_CACHE_OBJECT_STRUCTURAL`

The owned q8 producer/cache is now represented as a structural metadata object. It is not an implementation.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_producer_cache_object_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_object_result.json
```

## Structural Contract

| field | value |
|---|---|
| operation | rmsnorm output plus q8 sidechannel |
| elements | `4096` |
| block elems | `32` |
| blocks | `128` |
| block bytes | `36` |
| total bytes | `4608` |
| fields | `d_fp16`, `s_fp16`, `qs_i8x32` |
| reuse | `2` consumers: `ffn_gate`, `ffn_up` |
| policy | default off, fallback to existing decode |

## Gates

The object gate checks layout, reuse, fallback, lifecycle target, incremental target, quality target, and metadata-only
status. It passes.

Next:

```text
extra/qk_decode_owned_q8_producer_cache_reference_probe.py
```

That probe should freeze byte/reference semantics before any owned lowering candidate is built.
