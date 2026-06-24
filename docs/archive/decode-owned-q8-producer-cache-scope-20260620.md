# Decode Owned q8 Producer/Cache Scope - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_PRODUCER_CACHE_SCOPE_READY`

This scopes the first owned q8 successor build track. It is producer/cache only: no gate/up consumer schedule work, no
runtime route, no default change.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_producer_cache_scope.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_scope_result.json
```

## Contract

| field | value |
|---|---|
| source activation | post-norm decode activation |
| q8 format | `block_q8_1_or_artifact_compatible_q8` |
| elements | `4096` |
| q8 blocks | `128` |
| q8 bytes | `4608` |
| reuse | `2`, shared by `ffn_gate` and `ffn_up` |
| fallback | existing default tinygrad decode |
| initial default | off |

Producer operation:

```text
rmsnorm output + q8 sidechannel
scale = max(abs(vals[32])) / 127
d = fp16(scale)
s = fp16(0)
qs = round-nearest int8, clipped to [-128, 127]
```

## Targets

| target | value |
|---|---:|
| fused producer lifecycle | `<= 7.501us` |
| incremental producer overhead | `<= 4.8us` |
| measured incremental overhead | `0.923us` |
| quality threshold | dNLL `<= 0.01` |

## Phases

| phase | exit gate |
|---|---|
| OPC-1 structural object | shape, format, reuse, fallback, and targets pass structural gate |
| OPC-2 byte semantics reference | q8 bytes are `block_q8_1`-compatible and reference dequant error is bounded |
| OPC-3 owned lowering candidate | owned implementation matches semantics and lifecycle target |

Next executable probe:

```text
extra/qk_decode_owned_q8_producer_cache_object_probe.py
```
