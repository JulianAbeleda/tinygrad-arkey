# Decode Owned q8 Lifecycle Successor Object Result - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_LIFECYCLE_SUCCESSOR_OBJECT_STRUCTURAL`

The owned q8 lifecycle successor is now representable as a first-class metadata object. This is structural only:
no lowering, no runtime route, no default change, and no performance claim.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_lifecycle_successor_object_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_successor_object_result.json
```

## Object Rows

| field | value |
|---|---|
| producer | post-norm q8 activation producer/cache |
| q8 format | `block_q8_1_or_artifact_compatible_q8` |
| reuse | `2`, shared by `ffn_gate` and `ffn_up` |
| consumers | Q4_K packed q4/q8 dot4 gate/up consumers |
| policy | default off, fallback to existing tinygrad decode |
| ownership target | tinygrad-owned successor |
| lowering | metadata only, unwired |

## Structural Gate

The gate passes:

- producer is owned by the successor contract;
- reuse count is `2`;
- both `ffn_gate` and `ffn_up` consumers are named;
- both consumers target Q4_K `4096 -> 12288`;
- fallback and default-off policy are explicit;
- artifact parity targets are present;
- quality target passes against the current q8 artifact evidence;
- no performance claim is made.

## Next

The next local probe is the artifact parity harness:

```text
baseline default decode
q8 artifact hardened opt-in
owned-successor target row
```

Implementation is still blocked on actual owned producer/cache and packed q4/q8 consumer candidates. Search remains
blocked until a lowerable owned candidate exists.
