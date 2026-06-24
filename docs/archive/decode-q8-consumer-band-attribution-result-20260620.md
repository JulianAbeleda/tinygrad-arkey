# Decode q8 Consumer Band Attribution Result

Date: 2026-06-20

## Verdict

`BLOCKED_DECODE_Q8_CONSUMER_BAND_BIMODAL_SESSION_STATE`

Command:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_consumer_band_attribution.py --sessions 5
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_q8_consumer_band_attribution_result.json
```

## Result

Correctness passed in every child session. Static metadata for the hipcc/LLD `q8_mmvq_gateup` consumer is clean:

| field | value |
|---|---:|
| dot4 instructions | `16` |
| global load groups | `11` |
| shuffle groups | `5` |
| LDS bytes | `16` |
| private bytes | `0` |
| kernarg bytes | `40` |

Protocol reconciliation:

| protocol | median of session medians | best session | worst session | fast sessions | slow sessions | band |
|---|---:|---:|---:|---:|---:|---|
| first-N | `89.52us` | `47.96us` | `101.60us` | `4` | `1` | fast/bimodal |
| repeat same dispatch | `89.46us` | `47.96us` | `90.80us` | `5` | `0` | fast |
| after dummy | `89.14us` | `47.92us` | `90.74us` | `5` | `0` | fast |
| mixed order | `89.58us` | `47.92us` | `90.76us` | `5` | `0` | fast |

## Interpretation

This pass overturns the narrower consumer-only conclusion from
`decode-q8-consumer-only-reconciliation-result-20260620.md`.

The fused gate/up consumer does **not** reproduce a stable `~101us` slow band once first-N, repeat, after-dummy, and
mixed-order rows are separated. The steady consumer protocols reconcile at `~89-90us`, which is within the fast band.
One first-N session hit `101.60us`, so there is still session/startup sensitivity, but the repeated consumer body itself
is not the current stable blocker.

Static issue analysis also does not show a simple resource red flag: the artifact has the expected dot4 body, no private
spill, and only a 16-byte LDS reduction slot. PMC/SQTT counter decode remains unavailable in the existing artifacts, so
there is no counter-grade issue rate yet.

## Decision

Do **not** start a blind fused-consumer rewrite from the previous `~101us` consumer-only result.

The next decode reconciliation should return to the full q8 lifecycle with the `NT=1024` producer and split first-N vs
steady rows under the same protocol used here. If lifecycle remains `~122us` while the steady consumer is `~89-90us`,
the missing time is lifecycle composition/launch/session policy rather than the consumer kernel body.

## Boundary

No decode default changed. The q8 route remains opt-in/research-only until a repeated lifecycle policy clears the
`115.24us` target or the target/session policy is explicitly changed.
