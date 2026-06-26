# Decode Attention A3.2b X-Lane Score Result

## Verdict

`A3_2B_CROSS_LANE_NO_TRANSFER`

The scoped x-lane generated score path is now wired and captured as `flash_score_whole_cache_xlane_32_128`, but it is not promotable.

What passed:

- The generated whole-cache route remains clean.
- The owned attention tile/combine does not fire in the A3.2b arm.
- `E_49152` materialization stays absent.
- The x-lane score program is present in capture.
- Token correctness is preserved by the existing decode search gate.

What failed:

- The naive one-score-per-wave lane-partition score kernel collapses throughput.
- A3.2b is far slower than both the A2 generated skeleton and the owned attention route.
- This proves that merely exposing scoped lane ownership plus cross-lane sum is not enough; decode attention needs a better lifecycle primitive, likely LDS/tile-level staging rather than scalar score-per-token x-lane reduction.

## Command

```bash
PYTHONPATH=. .venv/bin/python -m py_compile extra/qk_flash_decode.py extra/qk_decode_attention_a3_2b_xlane_score_gate.py && \
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_2b_xlane_score_gate.py
```

## Artifact

- `bench/qk-decode-attention-a3-2b-xlane-score/latest.json`

## W==D Result

| ctx | owned tok/s | A2 generated tok/s | A3.2b x-lane tok/s | A3.2b vs A2 | A3.2b vs owned |
|---:|---:|---:|---:|---:|---:|
| 512 | 105.3 | 78.3 | 6.0 | 7.7% | 5.7% |
| 1024 | 103.3 | 75.5 | 3.2 | 4.2% | 3.1% |
| 2048 | 100.8 | 69.7 | 1.5 | 2.2% | 1.5% |
| 4096 | 95.8 | 60.5 | 0.7 | 1.2% | 0.7% |

## Interpretation

A3.2b answers the narrow question: can attention use the existing lane-partition/cross-lane building block in a scoped generated score program?

Answer: yes, mechanically, but no as a performance path.

The generated score program is present, but it serializes the score lifecycle into a shape that is much worse than A2. This rules out promoting this direct x-lane score form. The next decode attention purity step should not keep tuning this specific kernel. It should move to the next missing lifecycle primitive: generated LDS/tile staging and an attention tile representation that preserves the owned route's parallelism while remaining generated/search-owned.

## Decision

Promote nothing from A3.2b.

Next action: scope and implement an A3.3 LDS/tile-lifecycle candidate or blocker gate, using A2 as the clean generated skeleton and owned attention as the flatline oracle.
