# Decode Attention A3.1 v_dot2 Score Result

## Verdict

Raw gate verdict:

- `A3_1_VDOT2_SCORE_INCONCLUSIVE`

Practical interpretation:

- `A3_1_VDOT2_SCORE_NO_MATERIAL_TRANSFER`

The route is clean and the vdot2-named score program is captured, but W==D is flat versus A2 within measurement
spread. Do not promote. Continue to cross-lane/LDS work.

## Artifact

- `bench/qk-decode-attention-a3-1-vdot2-score/latest.json`
- Tool: `extra/qk_decode_attention_a3_1_vdot2_score_gate.py`

## W==D result

| ctx | owned tok/s | A2 tok/s | A3.1 tok/s | A3.1 / A2 | A3.1 / owned | delta vs A2 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 105.0 | 78.2 | 78.4 | 100.3% | 74.7% | +0.2 |
| 1024 | 103.0 | 75.6 | 75.6 | 100.0% | 73.4% | +0.0 |
| 2048 | 100.6 | 69.7 | 69.8 | 100.1% | 69.4% | +0.1 |
| 4096 | 95.6 | 60.6 | 60.5 | 99.8% | 63.3% | -0.1 |

## Route gate

| Check | A3.1 |
|---|---:|
| route clean | yes |
| tokens match | yes |
| owned tile fires | 0 |
| owned combine fires | 0 |
| `E_49152` present | no |
| selected-route buffer identity | yes |
| vdot2-named score program | yes |

Captured score program:

```text
flash_score_whole_cache_vdot2_32_128
```

## Interpretation

A3.1 answers the first performance question:

```text
Can the score path be separately routed and kept lifecycle-clean? yes
Does score vdot2 naming/lowering alone move W==D? no material transfer
```

This means the current gap is not closed by simply enabling the fdot2 lowering hook on the whole-cache score
program. The next likely blockers are:

- cross-lane cooperation
- LDS-staged K/V tile layout
- TILE+COMBINE lifecycle bundling

## Decision

Do not promote A3.1.

Next step:

```text
A3.2 cross-lane reduction / cooperation
```

The owned flatline route is still the oracle shape. A3.1 shows the generated path needs more than scalar score
dot lowering to approach it.
