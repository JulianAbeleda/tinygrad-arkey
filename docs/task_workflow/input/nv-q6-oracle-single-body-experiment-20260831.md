# NV Q6 single physical segment-body experiment

## Decision

`INCONCLUSIVE_GEOMETRY`; `NO_PROMOTION`.

The single physical body is correct and substantially reduces static code and
spill instructions, but the executable screening form changes the grid from
170 CTAs to `170x2` CTAs. Its timing regression therefore cannot falsify the
single-body mechanism. It also does not justify the source-level 170-CTA
`__noinline__` follow-up under the agreed test policy because it wins zero of
31 paired rounds.

## Exact candidate

The isolated experiment uses `blockIdx.y` as a uniform runtime segment
selector. Both segment values execute one physical K256 body; output slot,
tile, epoch start, epoch depth, Q6 base, and Q8 base are runtime-selected.
Inactive second segments are uniformly gated.

Against the authoritative duplicated-body baseline in the same process:

- Active partial slots match bit-for-bit. Inactive slots are NaN in both arms
  and are compared with `equal_nan=True`.
- Deterministic fixed-up output matches bit-for-bit.
- Candidate GPU fixup matches CPU slot-ordered reduction bit-for-bit.
- The existing direct-wide reference difference is unchanged: maximum
  absolute `0.1871337890625`, mean absolute `0.01368770468980074`.

## Alternating locked R31 screening

All GPU correctness and timing execution was serialized with:

```text
flock /tmp/nv-q6-oracle-gpu.lock -c '<qualifier>'
```

Call order reversed every round.

| measurement | duplicated 170-CTA baseline | single-body 170x2 candidate | delta |
| --- | ---: | ---: | ---: |
| main R31 median | 284.320 us | 413.696 us | +129.376 us |
| paired recovery median | - | - | -131.424 us |
| candidate wins | - | 0 / 31 | - |
| candidate pair median | - | 446.048 us | - |
| ratio to llama main | 1.413x | 2.056x | - |
| ratio to llama pair | - | 2.125x | - |

The timing result is a geometry screening failure, not a causal rejection:
the candidate launches 340 CTAs and loses the baseline's per-owner sequential
segment balance.

## SASS and resources

| static measurement | duplicated baseline | single body | delta |
| --- | ---: | ---: | ---: |
| instructions | 8,328 | 3,992 | -4,336 |
| registers | 255 | 255 | 0 |
| stack | 288 B | 264 B | -24 B |
| LDL | 251 | 129 | -122 |
| STL | 377 | 194 | -183 |
| IMMA | 512 | 256 | -256 |
| LDSM | 64 | 32 | -32 |
| BAR | 11 | 5 | -6 |

This proves static body duplication is removable and explains a real resource
reduction. It does not prove a runtime win at preserved geometry.

## Missing compiler substrate

The geometry-preserving expression is an outer two-segment runtime RANGE
around the existing dynamic K256 RANGE, with the accumulator bank reset and
reused between segments. The isolated UOp implementation reaches tinygrad
range simplification and fails before rendering:

```text
KeyError: UOp(Ops.END, ...)
tinygrad/codegen/simplify.py: simplify_merge_adjacent
```

The failure occurs while flattening the nested segment/epoch ranges containing
register accumulator lifecycle dependencies. This is the concrete missing
substrate for expressing one 170-CTA physical body directly in UOps.

## Next investment condition

Build or fix nested RANGE lifecycle lowering, then rerun the same alternating
R31 A/B with both arms at exactly 170 CTAs. Promotion requires bit-exact
baseline equivalence, a positive paired median, and at least 24/31 wins.

## Evidence

- `docs/task_workflow/evidence/nv-q6-oracle-single-body-20260831/qualification.json`
- `extra/llm_research/prefill/nv_q6_oracle_single_body_experiment.py`
- `extra/llm_research/prefill/bench_nv_q6_oracle_single_body_experiment.py`
