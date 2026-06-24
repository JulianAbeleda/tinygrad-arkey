# Decode Native Renderer DNR-3C7C Issue Interleaving Result - 2026-06-20

## Verdict

`BLOCKED_DNR3C7C_ISSUE_INTERLEAVING_PARTIAL_SIGNAL_NOT_PROMOTED`

DNR-3C7C built real issue-order variants for the q8 decode native renderer. The best candidate is correct and
directionally faster, but it does not clear the material promotion gate and remains far from the hipcc/LLD oracle.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c7c_issue_interleaving_probe.py --warmups 4 --iters 12
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7c_issue_interleaving_result.json
```

## Candidates

| variant | model |
|---|---|
| `predicate_hoist` | Hoist the q4 odd/even sub-lane predicate out of the eight-qword dot body. |
| `unpack_all_then_dot` | Do all q4 nibble selects first, then issue the dot4 accumulator stream. |
| `unpack_all_then_dot_dsload_b128` | Combine `unpack_all_then_dot` with the prior vector LDS cross-wave read. |

One bad splice was found during bring-up: an earlier `unpack_all_then_dot_dsload_b128` replacement used the wrong
instruction window and hung the AMD queue. The final probe fixes the splice by replacing the scalar LDS load window
only, then retests it as a normal candidate.

## Timing

Same-process interleaved timing, final confirmation run:

| variant | correct | median us | delta vs native | delta vs best static |
|---|---:|---:|---:|---:|
| native DNR-2 | yes | `355.975` | `0.000` | `+16.316` |
| best static DNR-3C6 | yes | `339.659` | `-16.316` | `0.000` |
| predicate hoist | yes | `336.688` | `-19.286` | `-2.970` |
| unpack all then dot | yes | `330.282` | `-25.693` | `-9.377` |
| unpack all then dot + dsload_b128 | yes | `327.125` | `-28.850` | `-12.534` |

Oracle reference from the q8 artifact contract:

| row | us |
|---|---:|
| hipcc/LLD oracle | `93.540` |
| best C7C candidate | `327.125` |
| remaining gap | `233.585` |

## Gates

| gate | result |
|---|---:|
| DNR-3C7B PMC ladder passed | yes |
| all variants correct | yes |
| issue order changed | yes |
| best schedule improves native by at least 30us | no (`28.850us`) |
| best schedule improves best static by at least 15us | no (`12.534us`) |
| best schedule reaches <=110% oracle | no |
| renderer default changed | no |

## Interpretation

This is the first decode native result that supports the issue/interleaving hypothesis: changing the dot-body issue
shape beats both native and the best static count rewrite in the final run.

It is still not promotion evidence:

- the best candidate misses both material gates;
- the absolute timing in this probe is slower than earlier C6 timing, so use the deltas directionally;
- the result is still more than `3x` slower than the oracle;
- no renderer defaults changed, and no native route should be promoted from this alone.

## Next

The only justified continuation is confirmation, not another count-matching loop:

1. run the C7B PMC ladder on `unpack_all_then_dot_dsload_b128`;
2. repeat timing under the prior C6 timing harness so the absolute scale is comparable;
3. promote only if the candidate clears the >=30us native gate or moves the same PMC wait/busy family materially.

If that confirmation fails, park the DNR-3C native renderer route until oracle resource metadata or SQTT body
timeline tooling is available.

No renderer defaults changed.
