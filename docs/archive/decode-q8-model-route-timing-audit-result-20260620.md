# Decode q8 Model Route Timing Audit Result

Date: 2026-06-20

## Verdict

`PASS_DECODE_Q8_MODEL_ROUTE_TIMING_AUDIT`

Command:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_model_route_timing_audit.py \
  --lanes auto,manual_peak \
  --modes baseline,q8 \
  --ckpts 512 1024 \
  --nmeas 20 \
  --warmups 8
```

One `manual_peak` baseline child hit an AMD allocation failure on the first aggregate run and was rerun successfully as
a single child. The final artifact was regenerated from the four completed child artifacts:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_model_route_timing_audit.py \
  --aggregate-existing \
  --lanes auto,manual_peak \
  --modes baseline,q8 \
  --ckpts 512 1024 \
  --nmeas 20 \
  --warmups 8
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_q8_model_route_timing_audit_result.json
```

## Result

| lane | ctxs | median q8 speedup | min q8 speedup | q8 median tok/s | q8 host-sync |
|---|---|---:|---:|---:|---:|
| auto | `512,1024` | `1.0584x` | `1.0554x` | `71.55` | `0.0%` |
| manual_peak | `512,1024` | `1.0609x` | `1.0601x` | `71.45` | `0.0%` |

Per-context rows:

| lane | ctx | baseline tok/s | q8 tok/s | speedup | q8 dispatch ceiling | q8 host-sync |
|---|---:|---:|---:|---:|---:|---:|
| auto | `512` | `68.4` | `72.6` | `1.0614x` | `68.7` | `0.0%` |
| auto | `1024` | `66.8` | `70.5` | `1.0554x` | `67.1` | `0.0%` |
| manual_peak | `512` | `68.1` | `72.2` | `1.0602x` | `68.4` | `0.0%` |
| manual_peak | `1024` | `66.6` | `70.6` | `1.0601x` | `67.2` | `0.0%` |

## Interpretation

The actual in-model graph route behaves like the prior W/D promotion artifacts:

- q8 route speedup survives in the model graph;
- `W` and `D` are effectively equal;
- per-token host sync is not the lever;
- `manual_peak` does not expose a hidden graph-route win over auto for whole-model decode at these contexts.

This is an important distinction from the isolated q8 FFN lifecycle microbench. The microbench is sensitive to
producer/consumer dispatch bands. The whole model route is dominated by the broader decode graph, where q8 contributes a
stable ~5-6% W/D speedup and host-sync residual is already `0%`.

## Decision

Do **not** start fused producer+consumer work to solve host waits. The in-model graph route already removes that as a
meaningful target.

If the goal is more whole-model decode speed, the remaining options are:

1. accept the current q8 route as a default-off opt-in with ~5-6% whole-decode speedup;
2. broaden the q8/native MMVQ primitive coverage beyond gate/up;
3. pursue larger route/runtime projects such as speculative decode or persistent decode.

True fused q8 producer+consumer is not justified by the current in-model timing evidence.

## Boundary

No decode default changed.
