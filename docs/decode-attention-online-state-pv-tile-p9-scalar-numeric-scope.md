# Decode Attention Online-State+PV Tile P9 Scalar Numeric Scope

## Goal

Prove or localize the scalar generated online-state tile numerics before any more x-lane or W==D work.

P8 showed both scalar and x-lane isolated paths can produce NaNs/numeric mismatch. Therefore the next question is not performance and not cross-lane. The next question is:

```text
Does flash_online_state_pv_tile_whole_cache_32_128 compute the same per-split m/l/PV state as a NumPy reference?
```

## Current blocker

Latest failed artifact:

- `bench/qk-decode-attention-online-state-pv-p8-numeric/latest.json`
- Verdict: `ONLINE_STATE_PV_P8_FAIL__NAN`

P8 compared scalar P5 vs x-lane P7 and found:

| Quantity | Observation |
|---|---|
| scalar active outputs | contain NaNs |
| x-lane active outputs | contain NaNs |
| `m/l/PV/out` | large mismatch |

This means x-lane debugging is premature. First prove scalar state recurrence.

## Scope boundary

P9 is isolated numeric proof only.

In scope:

- deterministic `q` and whole-cache `cache` tensors;
- generated score kernel output;
- scalar online-state tile output;
- state gmax/combine output;
- NumPy reference for score, per-split `m`, per-split `l`, per-split PV, and final output;
- active split/tail handling;
- NaN/Inf detection;
- first-failing-stage classification.

Out of scope:

- x-lane tile;
- W==D;
- ISA/resource performance;
- BubbleBeam promotion;
- changing default runtime behavior.

## Generated programs under test

```text
flash_score_whole_cache_32_128
flash_online_state_pv_tile_whole_cache_32_128
flash_state_gmax_32_128
flash_state_combine_32_128
```

## Reference math

For each query head `h`, split `s`, and token range:

```text
t0 = s * L
t1 = min(t0 + L, Tc)
kv = h // G
score[h,t] = dot(q[h], K[kv,t]) / sqrt(Hd)
m[h,s] = max_t score[h,t]
l[h,s] = sum_t exp(score[h,t] - m[h,s])
PV[h,s,d] = sum_t exp(score[h,t] - m[h,s]) * V[kv,t,d]
```

Final combine:

```text
gm[h] = max_s m[h,s]
den[h] = sum_s exp(m[h,s] - gm[h]) * l[h,s]
out[h,d] = sum_s exp(m[h,s] - gm[h]) * PV[h,s,d] / den[h]
```

## Required test cases

P9 must include at least these deterministic cases:

| Case | Purpose |
|---|---|
| exact split: `Tc=128,L=64` | no tail masking; simplest scalar recurrence proof |
| tail split: `Tc=130,L=64` | validates invalid-token handling |
| one split: `Tc=32,L=64` | validates partial split only |
| multi split: `Tc=256,L=64` | validates combine over several splits |

The harness may start with smaller `Hq/Hkv/Hd` only if the exact production shape also runs before pass. Current target remains:

```text
Hq=32,Hkv=8,Hd=128,MAXC=512,L=64
```

## Tolerances

| Quantity | Tolerance |
|---|---:|
| score | `2e-3` |
| per-split `m` | `2e-3` |
| per-split `l` | `2e-3` |
| per-split PV | `5e-3` |
| final output | `5e-3` |

Because generated kernels use fp32 accumulation from fp16 inputs and may reassociate floating-point operations, tolerance is not byte-exact. But NaN/Inf is always failure.

## Failure labels

```text
ONLINE_STATE_PV_P9_NUMERIC_PASS
ONLINE_STATE_PV_P9_FAIL__CAPTURE
ONLINE_STATE_PV_P9_FAIL__SCORE_NAN
ONLINE_STATE_PV_P9_FAIL__SCORE
ONLINE_STATE_PV_P9_FAIL__STATE_NAN
ONLINE_STATE_PV_P9_FAIL__M
ONLINE_STATE_PV_P9_FAIL__L
ONLINE_STATE_PV_P9_FAIL__PV
ONLINE_STATE_PV_P9_FAIL__OUT_NAN
ONLINE_STATE_PV_P9_FAIL__OUT
```

## Failure interpretation

| Failure | Meaning | Next action |
|---|---|---|
| score fails | score kernel/reference mismatch | fix score/reference or score buffer shape first |
| `m` fails | max recurrence/state column wrong | debug tile `m` update/store |
| `l` fails | denominator recurrence wrong | debug correction factor and invalid-token handling |
| PV fails | accumulator recurrence wrong | debug `acc*corr + p*V` and state column layout |
| only output fails | state columns are correct; combine is wrong | debug `flash_state_combine` |
| NaN anywhere | invalid state or unwritten active output | debug masking/store/cache shape before x-lane |

## Pass criteria

P9 passes only if all required cases pass score, `m`, `l`, PV, and final output tolerance with no NaN/Inf in active outputs.

## Expected artifact

```text
bench/qk-decode-attention-online-state-pv-p9-scalar-numeric/latest.json
```

The artifact must include:

- verdict;
- per-case errors;
- first failing case;
- first failing stage;
- shape metadata;
- generated program names if captured;
- next decision.

## Decision after P9

If P9 passes:

```text
Return to P7/P8 x-lane numeric debugging.
```

If P9 fails:

```text
Fix scalar online-state tile recurrence first. Do not touch x-lane, W==D, v_dot2, or promotion.
```
