# Prefill Actuals-Derived Gameplan

Date: 2026-07-10.

This plan is derived from existing measured artifacts, not a new theory branch.

## Authority Actuals

### Whole-Prefill

| Route | Provenance | pp512 tok/s | pp4096 tok/s | Notes |
|---|---|---:|---:|---|
| S9 hand/LDS2 path | `external_handwritten_kernel` | `4406.86` single, report median `4396.67` | report median `3228.92` | Fast authority, not pure. |
| Pure scheduler baseline | `tinygrad_scheduler_generated` | `1629.74` | `1420.14` | Pure generated baseline. |
| S10 composed generated primitive | `compiler_primitive_spec_owned` | `1493.40` | `1325.35` | Correct, but slower than pure baseline. |

Immediate readout:

```text
S10 composed is -8.37% vs pure baseline at pp512 and about 2.95x slower than S9.
Therefore S10 is not a promotion candidate. Treat it as a research path only.
```

### S9 Fast Micro Shape

The strongest S9 micro evidence is the `2x4` LDS2 shape from `bench/prefill-lds2-s9`:

| Metric | S9 `2x4` LDS2 |
|---|---:|
| TFLOPS | `73-75` pinned, best observed `75.09` |
| tile | `128x128`, `256` threads |
| WMMA/block | `64` |
| waits/WMMA | `0.281` |
| inst/WMMA | `9.047` |
| global/load + LDS store/WMMA | `0.25 / 0.25` in S9 search artifact |
| ds_load/WMMA | `1.5` |
| LDS bytes | `40960` |
| scratch end VGPR | `140` |

S9 roofline audit says the route is compute-bound:

```text
compute roof: 122.8 TFLOPS
S9 actual:    about 75 TFLOPS
headroom:     about 1.6x, but already in the right performance class
```

### Generated DBUF / K-Major Actuals

Existing generated rows for the same prefill class:

| Route | Shape | Correct | TFLOPS | WMMA | waits/WMMA | inst/WMMA | global/store/load per WMMA | D3 |
|---|---|---|---:|---:|---:|---:|---|---|
| generated DBUF baseline | `2x2` | yes | `7.89` | `16` | `3.312` | `39.062` | `2.0 / 2.0 / 4.0` | false |
| generated K-major | `2x2` | yes | `11-12.5` band | `16` | `2.875` | `34.625` | `2.0 / 2.0 / 2.0` | false |
| generated K-major | `4x2` | yes in recent sweep | `9.05` | `32` | `2.188` | `29.188` | `1.5 / 1.5 / 1.5` | false |
| generated K-major | `2x4` | yes in recent sweep | `8.24` | `32` | `2.188` | `29.188` | `1.5 / 1.5 / 1.5` | false |
| K-major + D3 marker | `2x2` | yes | `~10.3` | `16` | `4.562` | high | `3.125 / 3.125 / 2.0` | true |

Immediate readout:

```text
Generated K-major already solved part of the LDS-load density problem.
It did not solve the lifecycle-window problem:
  S9 has 64 WMMA per useful window.
  Generated has 16-32 WMMA per window.
  S9 waits/WMMA is 0.281-0.406.
  Generated waits/WMMA is 1.78-3.31.
  S9 inst/WMMA is about 9-11.
  Generated inst/WMMA is about 27-39.
```

## What The Numbers Say

### 1. S10 Is Not Blocked By A Tiny Wait Tweak

The needle test showed:

```text
K-major base waits:                  46
clustered LGKM wait waits:           41
unsafe skip wait:                    wrong output, rr=1.1e+00
```

Five fewer waits did not create a decisive timing win. The unsafe shortcut is invalid. So the primitive is not "delete
waits." Wait reduction must come from a legal lifecycle that emits more WMMA work between waits.

### 2. The Main Deficit Is Window Density

The S9 hand route amortizes fixed lifecycle work over a larger compute window:

```text
S9 hand LDS2:      64 WMMA, 18-26 waits depending artifact/window
S10 generated:     16-32 WMMA, 46-70 waits in bounded sweeps
```

This explains why instruction trimming alone is not the first lever. Generated has a similar absolute non-matrix
instruction count in some windows, but too little math is packed behind it.

### 3. The Current "Generated LDS" Path Is Worse Than The Pure Baseline

Whole-prefill actuals:

```text
pure baseline pp512:   1629.74 tok/s
S10 composed pp512:    1493.40 tok/s
delta:                 -8.37%
```

So a generated LDS/DBUF route should not be promoted unless it first beats the pure baseline and then moves toward S9.

## Derived Gameplan

### Track A: Preserve The Fast Authority

Keep S9 hand/LDS2 as the production authority until a candidate clears the numeric gates below.

Promotion gate:

```text
candidate pp512 >= 0.98 * S9 median = 4308 tok/s
candidate pp4096 >= 0.98 * S9 median = 3164 tok/s
```

This prevents replacing a 4.4k route with a 1.5k route because it is cleaner architecturally.

### Track B: Convert Ownership By Slices That Do Not Move Performance

Use S9 as executable oracle, but convert only low-risk ownership layers first:

| Slice | Convert Now? | Gate |
|---|---|---|
| route/spec selection | yes | route attribution changes, perf unchanged |
| register layout constants | yes | S9 authority remains within noise |
| wait policy constants | yes | S9 authority remains within noise |
| lifecycle template data | yes | S9 authority remains within noise |
| full instruction emitter / epoch scheduler | no | only after generated lifecycle reaches S9 counters |

This is the compromise path: reduce handwritten ownership without pretending the hard epoch scheduler is solved.

### Track C: Make Generated Research Earn Its Way Back

Generated S10 must clear these gates before whole-prefill:

| Gate | Required Actual |
|---|---|
| bounded correctness | finite, correct output |
| WMMA density | at least `64` WMMA per measured lifecycle window, or a proven equivalent |
| wait density | `waits/WMMA <= 0.6` first, target `<= 0.3` |
| instruction density | `inst/WMMA <= 14` first, target `<= 10-11` |
| memory density | no worse than generated K-major `4x2/2x4`, target S9-like `<= 1.5 ds_load/WMMA` |
| bounded TFLOPS | first beat pure schedule table `~32.6 TFLOPS`, then target S9 `~75 TFLOPS` |
| whole-prefill | first beat pure baseline `1629.74 tok/s`, then target S9 band |

If a generated candidate cannot beat the pure scheduler baseline, it should not be part of the S10 whole route.

### Track D: Next Measurement To Run

The next useful actual is not another 2x2 wait probe. It is a role-level attribution run that answers:

```text
How much of the 4406 -> 1493 tok/s loss is ffn_gate_up LDS,
and how much is pipe-role replacement / route policy?
```

Run matrix:

| Variant | Purpose |
|---|---|
| pure baseline | freeze current pure floor |
| S9 fast authority | freeze current target |
| S9 authority with only route/spec ownership slices converted | prove ownership conversion is free |
| pure baseline plus only one role replaced | identify role-level regressions |
| ffn_gate_up generated LDS only | measure whether LDS replacement alone is the loss |
| pipe roles generated, ffn_gate_up S9 oracle | measure whether pipe role path is safe |

Do not proceed to deeper DBUF codegen until the role-level loss is attributed.

## Current Decision

```text
Use S9 as the performance-preserving authority.
Use actuals gates to convert ownership slices around it.
Park generated S10 promotion until it beats the pure baseline on bounded and whole-prefill measurements.
```

The next engineering task is an actuals harness/report, not a new lowering theory:

```text
build/update one command that emits:
  whole tok/s for S9, pure baseline, S10 composed,
  per-role route attribution,
  per-role timing where available,
  structural counters for generated and S9 oracle,
  pass/fail against the gates above.
```

Implemented reusable command:

```bash
PYTHONPATH=. python3 extra/qk/prefill/baseline_audit_bundle.py --json
```

The bundle now includes:

```text
s9_authority
comparison
promotion_gates
route_census
shape_matrix
schedule_gate
```

Current generated decision from the bundle:

```text
promotion_gates.decision = keep_s9_authority_and_treat_candidate_as_research
```

## Refreshed S10 Baseline - 2026-07-10

Command:

```bash
PYTHONPATH=. DEV=AMD \
PREFILL_GRAPH_GEMM=1 \
PREFILL_WMMA_PIPE_PRIMITIVE=1 \
PREFILL_WMMA_LDS_PRIMITIVE=1 \
PREFILL_DBUF=1 \
PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE=1 \
python3 extra/qk/prefill_whole_synced.py \
  --mode authority \
  --require-route prefill_wmma_pipe_lds_dbuf_primitive_generated \
  --artifact bench/prefill-whole-synced/s10-composed-current-authority.json \
  --json \
  --pin-clock
```

Route binding passed:

```text
selected_route = prefill_wmma_pipe_lds_dbuf_primitive_generated
verdict        = PREFILL_ROUTE_BINDING_PASS
role routes    = attn_qo:pipe, attn_kv:generated_pipe_no_local_stage, ffn_down:pipe, ffn_gate_up:lds_dbuf
```

New authority row:

| Length | Pure baseline tok/s | S10 current tok/s | Delta |
|---:|---:|---:|---:|
| 512 | `1629.74` | `1332.22` | `-18.26%` |
| 1024 | `1587.88` | `1318.20` | `-16.98%` |
| 2048 | `1518.44` | `1271.21` | `-16.28%` |
| 4096 | `1420.14` | `1188.86` | `-16.29%` |

Against S9 authority:

| Gate | Actual | Target | Pass |
|---|---:|---:|---|
| pp512 vs `0.98 * S9` | `1332.22` | `4308.74` | no |
| pp4096 vs `0.98 * S9` | `1188.86` | `3164.34` | no |
| pp512 vs pure baseline | `1332.22` | `1629.74` | no |
| pp4096 vs pure baseline | `1188.86` | `1420.14` | no |

Actuals bundle artifact:

```bash
PYTHONPATH=. python3 extra/qk/prefill/baseline_audit_bundle.py \
  --candidate bench/prefill-whole-synced/s10-composed-current-authority.json \
  --output bench/prefill-baseline-audit/s10-current-actuals.json \
  --json
```

Current decision remains:

```text
keep_s9_authority_and_treat_candidate_as_research
```
