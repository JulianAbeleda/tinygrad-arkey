# NV full ledger after attention lifecycle audit

Date: 2026-08-24
Commit tested: `e9c3e4edd`
GPU: RTX 5090 (`sm_120`), depth 512

## Verdict

`DEVICE_PATH_STABLE_ENDPOINT_VARIABLE`.

The installed device path remains at the previously established 235-class
compute level, but two new unprofiled endpoint runs do not reproduce one clean
235 tok/s median. They measure 233.0-233.5 tok/s and show late-run latency
steps. No competing GPU process was present and the measured application
clocks remained 2895/14001 MHz.

The correct current statement is therefore:

- the optimized graph and device residence have not regressed;
- about 235 tok/s remains a demonstrated fast endpoint regime;
- the newest full-run endpoint authority is about 233-234 tok/s;
- the difference is variable endpoint overhead, not lost kernel time.

## Unprofiled endpoint authority

| run | median us/token | tok/s | remaining to 240 | gap to retained llama |
| --- | ---: | ---: | ---: | ---: |
| fresh reps 9 confirmation | 4282.038 | 233.534 | 115.372 us | 233.714 us |
| fresh reps 15 | 4291.985 | 232.992 | 125.318 us | 243.661 us |
| preceding clean attention Control A | 4250.508 | 235.266 | 83.841 us | 202.183 us |

The retained llama authority is 4048.325 us/token, or 247.016 tok/s. The
reps-9 confirmation is the newest primary endpoint because the reps-15 run is
visibly bimodal: its first seven windows are 4249.532-4259.465 us/token, then
later windows step into the 4411.536-4427.361 us/token range.

The reps-9 confirmation also ends with one 4434.348 us/token window. Its
median remains in the lower regime, but the tail confirms the same endpoint
variability. These samples are retained rather than silently discarded by an
arbitrary outlier rule.

## Fresh device ledger

The separate profiled authority closes exactly:

```text
node sum  4113.536 us/token
overlap      3.536 us/token
union     4110.000 us/token
```

The profile's own wall field is rejected because instrumentation perturbs
token delivery. Compared with the preceding attention control, device union
moves from 4110.750 to 4110.000 us/token, only -0.750 us. The optimized device
path is stable.

Against retained llama device union of 3888.240 us/token, the current device
gap is 221.760 us/token.

## Current dominant rows

| row | calls | us/token | us/call |
| --- | ---: | ---: | ---: |
| Q4 gate/up | 36 | 1273.856 | 35.385 |
| Q6 down, packed u4 | 18 | 491.360 | 27.298 |
| Q4 down, vector | 18 | 361.088 | 20.060 |
| vocab main | 1 | 313.856 | 313.856 |
| O projection | 36 | 304.864 | 8.468 |
| Q projection, ordinary | 19 | 159.680 | 8.404 |
| Q projection, shared | 17 | 139.904 | 8.230 |
| flash score | 36 | 238.176 | 6.616 |
| flash combine | 36 | 99.840 | 2.773 |
| Q norm/RoPE | 36 | 69.440 | 1.929 |
| K norm/RoPE/cache | 36 | 67.264 | 1.868 |

The row population and times reproduce the post-Q6-unroll ledger. Attention
boundary work produced no promotion, and this rebuild finds no hidden device
regression.

## What the gap means now

From the newest reps-9 endpoint, reaching 240 requires 115.372 us/token.
Only 221.760 us/token separates the device unions, so almost all parity work
still has to come from device-side row improvements. Endpoint variability can
move the observed throughput by roughly two tok/s, but removing variability
alone does not close the llama device gap.

The next clean test remains current Q/O projection cold-rate and compulsory
byte accounting. Those rows contribute 604.448 us/token together and retain
a comparison residual, while their known topology/body rewrites are closed.
Only a measured rate or byte mechanism should reopen them.

## Evidence

Committed summaries are under
`docs/task_workflow/evidence/nv-post-attention-full-ledger-20260824/`:

- `installed-wall-r9-confirm.json`
- `installed-wall-r15.json`
- `device-profile.json`

Raw device profile JSONL remains local and is represented by the hash in the
profile summary.
