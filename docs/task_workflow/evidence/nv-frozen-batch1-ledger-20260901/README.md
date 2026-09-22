# Frozen batch-1 decode ledger, 2026-09-01

## Decision

The current tinygrad batch-1 route is healthy after the runtime-input WAR
repair in `5c7f9355a`. At locked 2850/14001 clocks, nine unprofiled windows
measure **4071.481 us/token, 245.611 tok/s**. All nine windows were accepted
and have the historical token-stream hash
`7e0b38ec59adc79a268f665ee525b602869c1930794239eb45ba21e9e0a22e3e`.

This does not establish an unconditional llama win. The retained strict
batch-1 greedy llama authority is 4058.359 us/token, 246.405 tok/s, placing
tinygrad 13.122 us/token or 0.794 tok/s behind. Other retained llama samples
move much more than this gap: PDL-off settles near 243 tok/s, while PDL-on
moves from 240.615 tok/s on its first sample to 250.662 tok/s hot. A new
paired, frozen, same-session llama run is still required for a current winner.

## Measurement domains

Do not add values across these domains.

| domain | result | authority |
|---|---:|---|
| unprofiled frozen endpoint, R9 x 24 | 4071.481 us/token, 245.611 tok/s | throughput |
| pre-regression frozen endpoint, R9 x 24 | 4086.354 us/token, 244.717 tok/s | historical checkpoint |
| marker-light frozen wall, R24 | 4117.523 us/token | accounting only |
| marker-light device window, R24 | 3860.752 us/token | four graph groups plus marker cost |
| marker wall minus device window | 255.028 us/token | disjoint outside-window remainder |
| PROFILE node sum, 74 steady replays | 3886.592 us/token | per-kernel accounting only |
| PROFILE device union | 3880.625 us/token | per-kernel accounting only |
| PROFILE overlap | 5.967 us/token | per-kernel accounting only |

The marker's `pre_first_graph_us` and `post_last_graph_us` are enqueue-side
timestamps. `post_last_graph_us` includes the device drain and is not a
disjoint host component. Only `wall_us - device_window_us` is the disjoint
outside-window row.

## Fresh exact-policy kernel census

The selected endpoint produces four groups with the stable signature
`32/64/128/194`. The capture contains 77 complete replays; the first three are
discarded and the following 74 form the table. PROFILE instrumentation has a
large relative cost on very short kernels, so this table ranks current work
but is not substituted into the common-protocol llama comparison.

| current kernel family | calls/token | PROFILE us/token |
|---|---:|---:|
| Q4_K gate/up four-warp | 36 | 1269.248 |
| Q6_K packed FFN-down | 18 | 483.392 |
| Q4_K direct FFN-down | 18 | 349.952 |
| Q6_K vocabulary | 1 | 312.496 |
| attention O with residual | 36 | 296.976 |
| native RMSNorm | 55 | 222.368 |
| Flash score/PV | 36 | 209.216 |
| Q projection, G3 half | 18 | 144.880 |
| Q projection, cooperative Q8 | 18 | 143.728 |
| Q/K norm plus RoPE | 36 | 70.752 |
| K norm/RoPE plus KV-cache sink | 36 | 69.312 |
| K/V pair, Q4/Q4 cooperative | 10 | 48.784 |
| K/V pair, Q4/Q6 cooperative | 8 | 46.672 |
| K/V pair, Q4/Q4 G3 | 8 | 44.816 |
| Q6_K V projection | 10 | 42.592 |
| Q4_K V projection | 10 | 40.656 |
| Flash combine | 36 | 36.128 |
| shared Q8 RMS provider | 18 | 27.456 |
| native argmax | 1 | 8.416 |
| remaining elementwise/reduce kernels | 9 | 18.112 |
| **node sum** | **418** | **3886.592** |

The sum of independently selected per-family medians is 0.640 us below the
median per-replay node sum; medians are not additive.

The large-region check agrees with the prior ledger: gate/up is 1269.248 us
versus 1264.896 us historically, and combined FFN-down is 833.344 us versus
839.808 us historically. The route has not lost the dominant projection work.

## Common-protocol tinygrad versus llama ledger

This remains the only valid role-by-role cross-runtime comparison. It uses
the same cold CUDA/CUPTI active-duration definition for both runtimes.

| lifecycle region | tinygrad active | llama active | TG - llama |
|---|---:|---:|---:|
| gate/up | 1264.896 us | 1291.116 us | -26.220 us |
| down | 839.808 us | 880.972 us | -41.164 us |
| O | 288.000 us | 284.993 us | +3.007 us |
| Q + K + V + shared provider | 519.936 us | 536.321 us | -16.385 us |
| native norms | 110.880 us | 203.778 us | -92.898 us |
| Flash score, 96-MiB disturbed | 153.216 us | 157.824 us | -4.608 us |
| vocabulary, including llama quant | 317.402 us | 301.602 us | +15.800 us |
| **selected total** | **3494.138 us** | **3656.606 us** | **-162.468 us** |

Tinygrad is ahead in aggregate active work. The only measured kernel-body
losses are vocabulary, 15.800 us/token, and O, 3.007 us/token. Those total
18.807 us/token, while current tinygrad trails the retained strict llama
endpoint by only 13.122 us/token. They are valid small levers, but the sign
inversion between active work and endpoint service proves that lifecycle and
boundary behavior still determines the observed winner.

## What changed and what did not

- `4d117c8e0` made every repeated opaque writer require a pre-existing scratch
  epoch chain. Decode feedback intentionally writes a runtime-input assignment
  and later an argmax return to one physical scalar, so current HEAD failed
  before timing.
- `5c7f9355a` delegates only explicitly typed `RUNTIME_INPUT` reuse to the
  ordinary WAR repair. Untyped and candidate workspace still fail closed.
- The marker and Phase-0 tools previously forced direct-greedy and feedback
  policies after model construction. The recorded route does not do that.
  The tools now observe the selected production policy unchanged.
- No model kernel, route lease, or numerical policy changed in this ledger.

## Next measurement

Run llama and tinygrad in a frozen, alternating, same-session bracket with the
same greedy feedback and per-token delivery contract. Until that exists, the
current factual statement is **tinygrad 245.611 tok/s; retained strict llama
246.405 tok/s; result within 0.8 tok/s and smaller than llama thermal-state
movement**.

## Files

- `tinygrad-endpoint-r9.json`: unprofiled throughput authority.
- `tinygrad-marker-r24.json`: wall/device boundary partition.
- `tinygrad-profile-child.json`: exact route census and PROFILE wall samples.
- `tinygrad-profile-ledger.json`: 74-replay per-name accounting.
- `tinygrad-profile.jsonl.gz`: raw four-group HCQ profile.
- `*-gpu-before.txt`, `*-gpu-after.txt`: clock records.

Historical comparison sources:

- `docs/task_workflow/output/nv-active-body-ledger-phase2-result.md`
- `docs/task_workflow/evidence/nv-host-visible-token-delivery/batch1-greedy-vs-llama.json`
- `docs/task_workflow/evidence/nv-lifecycle-recovery-tests-20260826/llama-pdl-ab/pdl-off.json`
- `docs/task_workflow/evidence/nv-lifecycle-recovery-tests-20260826/llama-pdl-ab/pdl-on.json`
