# Common-protocol active-body ledger: projections

## Decision

Tinygrad does not need to make its dense projection kernels match llama's
CUPTI-active execution. On the rows now measured under one cold CUDA/CUPTI
protocol, tinygrad is already faster in aggregate.

The old ledger mixed tinygrad HCQ/QMD timestamp intervals with llama CUPTI
kernel-active durations. That boundary made Flash, norms, Q, O, and K/V look
like kernel-body debts. Exact installed cubins launched through the CUDA driver
show that most of those debts do not exist inside the kernels.

| lifecycle region | tinygrad cold active | charged llama active | TG - llama |
|---|---:|---:|---:|
| gate/up | 1264.896 us | 1291.116 us | **-26.220 us** |
| down | 839.808 us | 880.972 us | **-41.164 us** |
| O | 288.000 us | 284.993 us | **+3.007 us** |
| Q + K + V + shared provider | 519.936 us | 536.321 us | **-16.385 us** |
| native norms | 110.880 us | 203.778 us | **-92.898 us** |
| Flash score, 96-MiB disturbed | 153.216 us | 157.824 us | **-4.608 us** |
| vocabulary, including llama quant | 317.402 us | 301.602 us | **+15.800 us** |
| **selected total** | **3494.138 us** | **3656.606 us** | **-162.468 us** |

The shared provider is charged exactly once in the aggregate Q/K/V row. This
avoids the earlier ambiguity where assigning it independently to Q, K, and V
would double-count one producer.

## What is actually left

Only two clean active-body losses survive this pass:

1. Vocabulary: 15.800 us/token, the largest confirmed kernel target.
2. O: 3.007 us/token, too small to explain the endpoint gap alone.

Gate/up, down, aggregate Q/K/V, norms, and Flash score are closed as llama
kernel-body recovery pools unless a new common-protocol measurement overturns
these results.

Replacing the corresponding old tinygrad QMD-timestamp rows with exact active
bodies produces an adjusted tinygrad active-node estimate of 3701.818 us/token,
versus llama's measured 3878.210-us CUPTI node sum. Tinygrad is therefore about
176.392 us ahead in kernel-active work, while its unprofiled endpoint remains
38.802 us/token slower (4060.523 versus 4021.721 us/token).

The sign inversion is the important result:

```text
endpoint disadvantage       +38.802 us/token
active-body advantage      -176.392 us/token
implied boundary residual  +215.194 us/token
```

The 215.194-us residual is diagnostic, not a claim that all 215 us can be
removed. It crosses isolated CUDA-active and production graph boundaries and
contains activation/launch service, graph cadence, exposed gaps, and any
remaining rows not yet converted to exact CUPTI. It does prove that another
Flash or projection SASS rewrite is aimed at the wrong accounting layer.

## Token-rate translation

Parity requires recovering the observed 38.802-us endpoint difference:

```text
4060.523 us/token = 246.274 tok/s
4021.721 us/token = 248.711 tok/s
```

The full 215.194-us diagnostic residual would imply roughly 260 tok/s if it
were all removable, but that is deliberately not a ceiling or booking. The
next campaign must first split this residual into paid QMD activation service,
device-idle gaps, and non-converted active rows. Only the measured reducible
part may be translated into a token-rate claim.

## Protocol

Each row uses the exact installed tinygrad cubin and captured launch ABI. A
96-MiB streaming conditioner runs between target launches so repeated small
matrices do not become falsely L2-hot. Nine post-condition target instances
are read from `CUPTI_ACTIVITY_KIND_KERNEL`; the reported value is their median.
Llama charges come from the retained PDL-off production DAG on the same CUPTI
active-duration definition.

## Next test

Instrument one complete tinygrad token so every QMD interval has both an
actual hardware-active interval and its predecessor/successor gap. Then classify
the approximately 215-us residual into:

1. QMD submitted-to-active latency,
2. active-to-next-active device gaps,
3. graph replay/doorbell cadence,
4. active rows still using the old timestamp boundary.

That test is the shortest path to the real remaining lever. Vocabulary remains
the first independent kernel-body investment, with a measured ceiling near one
token per second.

## Evidence

- `docs/task_workflow/evidence/nv-active-body-ledger-20260827/phase2-projection-result.json`
- `docs/task_workflow/evidence/nv-active-body-ledger-20260827/tiny-projection-capture.json`
- `docs/task_workflow/evidence/nv-lifecycle-recovery-tests-20260826/llama-pdl-ab/pdl-off-dag.json`
- `docs/task_workflow/output/nv-active-body-ledger-phase1-result.md`
- `docs/task_workflow/output/nv-flash-score-common-protocol-result.md`
