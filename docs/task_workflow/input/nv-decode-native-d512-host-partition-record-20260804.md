# Native NV d512 host/outside-window partition record

Date: 2026-08-04. Route: `DEV=NV`; Qwen3-8B-Q4_K_M; depth 512;
RTX 5090 / driver 595.84. Status: **outside-device-window term reconciled.**

## Result

The canonical outer ledger measured 321.784 us/token outside the native device
window. A five-repetition, four-arm diagnostic extended
`scratchpad/nv_decode_group_window_ledger.py` with host clocks and an explicit
device drain immediately before `Tensor.item()`. All arms returned token 330.

The marker-free drain arm gives the disjoint partition:

| exposed component | median us/token |
| --- | ---: |
| preparation before first graph call | 247.557 |
| already-drained scalar copyout + `Tensor.item()` | 83.247 |
| Python append/yield tail | 3.096 |
| redundant synchronize after `next()` | 4.358 |
| **disjoint sum** | **338.458** |

The sum is 16.674 us above the canonical 321.784 us term, inside the campaign
tolerance of `max(50 us, 2%)`. The explicit drain perturbed marker-free wall by
+32.370 us (5648.916 vs 5616.546 us), so the 16.7 us closure error is bounded
and expected rather than an unaccounted parity-scale term.

## What is not additive

The same drain arm observed 202.541 us in the five host graph calls, 195.740 us
between graph-call returns and the next graph call, and 90.842 us from the last
graph return to `Tensor.item()`. These intervals occur while queued device work
is running. Adding them to the outside term would double-count CPU/GPU overlap.
The pre-item drain makes this distinction decisive: 4831.324 us of the apparent
4914.801 us `item()` interval was waiting for already-submitted device work;
only 83.247 us remained after the drain.

There is no separate four-byte host-to-device feedback copy on settled decode.
`Transformer.generate` passes the previous output tensor directly into the next
model invocation. The measured 83.247 us boundary is the opposite direction:
the scalar output copy/mapping plus `Tensor.item()`.

## Consequence

The native-versus-llama outside-window delta of 239.805 us is now directionally
explained: tinygrad exposes a large pre-first-graph preparation boundary and a
nontrivial scalar-copy/item boundary; Python itself is only about 7.5 us across
yield and the final synchronize. The exact llama-side semantic split is not
available, so this record does not assign a cross-runtime delta to either native
component beyond the measured 81.979 us llama outside total.

Raw evidence is `/tmp/nv_decode_host_partition_v2_20260804.json`, SHA-256
`e2fcefe16d1044b1ba20a68f2b34d38fad1fab228402508860aa3d466958b438`.
The compact payload is
`docs/task_workflow/output/nv-decode-native-d512-host-partition-20260804.json`.
