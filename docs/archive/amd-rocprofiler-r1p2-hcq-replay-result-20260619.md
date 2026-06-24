# AMD ROCprofiler R1-P2 HCQ Replay Result

Date: 2026-06-19

Artifacts:

- Probe: `extra/amd_rocprofiler_r1p2_hcq_replay.py`
- Result: `bench/amd-scheduler-tooling-backend/r1p2_hcq_replay.json`
- Predispatch debug: `bench/amd-scheduler-tooling-backend/r1p2_hcq_replay_predispatch.json`

## Scope

This phase tested the only bounded reopen left after the ROCprofiler ATT audit:

1. export v2 AQLprofile start/stop vendor AQL packets from a real HSA process;
2. allocate equivalent control, command, and trace-output buffers in tinygrad HCQ;
3. patch the exported packets/command stream to tinygrad-owned GPU VAs;
4. submit `start -> tinygrad kernel -> stop` on tinygrad's AQL queue;
5. require synchronized completion plus nonzero, decodable SQTT body packets.

Pass means tinygrad can import AQLprofile's mature ATT lifecycle without enabling the HIP/HSA runtime inside the tinygrad
process. Fail means the remaining path is native profiled-HCQ, not another SQTT register sweep.

## Result

Verdict: `PASS_BODY_ATTRIBUTION`.

| Gate | Result |
|---|---:|
| v2 helper packet export | PASS |
| tinygrad VA patching | PASS |
| HCQ submit + sync | PASS |
| nonzero trace output | PASS |
| decodable body packets | PASS |

Final trace summary:

| Metric | Value |
|---|---:|
| Trace bytes | 8,388,608 |
| Nonzero bytes | 338,882 |
| Body-like packets | 98,269 |
| `VALUINST` packets | 98,142 |
| `INST` packets | 105 |
| `WAVESTART` / `WAVEEND` | 1,070 / 558 |

The body dispatch was intentionally enlarged to `4096` workgroups with a long VALU loop. ATT samples one target
WGP/SIMD slice, so a four-workgroup smoke kernel can miss the selected WGP and produce lifecycle-only data even when the
replay path is correct.

## What Actually Blocked Us

The first reported P0 blocker was a local ABI bug, fixed in the previous R1-P2 result: `aqlprofile_att_profile_t.agent`
is a real `hsa_agent_t`, not an `aqlprofile_agent_handle_t`.

The P1/P2 blocker was not allocation visibility alone. The corrected vendor packet executed but faulted on the helper
process's old trace-output VA. The reason: AQLprofile's PM4 command buffer does not store every trace pointer as a raw
64-bit VA. The trace buffer base appears in PM4 register payloads as page-address fields (`VA >> 12`, low 32 bits). A
plain 64-bit pointer patch left those encoded PM4 page fields pointing at the helper process allocation.

The replay probe now patches both representations:

- raw 64-bit pointers in packets/control buffers;
- PM4 page-address words in command buffers.

After that, `start_only` synchronized and wrote SQTT lifecycle packets. Full `start_body_stop` then produced decodable
instruction packets.

## Consequence

The ROCprofiler ATT path is no longer blocked conceptually. We now have a working primitive-local thread-trace path for
tinygrad HCQ:

- no HIP runtime in the tinygrad process;
- no same-process HSA runtime, which remains mutually exclusive with tinygrad HCQ;
- AQLprofile remains a separate-process packet factory;
- tinygrad submits the imported vendor packets directly through HCQ.

This upgrades the scheduler/resource tooling track from "blocked by ROCprofiler integration" to "usable for targeted
primitive attribution." The next work should consume this trace on real decode/prefill primitives and align decoded
instruction/resource evidence with the existing PMC atlas, not reopen packet plumbing.

## Remaining Boundaries

- The exporter currently uses a separate HSA helper process because `hsa_init()` and tinygrad HCQ are mutually exclusive
  in-process.
- The PM4 patcher is enough for the captured gfx1100 ATT command shape, not a general PM4 relocation framework.
- ATT captures a selected WGP/SIMD target, so short kernels need enlarged or repeated dispatches to guarantee body
  attribution.
- This is observability tooling only. It does not by itself improve decode or prefill performance.
