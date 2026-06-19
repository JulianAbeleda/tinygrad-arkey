# AMD ROCprofiler R1-P1 AQLprofile replay result - 2026-06-19

Purpose: execute the bounded Track 1 reopen from `amd-rocprofiler-reopen-tracks-scope-result-20260619.md`: try to
recover tinygrad HCQ body instruction packets through AQLprofile-style packet replay before starting native
profiled-HCQ work.

Artifacts:

- `extra/amd_rocprofiler_r1p1_replay_proof.py`
- `bench/amd-scheduler-tooling-backend/r1p1_aqlprofile_replay_proof.json`

## Verdict

**BLOCKED_REQUIRES_V2_AQLPROFILE_PACKET_EXPORT_OR_NATIVE_PROFILED_HCQ.**

R1-P1 did not produce HCQ body instruction packets.

The useful split:

- forcing tinygrad `AMD_AQL=1` is stable, but still lifecycle-only;
- AQLprofile can generate nonzero gfx1100 ATT command buffers;
- the old `hsa_ven_amd_aqlprofile_*` command-buffer material is not a standalone replay blob for HCQ;
- direct replay now requires an AQLprofile v2 packet exporter that binds command/output/control buffers to
  tinygrad-owned GPU VAs, or a native profiled-HCQ implementation.

## Gates

| gate | result |
|---|---|
| AQL queue compatibility | pass |
| AQL queue body packets | fail |
| AQLprofile packet material exists | pass |
| directly replayable packet exists | fail |

The AQL queue compatibility run:

| metric | value |
|---|---:|
| capture ok | yes |
| SQTT events | 12 |
| SQTT bytes | 1,970,592 |
| decoded rows | 12 |
| raw body packet events | 0 |
| mapped body instructions | 0 |

So plain AQL submission is not the missing primitive.

## Packet Material

The earlier AQLprofile probe remains useful:

- working minimal parameter sets: `cu_se_only`, `cu_only`, `no_params`;
- command buffers contain nonzero gfx1100 SQTT PM4-like words;
- explicit token-mask parameter sets fail `start` with status `4096`;
- legacy `hsa_ven_amd_aqlprofile_legacy_get_pm4` returns status `4096` for start/stop/read packets.

That means the old command buffer is useful for diffing, but not directly sufficient as an HCQ replay unit. It embeds
trace output/control buffer assumptions and expects AQLprofile's trace-control protocol. We already tried transplanting
the obvious register values and got zero body packets, so replaying only the visible `MASK/TOKEN/CTRL` subset is closed.

## What Is Closed

- plain `AMD_AQL=1` as the fix;
- old `hsa_ven_amd_aqlprofile_*` command-buffer words as a directly reusable HCQ blob;
- another `SQ_THREAD_TRACE_MASK` / `TOKEN_MASK` / `CTRL` transplant.

## Remaining Reopen

The next executable replay, if funded, is narrower and deeper:

1. Build a small AQLprofile v2 packet exporter.
2. Use callbacks that bind command/output/control allocations to known tinygrad-owned or tinygrad-mappable GPU buffers.
3. Export exact start/stop vendor packet bodies plus trace-control layout.
4. Submit those packets around one HCQ dispatch.
5. Decode tinygrad's resulting SQTT payload.

Pass condition stays strict: nonzero body instruction packet classes from tinygrad HCQ, not just lifecycle packets.

If that fails, stop the HCQ-body-ATT route and use the split tooling model.

## Decision

Do not start native profiled-HCQ from this result alone. R1-P1 proved the cheap version is not enough, and it sharpened
the remaining path to a v2 AQLprofile packet-export problem.

For ongoing decode work, continue with the split tooling model: external ROCprofiler ATT for instruction-oracle evidence,
tinygrad-native PMCs for in-model HCQ attribution.
