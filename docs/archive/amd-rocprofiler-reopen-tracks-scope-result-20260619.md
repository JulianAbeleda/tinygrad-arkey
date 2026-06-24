# AMD ROCprofiler reopen tracks scope/result - 2026-06-19

Purpose: scope and execute the first decisive phase for the three reopen options from
`amd-rocprofiler-thread-trace-audit-result-20260619.md`.

Artifacts:

- `extra/amd_rocprofiler_reopen_tracks.py`
- `bench/amd-scheduler-tooling-backend/rocprofiler_reopen_tracks.json`

## Summary

The audit split into three tracks:

1. **AQLprofile packet import/replay** - bounded experimental reopen.
2. **Native profiled-HCQ** - project-level backend work.
3. **Split tooling model** - usable default now.

Executed first phase verdict:

| track | phase | verdict |
|---|---|---|
| 1. AQLprofile packet import/replay | R1-P0 packet/material audit | **GO_TO_R1_P1_PACKET_REPLAY_PROOF** |
| 2. Native profiled-HCQ | R2-P0 capability decomposition | **PROJECT_LEVEL_ONLY_NO_SMALL_PATCH** |
| 3. Split tooling model | R3-P0 contract check | **PASS_DEFAULT_OBSERVABILITY_MODEL** |

Recommendation: use Track 3 by default. Fund Track 1 only if HCQ body instruction packets are worth one more bounded
experiment. Do not start Track 2 until Track 1 proves a real packet/lifecycle route or the project explicitly funds
native profiled-HCQ work.

## Track 1 - AQLprofile Packet Import/Replay

Question: can we reopen HCQ body ATT by importing/replaying AQLprofile's packet lifecycle instead of sweeping SQTT
registers?

Why this is the only bounded reopen:

- AQLprofile is installed and linkable.
- Prior `T1b` generated nonzero AQLprofile SQTT command buffers on gfx1100.
- Working minimal parameter sets exist: `cu_se_only`, `cu_only`, `no_params`.
- The prior register transplant already proved raw `MASK/TOKEN/CTRL` values are not enough.
- The source audit says ROCprofiler's real win is the packet/lifecycle path, not a register value.

R1-P0 verdict: **GO_TO_R1_P1_PACKET_REPLAY_PROOF.**

R1-P1 scope:

1. Build a no-model tinygrad HCQ replay proof.
2. Generate AQLprofile start/stop command buffers in a separate helper or same process if HIP/HSA does not conflict.
3. Submit one tinygrad HCQ kernel wrapped by the imported start/stop command stream.
4. Decode the captured SQTT blob with the existing parser.
5. Pass only if HCQ produces body instruction packet classes, not just lifecycle packets.

Kill gate: if the replayed AQLprofile command stream still produces zero body packets, close packet import and keep
external ATT only.

Important constraint: no model route, no default runtime change, no scheduler/codegen work.

## Track 2 - Native Profiled-HCQ

Question: should tinygrad implement ROCprofiler's profiled-HSA-queue lifecycle natively for KFD/HCQ?

R2-P0 verdict: **PROJECT_LEVEL_ONLY_NO_SMALL_PATCH.**

The missing components are high-confidence and structural:

- HSA queue interposition around dispatch packets;
- `hsa_amd_profiling_set_profiler_enabled`;
- profiler-active queue packet;
- AQLprofile-generated vendor AQL start/stop packets;
- trace-control buffer status/WPTR protocol;
- SQTT begin/end ordering including `BuildPrimeL2`, status clear, control-buffer reads, and cache flush;
- ROC decoder metadata markers.

This is not a good next step as a small primitive patch. It is a backend/tooling project with a larger surface than our
current decode objective.

Start gate:

- R1-P1 passes enough to show imported packet lifecycle works through HCQ; or
- the project explicitly accepts a native profiled-HCQ backend/tooling investment independent of immediate decode gain.

Kill gate:

- do not start from another `SQ_THREAD_TRACE_MASK`, `TOKEN_MASK`, `CTRL`, `SQTT_MODE`, or decoder-filter patch.

## Track 3 - Split Tooling Model

Question: can we proceed with external ROCprofiler ATT as the instruction oracle plus tinygrad PMCs for in-model
attribution?

R3-P0 verdict: **PASS_DEFAULT_OBSERVABILITY_MODEL.**

This is the usable path now:

- external ROCprofiler ATT works on HIP controls after the decoder/toolchain repair;
- the oracle-to-HCQ diff proved the external path is instruction-rich;
- tinygrad-native PMCs already give in-model primitive attribution;
- the split matches the evidence boundary: HIP ATT explains codegen/resource behavior, while HCQ PMCs explain real
  tinygrad model behavior.

Do not overclaim it. External HIP ATT is an oracle/control path, not direct proof that tinygrad HCQ emitted body
instruction packets.

## Decision

Use Track 3 for ongoing decode/prefill analysis.

If we still need HCQ body instructions, fund exactly one bounded Track 1 replay proof. If that fails, stop. If it passes,
then Track 2 becomes a real backend transfer candidate.

Do not spend more time on simple register sweeps or decoder availability.
