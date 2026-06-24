# AMD SQTT oracle-to-HCQ diff scope - 2026-06-19

Purpose: use the now-working ROCprofiler ATT oracle to repair tinygrad HCQ SQTT body-instruction attribution, if the
missing piece is bounded and observable.

This is **tooling only**. It does not change decode performance by itself. Its value is that it can turn the q8/AMD
scheduler investigation from lifecycle/PMC evidence into instruction-level body attribution.

## Current State

Solved:

- `rocprofv3 --att` now works on a HIP control kernel with ROCprof Trace Decoder `0.1.6`.
- The HIP control is built coherently against ROCm 7.2.4, avoiding the Ubuntu HIP 5.7 shadow path.
- The ATT run emits `.att`, decoded UI files, wave JSON, occupancy JSON, and a result JSON.

Still blocked:

- tinygrad HCQ SQTT produces lifecycle-style packets for the q8 kernel, but not mapped body instruction packets.
- Prior raw-register transplants from AQLprofile changed trace volume but still produced no `INST` / `VALUINST` /
  `IMMEDIATE` / `VMEMEXEC` / `ALUEXEC` body packet classes.

Known non-causes:

- not a missing decoder anymore;
- not a simple `SQTT_MODE` / `SQTT_TTRACE_EXEC` knob;
- not just `SQ_THREAD_TRACE_MASK`, `SQ_THREAD_TRACE_TOKEN_MASK`, or `SQ_THREAD_TRACE_CTRL`;
- not a failure to write raw SQTT registers through HCQ.

## Success Definition

Final pass:

```bash
PYTHONPATH=. python3 extra/amd_sqtt_t1_body_mapping_proof.py
```

must produce:

- `raw_body_packet_events_top20 > 0`;
- `body_instruction_events > 0`;
- q8 body instructions mapped for `q8_b2b_fullrow_reduce`, not only `S_ENDPGM`.

Intermediate pass:

- ROCprofiler ATT oracle output is archived in a stable, parseable artifact;
- decoded wave files prove body instruction records exist for the HIP control;
- a concrete ROCprofiler-vs-HCQ setup delta is named beyond the three already-tested registers.

## Non-Goals

- Do not restart random SQTT register sweeps.
- Do not land default runtime behavior changes.
- Do not claim a scheduler/codegen primitive from lifecycle-only SQTT.
- Do not require ROCprofiler tooling for normal tinygrad operation.
- Do not broaden into decode kernel changes unless this scope first proves body attribution.

## Phase O0 - Stable Oracle Capture

Deliverables:

- `extra/amd_sqtt_oracle_hcq_diff.py`
- `bench/amd-scheduler-tooling-backend/att_oracle_capture.json`
- archived small subset of oracle output metadata, not the full temporary decoder work tree

Actions:

1. Reuse the decoder path selected by `att_decoder_binary_probe.json`.
2. Run a minimal HIP control with `rocprofv3 --att`.
3. Inventory the output directory:
   - raw `.att`;
   - code-object `.out`;
   - `*_results.json`;
   - `ui_output_agent_*` files;
   - wave-state / instruction JSON files.
4. Parse enough JSON to prove instruction records exist and identify the stable fields needed for comparison.

Gate:

- oracle run exits `0`;
- at least one decoded wave/instruction file contains non-empty instruction/body records.

Kill:

- oracle output exists but contains only lifecycle/occupancy metadata;
- output format cannot be parsed enough to prove instruction records.

## Phase O1 - tinygrad HCQ Baseline Capture

Deliverable:

- `bench/amd-scheduler-tooling-backend/hcq_sqtt_baseline_capture.json`

Actions:

1. Run `extra/amd_sqtt_t1_body_mapping_proof.py` unchanged.
2. Preserve per-config SQTT blob sizes, packet top counts, mapped instruction counts, env knobs, and program hashes.
3. Keep the current verdict if no body packets appear.

Gate:

- baseline reproduces lifecycle-only behavior on the current tree.

Kill:

- baseline now passes unexpectedly; in that case stop and document the natural pass before patching anything.

## Phase O2 - Command/Setup Diff

Deliverable:

- `bench/amd-scheduler-tooling-backend/att_hcq_setup_diff.json`

Diff targets:

- SQTT register values:
  - `SQ_THREAD_TRACE_BUF0_SIZE`;
  - `SQ_THREAD_TRACE_BUF0_BASE`;
  - `SQ_THREAD_TRACE_MASK`;
  - `SQ_THREAD_TRACE_TOKEN_MASK`;
  - `SQ_THREAD_TRACE_CTRL`;
  - `COMPUTE_THREAD_TRACE_ENABLE`;
  - `COMPUTE_STATIC_THREAD_MGMT_SE*`.
- command ordering around:
  - trace buffer setup;
  - GRBM / SE / SA / WGP selection;
  - SPI config;
  - start event;
  - target dispatch;
  - stop event;
  - `THREAD_TRACE_FINISH`;
  - wait-for-finish / wait-for-busy;
  - WPTR copyout.
- target selection:
  - SE;
  - SIMD;
  - WGP;
  - CU mask;
  - whether the traced SIMD actually receives the target wave.
- serialization:
  - barriers before start;
  - barriers after start;
  - partial flush before/after stop;
  - event write ordering.

Actions:

1. Extract whatever the oracle exposes directly from ATT/UI output.
2. Use AQLprofile command recovery as the low-level register/PM4 side input.
3. Compare against the tinygrad `AMDComputeQueue.sqtt_start/sqtt_stop/sqtt_setup_exec` sequence.
4. Classify every diff:
   - already tested;
   - observable and patchable;
   - observable but too broad;
   - not observable from available tooling.

Gate:

- names at least one **new** patchable difference that could plausibly affect instruction body packets.

Kill:

- all observed differences reduce to already-tested registers;
- oracle proves body packets but does not expose enough setup information to produce a bounded patch.

## Phase O3 - One Env-Gated Patch

Deliverables:

- minimal patch in `tinygrad/runtime/ops_amd.py`;
- result artifact `bench/amd-scheduler-tooling-backend/hcq_sqtt_oracle_patch_result.json`.

Rules:

- Must be env-gated, for example `SQTT_ORACLE_STYLE=1`.
- Must not affect default behavior when unset.
- Must patch one named setup/order delta, not a broad rewrite.
- Must preserve existing raw-register env overrides.

Likely patch classes, depending on O2:

- different start/stop event ordering;
- additional wait/flush before target dispatch;
- altered `COMPUTE_THREAD_TRACE_ENABLE` timing;
- altered static-thread-management mask to force dispatch onto the traced SIMD;
- different GRBM selection lifetime around setup/start/stop.

Gate:

- `extra/amd_sqtt_t1_body_mapping_proof.py` has at least one passing config with the new env flag.

Kill:

- patch only changes byte volume, not body packet classes;
- patch causes hangs, corrupt traces, or unstable output;
- more than one unrelated patch would be needed before any signal.

## Phase O4 - Attribution Usability Check

Deliverable:

- `bench/amd-scheduler-tooling-backend/q8_body_attribution_smoke.json`

Actions:

1. With the passing SQTT setup, decode q8 kernel body instructions.
2. Join mapped instructions to the existing q8 oracle/tinygrad schedule artifacts.
3. Check whether the attribution is good enough to answer feature questions:
   - load coalescing;
   - dot4 issue density;
   - waitcnt / stall regions;
   - clause/delay usage;
   - occupancy/resource pressure.

Gate:

- produces a feature-level attribution table with at least one row that was previously PMC-only or inferred.

Kill:

- body packets exist but cannot be mapped back to useful program PCs/instructions.

## Phase O5 - Closeout

Deliverable:

- `docs/amd-sqtt-oracle-hcq-diff-result-20260619.md`

Possible verdicts:

| verdict | meaning | next |
|---|---|---|
| `PASS_BODY_ATTRIBUTION` | tinygrad HCQ SQTT can now emit q8 body packets | use it for q8 scheduler/resource attribution |
| `PASS_ORACLE_ONLY` | ROCprofiler oracle works, HCQ remains lifecycle-only | keep external oracle for controls, do not patch tinygrad |
| `KILL_NO_PATCHABLE_DIFF` | oracle exposes no bounded HCQ-repair delta | close Track T; use PMC + static disasm |
| `KILL_PATCH_NO_BODY` | bounded patch tried and still no body packets | close Track T until deeper ROCprofiler integration |

## Decision Boundary

If O3 passes, Track T becomes a reusable primitive-observability tool.

If O3 fails or O2 cannot name a patchable difference, stop. That means this is not a small primitive tooling fix; it is
a broader ROCprofiler command-service integration project. The decode path should continue using the already-converged
evidence: PMCs, static disassembly, isolated-vs-in-model bandwidth reconciliation, and q8 lifecycle artifacts.
