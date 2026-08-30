# NVIDIA pp512 Q6-down boundary capture result

## Verdict

**BLOCKED.** The exact 18-role replay wrapper is now present, but no run has
yet supplied all four device timestamp and allocation/copy/materialization
observations.

## Implemented seam

- `extra/llm_research/prefill/nv_compiler_q6k_pp512_binding.py` now defines the
  distinct diagnostic ABI `tinygrad.nv_compiler_q6k_boundary_capture.v1` and
  emits one record for each required cut.
- `extra/llm_research/prefill/nv_compiler_q6k_model_arm.py` exposes it only when
  `NV_COMPILER_Q6_BOUNDARY_CAPTURE=1`.
- The seam is packet-local and default-off. It does not alter arithmetic,
  geometry, routing, graph policy, or promotion records.
- `new_boundary_replay(Device["NV"]).capture(calls)` is the capture-local
  wrapper. It enforces the 36-role Q6 V/down census and emits one record for
  each boundary; optional submission observations are required for PASS.

## Exact limitation

The existing captured PROGRAM list has no device-side begin/end marker ABI,
queue-ready/dependency timestamp fields, allocation/copy/materialization
census, or stable identity for output publication and the rank-preserving
residual epilogue. Therefore the seam reports `UNAVAILABLE` for those cuts and
cannot claim a boundary measurement. It also cannot claim a PASS for producer
or Q6-main device timing; host observation is not substituted for device time.

The runner now emits an explicit observation record for both `hot` and
`rotated_cold` phases and confirms the exact 36-role producer census. The
adapter is fail-closed: it does not substitute host time or standalone marker
launches for real boundary observations. D0 remains blocked because the
runner has no per-submission HCQ event handles or allocator/copy/materialization
observer.

## Execution status

The marker-only device gate was run on 2026-08-29:

| Boundary | PROGRAM | uint32 sink after launch |
|---|---|---:|
| compact Q8 producer | `nv_q6k_boundary_marker_compact_q8_producer_v1` | 1 |
| Q6 main | `nv_q6k_boundary_marker_q6_main_v1` | 2 |
| output publication | `nv_q6k_boundary_marker_output_publication_v1` | 3 |
| residual epilogue | `nv_q6k_boundary_marker_residual_epilogue_v1` | 4 |

This proves compilation and launchability of all four distinct marker cuts,
but does not measure the real cuts. The runner-side record confirms the exact
36-role census for both phase labels, while the following fields remain
unavailable for both phases: device begin/end, queue-ready/dependency
timestamps, and allocation/copy/materialization counts and bytes. The exact
execution boundary is still the missing submission-layer wrapper around the
producer, Q6-main, publication, and residual submissions during a real
Q6-down replay. D1 remains closed.
