# NVIDIA pp512 Q6-down boundary marker seam

## Verdict

**BLOCKED.** The diagnostic ABI is structurally defined, but this packet did
not execute the four marker launches on an NVIDIA device. D0 therefore cannot
claim that all four markers are launchable.

## Added seam

`extra/llm_research/prefill/nv_compiler_q6k_pp512_binding.py` now exposes the
opt-in helper `compile_boundary_markers(dev)`. It creates one no-op native NV
PROGRAM for each required cut:

| Boundary | Stable marker name |
|---|---|
| compact Q8 producer | `nv_q6k_boundary_marker_compact_q8_producer_v1` |
| Q6 main | `nv_q6k_boundary_marker_q6_main_v1` |
| output publication | `nv_q6k_boundary_marker_output_publication_v1` |
| residual epilogue | `nv_q6k_boundary_marker_residual_epilogue_v1` |

The ABI is `tinygrad.nv_compiler_q6k_boundary_marker.v1`. Each marker has a
single `uint32*` diagnostic sink and is compiled with a distinct cache key.
`boundary_capture_from_calls` publishes the same identities in the existing
capture record.

## Safety and scope

- Marker compilation is explicit; `binding_for()` does not compile or install
  markers.
- No arithmetic, geometry, queue policy, model integration, or default route
  changed.
- The marker programs are diagnostic-only and must be launched around the
  exact 18-role Q6-down population by a capture driver.

## Seam checks

- Python module syntax/import surface: PASS.
- ABI constants and all four stable identities: PASS.
- Device compilation and launch of all four markers: **NOT RUN**.

## Remaining boundary

A CUDA-capable D0 runner must call `compile_boundary_markers(Device["NV"])`,
allocate one uint32 sink, launch each returned PROGRAM, synchronize, and
verify the sink increments. Only that result can change this verdict to PASS.

