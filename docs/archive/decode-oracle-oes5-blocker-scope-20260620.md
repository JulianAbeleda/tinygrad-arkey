# Decode Oracle OES-5 Blocker Scope - 2026-06-20

Verdict: `BLOCKED_OES5_NEEDS_ROCPROF_VISIBLE_ORACLE_RUNNER`

OES-4 is complete, but OES-5 cannot be honestly run from the current oracle artifact path. The extracted `q8_mmvq_gateup` artifact is loaded through tinygrad `AMDProgram`/HCQ, and this repo's profiling docs already establish that `rocprofv3` cannot trace tinygrad HCQ/KFD dispatches. The profiling tools are installed; the missing piece is the right runner surface.

## What Is Ready

| item | status |
| --- | --- |
| Oracle HSACO identity | pass |
| Oracle metadata/resource envelope | pass |
| Oracle semantic ISA map | pass |
| `rocprofv3` | present |
| `rocprof-compute-viewer` | present |
| LLVM object tools | present |

## What Blocks OES-5

| missing item | why it matters | construction path |
| --- | --- | --- |
| `rocprof_visible_oracle_runner` | Current oracle route uses `AMDProgram`/HCQ, which `rocprofv3` does not attribute. | Build a minimal HIP host executable that launches the same `q8_mmvq_gateup` source/object with the same geometry and deterministic buffers. |
| `native_pc_join_surface` | OES-4 names oracle PC stages, but native/C7C PCs still need the same semantic labels before stage comparison is meaningful. | Export native/C7C disasm PCs from existing q8 native probes and map them to S1/S2/S3/S4/S5 equivalents. |
| ATT/counter fallback policy | Prior ATT docs show thread trace can be environment-sensitive. | Prefer ATT PC timeline; if blocked, collect kernel trace resource fields plus available PMC counters and stop at coarse stall-family attribution. |

## Next Executable Plan

1. Create a minimal HIP host runner for `q8_mmvq_gateup` using launch geometry `(12288,2,1)` / `(32,4,1)` and synthetic deterministic buffers.
2. Run `rocprofv3 --kernel-trace` first to prove the dispatch is visible and resources match the extracted artifact envelope.
3. Run `rocprofv3 --att` if the decoder path works; otherwise record the ATT blocker and fall back to coarse resource/counter attribution.
4. Join oracle PCs to OES-4 semantic stages and compare against native/C7C stage PCs before reopening native scheduling.

Probe: `extra/qk_decode_oracle_oes5_blocker_scope.py`

