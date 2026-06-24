# Decode Oracle Extraction Scope - 2026-06-20

Verdict: `SCOPE_DECODE_ORACLE_EXTRACTION_READY`

This scopes the next decode path after native DNR-3C/C7C stopped producing material gains. The decision is to stop guessing from static native shape similarity and extract the HIP/LLD oracle directly: code object, resource envelope, semantic ISA, and PC/stall attribution.

## Source Basis

The scope is based on public tool documentation:

- LLVM AMDGPU Usage: AMDGPU code objects expose metadata/note records, kernel descriptors, and VGPR/SGPR metadata.
- ROCm workload optimization docs: saved ROCm code objects can be disassembled with `llvm-objdump --disassemble-all`.
- ROCprofiler-SDK `rocprofv3`: kernel trace output exposes kernel launch/resource fields including LDS, scratch, VGPR, SGPR, workgroup, and grid data.
- ROCprofiler-SDK thread trace: thread trace is intended for instruction timing, execution path, wave scheduling, stalls, and hotspots.
- ROCprof Compute Viewer: viewer output provides ISA source visualization, hotspots, waitcnt dependency views, and occupancy visualization.

## Local Tool Inventory

The scope probe found the required tools locally:

| tool | path |
| --- | --- |
| `llvm-objdump` | `/opt/rocm/llvm/bin/llvm-objdump` |
| `llvm-readobj` | `/opt/rocm/llvm/bin/llvm-readobj` |
| `llvm-objcopy` | `/opt/rocm/llvm/bin/llvm-objcopy` |
| `clang-offload-bundler` | `/opt/rocm/llvm/bin/clang-offload-bundler` |
| `rocprofv3` | `/opt/rocm/bin/rocprofv3` |
| `rocprof-compute-viewer` | `/usr/local/bin/rocprof-compute-viewer` |

## Phases

| phase | purpose | pass gate |
| --- | --- | --- |
| OES-1 oracle kernel identification | Identify the exact HIP/LLD q8 gate/up oracle dispatch and symbol. | One q8 gate/up dispatch accounts for the oracle timing row and has stable identity. |
| OES-2 code object extraction | Recover the loadable gfx1100 oracle code object. | `llvm-objdump` can read it and the selected symbol is present. |
| OES-3 metadata and descriptor extraction | Extract VGPR, SGPR, LDS, scratch, kernarg, workgroup, and descriptor metadata. | Resource envelope is comparable to native, best-static, and C7C-best. |
| OES-4 semantic ISA map | Annotate oracle ISA into q8 stages: load, unpack/select, dot4, scale/min, reduction, wait, branch, store. | One unimplemented semantic mechanism is named, or oracle reduces to already-tested native patterns. |
| OES-5 PC timeline and stall attribution | Join thread trace/ATT PCs to oracle and native ISA. | One stall/dependency family with plausible material movement is identified. |
| OES-6 fair oracle comparison | Compare oracle/native/best-static/C7C under one timing and clock policy. | Remaining gap is assigned to body schedule, resources, runtime/launch, or route policy. |

## Decision Policy

Resume native decode only if one DNR-3C9 reopen gate passes from OES output. Otherwise keep DNR-3C parked and move to route-level decode work.

Do not start BEAM/search from static opcode similarity. Search needs a measurable objective from oracle extraction: resource envelope, semantic schedule mechanism, PC stall family, or launch/runtime delta.

Probe: `extra/qk_decode_oracle_online_scope.py`

