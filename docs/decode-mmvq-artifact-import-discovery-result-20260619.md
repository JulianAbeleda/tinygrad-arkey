# Decode MMVQ artifact/import discovery result - 2026-06-19

Purpose: execute the large-path L1 inventory from `decode-large-small-paths-scope-20260619.md`.

No kernels were built. No model route or default changed.

Artifacts:

- `extra/qk_decode_path_split.py`
- `bench/qk-decode-path-split/large_artifact_inventory.json`

## Verdict

`NO_READY_HCQ_ARTIFACT__SOURCE_IMPORT_OR_RENDERER_PROJECT_LEVEL`.

The local llama.cpp checkout contains the mature decode MMVQ source family and linked build objects, but it does not
contain a standalone Tensile-like Q4_K/Q6_K MMVQ code-object family that can be mechanically extracted with the TPE
method.

## Inventory

Found source family:

- `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`
- `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmvq.cuh`
- `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu`
- `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu`

Found build objects:

- `build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/mmvq.cu.o`
- `build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/mmvq.cu.o.0.hipv4-amdgcn-amd-amdhsa--gfx1100`
- `build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/template-instances/mmq-instance-q4_k.cu.o`
- `build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/template-instances/mmq-instance-q6_k.cu.o`

Not found:

- standalone `.hsaco` / `.co` MMVQ code-object family;
- fixed per-shape descriptor set analogous to rocBLAS/Tensile;
- simple extracted kernarg contract ready for tinygrad `AMDProgram` / HCQ launch.

## Interpretation

This is meaningfully different from the prefill Tensile extraction.

Tensile gave us mature backend kernels as compiled artifacts with recoverable descriptors, kernargs, and fixed launch
contracts. llama.cpp's decode path gives us mature source templates and linked HIP build objects. That is useful, but
it is not a bounded artifact-import path.

The large path now has only two honest forms:

1. source-contract import: compile/extract the llama MMVQ source family into tinygrad-loadable code objects, recover
   launch contracts, and maintain the external source boundary;
2. native renderer/scheduler work: make tinygrad generate the same MMVQ lifecycle contract itself.

Both are project-level. Neither is a small machine-search primitive.

## Decision

Do not start TPE-style HSACO extraction for decode until a real standalone MMVQ code-object family is identified.

If the project funds the large path, scope it as source import or renderer/scheduler ownership, with the measured
target of moving in-model weight-GEMV efficiency from about `44%` toward llama's `54%`.
