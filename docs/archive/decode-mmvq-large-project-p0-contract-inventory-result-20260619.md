# Decode MMVQ large project P0 contract inventory result - 2026-06-19

Purpose: execute P0 from `decode-mmvq-large-project-scope-20260619.md`.

No kernels were launched. No model route or default changed.

Artifacts:

- `extra/qk_decode_mmvq_large_project.py`
- `bench/qk-decode-mmvq-large-project/contract_inventory.json`
- `bench/qk-decode-mmvq-large-project/summary.md`

## Verdict

`P0_PASS__SOURCE_IMPORT_P1_IS_LOADABLE_DESCRIPTOR_SMOKE`.

The funded large path should start with source/object import before native renderer work.

## What Was Found

The local llama.cpp build contains a gfx1100 AMDGPU object:

```text
/home/ubuntu/env/llama.cpp/build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/mmvq.cu.o.0.hipv4-amdgcn-amd-amdhsa--gfx1100
```

It is an `elf64-amdgpu` object with:

- Q4_K/Q6_K candidate functions: `22`;
- Q4_K/Q6_K `.kd` descriptors: `22`;
- `.note` AMDGPU metadata;
- `.rodata` descriptor section;
- `.text` kernel code section;
- `144` byte kernarg segments for the selected MMVQ kernels.

Representative candidates:

| type | ncols | bools | VGPR | SGPR | kernarg | max wg |
|---|---:|---|---:|---:|---:|---:|
| Q4_K | 1 | `0/0` | `23` | `24` | `144` | `32` |
| Q4_K | 1 | `1/0` | `34` | `42` | `144` | `32` |
| Q4_K | 7 | `0/0` | `78` | `28` | `144` | `32` |
| Q6_K | 1 | `0/0` | `26` | `24` | `144` | `64` |
| Q6_K | 1 | `1/0` | `33` | `42` | `144` | `64` |
| Q6_K | 7 | `0/0` | `66` | `30` | `144` | `32` |

The low-VGPR ncols-1 candidates are the first P1 loader targets because they most closely match the traced llama
one-wave decode contract.

## Why This Matters

The prior artifact inventory correctly said there is no packaged Tensile-like `.hsaco/.co` family. P0 refines that:
there is still a compiled object with descriptors and metadata. That makes source/object import a real first track.

The next unknown is not "does code exist?" It is:

```text
Can tinygrad HCQ load a selected llama `.kd` descriptor by name from this object and later launch it with captured
kernargs, without HIP runtime in-process?
```

## Next Phase

P1: single-kernel HCQ loader smoke.

Gate:

- load one selected Q4_K descriptor and one selected Q6_K descriptor by name;
- no in-process HIP runtime;
- no unsupported relocations;
- no model route changes.

Do not launch in P1 unless descriptor load is proven. Launch correctness belongs to P3 after P2 captures the real
kernarg and launch contract.
