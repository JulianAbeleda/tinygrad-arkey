# Hard-Fork Prune Manifest (complete scope) — 2026-06-16

Full file-level scope of the non-AMD prune, synthesized from 5 parallel scopers
(KEEP-set closure, backend units, autogen map+edits, `extra/` classification,
test footprint). Supersedes the rough numbers in
[hardfork-prune-plan.md](hardfork-prune-plan.md). **Read this before executing.**

## Headline totals

`extra/` is **426 files / ~204k LOC** once vendored `.h` headers count (not the
21k .py-only figure). Total addressable non-AMD surface is **~150–180k LOC**, in
three confidence tiers:

| tier | LOC (approx) | what |
|---|---:|---|
| **CLEAN** (no AMD coupling, ungated) | **~90–100k** | non-AMD backends + their exclusive autogen + non-AMD `extra/` + non-AMD tests |
| **GATED** (needs a small refactor first) | **~38k** | `mlx5`/rdma, `mesa`/`nir`/`ptx`/`wgsl`, `nv_pma`, `torch_backend` |
| **BORDERLINE** (human call: AMD-family-but-unused / upstream model zoo) | **~55k** | `hiprtc` 15k, `hip_gpu_driver` redundant headers ~36k, `models`/`datasets`, `hcq_smi`, `fp8`, `mmapeak` |

The CLEAN tier alone roughly **halves the repo's non-autogen-AMD footprint**.

## KEEP-SET (empirical anchor — the complement is deletable)

Captured via `sys.modules` after running the real AMD path + host devices + all 49 fork tests (all import-clean; baseline green).
- **Backends:** `ops_amd`, `ops_python`, `ops_cpu`, `ops_disk`, `ops_null`, `ops_npy`, `ops_tinyfs`, `ops_hip` (AMD-family; fork imports it), **`ops_rdma`** (⚠ hard-imported by `graph/hcq.py:9` on the AMD path).
- **support/:** `c`, `memory`, `elf`, `hcq`, `system`, `usb`, `amd`, `compiler_amd`, `compiler_cpu`, `autogen`, `am/*`, **`mlx/mlxdev`** (⚠ via `ops_rdma`).
- **renderer/:** `cstyle` (the AMD `HIPRenderer`/`HIPCCRenderer` + `ClangRenderer` live here — **KEEP the file**, in-file-prune the non-AMD classes), `llvmir` (`ops_amd.py:13`), `isa/*`, `amd/*` (ISA assembler, on the decode path).
- **autogen/:** `amd_gpu`, `amdgpu_drm`, `amdgpu_kd`, `kfd`, `hsa`, `sqtt`, `comgr`, `comgr_3`, `libc`, `llvm`, `pci`, `vfio`, `io_uring`, `libusb`, `iokit`, `corefoundation`, `hip`, `rocprof`, `ggml_common` (⚠ fork gguf decode), `libclang`, **all `am/*`** (GPU-variant tables loaded by runtime target-selection — invisible to single-GPU capture), **all `amd/*`** (RDNA/CDNA ISA encoders), **`mlx5`** (⚠ gated, see below).
- The fork-added `extra/qk_*`/`llm_*`/`q4_k_*`/`q6_k_*` (22,075 LOC) + `tinygrad/llm` decode/codegen diff + `viz.serve` (on the decode path).

## CLEAN delete-units (no AMD coupling — execute in this order)

Order respects shared-autogen orphaning (delete importers before the shared module).

1. **`extra/` non-AMD (~70k clean, excl. gated `nv_pma`/`torch_backend`)** `[test]` — DELETE: `qcom_gpu_driver` 19.7k, `hiprtc` 15.1k*(borderline)*, `dsp` 5.6k, `nv_gpu_driver` 5.4k, `webgpu` 4.3k, `models` 4.2k*(borderline)*, `hevc`, `datasets` 1.4k*(borderline)*, `mlx_driver`, `usbgpu` 1.2k, `optimization`, `perfetto`, `testsig`, `mmapeak`, `viz/kernel_graph.py`, `mesa/`(epilog), the gemm non-AMD files (`cuda_matmul`/`metal_*`/`triton_nv`/`torch_gemm`/`tvm`/`halide`/`max_matmul`), the thunder `metal/`(10k)+`cuda/`, and the upstream example .py orphans (`export_model`, `thneed`, `hook_cuda`, `archprobe`, `multitensor`, `training`, `gradcheck`, `introspection`, `lr_scheduler`, `onnx_helpers`, `bench_log`, `f16_decompress`, `hip_large_kernel`, `huggingface_onnx`, `torch_hook`) + non-AMD `*.sh`/`.h` (`nvJitLink.h`, `tinydreno.h`, nv/dsp setup scripts). **KEEP** the AMD set: `amdpci`, `hip_gpu_driver`(core), `gemm`(amd_*/cdna/rdna4/mi350x/simple), `thunder/{amd,tiny}`, `llama_kernels`, `sqtt`, `hcq2`, `tinyfs`, `remote/amd_*`, fp8*(borderline)*.
2. **mockgpu non-AMD** `[test]` — `test/mockgpu/nv/*` (514), `test/mockgpu/cuda/cuda.py` (174); edit `test/mockgpu/mockgpu.py` (drop NVDriver import+entry) and `helpers.py` (`ptx_run`).
3. **NV stack** `[runtime]` — `ops_nv.py` (845), `support/nv/` (811), `test/mockgpu/nv` done in 2; autogen `nv_580` (26k), `nv` (4.9k), and `nv_570` (24.9k, now orphaned after units 1–2). Strip NV branch in `external_test_hcq.py` (keep file).
4. **CUDA** `[runtime]` — `ops_cuda.py` (133), `graph/cuda.py` (74), `support/compiler_cuda.py`, in-cstyle `CUDARenderer`/`NVCCRenderer` + `renderer/ptx.py` (242, orphaned after NV), autogen `cuda` (2.2k)+`nvrtc`+`nvjitlink`.
5. **QCOM + DSP** `[runtime]` — `ops_qcom.py` (412), `compiler_qcom.py`, in-cstyle `QCOMCLRenderer`, autogen `kgsl`(786)+`llvm_qcom`(103); `ops_dsp.py` (313), autogen `qcom_dsp` (558).
6. **CL** `[runtime]` — `ops_cl.py` (132), in-cstyle `OpenCLRenderer` (now orphaned after QCOM), autogen `opencl` (811). Tests: `test/device/test_ocl.py`, `external_cl_half_max.py`, `external_osx_profiling.py`, `external_test_image.py`, `external_gpu_fail_osx.py`.
7. **WEBGPU** `[runtime]`/`[codegen]` — `ops_webgpu.py` (221), `renderer/wgsl.py` (117), autogen `webgpu` (2.6k), `test/web/test_webgpu.js`, strip wgsl case in `test_renderer_failures.py`.
8. **METAL** `[runtime]` — `ops_metal.py` (192), `graph/metal.py` (113), `support/objc.py` (73), in-cstyle `MetalRenderer`, autogen `metal` (1.5k); tests `test/device/test_metal.py`, `test/unit/test_metal_graph.py`, `test/unit/test_objc.py`, `external_metal_compile_fail.py`. (Leave `iokit`/`corefoundation` — host `system.py`.)
9. **autogen generator edits** `[runtime]` — `autogen/__init__.py` (drop the `case`/source-dict entries for every deleted module) + `.github/workflows/autogen.yml` (drop the import/regen lines) — exact lines enumerated in the autogen scoper output.

## GATED deletes — each needs ONE small refactor first (then they're clean)

| target | LOC | gate (do this first) |
|---|---:|---|
| `mlx5` + `ops_rdma` + `support/mlx` | ~10.9k | make `graph/hcq.py:9` `RDMACopyQueue` import **lazy** (AMD path imports but never exercises it) |
| `mesa` + `renderer/nir.py` + `compiler_mesa` (after NV+QCOM) | ~11k | resolve why host `CPU`/`NULL` capture pulls `nir`+`mesa` (likely an eager renderer import) — make lazy, or drop `Device['CPU']`/`['NULL']` if the fork doesn't need them |
| `nv_pma` (cupti 14.2k) | ~14.8k | excise the function-local `from extra.nv_pma.decode import decode` branch in `viz/serve.py:501` (keep the file) |
| `torch_backend` | ~2.3k | guard/remove the `try: import extra.torch_backend.backend` in `tinygrad/nn/torch.py:4` |

## BORDERLINE — human call (AMD-family-but-unused, or upstream model zoo)

1. **`hip_gpu_driver` internal headers (~36k of 68.7k)** — `gc_10_3_0_offset.h` (13.6k), `soc21_enum.h` (22.5k) and some `*_sdma_pkt_open.h` show no keep-side ref; autogen loads the AMD headers from the `$AMD` ROCm path, so the `extra/` copies may be redundant. Sub-prune needs a real-autogen build check. KEPT whole for safety.
2. **`hiprtc/` (15.1k)** — AMD-family HIP-RTC header, zero live references. Plan says delete; confirm `autogen/__init__.py` has no dynamic `hiprtc` loader first.
3. **`models/` (4.2k) + `datasets/` (1.4k)** — upstream model zoo used only by non-AMD-decode model/training tests; fork uses `tinygrad/llm/model.py`. Delete with the test surface, but confirm `test/null/test_real_world.py` isn't a kept smoke.
4. **`hcq_smi`, `hcqfuzz`, `fp8`, `mmapeak`, `remote/bench.py`+`serve.py`** — zero/near-zero importers; AMD-family-but-unused vs generic-upstream. Policy call.

## Required edits to KEPT files (not deletions)

`autogen/__init__.py`, `.github/workflows/autogen.yml`, `tinygrad/nn/torch.py` (torch_backend guard), `tinygrad/viz/serve.py` (nv_pma branch), `tinygrad/graph/hcq.py` (gated, RDMA lazy), `renderer/cstyle.py` (in-file class prune each unit), `test/mockgpu/mockgpu.py`+`helpers.py`, and method-level trims in `external_test_hcq.py` / `test_renderer_failures.py`.

## Verification gate (run before, and after EVERY unit)

```
.venv/bin/python -c "from tinygrad import Device; print(sorted(Device._devices))"        # AMD/PYTHON/CPU/NULL present, clean import
DEV=AMD .venv/bin/python -c "from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())"   # [2,3,4]
.venv/bin/python -m pytest test/external/ -q 2>&1 | tail -3                               # fork suite stays green (246)
git grep -n "<deleted_module>" -- tinygrad/ extra/ test/                                  # empty
```
Plus the decode smoke (tiny fixed-seed rollout) after the NV / renderer / codegen-touching units. Never commit red; one owning prefix per commit (`[runtime]` backends, `[codegen]` renderer/uop/autogen, `[test]` extra/tests). Pull-rebase before push.

## Decisions needed before execution

1. **Host devices:** keep `Device['CPU']`/`['NULL']`? (gates `mesa`/`nir`/`ptx`/`wgsl` ~11k). The fork's tests use them, so likely keep → resolve via lazy renderer import, not deletion.
2. **Do the 4 gated refactors?** (unlocks ~38k: mlx5/rdma, mesa/nir, nv_pma, torch_backend).
3. **Borderline policy:** prune AMD-family-but-unused (`hiprtc`, `hip_gpu_driver` redundant headers, `hcq_smi`) or keep for AMD utility? Prune the upstream model zoo (`models`/`datasets`) + its tests?
4. **Execution:** I drive it unit-by-unit (verified, the framework units need care) — start with Unit 1 (`extra/` clean) + the backend units; gated/borderline after your calls.
