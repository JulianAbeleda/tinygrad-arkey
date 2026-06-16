# Hard-Fork Prune Plan — non-AMD surface removal

**Decision recorded:** this is a **hard fork** (no upstream merges) → the
divergence cost is gone, so pruning the non-AMD surface in-place is justified.
Target ≈ **120k+ LOC** (autogen non-AMD ~91k + non-AMD `extra/` ~26k + non-AMD
backends ~3k + their tests). See [upstream-audit-2026-06-16.md](upstream-audit-2026-06-16.md).

**This is surgery, not a sweep.** Backends are dynamically loaded (`device.py`
scans `runtime/ops_*.py`), but autogen is shared and cross-referenced. Two grep
traps already hit — do not trust a single grep:
- `nv_580` looked zero-import but `ops_nv.py:13` imports it on a multi-name line
  (`from ...autogen import nv_570, nv_580, mesa`).
- A `\bnv\b` pattern false-*matched* `ops_amd.py` (substring `navi`). **Verify every
  deletion against the AMD path.**

## Keep-set (the AMD + host floor — never delete)

- **Backends:** `ops_amd`, `ops_python`, `ops_cpu`, `ops_disk`, `ops_null`,
  `ops_npy`, `ops_tinyfs`. Keep `ops_hip` (AMD-family, tiny) unless proven unused.
- **AMD/host autogen:** `amd_gpu`, `amdgpu_drm`, `amdgpu_kd`, `kfd`, `hsa`, `sqtt`,
  `am/*`, `pci`, **`libc` (21 users)**, **`llvm` (4)**, `comgr`, `libclang`,
  `io_uring`. (`libc`/`llvm`/`pci` are shared host infra — KEEP.)
- **Renderers/support the AMD path uses** (verify before touching any renderer).
- The fork-added `extra/qk_*`/`extra/llm_*` + `tinygrad/llm` decode/codegen diff.

## Shared-autogen ordering traps (delete users BEFORE the shared module)

| shared autogen | LOC | users (all must go first) |
|---|---:|---|
| `mesa` | 10,532 | `ops_nv`, `ops_qcom`, `renderer/nir.py`, `support/compiler_mesa` |
| `cuda` | 2,155 | `ops_cuda`, `support/compiler_cuda`, `extra/gemm/*`, `extra/hook_cuda`, `extra/nv_pma` |
| `nv_570` | 24,866 | `ops_nv`, `support/nv/ip.py`, `extra/nv_gpu_driver`, `extra/hevc`, `test/mockgpu/nv/*` |
| `opencl` | 811 | `ops_cl`, `extra/archprobe`, … |

## Ordered delete-units (each = one commit, verified, dependency-order)

Run **Phase 0** verification first and after EVERY unit (below). Order matters:
extra/ importers → backend → its now-orphaned autogen → shared autogen last.

1. **`extra/` non-AMD bindings/tools (~26k)** `[test]` — `nv_pma` (14.7k),
   `gemm/{cuda,metal,max}_matmul` (keep `cdna_asm_gemm`), `torch_backend` (2.2k),
   `qcom_gpu_driver` (1.4k), `dsp` (1.3k), `nv_gpu_driver`, `hevc`, `hook_cuda`,
   `archprobe`, `mlx_driver`, `usbgpu`, `torch_hook`, `testsig`, `mmapeak`,
   `optimization`, `perfetto`, `viz` + the upstream example orphans. (Removes the
   `extra/` static importers of non-AMD backends/autogen.)
2. **NV stack (~57k)** `[runtime]` — `ops_nv`, `support/nv/`, `support/compiler_cuda`
   (if NV-only), autogen `nv_580`+`nv`+`nvrtc`+`nvjitlink` (+`nv_570` once units 1 &
   `test/mockgpu/nv` are gone), `test/mockgpu/nv/*`, `external_test_hcq` NV parts,
   and the `nv_570`/`nv_580`/`nv` lines in `autogen/__init__.py` + `.github/workflows/autogen.yml`.
3. **CUDA (~2.5k)** `[runtime]` — `ops_cuda`, `cuda`/`nvrtc`/`nvjitlink` autogen (now orphaned), `__init__`/workflow.
4. **QCOM + DSP (~3k)** `[runtime]` — `ops_qcom`, `ops_dsp`, `kgsl`, `qcom_dsp` autogen.
5. **CL + WEBGPU + METAL (~6k)** `[runtime]` — `ops_cl`/`opencl`/`cl`, `ops_webgpu`/`webgpu`/`renderer/wgsl`, `ops_metal`/`metal` (+ `renderer/` entries; verify AMD doesn't use them).
6. **MESA (~12k)** `[codegen]` — only after NV+QCOM gone: `mesa` autogen, `renderer/nir.py`, `support/compiler_mesa` (verify nothing AMD imports `nir`).
7. **RDMA/mlx5 (~12k)** `[runtime]` — `ops_rdma`, `mlx5`, `extra/remote`/networking (verify not used by the decode path).
8. **non-AMD tests** `[test]` — `test/` backend/hw tests for removed backends (KEEP AMD hw tests under `test/amd/`).

## Verification gate (Phase 0 — run before, and after EVERY unit)

```
.venv/bin/python -c "from tinygrad import Device; print(sorted(Device._devices))"   # AMD/PYTHON/CPU present, no import error
DEV=AMD .venv/bin/python -c "from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())"   # [2, 3, 4]
.venv/bin/python -m pytest test/external/ -q 2>&1 | tail -3   # fork suite stays 246 (minus intentionally-deleted non-AMD tests)
git grep -n "<deleted_module>" -- tinygrad/ extra/ test/   # empty
```
Plus a real decode smoke (the tiny fixed-seed rollout) after the NV/MESA/renderer units, since those touch codegen. **Never commit red. One owning prefix per commit** (`[runtime]` backends, `[codegen]` renderers/autogen-uop, `[test]` extra/tests). Pull-rebase before push.

## Recommendation

Execute **unit-by-unit, verified** (not one blast) — the AMD path is the fork's
reason to exist and the entanglement (shared autogen + renderers + grep traps) is
real. Start with Unit 1 (`extra/` non-AMD, lowest risk, ~26k) to prove the gate,
then the NV stack (biggest, ~57k). Each unit is a clean, separately-verifiable
commit. Could be driven by a per-unit Codex packet, but the framework units
(2,6,7) need careful AMD-path verification each.
