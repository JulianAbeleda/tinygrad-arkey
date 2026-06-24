# EBT-1 RESULT — Lane A KILL: HIP runtime and tinygrad HCQ/KFD are mutually exclusive in one process

Executed Phase EBT-1 of `prefill-external-rawhip-tensile-boundary-scope-20260619.md` (Lane A: HIP-runtime BLAS on
tinygrad `HCQBuffer.va_addr` pointers, no copies). **Verdict: KILL.** The blocker is not pointer validity — the HIP
runtime cannot even initialize in the same process as tinygrad `DEV=AMD`. Research-only; no model route, no defaults,
decode untouched. Probe: `extra/qk_prefill_bridge_{shim.cpp,interop.py}`; artifact:
`bench/qk-prefill-external-bridge/interop.json`.

## What was built
- C shim (`extra/qk_prefill_bridge_shim.cpp`, host-compiled `.so`, ctypes): `hipPointerGetAttributes` + `rocblas_gemm_ex`
  directly on caller-supplied VA pointers; row-major `C[M,N]=A·B` via the col-major swap.
- Python driver (`extra/qk_prefill_bridge_interop.py`): allocates A/B/C as **tinygrad AMD tensors**, extracts
  `t.uop.buffer._buf.va_addr`, `Device.synchronize()`, calls the shim, verifies vs a tinygrad fp16 oracle.

## Result [M] (ffn_gate/up 512×4096→12288, tinygrad-owned VA pointers)

| check | result |
|---|---|
| `hipPointerGetAttributes(A/B/C)` | **err 100 = hipErrorNoDevice** for all three |
| `rocblas_create_handle` | **failed** (no device) |
| GEMM ran | no |
| correctness | n/a (gemm did not run) |
| timing | n/a |

**Gates: hip_accepts_pointers=False, gemm_ran=False → KILL.**

## Root cause (proven both directions) [M]

The failure is mutual exclusion of the two device-management stacks, confirmed by a direct coexistence probe
(`hipGetDeviceCount` via ctypes vs tinygrad init):

- **HIP first:** `hipGetDeviceCount` → err 0, **count = 1** (HIP sees the GPU). Then importing/initializing tinygrad
  `DEV=AMD` **fails** ("AMD is not available") — tinygrad's KFD/amdgpu open is locked out because HIP/ROCr already
  claimed the device.
- **tinygrad first (the EBT-1 run):** tinygrad opens the GPU via KFD/HCQ; the in-process HIP runtime then reports
  **hipErrorNoDevice (100)** — it sees no usable device, so rocBLAS can't init and never touches the pointers.

So it is **not** that HIP rejects HCQ/KFD VA pointers — HIP never gets far enough to look at them. tinygrad `DEV=AMD`
(KFD/HCQ/HSA, `tinygrad/runtime/ops_amd.py`) and the HIP runtime (ROCr) **cannot share the GPU in one process**;
whichever initializes first excludes the other. The standalone PXB-1 ceiling worked precisely because it was a
*separate, HIP-only* process.

## Verdict + next step

**EBT-1 KILL. Close HIP-runtime Lane A** (in-process rocBLAS/hipBLASLt on tinygrad pointers is impossible without a
separate process — which reintroduces IPC/copies and defeats the no-copy goal). The external BLAS ceiling
(~70 TFLOPS) is real but **unreachable through the HIP runtime inside `DEV=AMD`.**

The only remaining external path is **Lane B — extract the Tensile HSACO and load it through tinygrad's HCQ**
(`Ops.PROGRAM`/custom_kernel raw-code, precedent `extra/qk_wmma_custom_smoke.py`), i.e. use rocBLAS's *compiled
kernel* without the rocBLAS/HIP *runtime*. That sidesteps the mutual-exclusion wall (no HIP runtime in-process) but
carries its own cost: selecting/extracting the right per-shape Tensile kernel + its launch params + arg layout, and
it is still an external-artifact dependency. Recommendation: **only pursue Lane B if the project accepts a
Tensile-HSACO artifact dependency**; otherwise the prefill matmul rests at the pure-tinygrad ceiling (~42 TFLOPS /
PREFILL_V2 ~70–83% llama) with the external ~1.34× pp now characterized but not bridgeable in-process.

Follow-on scope: `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md`.

## Exact commands
```sh
# build shim (.so)
g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -shared -fPIC -I/opt/rocm-7.2.4/include -L/opt/rocm-7.2.4/lib \
    -Wl,-rpath,/opt/rocm-7.2.4/lib extra/qk_prefill_bridge_shim.cpp -lamdhip64 -lrocblas -o /tmp/qk_bridge.so
# run EBT-1
DEV=AMD ROCBLAS_TENSILE_LIBPATH=/opt/rocm-7.2.4/lib/rocblas/library LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib \
    PYTHONPATH=. .venv/bin/python extra/qk_prefill_bridge_interop.py
```

## Files
`extra/qk_prefill_bridge_shim.cpp`, `extra/qk_prefill_bridge_interop.py`, `bench/qk-prefill-external-bridge/interop.json`,
this doc. Provenance: `prefill-external-rawhip-tensile-boundary-scope-20260619.md`, `prefill-external-blas-result-20260619.md`.
No kernel/model/default changes.
