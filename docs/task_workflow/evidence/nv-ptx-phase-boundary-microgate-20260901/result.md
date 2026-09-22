# NV PTX phase-boundary synthetic microgate (2026-09-01)

## Verdict: `SYNTHETIC_PASS_RELEASE_ONE_REAL_Q6_STATIC_RERUN`

`split_register_bridge` satisfies deterministic cubin, exact traffic, barrier ordering, <=160 span, and spill gates. It releases one CUDA-only real-Q6 static compile, not execution or timing.

| Arm | LDG->STS span | REG | Stack | Instructions | LDG opcode | Stable | Hard pass |
|---|---:|---:|---:|---:|---|---:|---:|
| `regionload_control` | 79 | 64 | 0 | 864 | `{'LDG.E': 18}` | True | False |
| `split_register_bridge` | 104 | 64 | 0 | 832 | `{'LDG.E': 18}` | True | True |
| `fused_ptx_region` | 104 | 64 | 0 | 832 | `{'LDG.E': 18}` | True | True |

## Contract

- All arms use sm120, launch_bounds(256), one 64-accumulator/8-FMA live phase body, 18 affine global-to-shared copies, and 18 downstream cross-thread shared consumers.
- Exact physical census is 18 LDG, 18 STS, 18 LDS, 1 STG, and 2 BAR; no stack/local/LDL/STL/MEMBAR/ATOM traffic is admitted.
- Candidate loads must precede overwrite BAR0 and candidate stores must lie strictly between overwrite BAR0 and publication BAR1.
- Source contains no const/restrict pointer, volatile C memory object, noinline helper, or function split. Inline PTX asm is volatile by design.

## Toolchain

- NVRTC: `13.2`
- NVRTC options: `['--gpu-architecture=sm_120', '-I/usr/local/cuda/include', '-I/usr/include', '-I/opt/cuda/include', '--minimal']`
- nvdisasm: `Cuda compilation tools, release 12.8, V12.8.55`
- cuobjdump: `Cuda compilation tools, release 13.2, V13.2.86`

This is a CUDA/PTX compiler-boundary experiment only. No kernel was launched and no Q6 or production source was modified.
