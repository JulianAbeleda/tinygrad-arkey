# Flash score exact common-protocol result

## Decision

The installed tinygrad Flash score body is not slower than llama. The prior
approximately 30--31 us/token row difference compared incompatible timing
boundaries and is not kernel-body recovery.

Under one CUDA context, stream, allocation shape, input state, extent, and
timer, the exact installed binaries measure:

| state | tinygrad exact NVRTC cubin | llama exact release cubin | tiny - llama |
|---|---:|---:|---:|
| hot, CUDA-event A/B/B/A | 4.191 us | 4.264 us | **-0.074 us/layer** |
| 96-MiB disturbed, CUPTI active | 4.256 us | 4.384 us | **-0.128 us/layer** |

The zero-valued FP32-Q/FP16-KV semantic arm produced finite llama metadata and
an exactly zero output vector. Both kernels use S6, 192 CTAs, 128 threads, one
Q head per CTA, and a physical extent of 768 tokens.

Binary authorities:

- tinygrad NVRTC cubin SHA-256 `b18b18d244ae9f8b65e7bdb4627f2f1915b6b8eeb3a9b9cd1dc56b340eb14fbe`;
- llama release cubin SHA-256 `cc76cd433d9b56dc83fc80813631c450dd703972cdd994f9c193124d3af263e9`.

## Cause of the false debt

The tinygrad production row is a native HCQ timestamp-to-timestamp QMD
interval. The llama row is CUPTI kernel-active start-to-end time. The former
contains timestamp/boundary service that the latter excludes. The uniform
per-layer difference was the signature of this fixed boundary cost.

| exact current tinygrad measurement | result |
|---|---:|
| CUDA/CUPTI hot active body | 3.872 us |
| native HCQ hot timestamp interval | 4.432 us |
| interval minus active body | about 0.560 us/launch |
| native full-entry interval | 4.960 us |

The chained unprofiled HCQ slope is 4.143 us/kernel versus a 4.127-us CUDA
event slope. Native unprofiled execution is therefore not 0.56 us slower per
kernel; most of that difference belongs to per-kernel timestamp instrumentation.

The quoted llama 162.948-us row was also a low replay rather than its retained
median of about 163.587 us, but this aggregation error was small relative to
the boundary mismatch.

## Invalidated probe

The old standalone llama probe supplied FP16 Q storage to a kernel that reads
FP32 `float2`, declared one Q head while launching 32, and used invalid head
strides. Its old isolated timings are not semantic authorities. The probe now
uses FP32 Q, correct dimensions/strides, and output validation.

## Ledger consequence

- Remove the 30--31 us/token Flash score row from demonstrated body debts.
- Do not subtract it from the endpoint: this is an accounting correction, not
  a new production optimization.
- The cross-runtime 47.570-us device-node-sum difference is not a clean body
  comparison because its rows use mixed timing boundaries.
- The endpoint remains 4.060523 ms/token / 246.274 tok/s versus llama at
  4.021721 ms/token / 248.711 tok/s.
- Flash SASS scheduling is closed. Localize the remaining wall gap using
  common-protocol rows or causal wall brackets, beginning with vocab and norms.

## Evidence

- `docs/task_workflow/evidence/nv-flash-common-protocol-20260827/exact-common-hot-r4.json`
- `docs/task_workflow/evidence/nv-flash-common-protocol-20260827/exact-common-cold96-r4.json`
- `docs/task_workflow/evidence/nv-flash-common-protocol-20260827/exact-common-cold96-cupti.json`
- `docs/task_workflow/evidence/nv-flash-common-protocol-20260827/tiny-exact-entry.json`
- `docs/task_workflow/evidence/nv-flash-common-protocol-20260827/tiny-exact-hcq-slope-r5.json`
