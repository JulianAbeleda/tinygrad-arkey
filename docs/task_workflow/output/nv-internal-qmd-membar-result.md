# Internal dependent-QMD membar result

## Outcome

The CUDA graph cadence advantage was mostly caused by tinygrad requesting an
L1 system-memory barrier at every grid completion. A grid that already points
to a same-queue dependent QMD does not need that external-completion barrier.
The promoted Blackwell policy clears `cwd_membar_type` only on such internal
QMDs. The final QMD retains the system membar before its timeline signal and
host visibility.

`NV_RELAX_INTERNAL_QMD_MEMBAR=0` is the explicit rollback.

## Causal ladder

| test | control | candidate/reference | result |
|---|---:|---:|---|
| 208 no-op grids | native 141.028 us | CUDA graph 104.128 us | 36.900 us is pure node-service debt |
| remove redundant cache invalidations | 141.028 us | 140.152 us | cache invalidations are not the cause |
| remove internal system membars, no-op | 141.028 us | 99.851 us | native becomes faster than CUDA graph |
| 208 real projection/provider calls | 2686.277 us | 2635.776 us | 50.501 us recovered, 46/46 hashes equal |
| 511 producer/consumer edges | 459.103 us | 332.855 us | zero visibility errors, 126.248 us recovered |
| production reverse bracket | 4178.963 us/token | 4111.433 us/token | -67.530 us/token, identical token hashes |

The producer/consumer test repeatedly overwrites and rereads the same 32 KiB
region with a new value across 256 pairs per replay. It is deliberately more
hostile to stale-cache visibility than the model route.

## Accounting

The fresh production bracket measures 239.294 -> 243.224 tok/s. Applying the
same measured wall movement to the earlier installed authority gives a
continuity estimate of 3992.993 us/token, or 250.439 tok/s. The latter is a
normalized projection, not a new endpoint measurement.

The matched bridge still has about 20 us between relaxed native and CUDA graph.
That residual remains open. It is no longer the highest-recovery explanation
for the token gap.

A second dense Q8 0.6B route also moved in the expected direction
(7328.811 -> 7241.665 us/token). Its token-stream comparison is not a semantic
authority: the two unchanged controls also generated different streams and
even different warmup tokens. Generality is therefore positive for performance
but inconclusive for that harness's cross-process exactness. The independent
producer/consumer visibility gate remains the generic semantic authority.

## Evidence

- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/noop-nv-bound-r2-reverse.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/noop-cuda-r2-reverse.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-nv-membar-internal-none-r1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/membar-semantic-system-r1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/membar-semantic-internal-none-r1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/membar-production-wall-r9.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/membar-dense-q8-06b-wall-r7.json`
