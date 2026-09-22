# NV unprofiled HCQ dispatch slope result (2026-08-22)

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`
GPU: RTX 5090

## Verdict

`NOT_GLOBAL`. The apparent `~1.4 us` HCQ dispatch tax inferred from the Q/K
norm comparison is not a fixed unprofiled backend cost. In a clean chained
replay of the exact production cubin, the unprofiled HCQ dispatch floor is
`~0.65 us/kernel`, and HCQ profiling timestamps add roughly zero GPU time.

The earlier `2.5 us` Q/K command wall is therefore mostly production-schedule
cache/serialization around the kernel, not a broad QMD-dispatch tax. Fixing
the HCQ command path is not the route to 240 tok/s.

## Measurements

The exact production `reduce_output_rmsnorm_8_128` cubin was replayed as N
chained QMDs on a real `NVComputeQueue`, once plain (unprofiled) and once with
two timestamp semaphores per kernel (profiled). Drain slope is the marginal
GPU time per additional kernel.

| program / arm | drain us/kernel | intercept us |
| --- | ---: | ---: |
| q/k cubin, plain | 1.698 | 4.97 |
| q/k cubin, timestamp-bracketed (reused signals) | 1.495 | 7.31 |
| no-op floor, plain | 0.649 | 5.10 |
| no-op floor, timestamp-bracketed | 0.651 | 5.56 |

The no-op floor isolates the HCQ QMD dispatch/tail cost: about `0.65 us`.
The q/k plain slope is that floor plus the hot kernel body. A faithful
distinct-signal HCQ profile replica reports a clean-chain per-kernel duration
of `1.696 us` median at N=64 and N=128 (mean `1.71..1.75 us`), matching the
plain slope and confirming the timestamp commands add no GPU tax. The reused-
signal arm's lower `1.495 us` is a queue-front-end artifact of that variant,
not the faithful profile; both bounds are far from the production `2.5 us`.

## Decomposition

```text
unprofiled HCQ dispatch floor       ~0.65 us/kernel
q/k plain total slope              =1.70 us/kernel
q/k hot body (plain - floor)       ~1.05 us/kernel
reference CUDA-driver body          1.18..1.20 us/kernel
profiling timestamp GPU tax        ~0.00..0.15 us/kernel

production Q/K command wall         2.50 us/kernel
clean-chain profiled duration      -1.70 us/kernel
production schedule/cache residual ~0.80 us/kernel
```

The `0.50..0.65 us` spread on the dispatch floor comes from whether the
q/k hot body is measured with the CUDA-driver reference (`1.196 us`) or the
no-op floor geometry. Either way it is less than half of the inferred
`~1.4 us`, and profiling does not create the remainder.

## 240 tok/s arithmetic

Even if every one of the 596 launches exposed the full `0.65 us` floor, the
maximum removal is:

```text
596 kernels x 0.65 us = 386.8 us
4747.5 - 386.8        = 4360.7 us/token = 229.3 tok/s
required for 240      = 580.8 us removal
shortfall             = 194.0 us
```

The HCQ command path alone is ~19 tok/s at the absolute best, not the 30 tok/s
needed. In production the larger projection/flash kernels already hide most of
their dispatch behind body execution, so the realistic recoverable amount is
smaller still. Q/K fusion remains a separate `~101 us / ~4-5 tok/s` change,
also not parity.

## Decision

The measured slope puts the decision in the "not global" branch:

1. A backend-level QMD-chain fix is not the prerequisite to 240.
2. Q/K fusion stays a modest optimization, not the parity route.
3. The remaining work returns to the production Q/K schedule/cache residual
   and the projection/flash body and serialization rows in the frozen census.

No production model, renderer, scheduler, or runtime code was changed.

## Evidence

- Dispatch slope: `docs/task_workflow/evidence/nv-hcq-dispatch-slope-20260822/nv_hcq_dispatch_slope.json`
- Per-kernel profile replica: `docs/task_workflow/evidence/nv-hcq-dispatch-slope-20260822/nv_hcq_profile_per_kernel.json`
- Tools: `extra/llm_research/decode/nv_hcq_dispatch_slope.py`, `extra/llm_research/decode/nv_hcq_profile_per_kernel.py`
