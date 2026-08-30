# NV installed-island Phase 1 fresh endpoint and profiler-tax bracket

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`
GPU: RTX 5090, UUID `GPU-c800ade9-21ea-2e55-f75c-6d7a458fb186`

Evidence: `docs/task_workflow/evidence/nv-installed-islands-20260822/phase1/`

## Verdict

`MEASURED` The fresh control replaces the working tinygrad wall with
`4771.423 us/token`, and the profiler tax is `+1821.731 us/token`
(+38.18%). Profiling is material on this workload and cannot be treated as a
small uniform correction.

## Fresh endpoint walls

All arms are fresh processes, locked clocks (2842 / 14001 MHz, P0), depth 512,
serialized under `flock /tmp/gpu-bench.lock`. Tinygrad token SHA is identical
across all three arms:
`dbd3026bb808fd57a5c7466963b0309168ee3d5ca58ccaf829cacdea9e0d491b`.

| arm | wall us/token | tok/s |
| --- | ---: | ---: |
| tinygrad control A (unprofiled) | 4776.511 | 209.36 |
| tinygrad candidate (PROFILE=1) | 6593.154 | 151.67 |
| tinygrad control B (unprofiled) | 4766.335 | 209.80 |
| llama PDL-on (default) | 4037.319 | 247.75 |
| llama PDL-off (`GGML_CUDA_PDL=0`) | 4352.438 | 229.76 |

`MEASURED` The unprofiled controls bracket stably: spread `10.176 us`, midpoint
`4771.423 us`. The profiler-tax delta is therefore
`6593.154 - 4771.423 = +1821.731 us/token` (+38.18%).

`MEASURED` The fresh tinygrad-to-llama (PDL-on) gap is
`4771.423 - 4037.319 = +734.104 us/token`. This is the same-session authority
gap for the rest of the campaign.

## Updated recovery budget

```text
working control wall    4771.423 us/token
240 tok/s target        4166.666667 us/token
remaining_to_240        604.756 us/token
```

The prior scope-carryover control of `4747.5 us` is superseded by
`4771.423 us` for this session.

## Installed timeline aggregates

`MEASURED` tinygrad decode replay cardinality is `596` nodes across the
canonical group pattern `[32, 64, 128, 256, 116]`, matching the locked role
census. In the profiled domain the steady median is:

```text
node_sum   = 4619.712 us
union      = 4614.000 us
overlap    = 5.023 us
span       = 5674.500 us
```

`MEASURED` llama PDL-off has `762` nodes with profiled
`node_sum = 3897.896 us` and `span = 4039.689 us`. The two node counts and
group counts are the matched boundary basis for Phase 2.

## Profiler observable statement

`MEASURED` The HCQ graph profile inserts two GPU timestamp semaphores per
kernel and reports `duration = end_timestamp - start_timestamp`. That interval
contains the kernel body plus the timestamp-command front-end/dispatch
resolution and any dependency wait. It is a body+front-end composite, not a
pure body measure. This is the same relationship quantified in Phase 0
(Q/K clean chained HCQ `1.698 us` versus faithful per-kernel profile
`1.696 us`), but here it is the whole-token tax of `+1821.731 us`, so the tax
is never silently subtracted from individual rows.

## Ledger snapshot

```text
node_sum   = 4619.712 us (tinygrad, profiled domain)
union      = 4614.000 us (tinygrad, profiled domain)
overlap    = 5.023 us (tinygrad, profiled domain)
wall       = 4771.423 us (tinygrad, fresh unprofiled control)
host_gap   = unmeasured in a single domain (profiled wall 6593.154 vs unprofiled 4771.423)
useful_body = unmeasured
booked_recovery = 0.000 us
remaining_to_240 = 604.756 us
```
