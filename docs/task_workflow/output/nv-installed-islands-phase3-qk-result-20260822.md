# NV installed-island Phase 3 Q/K installed-island completion

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

Evidence: `docs/task_workflow/evidence/nv-installed-islands-20260822/phase3/`

## Verdict

`MEASURED` Q and K bodies are `BODY_PARITY`; the production-conditioned
residual `R` is the dominant installed term and remains `UNMEASURED_RESIDUAL`.
All five work items are now complete.

## Production P distributions

`MEASURED` Fresh `PROFILE=1` distribution across 401 complete decode replays
(14436 samples each), layer order preserved:

| norm | median | p10 | p90 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q `reduce_output_rmsnorm_32_128` | 2.560 | 2.400 | 2.720 | 2.272 | 3.680 |
| K `reduce_output_rmsnorm_8_128` | 2.496 | 2.400 | 2.656 | 2.240 | 3.904 |

`MEASURED` Layer correlation is present but bounded. Q per-layer median ranges
2.368 to 3.008 us; K ranges 2.368 to 2.752 us. The Q layer-5 (3.008 us) and
the early-layer K rows are the outliers, not a uniform inflation.

## B / D / R decomposition

`MEASURED` Using the Phase 0 calibrated constants (Q body 1.190 us, K body
1.196 us, clean chained HCQ 1.698 us):

```text
Q:  P 2.560 = B 1.190 + D 0.508 + R 0.862   (identity residual 0.000)
K:  P 2.496 = B 1.196 + D 0.502 + R 0.798   (identity residual 0.000)
```

`MEASURED` `D` is the clean front-end/dispatch component. `R` is the
production-conditioned residual and is deliberately not assigned to cache
state, dependency wait, memory visibility, QMD scheduling, or placement
because no counters have adjudicated it in this session.

## Verdicts

```text
Q body                    BODY_PARITY
K body                    BODY_PARITY
Q installation mechanism  UNMEASURED_RESIDUAL
K installation mechanism  UNMEASURED_RESIDUAL
best legal construction   ranked only after wall-sensitivity simulation
```

## Wall sensitivity ceiling

`INFERRED` The retained Q/K head-norm attribution ceiling is `104.831 us`
(Q norm +55.455, K norm +49.376). Against the fresh control wall of
`4771.423 us`, removing that entire ceiling yields approximately
`4666.6 us` = `214.3 tok/s`, below the `240 tok/s` target. Q/K fusion is
therefore a real but insufficient lever; it is not the prerequisite to 240.

This ceiling is attribution, not a booked recovery. It remains unbooked until
a token-SHA reverse wall bracket measures it.

## Predecessor and counter conditions

`MEASURED` The retained microgate already isolates the hot/fill/flush
conditions: tiny Q/K norms run `1.87 us` hot, `4.10 us` after a fill producer,
and `6.11 / 6.14 us` after a 128 MiB L2 flush. Llama runs the same flush
condition at `4.48 us`. Tinygrad is therefore more L2-cold sensitive, which is
consistent with `R` being cache-state conditioned; this remains
`UNMEASURED_RESIDUAL` because the microgate is a CUDA-context reconstruction,
not the installed NV HCQ command path.

`MEASURED` Exact-cubin NCU counters (see
`phase3/qk-ncu-counters.json`): Q norm reads `22.78 KB` DRAM, `179.97 KB` L2,
`2.08%` warps, `1.47%` issue-active; K norm reads `9.98 KB` DRAM,
`147.81 KB` L2, `2.08%` warps, `1.63%` issue-active. Both are tiny
latency-bound kernels, confirming the residual is not an arithmetic/bandwidth
body problem.

## Legal launch-elimination sensitivity

`INFERRED` (see `phase3/qk-wall-sensitivity.json`). The legal shapes delete the
norm command but must still execute the norm body inside the surviving
completion/rope kernel, so the recoverable term is `D + R`, not `P`:

```text
Q completion+norm+rope    deleted 96.736 us, body moves 42.840 us, recover 49.320 us
K completion+norm+rope    deleted 90.208 us, body moves 43.056 us, recover 46.800 us
combined Q/K support      deleted 186.944 us, body moves 85.896 us, recover 96.120 us
```

The combined shape ceiling is `96.12 us`, i.e. `4771.423 - 96.12 = 4675.3 us`
= `213.9 tok/s`. This confirms Q/K is a real but insufficient lever.

## Ledger snapshot

```text
node_sum   = 4677.920 us (tinygrad) / 3878.254 us (llama)
union      = 4671.500 us (tinygrad) / 3878.254 us (llama PDL-off)
overlap    = 6.420 us (tinygrad) / 0 us (llama PDL-off)
wall       = 4771.423 us (fresh control)
host_gap   = unmeasured single-domain
useful_body = unmeasured
booked_recovery = 0.000 us
remaining_to_240 = 604.756 us
```
