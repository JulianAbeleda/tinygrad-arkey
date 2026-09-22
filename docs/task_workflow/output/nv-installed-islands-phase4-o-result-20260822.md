# NV installed-island Phase 4 O projection decomposition

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

Evidence: `docs/task_workflow/evidence/nv-installed-islands-20260822/phase4/`

## Findings

`MEASURED` The O projection island (`I_O`, 36 semantic calls, 36 physical
kernels on both sides) is **not** an arithmetic problem. The exact production
cubin body is `7.584 us` median (2000 reps), versus llama's O body of
`7.232 us` (36 PDL-off nsys nodes). The body delta is `+0.352 us/call`
(`+4.9%`), inside the 10% parity band.

`MEASURED` The installed delta is dominated by the production-conditioned
residual. Fresh `PROFILE=1` production command interval is `9.184 us` median
(14436 samples). The clean chained HCQ drain slope is `7.698 us/kernel`, so:

```text
P 9.184 = B 7.584 + D 0.114 + R 1.486   (identity residual 0.000)
```

`R` is `92.9%` of the installed delta `P - B = 1.600 us`. `D` (clean
front-end/dispatch) is only `0.114 us`. This is `INSTALL_DOMINANT`: the
flash-combine-to-O producer/consumer handoff, not the GEMV body.

`MEASURED` Counters confirm the body is neither DRAM- nor pipe-bound at peak:
`9.47 MB` DRAM read per call (the 9 MB Q4 weight matrix, read once),
`0` DRAM write, `11.34 MB` L2 traffic, `32.22%` SM throughput,
`39.80%` achieved warps, `39.75%` issue-active. This is a latency/occupancy
shaped GEMV, not a bandwidth wall.

## Verdicts

```text
O body                    BODY_PARITY       (+0.352 us/call, +4.9%)
O clean dispatch          DISPATCH_CLEAR    (D = 0.114 us/call)
O installation mechanism  INSTALL_DOMINANT  (R = 1.486 us/call, 92.9%)
```

## Like-for-like comparison

| field | tinygrad | llama | note |
| --- | ---: | ---: | --- |
| physical cardinality | 36 | 36 | matched |
| installed command interval | 9.184 us | unmeasured | llama has no separate combine->O command boundary |
| exact body | 7.584 us | 7.232 us | nsys exact cubin vs PDL0 body |
| clean HCQ drain slope | 7.698 us | n/a | plain QMD chain |

## Wall sensitivity

`INFERRED` Zero-cost ceiling is `36 x 1.600 = 57.6 us/token`, but the legal
mechanism ceiling is the production residual only:

```text
legal O ceiling = 36 x 1.486 = 53.5 us/token   (before alternate-path takeover)
```

This is attribution, not booked recovery. It is above the `20 us/token`
demotion threshold, so O is retained for the 240 campaign. The selected
mechanism is a flash-combine-to-O typed-boundary/handoff scope, **not** an
arithmetic rewrite: an SM120-native Q4 topology would save at most the
`0.352 us/call` body delta (`12.7 us/token`) and leaves the 1.486 us residual
untouched.

## Decision

`INSTALL_DOMINANT`. The follow-on scope is a flash-combine-to-O
typed-boundary/handoff scope with a reverse wall bracket, targeting the
production-conditioned residual `R`, not the GEMV topology.

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
