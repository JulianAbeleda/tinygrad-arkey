# NV installed-island Phase 10 corrected causal wall ledger

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

Evidence:
`docs/task_workflow/evidence/nv-installed-islands-20260822/phase10/`

## Verdict

`240_UNMEASURED`. The device-side gap is fully named (zero unexplained
residual), so the accounting closes. But the single largest recoverable term
is the production-conditioned residual `R`, and its causal mechanism is still
unmeasured across Q, K/V, O, and flash. The aggregate legal ceilings exceed
the required recovery, so 240 is not `240_UNCLOSED`; it is `240_UNMEASURED`
until the `R` mechanism is named and one reverse wall bracket is booked.

## Identity closure

```text
wall        = 4771.423 us/token   (fresh Phase 1 control)
240 target  = 4166.667 us/token
required    =  604.756 us/token

node_sum    = 4677.920 us          (596 tinygrad kernels)
union       = 4671.500 us
overlap     =    6.420 us          (node_sum - union)
host_gap    =   99.923 us          (wall - node_sum; single-domain, unmeasured)

tinygrad device delta vs llama      = 793.246 us union (799.666 us node_sum)
```

The nine named census rows reproduce `646.838 us`; the remainder rows
(norm/rope/quant) contribute `152.828 us`. Both are now fully decomposed.

## Disjoint island ledger

Every row is a disjoint census role. `census delta` is tinygrad minus llama in
the node_sum domain. `B/D/R` are the measured exact-body / clean-dispatch /
production-residual terms per token. `legal ceiling` is the projected
recoverable term; it is attribution, never booked recovery.

| island | census delta | body delta | install (D+R) | legal ceiling | mechanism |
| --- | ---: | ---: | ---: | ---: | --- |
| gate/up GEMV | +101.33 | faster (L2-hot) | - | 90.97 | DRAM cold streaming |
| down GEMV | +74.55 | faster (L2-hot) | - | 72.56 | DRAM cold streaming |
| Q projection | +84.41 | -25.2 (coop faster) | +107.4 | 107.4 | R unnamed |
| O projection | +75.23 | +12.7 | +57.6 | 53.5 | R unnamed |
| K/V projection | +112.22 | ~parity | +112.2 | 112.2 | R + completion |
| vocab main+tail | +67.61 | +15.1 | - | 58.3 | reduction topology |
| flash combine | +66.94 | +46.1 | +20.8 | 66.9 | body topology |
| flash score | +64.54 | faster | +94.1 | 94.1 | R unnamed |
| Q head norm | +55.45 | ~+1.4 | +54.0 | 49.3 | launch elimination |
| K head norm | +49.38 | ~+1.4 | +48.0 | 46.8 | launch elimination |
| rope + K/V store | +28.61 | unmeasured | unmeasured | unmeasured | positional |
| norm 4096 (coupled) | +125.44 | +16.4 | +109.0 | net +11.0 | provider-coupled |
| activation quant | -114.43 | - | - | - | tinygrad advantage |
| misc / embedding | +8.38 | unmeasured | unmeasured | unmeasured | positional |
| **total** | **+799.67** | | | **~793** | |

## Named versus unnamed recovery

`MEASURED` The recoverable ceilings split into a named set and an unnamed set:

```text
named mechanisms                          ceiling
  FFN GEMV DRAM streaming                 163.53 us   (measured rate gap)
  flash combine body parity                46.08 us   (measured 2.25x body)
  vocab tail reduction topology            58.27 us   (measured single-warp)
  Q/K head norm launch elimination         96.12 us   (measured D+R)
  attn norm body (secondary, coupled)      40.10 us   (measured 4.86 vs 2.75)
  named sum                               404.10 us

unnamed production-conditioned residual   ceiling
  Q coop/G3 install                       107.4 us
  O residual                               53.5 us
  K/V residual + completion               112.2 us
  flash score residual                     94.1 us
  rope/misc positional                     37.0 us
  unnamed sum                             404.2 us
```

`MEASURED` The named set alone (`~404 us`) reaches
`4771.423 - 404.1 = 4367.3 us` = `229.0 tok/s`, short of 240. Reaching 240
requires approximately `200 us` of the unnamed residual to be real and
recoverable in addition to the named set.

## Critical-path finding

`MEASURED` The retained full-token dependency DAG
(`probe2-tinygrad-capture.json`, 596 nodes, 1230 edges) gives:

```text
serialized node_sum      4677.920 us
critical path (3-queue)  4205.376 us
unexploited overlap       472.544 us
```

tinygrad's installed schedule achieves only `6.420 us` of that overlap. The
critical path (`4205.4 us`) is only `38.7 us` above the 240 target. This is
the strongest quantitative statement of the campaign: tinygrad has `~472 us`
of its own dependency-legal parallelism that the installed schedule
serializes, and llama's PDL hides its equivalent support work on top of the
DRAM-bound GEMVs.

`INFERRED` The `~404 us` unnamed residual and the `~472 us` overlap are two
views of the same gap: support kernels running behind DRAM-bound GEMVs instead
of on top of them. This is a scheduler/concurrency mechanism, not an
arithmetic mechanism. It remains `INFERRED` because no wait-exit timestamps or
PDL counter evidence adjudicated it in this campaign.

## 240 feasibility gate

```text
measured booked recovery            0.000 us
sum of non-overlapping legal ceilings (named)   404.1 us
required recovery to 240           604.756 us
named-set residual to 240          200.7 us
conditional path                    unnamed R (404 us) or overlap (472 us)
```

## Ledger snapshot

```text
node_sum   = 4677.920 us (tinygrad) / 3878.254 us (llama)
union      = 4671.500 us (tinygrad) / 3878.254 us (llama PDL-off)
overlap    = 6.420 us (tinygrad) / 0 us (llama PDL-off)
wall       = 4771.423 us (fresh control)
host_gap   = 99.923 us (single-domain arithmetic residual, unmeasured)
useful_body = unmeasured
booked_recovery = 0.000 us
remaining_to_240 = 604.756 us
```

## Next action

Exactly one: measure the production-conditioned residual `R` against a PDL /
concurrency counter experiment. If `R` is a serialization artifact, a
scheduler/placement scope becomes the primary route to 240 and the verdict
upgrades to `240_BUILDABLE`. This is written as the Phase 11 handoff.
