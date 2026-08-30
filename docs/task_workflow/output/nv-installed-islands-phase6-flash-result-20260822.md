# NV installed-island Phase 6 flash score, combine, and attention handoff

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

Evidence: `docs/task_workflow/evidence/nv-installed-islands-20260822/phase6/`

## Findings, ordered by wall severity

`MEASURED` The attention island `I_ATTN` has 72 tinygrad nodes and 108 llama
nodes. Node-sum attribution is `+107.451 us/token`, split exactly as:

```text
flash score   tinygrad 227.488 - llama 162.948 = +64.540 us
flash combine tinygrad 104.000 - llama  37.057 = +66.943 us
```

These are disjoint census rows, not additive wall recovery.

### Flash score

`MEASURED` tinygrad's score body is **faster** than llama's, but its installed
command interval is slower. The exact production cubin
`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` runs at
`3.840 us` median (2000 nsys reps), versus llama's `4.512 us` body. The
production `PROFILE=1` command interval is `6.272 us` median (14436 samples):

| field | tinygrad | llama | delta |
| --- | ---: | ---: | ---: |
| exact body (nsys) | 3.840 us | 4.512 us | -0.672 us (tinygrad faster) |
| production command interval | 6.272 us | 4.512 us (body) | +1.760 us/call |
| clean HCQ drain slope | 3.658 us | n/a | n/a |

`MEASURED` The positive installed delta is **not** arithmetic and **not** clean
dispatch. `P - B = 2.432 us/call`; the production-conditioned residual
`R = P - C = 2.614 us/call` dominates. The clean HCQ slope `C = 3.658 us`
sits below the CUDA-runtime-replayed body `B = 3.840 us`, so the difference is
instrumentation, not removable dispatch; `D` is clamped to zero and the whole
delta is assigned to `R`.

`MEASURED` Cold (cache-flushed) NCU replay lands at `6.08 us`, close to the
production `6.272 us`, and shows the shape: `2.13 MB` DRAM read, `19.07%`
DRAM throughput, `16.66%` warps active, `16.79%` achieved occupancy, grid
`384` blocks x `128` threads (0.21 waves). The score is a low-occupancy,
latency-shaped kernel; its production penalty is the DRAM-cold/back-to-back
producer boundary, not a body rewrite.

### Flash combine

`MEASURED` combine is the opposite: the body itself is the slow term.
`flash_fused_gmax_combine_f16_32_128` runs at `2.304 us` median versus llama's
`1.024 us` body (`+1.280 us/call`, `2.25x`). Production command interval is
`2.880 us`:

```text
P 2.880 = B 2.304 + D 0.302 + R 0.274
```

`MEASURED` The body term is `80%` of `P` and `69%` of the installed delta
versus llama. NCU shows why: `0.79%` SM throughput, `2.08%` warps active,
`2.08%` achieved occupancy, grid `32` blocks x `32` threads (0.01 waves).
This is a single-warp, latency-bound body-shape problem, not a launch or
bandwidth wall.

## Verdicts

```text
flash score body        BODY_FASTER        (-0.672 us/call vs llama)
flash score mechanism   INSTALL_DOMINANT   (R = 2.614 us/call dominates)
combine body            BODY_DOMINANT      (2.304 us, 2.25x llama, 80% of P)
combine clean dispatch  DISPATCH_CLEAR     (D = 0.302 us/call)
complete island span    UNMEASURED         (no wait_exit timestamps)
```

The `Q/K/V-ready -> O-input-ready` installed span is left `UNMEASURED` because
this phase has no wait-exit timestamps for the producers. The prior
`nv-flash-counter-ab-20260821` conclusion (score is SM/pipe-bound at low
occupancy, not DRAM-bound at peak) is consistent with the cold NCU counters
above.

## Like-for-like comparison

| field | tinygrad | llama | note |
| --- | ---: | ---: | --- |
| score physical cardinality | 36 | 36 | matched |
| combine physical cardinality | 36 | 36 | matched |
| score installed command interval | 6.272 us | unmeasured | llama has no separate command boundary |
| score exact body | 3.840 us | 4.512 us | nsys exact cubin vs PDL0 body |
| combine installed command interval | 2.880 us | unmeasured | llama folds combine into attention |
| combine exact body | 2.304 us | 1.024 us | nsys exact cubin vs PDL0 body |

## Wall sensitivity

`INFERRED` Legal, non-double-counted ceilings per token:

```text
score residual   36 x 2.614 = 94.1 us   (attribution; not booked)
combine body     36 x 1.280 = 46.1 us   (body parity to llama)
combine dispatch 36 x 0.302 = 10.9 us   (launch elimination)
combine residual 36 x 0.274 =  9.9 us   (boundary)
```

The score body is already faster than llama, so there is no score arithmetic
recovery to take. The combine body is the cleanest measured target in this
island. Neither row is booked.

## Decision

```text
score    INSTALL_DOMINANT  -> score->combine / producer->score handoff scope
combine  BODY_DOMINANT     -> latency-bound combine body/topology scope
```

One non-double-counted island ranking is deferred to Phase 10, where the
combine body and the flash score residual are placed against the FFN, O, and
Q/K ceilings on a single alternate-path ledger.

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
