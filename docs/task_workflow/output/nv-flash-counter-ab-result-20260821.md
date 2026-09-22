# NV flash score counter A/B result 20260821

## Verdict

The flash score body is **not DRAM-bound** on either side. Tinygrad's
production tile is **SM/pipe-bound at low occupancy**; llama's `flash_attn_ext_vec`
is **latency/occupancy-bound** while streaming K/V from L2. The counters do not
put the flash gap in memory bandwidth, so the in-window `flash_score` exposure
difference is not a bandwidth fix.

The two bodies stay near wall parity (the earlier matched-isolated measurement:
tinygrad 4.16-4.19 us vs llama 4.10 us) by opposite means. Tinygrad executes
about 2.7x more warp instructions and keeps ~60% of an SM pipe busy at ~19%
occupancy; llama executes fewer instructions and stalls to ~7% SM utilization at
~8% occupancy. This is a body-shape difference, not a traffic-bound gap.

## Method

`ncu` under passwordless sudo (required by `RmProfilingAdminOnly=1`), RTX 5090,
sm_120, `--cache-control all`, cold and warm d512 production shapes.

| probe | kernel | grid / block |
| --- | --- | --- |
| tinygrad | `flash_score_tile_peak<32,4,16,1,2,16,8>` | `(48,8,1)` / `(32,4,1)` |
| llama | `flash_attn_ext_vec<128,1,F16,F16,false>` | `(1,4,32)` / `(32,4,1)` |

Tinygrad's cold row was profiled on the harness warmup launch, which loops 64
internal iterations. DRAM and L2 byte totals are dominated by the first cold
iteration; instruction and L1 totals were divided by 64 to get one pass. The
warm 1000-iteration profile independently agrees with the per-pass values.

## Counters, per launch, d512 cold

| metric | tinygrad tile | llama vec | read |
| --- | ---: | ---: | --- |
| DRAM read bytes | 2.20 MB | 0.29 MB | both tiny |
| DRAM write bytes | 0 | 0 | - |
| DRAM throughput, % of peak sustained | 0.76 | 3.19 | neither bound |
| L2 (`lts`) bytes | 3.00 MB | 8.91 MB | llama streams from L2 |
| L1 (`l1tex`) bytes | 2.95 MB | 8.98 MB | - |
| warp instructions | 1.60 M | 0.60 M | tinygrad does ~2.7x more |
| SM throughput, % of peak sustained | 59.79 | 6.73 | tinygrad pipe-heavy |
| achieved occupancy, % | 18.79 | 8.30 | both low |

Warm cross-check (same shapes, warm L2): tinygrad SM 61.80% / occupancy 18.80% /
1.60 M warp instructions per pass; llama SM 6.80% / occupancy 8.30% / 0.60 M.
The classification is stable across cold and warm.

## What this means for the gap

The ledger's `flash_score` in-window mass is `266.75 us` (tinygrad) versus
`74.56 us` (llama). These counters rule out the hypothesis that this is a DRAM
bandwidth limit: both bodies use under 4% of peak DRAM. The tinygrad body is
instruction/pipe-heavy, so its recoverable body win is instruction reduction or
higher occupancy, not more bandwidth. The larger in-window difference remains a
timeline exposure effect: tinygrad serializes these launches while llama hides
them, exactly the conclusion of the shadow-split and overlap work.

No production change or performance promotion follows from this record.

## Labels

- `observed`: all ncu counter values and percentages, kernel identities, grid
  and block shapes, compute capability.
- `inferred`: per-pass tinygrad instruction/L1 values from the 64-iteration
  warmup launch; traffic-versus-occupancy classification.
- `unmeasured`: full-token per-family DRAM attribution for the `5.04 vs 4.70
  GB/token` ledger, `flash_combine` counters, and installed-launch cold-start
  interaction with surrounding graph traffic.

## Evidence

- [counter-ab.json](/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-flash-counter-ab-20260821/counter-ab.json)
- [sha256.txt](/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-flash-counter-ab-20260821/sha256.txt)
- raw ncu CSVs in [evidence/nv-flash-counter-ab-20260821](/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-flash-counter-ab-20260821)
