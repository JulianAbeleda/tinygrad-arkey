# NV installed-island Phase 5 gate/up and down FFN island

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

Evidence: `docs/task_workflow/evidence/nv-installed-islands-20260822/phase5/`

## Findings, ordered by wall severity

`MEASURED` The FFN island is **DRAM-bound on both sides**, and tinygrad's
kernels are **not slower in arithmetic**. The exact production cubin bodies,
run L2-hot (2000 reps), are at or below llama's production bodies:

| kernel | tinygrad body (L2-hot) | tinygrad production P | llama production body |
| --- | ---: | ---: | ---: |
| gate/up `w1w3fused16_12288_4096` | 22.945 us | 37.728 us | 35.201 us |
| down q6 `mmvq_direct_4096_12288` | 21.184 us | 30.560 us | 28.529 us |
| down q4 `mmvq_direct_4096_12288` | 17.568 us | 20.896 us | 18.896 us |

`MEASURED` The production gap is therefore **not body parity**. The L2-hot
body is up to `12.3 us` faster than llama, but production runs DRAM-cold and
lands at:

```text
gate/up  P 37.728 vs llama 35.201 = +2.527 us/call  x36 = +90.97 us
down q6  P 30.560 vs llama 28.529 = +2.031 us/call  x18 = +36.56 us
down q4  P 20.896 vs llama 18.896 = +2.000 us/call  x18 = +36.00 us
                                                            ---------
GEMV production delta                                      +163.53 us
```

`MEASURED` NCU cold (cache-flushed) replay confirms the DRAM-streaming story:
gate/up reads `56.64 MB` DRAM at `6.39%` L2 hit, down q6 `41.34 MB` at
`16.59%`, down q4 `28.36 MB` at `17.84%`. The weights are the full Q4/Q6
matrices streamed from DRAM every token; they do not stay L2-resident.

`MEASURED` DRAM streaming efficiency (production wall) against llama:

| kernel | tinygrad | llama | tinygrad / llama |
| --- | ---: | ---: | ---: |
| gate/up | 1.501 TB/s (83.9%) | 1.609 TB/s (89.9%) | 0.933 |
| down q6 | 1.353 TB/s (75.6%) | 1.449 TB/s (81.0%) | 0.934 |
| down q4 | 1.357 TB/s (75.8%) | 1.501 TB/s (83.8%) | 0.904 |

`MEASURED` The topology difference is concrete: tinygrad gate/up launches
`[12288,1,1] / [32,1,1]` (one warp per output row), while llama G launches
`[12288,1,1] / [32,4,1]` (four warps per row). The one-warp shape leaves
`45.60%` warps active and `52.96%` SM throughput, ~7% short of llama's DRAM
utilization. The down kernels are four-warp on both sides yet still trail
llama's rate, so warp count is one factor, not the whole story.

`MEASURED` The clean dispatch component is small: gate/up `D = 0.398 us`,
down q6 `0.606 us`, down q4 `0.420 us` (clean chained HCQ minus L2-hot body).
This is not a launch-dispatch problem.

## Support structure (deferred to Phase 9)

`MEASURED` The FFN fold also contains tinygrad's separate norm/reduction/
elementwise support: `reduce_output_rmsnorm_1_4096` (19 nodes, 136.7 us),
`r_16_256` (37, 123.2 us), `E_32_32_4` (38, 68.9 us), and the fused
`rmsnorm_q8_1_llama_provider_4096` (17, 30.7 us), totaling `359.5 us`.
Llama folds the equivalent into `rms_norm_f32` + `quantize_q8_1` totaling
`149.2 us`, a `+210.3 us` structural delta.

This is the coupled norm/provider row; its accounting is Phase 9, not here.
The GEMV rows above (`+163.5 us`) are independent of it.

## Verdicts

```text
gate/up body         BODY_PARITY_OR_FASTER  (L2-hot 22.945 vs llama 35.201)
gate/up mechanism    DRAM_COLD_STREAMING    (1.501 vs 1.609 TB/s)
down q6 mechanism    DRAM_COLD_STREAMING    (1.353 vs 1.449 TB/s)
down q4 mechanism    DRAM_COLD_STREAMING    (1.357 vs 1.501 TB/s)
clean dispatch       DISPATCH_CLEAR         (D <= 0.606 us/call)
```

The FFN gap is a memory-system problem (DRAM-cold streaming efficiency plus
the non-folded support kernels), **not** an arithmetic/body problem and **not**
a launch-dispatch problem.

## Wall sensitivity

`INFERRED` The GEMV legal ceiling is bounded by closing the DRAM-rate ratio to
llama's `~90%` of peak. At best that recovers `163.5 us`; a realistic
four-warp gate/up alone is the first arm. The support-structure `210.3 us`
belongs to Phase 9 and is not summed here. Neither term is booked.

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
