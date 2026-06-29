# Phase N2 — dynamic stall attribution (owned vs native ISA decode tile)

**Verdict:** `AMD_ISA_PHASE_N2_PASS_DEGRADED_TIMING_ATTRIBUTION`  ·  backend: sqtt  ·  ctx profiled: 512

W==D: owned 103.77/94.84 vs native 61.09/57.92 tok/s (native 58.9%/61.1% of owned)
token_match=True · deterministic=True · route_bound (owned+native captured)

_UNAVAILABLE: the vendored rocprof-trace-decoder (0.1.6) build returns wave events with instructions_size==0 (occupancy-only; verified roc.py --kernel yields OCC but no per-PC Stall table). Dynamic attribution is therefore OCCUPANCY + WAVE-CYCLE timing (degraded), not per-instruction stall categories._

## Dynamic diff (occupancy + wave-cycle timing)

| row | owned | native | ratio | lever |
|---|---|---|---|---|
| total_kernel_time_gpu_ms_capture | 1202.0 | 19305.48 | 16.06 | lower wall time |
| active_cu_count_sampledSE | 48 | 48 | 1.0 | occupancy/CU coverage (N3-occupancy) |
| traced_waves_sampledSE | 1536 | 1536 | 1.0 | grid/wave mapping |
| median_wave_cycles | 14434 | 372999 | 25.84 | reduce per-wave latency |
| mean_wave_cycles | 14172 | 367753 | 25.95 | reduce per-wave latency |
| vmem_stall / lds_stall / waitcnt_stall (per-PC categories) | unavailable | unavailable | None | needs PMC counters or a fixed itrace decoder to split VMEM vs LDS vs waitcnt |
| lane_utilization | unavailable | unavailable | None | — |

**Interpretation:** native median wave lifetime 372999 cyc vs owned 14434 cyc (x25.84); native occupies 48 CUs vs owned 48 (sampled SE). With native doing FEWER static instructions (N0: 324 vs 557) but its waves living LONGER, the gap is per-wave LATENCY/STALL-bound, not instruction-count or occupancy. The exact stall source (VMEM vs LDS vs waitcnt) is NOT resolved (decoder occupancy-only). CAVEAT: median_wave_cycles is a wave-RESIDENCY proxy (occupancy start->end) pooled over launches; it can include occupancy-overlap idle, so the absolute ratio overstates pure-compute cycles -- the DIRECTION (native waves live much longer => latency/stall-bound) is robust, the magnitude is approximate. End-to-end W==D gap is 1.6-1.7x (attention is one of many decode kernels), consistent with a much slower attention tile that is a fraction of the decode.

**Strongest suspected bottleneck:** per-wave latency: native waves take 25.84x owned's cycles despite fewer static instructions -> stall-bound (exposed load-use / memory latency). Category unresolved by occupancy-only decode.

**Next implementation phase:** N2.1 (finer counters): get a working itrace/PMC decode to split VMEM vs LDS vs waitcnt stall, THEN N3A (memory/coalescing) or N3C (waitcnt/load-use scheduling). Strong prior hint: per-token ds_bpermute reduce latency exposed.

