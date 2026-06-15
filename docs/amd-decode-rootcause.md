# Decode root cause — SINGLED OUT: memory-level-parallelism (prefetch), not compute, not gaps

Date: 2026-06-15. The clinching measurement that narrows the entire decode investigation to one cause.

## The decisive measurement
During sustained single-stream decode (Qwen3-8B Q4_K), sampled `rocm-smi` GPU use:
**100% busy** (steady state). Yet effective bandwidth = ~32% of peak (89 GB/s this degraded session, ~278
historical, vs 859 peak). So the GPU is FULLY OCCUPIED but NOT saturating HBM.

## What this rules in / out (with the prior measurements)
| evidence | conclusion |
|---|---|
| GPU 100% busy during decode | NOT idle gaps / launch overhead -- no inter-kernel idle time |
| only ~32% of peak bandwidth | kernels busy but not saturating HBM -> memory-LATENCY-bound |
| READRAW (pure read, no dequant) = 85% of peak | the access pattern CAN saturate; HBM isn't the wall |
| fp GEMV (dequant) = 42%; v_dot4 faster dequant = 0 e2e gain | the dequant COMPUTE is not the bottleneck |
| packed_load (wider loads) = +6% | more bytes/request = slightly more memory-level parallelism (right lever) |

## The single root cause
**The decode GEMV is bound by the `load -> dequant -> accumulate` DEPENDENCY CHAIN.** The GPU stays 100%
busy traversing it, but does not keep enough loads in flight (low memory-level parallelism) to saturate
HBM -- so it runs at ~32% of peak with the cores fully occupied waiting on dependent loads. It is
memory-LATENCY-bound, not bandwidth-bound, not compute-bound, not gap-bound.

This is consistent with EVERY prior result:
- v_dot4 / TC null e2e: they speed COMPUTE; the chain is load-latency-bound, so compute speed is irrelevant.
- READRAW saturates (85%): remove the dependent dequant and the loads stream freely.
- packed_load +6%: wider loads = fewer requests = more MLP (the right lever, small dose).
- llama.cpp at 54%: its kernels keep more loads in flight (prefetch) -> closer to saturation, same batch-1.

## The lever (specific, finally)
NOT a compute lever (v_dot4/TC/fusion -- all ruled out by measurement). The lever is **software-pipelined
prefetch / memory-level parallelism**: issue many weight loads ahead so HBM stays saturated while the
dequant runs on already-loaded data, decoupling load latency from compute. This is what llama.cpp does to
reach 54% at batch-1, and what an async-copy / pipelined-load primitive (TileLang-class) expresses.

## Why this is "a good spot"
We have narrowed decode from "slow / unknown" to a single, measured cause with a specific lever:
- It is NOT in the compute (every compute lever measured-null).
- It is NOT idle gaps (GPU 100% busy, measured).
- It IS the load->dequant dependency chain limiting memory-level parallelism (prefetch).
The open question is now precise and small: can the GEMV be made to prefetch / keep more loads in flight
(in tinygrad's expressible opts, or via a pipelined-load primitive), to raise 32% -> toward llama.cpp's 54%?
That is the next make-or-break, and unlike the compute levers, it directly targets the measured bottleneck.

## Caveat
GPU bandwidth-utilization counters weren't available (`rocm-smi` Memory Activity = N/A) and
GlobalCounters.time_sum_s is 0 in JIT replay, so "32% of peak" is bytes/wall-time, and "100% busy" is the
GPU-use% counter (occupancy, includes memory-wait). The diagnosis (busy-but-unsaturated -> latency/MLP)
follows from the combination of these + READRAW + the v_dot4 null, not any single number.
