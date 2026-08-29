# NVIDIA pp512 exact cross-runtime lifecycle trace scope

## Objective

Replace the remaining prefill estimates with one event-level accounting of the
current tinygrad compiler-packed stack and llama.cpp. The accounting must
explain the measured wall as executed work, overlap, device idle, and host
residual. It must not infer recoverable token rate from bytes, rooflines,
profile shares, or isolated kernels.

Frozen workload:

- Qwen3-8B Q4_K_M;
- prompt length and ubatch 512;
- RTX 5090 sm_120;
- Flash Attention enabled;
- tinygrad commit `4d117c8e0`, admitted default-off gate/up + K + Q/O arm;
- llama.cpp CUDA build `ac4cddeb0` unless a fresh build is proven equivalent;
- model `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`.

The retained walls, 69.378 ms tinygrad and 36.608 ms llama, are hypotheses to
reproduce, not constants to force into the result.

## Accounting identity

Every traced run must close this identity on one common monotonic timeline:

```text
traced wall
  = union(all GPU execution intervals)
  + GPU idle between first and last relevant device event
  + pre-device boundary residual
  + post-device boundary residual
```

GPU intervals are also labeled by lifecycle region. Category sums are active
time and may overlap; the interval union is the only additive GPU total.
Overlap is reported explicitly rather than counted twice.

The unprofiled wall remains the performance authority. A profiled wall is used
only to account for its own trace and is never substituted for the unprofiled
wall. The difference between profiled and unprofiled runs is a measured
instrumentation delta, not model work.

## Common semantic regions

Each device launch must receive exactly one primary region and may receive
secondary dependency tags:

1. input/embed and graph setup;
2. RMSNorm and activation conversion;
3. Q;
4. K;
5. V;
6. Flash score/reduction;
7. O;
8. gate;
9. up;
10. activation/multiply;
11. down;
12. residual, RoPE, KV write, and other support;
13. final-row gather/prune;
14. vocabulary;
15. output/token transfer;
16. unknown.

Dense launches also carry format (`Q4_K`, `Q6_K`, FP16), layer, role, Q8
producer/main/fixup identity, grid, block, stream/queue, and graph-node ID.
`unknown` must be zero before the result is accepted.

## Llama trace lane

The llama lane owns these deliverables:

1. Reproduce a fresh unprofiled R9 pp512 wall and retain every sample.
2. Capture a fresh Nsight Systems run with CUDA graph node tracing, CUDA API,
   kernel intervals, stream IDs, and graph-node IDs.
3. Export the raw SQLite/report plus kernel, CUDA API, graph, and interval CSVs.
4. Retain the exact command, environment, binary SHA, model SHA, GPU/driver
   identity, clocks if observable, stdout, and profiler version.
5. Parse every launch into the common semantic regions using kernel template,
   launch geometry, graph order, source mapping, and the known 36-layer role
   sequence. Kernel-name substring matching alone is insufficient for roles
   sharing one MMQ specialization.
6. Reconstruct the serialized/overlapped interval union and all device gaps.
7. Reconcile the profiled wall to the trace identity and separately report the
   profiler delta relative to the unprofiled R9 distribution.
8. Emit per-layer and per-role rows, not only aggregate totals. The final-layer
   M=1 prune must be identified from executed shapes/launches.

Required llama artifacts:

```text
llama-unprofiled-r9.json
llama-trace.nsys-rep
llama-trace.sqlite
llama-kernels.csv
llama-cuda-api.csv
llama-graph.csv
llama-intervals.json
llama-role-map.json
llama-accounting.json
llama-accounting.md
```

## Tinygrad trace and causal lane

The matched tinygrad lane will use the admitted 69.378-ms safe-cut graph and
must retain its six HCQ graph segments. It will enumerate all 180 compact-Q8
producers, 180 compiler mains, remaining 72 V/down FP16 overlays, Flash and
support calls, queue IDs, dependencies, and already-bound buffers.

Because external CUPTI does not currently observe native HCQ launches, this
lane must use device-side interval timestamps or another GPU-clock mechanism
attached to actual HCQ submissions. `PROFILE=1` duration sums alone are not an
acceptable common clock. If exact GPU timestamps cannot be obtained, the
result must stop at a named substrate wall rather than mix profiler regimes.

The causal wall ledger uses fresh-process R9 rollback arms on the same commit:

- gate/up packed -> FP16;
- K packed -> FP16;
- Q/O packed -> FP16;
- admitted safe queue cut -> exact primary-only control;
- later, V or down replacements only after each is correctness-qualified.

Each rollback must pass the same token, full-logit tolerance, graph census,
canonical-weight, and replay gates. Marginal values are reported in their
measured composition order and are not assumed additive under another order.

## Output table

The final cross-runtime ledger must contain measured values only:

| region | tiny active time | tiny charged wall | llama active time | llama charged wall | exact charged debt | overlap/idle note | causal rollback |
|---|---:|---:|---:|---:|---:|---|---:|

`Charged wall` is derived from an explicit interval partition or a causal
rollback, never by scaling profile percentages onto an unprofiled wall. If a
region cannot be uniquely charged because of overlap, show the exact interval
intersection and leave the charged value unassigned.

## Acceptance gates

- fresh unprofiled R9 for both runtimes;
- exact binary/model/commit provenance;
- 100% launch classification;
- no duplicate interval charging;
- profiled trace reconciles to its own wall within timestamp resolution;
- instrumentation delta reported separately;
- final-row lifecycle proved from executed launches;
- no estimated recovery or roofline-derived token rate;
- all unknowns and non-comparable clocks named explicitly;
- raw artifacts and deterministic parser retained.

The result is complete only when it either produces the closed accounting
table or identifies one precise missing timestamp/ownership substrate needed
to do so.
