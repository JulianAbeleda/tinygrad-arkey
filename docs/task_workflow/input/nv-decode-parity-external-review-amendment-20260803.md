# NV decode parity external review - amendment

Date: 2026-08-03

Status: external-review amendment, docs only. This document is the
response-of-record to `nv-decode-parity-external-review-scope-20260803.md` at
commit `24deaffd7`. It corrects the blocker classification, reconciles the
wall-gap arithmetic, gives verdicts on assumptions 1-8, and replaces the
proposed decisive experiments. It authorizes no code change, GPU use, route
record change, or promotion to `dev`/`exp`/`master`. Branch boundary: tinygrad
`nvidia-bringup-20260731` at `24deaffd7`.

The original scope remains unchanged as the request-of-record. This amendment
must be read with it and supersedes its sections 1, 2.3, 4-6, and 8 wherever
the blocker, arithmetic, evidence class, or forward experiment differs.

---

## 1. Verdict

**REVISION REQUIRED before the scope is used to close graph-level overlap or
to search for an additional unexplained parity term.**

The measured gap does not presently require a hidden sixth lever. On the
available spans, it is already reconciled by two known terms: more serialized
GPU work on the tinygrad route and the absence of llama's graph concurrency.
The remaining wall residual is approximately the difference between each
side's wall and profiled GPU span.

The current blocker is also classified too strongly. The native probe proves
that multiple GPFIFOs created through the fork's current shared-context/channel
construction serialize. It does not prove that GB202 or driver 595.84 lacks
concurrent-kernel capability. llama.cpp demonstrates concurrency on the same
device, while the native independently scheduled context/subcontext variant
never completed successfully.

Corrected blocker statement:

> Graph-level overlap is blocked on reproducing CUDA-equivalent stream/channel
> scheduling through the native RM substrate. Shared-context native GPFIFOs
> measured serialized; a correctly bound independently scheduled native
> context/subcontext remains unproven. This is not a hardware no-concurrency
> verdict.

## 2. Findings, ordered by severity

### 2.1 The hardware-blocked conclusion is not earned

The probe's canonical E3-E5 rows are valid observations about the tested
configuration: two or three GPFIFOs under the existing channel construction
executed with span equal to node-sum. The inference that the GPU therefore
cannot co-schedule kernels is not valid.

At llama.cpp commit `ac4cddeb0`, `ggml-cuda.cu` explicitly:

- recognizes the three-way Q/K/V fan-out rooted at `attn_norm`;
- maps each branch to a separate CUDA stream;
- records a fork event on the main stream;
- waits on that event from the branch streams; and
- joins those streams back into the main stream with events.

The relevant implementation is
`/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4289-4304`,
`:4381-4388`, and `:4662-4803`. It targets QKV concurrency specifically; it is
not generic overlap of every non-GEMV node.

The native probe record itself states that the separate-ctxshare channels did
not execute and that their binding/scheduling setup is incomplete. Therefore:

- the shared-context serialization result is OBSERVED;
- "channel-level serialization in this construction" is INFERRED and
  supported;
- "one GR engine means no concurrent kernels" is FALSE; and
- "hardware/driver prevents concurrency" is FALSE as written.

One physical GR engine is not a concurrency gate. CUDA streams can co-schedule
kernels on the same GR engine. The next native question is how CUDA's streams
are represented by RM channel groups, context shares/subcontexts, runlists, and
scheduling controls, not whether GR1-GR7 exist.

### 2.2 The measured spans already reconcile the parity gap

Use the current GPU timestamps and unprofiled wall rows, not the inflated
`DEBUG=2` per-kernel sum:

| quantity | tinygrad | llama | basis |
| --- | ---: | ---: | --- |
| serialized/node GPU sum | 5.367 ms | 5.006 ms | native HCQ timestamps / CUPTI node trace |
| replay critical span | 5.367 ms | 3.890 ms | native HCQ timestamps / CUPTI node trace |
| unprofiled wall | 5.627 ms | 3.971 ms | 177.72 / 251.8 tok/s |

The GPU-span gap decomposes as:

```text
extra serialized work       = 5.367 - 5.006 = 0.361 ms
missing overlap             = 5.006 - 3.890 = 1.116 ms
GPU-span gap                =                   1.477 ms

wall gap                    = 5.627 - 3.971 = 1.656 ms
wall/profile residual delta =                   0.179 ms
```

Within the limitations of cross-profiler data, `1.477 + 0.179 = 1.656 ms`.
This does not prove that llama's schedule can be transferred to the tinygrad
DAG, but it refutes the claim that the current accounting necessarily lacks a
large third term.

The scope's section 8.6 calculation starts from `6.02 ms`, even though section
5.3 already establishes that this `DEBUG=2` sum carries per-kernel
launch/synchronization overhead. It then treats 22% overlap as a uniform
multiplier. Both operations are invalid for a critical-path forecast.

### 2.3 llama's graph-optimization environment is missing provenance

At commit `ac4cddeb0`, QKV concurrent-stream discovery is gated by:

```text
GGML_CUDA_GRAPH_OPT=1
```

The gate is at
`/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4605-4618`.
The external-review scope does not say whether this variable was set for either
the unprofiled llama authority rows or `/tmp/llama_nsys_d512.nsys-rep`.

Until that provenance is recorded, the 22% trace and the 251.8 tok/s authority
row cannot be assumed to represent the same llama execution policy. If the
variable differed, the overlap attribution is not an apples-to-apples
explanation of the wall row. If it matched, the exact value must be added to
the protocol.

### 2.4 The kv-store result is not evidence that count-reducing fusion is neutral

The kv-store experiment measured 948 kernels in both arms. It discovered that
the legacy expression had already been elementwise-fused into one store kernel
per layer, so the candidate was a 1:1 replacement. Its wall-neutral result
does not establish that a real 2:1 RMSNorm fusion, residual epilogue absorption,
or another count-reducing fusion would be neutral.

Lever 3 should therefore read:

> Actual count-reducing fusion remains unproven at wall. The kv-store 1:1
> replacement was neutral; earlier M3/M4/M5/Path-3 attempts were defeated by
> boundary materialization or candidate-specific work, not by a demonstrated
> general law that kernel-count reductions do not convert.

### 2.5 The token-time and fusion models are lower bounds, not additive forecasts

For a heterogeneous token graph, a single route-wide `BW` and `R` cannot
predict wall time. The critical-path form is closer to:

```python
T_token = critical_path(
  node_time[i] = max(B_i / BW_i, F_i / R_i) + boundary_i,
  dependencies=route.dependencies,
  resource_constraints=route.resources,
)
```

Likewise, `saved_bytes / BW + saved_ops / R` can double-count benefits when
bytes and operations belong to the same roofline term. A fusion ledger must
compare the before/after `max(...)` and then recompute the route critical path.
The section 2 functions remain useful as lower-bound discipline if this
limitation is made explicit.

### 2.6 Several facts are stale or dimensionally mislabeled

- `B_route ~ 46% of BW` is dimensionally invalid: `B_route` is bytes, while
  46% is an estimated achieved-bandwidth fraction.
- The `824 GB/s = 46%` row comes from the older 5.04 GB / 6.12 ms route. Using
  the same estimated bytes with the current 5.63 ms wall gives approximately
  895 GB/s, or 50% of 1792 GB/s.
- The 5.04 GB/token numerator is an accounting estimate, not a hardware DRAM
  counter. Thus the route bandwidth is INFERRED, not OBSERVED.
- Current HCQ profile artifacts contain five groups of
  `32/64/128/256/468 = 948` kernels. The scope's six-group
  `32/64/128/256/512/29` description is the older 1021-kernel state and must
  not be labeled as the current 948-kernel route.
- The scope calls the harness prompt `[1] * depth`, while
  `decode_runtime_overhead.py` constructs a repeated tokenizer-derived text
  prompt. llama-bench fills depth with pseudo-random vocabulary tokens.
- "W==D method" is misleading. W and D are different diagnostics, and the
  harness explicitly refuses W-D host attribution when D is slower than W.

## 3. Assumption verdicts

| # | verdict | review basis | evidence needed to settle/reopen |
| --- | --- | --- | --- |
| 1 | UNPROVEN | The 1.116 ms span reduction is the largest measured term, but llama targets QKV branches and our achievable DAG critical path has not been computed. | Same-policy llama A/B plus tinygrad dependency-DAG critical-path simulation. |
| 2 | UNPROVEN | The approximately 0.6 ms cap mixes tinygrad `DEBUG=2` median-by-name timing with llama CUPTI data, including a reused trace. It is diagnostic, not an additive wall term. | Same-session, per-class GPU timestamps with instrumentation overhead calibrated on both sides. |
| 3 | FALSE | The tested native shared-context channels serialize, but llama demonstrates device concurrency and the independently scheduled native setup never completed. | Correctly bound native context/subcontext E1-E5 run. |
| 4 | TRUE, narrowly | Current native timestamps give 5.367/5.63 ms = 95.4% GPU busy. This establishes a GPU-bound route, not that every kernel is bandwidth-bound. | Repeat on the exact final HEAD used for any promotion. |
| 5 | UNPROVEN | Same model/device/session discipline is good, but statistics, prompt/KV contents, and graph-opt provenance differ. | Align median-versus-median reporting, prompt policy where possible, cache settings, and all environment variables. |
| 6 | FALSE | The levers are not additive; prefill is not a d512 decode lever; depth scaling has little d512 mass; and overlap is a DAG critical-path effect. The 0.94x result is created by the inflated 6.02 ms input. | Replace with the matched-span reconciliation in section 2.2. |
| 7 | FALSE as written | `B_route` is not 46%; the bandwidth percentage is stale and inferred from estimated bytes. Aggregate decode may be weight-bandwidth-bound while individual kernels remain latency, occupancy, or instruction-throughput bound. | Hardware DRAM-byte counters or a validated per-kernel byte ledger plus achieved-bandwidth measurements on the current route. |
| 8 | UNPROVEN, supported | fp16 KV causally helps at d4096 and is nearly neutral at d512, supporting a KV-read component, but it closes only about half the depth delta. | Per-depth class spans with matching fp16 cache policy and a controlled flash/KV kernel A/B. |

## 4. Evidence-class corrections

The following claims must be relabeled from OBSERVED to INFERRED or qualified:

1. The 1.67 ms wall gap is arithmetic from observed throughput rows: INFERRED.
2. The 22% overlap fraction is arithmetic from observed node intervals: INFERRED.
3. "Non-GEMV classes are hidden behind the mmq chain" is an attribution from
   interval structure and source behavior: INFERRED until class-pair overlap is
   reported directly.
4. The 0.5-0.6 ms like-for-like cap is cross-instrument arithmetic: INFERRED.
5. `824 GB/s`, `46%`, and 5.04 GB/token are derived accounting, not measured
   traffic: INFERRED.
6. "Compute-bound matmul serialization rules out DRAM contention" is a
   reasonable probe inference, not an observation.
7. "One physical GR engine is addressable, therefore concurrency is blocked"
   is unsupported and must be removed.
8. "Host dispatch is fully overlapped" is inferred from GPU span, wall, and
   submit measurements; the individual measurements remain OBSERVED.

The source observations, timestamp intervals, correctness pins, kernel counts,
and raw throughput samples remain OBSERVED when their exact session and HEAD are
named.

## 5. Replacement for assumption 6

Replace the current 0.94x sanity arithmetic with:

> Using matched GPU spans, tinygrad is 5.367 ms serialized while llama is
> 5.006 ms node-sum and 3.890 ms critical span. The measured GPU gap therefore
> decomposes into approximately 0.361 ms of additional serialized work plus
> 1.116 ms of absent overlap. Together with the approximately 0.179 ms
> wall/profile residual delta, these terms account for the current 1.656 ms
> wall gap. Does this eliminate the need for an additional missing term, and
> how much of llama's 1.116 ms is achievable on tinygrad's actual dependency
> DAG under realistic queue/resource constraints?

This replacement is still a reconciliation, not a composed endpoint forecast.
No parity claim follows until the mechanisms are implemented and measured at
wall.

## 6. Three cheapest decisive experiments

### E1 - llama graph-optimization A/B

Run the identical d512 llama-bench command in one session with
`GGML_CUDA_GRAPH_OPT=0` and `=1`. For each arm record raw wall samples, median
and mean tok/s, nsys node-sum, replay span, stream/graph identity, and the full
environment.

Belief-flip measurement:

- If `=1` reduces replay span and wall by a parity-scale amount, the overlap
  attribution is confirmed.
- If the span changes but wall does not, or the wall delta is below 0.2 ms,
  the 22% trace is not a parity-scale wall lever.
- If the authority row used `=0` while the trace used `=1`, the current
  overlap decomposition is invalid and must be rebuilt from matched arms.

### E2 - tinygrad dependency-DAG critical-path simulation

Capture one complete d512 native HCQ dependency graph with per-node GPU
durations. Offline, calculate the serialized span, unlimited-resource critical
path, and deterministic two-queue and three-queue list schedules. Keep graph
groups and cross-group dependencies explicit.

Belief-flip measurement:

- Reopen overlap as parity-scale if a realizable two/three-queue schedule saves
  approximately 0.8-1.1 ms before implementation.
- Downgrade overlap if even the unlimited-resource critical path saves less
  than approximately 0.4 ms.

### E3 - CUDA-stream versus corrected native-channel control

Run equivalent independent elementwise, partial-SM, and compute-heavy kernels
through two CUDA streams and through a correctly bound native independent
context/subcontext construction. Use the same span/node-sum criterion as the
existing E3-E5 probe.

Belief-flip measurement:

- CUDA overlap with native serialization proves a native RM
  construction/scheduling gap, not a hardware limitation.
- At least 5% native overlap with correct numerics reopens D2-D4.
- Zero overlap in a correctly validated native construction narrows the
  blocker to the native RM/driver path, but still does not imply that the GPU
  cannot run concurrent kernels through CUDA.

## 7. Corrected forward state

Until the experiments above settle the mechanism:

- L1 graph overlap: **UNPROVEN ON NATIVE RM; do not classify as hardware
  blocked**.
- L2 GEMV delta: **diagnostic cap only; wall conversion pending**.
- L3 count-reducing fusion: **open only through variant-specific boundary and
  endpoint gates; kv-store 1:1 neutrality is not a general precedent**.
- L4 depth scaling: **fp16 KV landed and beneficial at depth; residual open**.
- L5 prefill: **separate regime; exclude it from decode additive arithmetic**.
- Additional hidden parity term: **not presently required by the reconciled
  measurements**.

HARD STOP on declaring native overlap impossible, composing a parity endpoint,
or promoting a route from this amendment. The next authorized artifact is a
measurement scope for E1-E3 or a corrected external-review brief incorporating
these findings.
