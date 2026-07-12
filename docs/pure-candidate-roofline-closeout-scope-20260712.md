# Pure candidate roofline closeout scope

## Objective

Move the generated exact-anchor candidate from the current pinned median of
68.46 TFLOP/s toward or beyond the generated-path oracle band near 75 TFLOP/s,
then expose the proven controls to BoltBeam machine search. Preserve strict-pure
Tinygrad lowering, dynamic candidate identity, and candidate/binary/commit joins.

## Established baseline

- Workload: fp16 GEMM M512/N12288/K4096, 51,539,607,552 FLOPs.
- Original pure scheduler: 2.64668 ms, 19.47 TFLOP/s in the matched run.
- Current exact candidate: 0.75288 ms, 68.46 TFLOP/s median; 77.34 TFLOP/s best.
- Matched median speedup: 3.5154x.
- Current binary: tile128x128x32, 4x2 waves, 256 threads, one 20,480-byte LDS buffer.
- Current resources: 113 VGPR, 18 SGPR, zero spill/scratch.
- Exact route, full-output correctness, compiled resources, runtime binary, and
  kernel-only timing have a passing five-stage authority join.

The earlier 54.10-TFLOP/s result was an auto-clock run. Use the pinned,
randomized, interleaved 68.46-TFLOP/s measurement for baseline comparisons.

## Roofline verdict

The classical compulsory-byte roofline remains compute-bound:

- Compulsory bytes: 117,440,512.
- Compulsory operational intensity: 438.857 FLOP/B.
- At 960 GB/s, the memory roof is about 421 TFLOP/s, above compute peak.

The emitted tiled kernel requests about 817.9 MB globally because A and B tiles
are replicated across workgroups. Cache reuse prevents treating this as HBM
traffic. The current artifacts do not contain same-binary HBM/L2/LDS counters,
so no fabric-level bandwidth claim is authoritative.

The active limitation is compiler lifecycle and issue efficiency:

- one produce/barrier/consume epoch per K32 tile;
- no overlap between the next global stage and current WMMA compute;
- about one wait per WMMA versus about 0.281 in the oracle;
- about 29.6 static instructions per WMMA versus about 9.05 in the oracle;
- scalar construction before late b128 packing and a scalar-heavy epilogue.

Occupancy is not the primary limiter. The candidate has lower VGPR and higher
possible LDS residency than the faster double-buffer oracle.

## Invariants

Every phase must retain:

1. Strict-pure generated surface: no `Ops.INS`, handwritten ASM, HIP kernel, or
   oracle fallback in a candidate path.
2. Canonical dynamic candidate hash in `KernelCandidateContext`, compiler cache,
   route evidence, resources, correctness, timing, and BoltBeam records.
3. Emitted semantic proof derived from actual UOps, not the requested payload.
4. Full-output correctness before timing; nonconstant row/column data is required.
5. Exact executed-binary equality and clean source-commit join.
6. Candidate failure taxonomy: invalid, unsupported, compile failure, emission
   mismatch, resource failure, correctness failure, timing failure, passed.
7. Ordinary Tinygrad TC lowering remains unchanged outside an admitted plan.

## Phase R1: vectorized cooperative producer

Replace the correctness-scaffold scalar producer with a core vector transport
that lowers to four b128 A/B stores per thread for the 128x128x32 candidate.
Reuse AMD packing/lowering; do not construct renderer-private ISA markers.

Required proof:

- 1,024 unique aligned 16-byte vectors cover A/B data exactly once;
- padding is neither written nor read;
- final ISA contains the expected global/DS b128 transport;
- source/ISA instruction count decreases without increasing spills;
- both full-output cases pass on the same binary;
- pinned interleaved timing does not regress beyond noise.

## Phase R2: typed two-buffer lifecycle

Extend the core execution plan with buffer count, stage count, slot expression,
and epoch ownership. Implement buffer2/stage1 only before deeper pipelines.

The generated graph must express prologue, steady-state, and drain:

- prologue produces slot0;
- steady state computes slot `k&1` while producing `(k+1)&1`;
- overwrite occurs only after all consumers of a slot complete;
- drain consumes the final staged tile;
- both operands share one epoch and slot identity;
- barriers/waits derive from DAG dependencies.

Required proof:

- two nonoverlapping 20,480-byte slots, total LDS40,960;
- exhaustive epoch/slot producer-consumer coverage and no overwrite hazard;
- emitted UOps, AMD ISA, and metadata agree on allocation and lifecycle;
- zero spill/scratch and legal workgroup resources;
- adversarial correctness cases pass before timing.

## Phase R3: overlap and K-body depth

After buffer2 correctness, expose K-unroll/prefetch distance values 1, 2, and 4.
Prove each schedule separately. The current candidate is the unroll1 baseline;
the oracle effectively amortizes control across a deeper K body.

Measure:

- waits, barriers, global/DS operations, and static instructions per WMMA;
- VGPR/SGPR/LDS and theoretical residency;
- kernel timing distribution, not only the minimum;
- correctness and binary identity for every candidate.

Promotion target: reduce wait/WMMA and instruction/WMMA materially while
preserving median performance. Reject fast outliers with unstable medians.

## Phase R4: dependency policy

Only after the generated overlap DAG is correct, expose a small set of semantic
wait policies. Policies name dependency behavior; they do not inject raw waitcnt
strings. AMD lowering remains responsible for instructions.

Each policy needs:

- producer/consumer and overwrite proof;
- final ISA wait/barrier audit;
- repeated adversarial correctness;
- pinned median timing and variance;
- fail-closed descriptor drift handling.

## Phase R5: epilogue

Vectorize/coalesce fp16 output stores through core UOps. Prove exact accumulator
ownership and conversion, aligned nonoverlapping output coverage, and absence of
extra output kernels. Compare store-site and instruction counts before timing.

## Phase R6: bounded machine search

BoltBeam enumerates canonical plans; Tinygrad is the sole capability and
execution authority. Initial search dimensions:

- buffer count: fixed2 after R2;
- K unroll/prefetch distance: proven R3 values;
- semantic wait policy: proven R4 values;
- tile M/N/K and waves M/N from the admitted divisible single-buffer family,
  extended to buffer2 only after resource validation;
- LDS padding from a small aligned set;
- vector width fixed16 until R1 proof is generalized.

For each candidate, isolated subprocess stages are admission, compile/semantic
proof, resources, correctness, and timing. Failed candidates do not stop later
candidates. Artifacts join on candidate hash, capability ID, source commit,
source/binary hash, and executed binary.

## Measurement closeout

Run candidate, original pure baseline, and oracle randomized/interleaved under
the same clock policy. Record at least 5 warmups and 21 waited samples. Add
same-binary counters when available:

- HBM read/write bytes and L2 hit/service traffic;
- LDS read/write transactions or bytes;
- WMMA/VALU/SALU issue counts;
- wait/barrier and stall cycles;
- active waves/CU and achieved occupancy;
- clock samples across the measurement window.

Without these counters, retain `compute/lifecycle/issue limited` as the diagnosis
and treat LDS pressure as plausible but unproven.

## Completion criteria

This scope is complete when:

1. The generated path reaches or exceeds the oracle median under the matched
   protocol, or a counter-bound practical ceiling explains the residual.
2. At least two distinct dynamic candidates complete every evaluator stage.
3. BoltBeam can generate, admit, evaluate, persist, and rank candidates without
   fixed schedule hashes or duplicated Tinygrad legality.
4. Whole-prefill route binding confirms the winning kernel is used in the
   intended 8B role with no hybrid/oracle fallback.
5. All evidence is committed and reproducible from clean revisions.
