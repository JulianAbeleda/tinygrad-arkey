# NV decode parity - external review scope (self-contained brief for a fresh-eyes agent)

Date: 2026-08-03
Status: review scope, docs only. SUPERSEDED IN PART by
`nv-decode-parity-external-review-amendment-20260803.md` (2026-08-03): the
amendment corrects the blocker classification, reconciles the gap arithmetic on
matched GPU spans, adds the `GGML_CUDA_GRAPH_OPT` provenance requirement,
returns verdicts on assumptions 1-8, corrects evidence classes, replaces
assumption 6, and replaces the decisive experiments with E1-E3. The original
text below remains the request-of-record and is unchanged; read the amendment
first. Authorizes no code change, no GPU use, no promotion to
`dev`/`exp`/`master`. Purpose: explain the current design approach and measured
state so an external agent with no repo history can run an adversarial review.
The specific ask: identify the clear reason we are failing to reach decode
parity that we cannot see.
Branch boundary: tinygrad `nvidia-bringup-20260731` @ `b2d7550e2`. Supersedes
nothing. Grounding records listed in section 9; every number below is OBSERVED
unless marked INFERRED.

---

## 1. The ask (read this first)

Decode wall time is 1.44-1.52x behind llama.cpp on the same box, same model,
same session. We have a measured decomposition (section 5), five scoped levers
(section 6), and a probe result that closed our main lever (section 6.1).
Every lever we have actually built either landed wall-neutral or is blocked by
hardware. Our working hypothesis: **we are missing a clear structural reason
for the gap that the measurements are not surfacing.**

Your job as reviewer:

1. Read sections 2-7 and the grounding artifacts in section 9.
2. Challenge every assumption in section 8 explicitly. State which are false
   or unproven, and what evidence would settle each.
3. Answer the review questions in section 10.
4. Rank the three cheapest decisive experiments that would confirm or refute
   your main hypothesis.

Do not assume the repo docs are correct. The evidence classes (OBSERVED /
INFERRED) are the authors' labels; part of the review is checking whether the
labels are earned.

## 2. The model we are applying (pseudocode)

This is the canonical token-time model used across the campaign. It is
target-agnostic; it applies to any backend.

```python
# 2.1 TOKEN-TIME MODEL
def token_time(route: Route) -> Time:
    """Lower bound on time per token for the selected route."""
    T_bytes = route.B_route / BW        # bytes actually moved / achieved bandwidth
    T_flops = route.F / R               # ops required / achieved op rate
    T_bulk  = max(T_bytes, T_flops)     # roofline: the binding resource
    T_boundary_critical = route.non_overlapped_boundary_time()
    return T_bulk + T_boundary_critical # lower bound, not a predictor
```

Key phrases:

- `B_route`, not `B_min`. Compulsory weight bytes are the floor; what costs
  wall time is what is actually moved: intermediates, copies, re-reads.
  The game is shrinking `B_route` toward `B_min`.
- `max(T_bytes, T_flops)` is why decode and prefill are different problems.
  Decode (`M=1`): each weight byte is read once, so `T_bytes` dominates.
  Prefill (`M=512`): bytes amortize over 512 rows, so `T_flops` dominates.
- `non_overlapped_boundary_time()`: graph replay can hide launches/syncs
  behind bulk work, so boundaries only count when they are on the critical
  path. Whether a boundary is hidden must be measured, not assumed.

```python
# 2.2 REGIME CROSSOVER (model size cancels)
def crossover_batch(w_bits: float, R: Rate, BW: Bandwidth) -> float:
    """Solve 2*M*P/R == P*w/8/BW for M."""
    return (w_bits / 16) * (R / BW)     # P cancels

# decode:  M=1   << M*   -> bandwidth-bound
# prefill: M=512 >> M*   -> compute-bound
```

```python
# 2.3 FUSION LEDGER (is a fusion worth it?)
def fusion_ledger(f: Fusion) -> Time:
    benefit = (f.saved_binding_bytes / BW
             + f.saved_ops / R
             + f.saved_non_overlapped_boundary_time)
    cost = (f.added_binding_bytes / BW
          + f.added_ops / R
          + f.resource_penalty)         # occupancy, spills, primitive throughput
    return benefit - cost               # winner is measured in the route, not the kernel
```

Worked example with our measured numbers (section 5):

```python
# d512, Qwen3-8B-Q4_K_M, RTX 5090 (sm_120), driver 595.84
ours  = token_time(B_route ~ 46% of BW,   boundary hidden 0.0 ms)  # 5.63 ms wall, 177 tok/s
llama = token_time(B_route ~ same model,  boundary hidden 1.1 ms)  # 3.97 ms wall, 252 tok/s

# the two levers the ledger names:
#   saved_boundary (overlap)      -> blocked: native multi-queue probe FAILED (6.1)
#   saved_binding_bytes (GEMV BW) -> like-for-like cap ~0.6 ms; wall conversion unproven (6.2)
```

The model's discipline: name the binding bound, shrink `B_route`, raise
achieved `BW`/`R`, hide boundaries, and measure the route. Isolated kernel
speed is diagnostic; same-run endpoint time is authoritative.

## 3. Goal and environment

- Target: decode wall parity with llama.cpp, Qwen3-8B-Q4_K_M, RTX 5090
  (sm_120), driver 595.84, CUDA 13.2, nvcc 13.2, nsys 2026.1.3.
- Model file: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf` (identity sha
  `b8ef0be84bfa0588efae9fb84a3b3e5b7beb53f5620ada7d8c48bd3a26633605`).
- llama baseline: llama-bench CUDA build `ac4cddeb0`, `-ngl 99 -fa 1
  -pg 0,10 -d <depth> -r 5`, decode row, mean tok/s. Same session back to
  back with every tinygrad row.
- tinygrad harness: `extra/llm_research/decode/decode_runtime_overhead.py`,
  W==D method (W = production generate wall, D = same JIT with final sync),
  fixed-depth prompt `[1]*depth`, chunk 32, temperature 0.0, nmeas 20,
  reps 3, median tok/s.
- Correctness pins hold at every depth: token sha256
  `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` 3/3,
  first token `151936` 3/3.

House convention: fused prefill attention is OFF for every decode run (the NV
fused prefill ABI is deterministically broken at HEAD). The prefill regime is
out of scope for this review except where noted.

## 4. Execution model (what the code actually does)

- The route runs on `DEV=NV` (tinygrad's native NVIDIA backend,
  `runtime/ops_nv.py`): kernels go through native NVIDIA ioctls, QMD command
  construction, and direct GPFIFO submission. There is no CUDA Runtime/Driver
  API call, so CUPTI/Nsight cannot see the kernels (`nsys --cuda-graph-trace
  =node` produces no GPU data; this is a backend/profiler mismatch, not
  missing tooling).
- Decode is graph-replayed: GraphRunner + HCQGraph (`runtime/graph/hcq.py`)
  replay 6 graph groups per token (batched 32/64/128/256/512/29 = 948
  programs/token after the kv-store fusion; 1021 before).
- HCQGraph creates **one compute queue per device** (`hcq.py:68`), assigns
  every kernel to it (`hcq.py:134`), and the NV queue submits to **one
  compute GPFIFO** (`ops_nv.py:205`, `ops_nv.py:633`). Execution is
  stream-serialized: 0.0% overlap between kernels (measured, section 5.3).
- The opaque custom-kernel boundary (`UOp.custom_kernel`,
  `uop/ops.py:1264-1273`) calls `.contiguous()` on every non-identity input,
  materializing copies for lazy producers. This copy tax made M3/M4/M5/Path3
  all measured non-landings (section 7).
- GPU busy is ~95% of wall during decode (5.83 ms kernel sum vs 6.12 ms
  wall), so the wall is kernel execution, not host launch cost.

## 5. Measured parity state

### 5.1 Wall parity (same-session authority rows)

| depth | tinygrad tok/s | llama tok/s | ratio |
| --- | ---: | ---: | ---: |
| d512 | 172.8-177.2 | 248.2-251.8 | 0.696-0.704 |
| d2048 | 161.5-164.8 | 235.1-237.8 | 0.687-0.693 |
| d4096 | 149.0-152.1 | 226.0-228.3 | 0.659-0.666 |

The d512 wall gap is 1.67 ms/token (INFERRED arithmetic: 1/177.2 - 1/251.8).

### 5.2 Kernel census and node-sum (d512, per token)

| side | kernels/nodes | node-sum | replay wall | overlap |
| --- | ---: | ---: | ---: | ---: |
| tinygrad | 948 kernels | 5.981 ms (DEBUG=2 tm sum) | 5.63 ms W wall | 0.0% (HCQ profile) |
| llama | 762 nodes | 5.006 ms | ~3.89 ms span | ~22% below node-sum |

llama class split: mmq 217 nodes/3.578 ms, quantize_q8_1 217/0.551, flash 72/
0.363, rms_norm 145/0.308, rope 72/0.126, kv_ops 39/0.080. Our split: GEMV-
class 216-252 kernels/3.775-3.836 ms, flash 72/0.367, other 655/1.424.
Caveat on llama's 22%: under-nsys llama-bench runs ~7% slower than
unprofiled (228.6 vs 246.2 avg tok/s), so llama node durations are inflated;
the direction of any resulting bias in the 22% figure is a review question
(Q2).

### 5.3 Overlap measurement (our side)

Native HCQ timestamp/dependency tracing (the `HCQ_GRAPH_PROFILE_JSON`
exporter) shows replay overlap 0.0% at d512 and d4096: per-group span equals
node-sum to the microsecond. An earlier ~6% "overlap" fallback (DEBUG=2
kernel-sum 5.981 ms vs W wall 5.63 ms) was refuted: the 0.35 ms delta is
per-kernel launch overhead in the DEBUG=2 tm values (~0.65 us x 948), not GPU
concurrency.

### 5.4 Per-kernel like-for-like (quantize-excluded)

llama bare GEMV class (mmq, minus vocab and quantize): **3.24 ms**. Ours
(GEMV-class census, q4k lanemap + q6k coop/partial): **3.836 ms**. The
like-for-like cap on this piece is ~0.5-0.6 ms of the 1.67 ms gap.

Microbenchmark reality check: seven CUDA microbenches reproduce llama-class
ceilings within ~0.3% (dp4a ~952 G dp4a/s, NACC-invariant; L2 Q6K partial
12.91 us installed / 7.39 us best row; flash tile anchor 14.13 us vs llama
floor 3.2 us; vocab coop 331.2 us = 86% of 1792 GB/s). So the raw instruction
and bandwidth ceilings are reachable on this GPU in isolation; the route does
not reach them (824 GB/s = 46% of 1792 GB/s). Whether the route-level 46% is
a per-kernel problem or a schedule problem is open (Q5).

### 5.5 Depth scaling (d512 -> d4096)

ours -14.2% (fp32 control; fp16 cache route -11.9%) vs llama -9.3%. The
fp16-cache route (capability-based, landed) closes about half the depth-penalty
delta. The flash score kernel grows 7.78 -> 32.4 us/median over that range;
llama's ext_vec 4.80 us.

## 6. The five levers and their status

Ranking basis: expected wall impact weighted by conversion confidence, not
ceiling alone (ceilings are not on one scale, and kernel-sum savings do not
automatically convert to wall: the kv-store fusion proved 948 -> 948 kernels,
wall-neutral 177.8 vs 178.4 tok/s).

### 6.1 Lever 1 - graph-level overlap. **BLOCKED, gate closed.**

- P0 answered: our replay has 0.0% overlap (5.3). llama's ~22% is the
  reference.
- Primitive route (generic, backend-agnostic): native multi-compute-queue
  execution. Device-level probe E1-E5 on this host:
  - E1 PASS: cross-GPFIFO memory-semaphore dependencies work, numerics exact
    (max err 0.00e+00).
  - E2 PASS: serial calibration (span == node-sum, as expected).
  - E3/E4/E5 FAIL: 2-queue, 3-queue, and 2-queue matmul all show span ==
    node-sum to <0.3%. Partial-SM grids (grid-div 2/4/8) serialize exactly
    too; the compute-bound matmul flavor serializing as hard as the DRAM-bound
    elementwise flavor rules out DRAM contention. Verdict: channel-level
    serialization.
  - Engine sweep: `engineType` 0/1 (GR0/GRAPHICS) allocate but serialize;
    GR1-GR7 (`engineType` 2-8) rejected by RM with `NV_ERR_INVALID_ARGUMENT`.
    One physical GR engine is addressable on this consumer part.
  - Separate-ctxshare variant (own FERMI_CONTEXT_SHARE_A per channel): extra
    channels did NOT execute (E1 hang, join target stuck at 11 of 12).
    Recorded as a setup gap, not a hardware verdict.
- Verdict: GATE CLOSED on this hardware/driver via the native primitive,
  unless RM-level channel scheduling control is found. Non-primitive
  alternative: route through DEV=CUDA and the existing CUDAGraph lowerer
  (a fork from the native substrate; NVIDIA-specific, not generic).

### 6.2 Lever 2 - GEMV per-kernel efficiency. Cap ~0.6 ms, conversion unproven.

- L2 Q6K partial single-pass: CLOSED NO-GO (best standalone 7.38 us vs the
  3.3 us llama-class floor).
- L4 vocab substrate fusion: GO as a landing warrant (fused 315.9 us vs llama
  303.75 us, bit-identical); isolated d512 wall measurement PENDING.
- Flash score tile structure: CLOSED NO-GO (best zero-load 5.311 us vs the
  3.2 us floor); tile structure confirmed SUBSTRATE.
- Like-for-like cap discipline: bounds the total GEMV-class claim to ~0.6 ms;
  the microbenchmarks reproduce ceilings but are not wall A/Bs, so the 46%
  route-level bandwidth figure is unchanged.

### 6.3 Lever 3 - kernel count. 948 vs 762. Fusion precedent says expect wall-neutral.

- kv-store chain fusion: LANDED closed-default, measured wall-neutral (the
  legacy chain was already elementwise-fused; realistic best case was 1:1).
- rmsnorm: 145 kernels / 431 us, 2/layer vs llama's 1.
- residual add + ffn: 144 kernels / 252 us (llama has zero add/residual
  kernels).
- vocab head + scatter: 5 kernels / 383 us (fused vocab head 325.09 us vs
  llama 303.75 us).
- q-rope + q-cast remainder: ~90 kernels.

### 6.4 Lever 4 - depth scaling. fp16 cache halves the depth-penalty delta.

Landed capability-based (cache dtype now from renderer `supported_dtypes()`,
never a backend string). Measured: fp16 route -11.9% vs fp32 control -14.0%
(d512 -> d4096), llama -9.3%. +2.9% at d4096.

### 6.5 Lever 5 - prefill host-side replay (B3). Separate regime.

The historical fused-ON pp512 rows (warm wall 44-46 ms vs busy 24.1 ms) are
unreproducible: the fused prefill path fails deterministically at HEAD with a
`PACKED_FRAGMENT_LOAD` UOp verification failure. Measured baseline is the
fused-OFF tuned schedule (160.70 ms warm wall, 138.0 ms busy, 6 graph
groups); its wall-minus-busy residual (22.53 ms, 99.3% INFERRED) is not
explained by submit (0.17 ms OBSERVED) or poll (0.00 ms OBSERVED). Prefill
ratio today: 0.77x at pp512, 0.97x at pp1024, 1.05x at pp2048, 0.99x at
pp4096 (historical rows).

## 7. Key design decisions and constraints

- **Primitive route preference.** The generic, target-agnostic path is
  preferred (multi-compute-queue on the native NV substrate). DEV=CUDA is the
  non-primitive fallback and changes the story from generic to NVIDIA-
  specific. The probe (6.1) was built to test the primitive route and it
  failed on this host.
- **Closed-default discipline.** Shared runtime changes must keep behavior
  identical at defaults; variants land behind closed defaults with tests and
  are promoted only with isolated same-session wall benefit.
- **Evidence classes.** OBSERVED (measured, with session provenance),
  INFERRED (arithmetic/attribution from OBSERVED rows). No promotion without
  measured endpoint winners; node-sum projections rank experiments only.
- **The copy tax.** The opaque custom-kernel boundary materializes copies for
  lazy producers. Measured consequences: M3 fused RMSNorm -3% wall, M4 q4k
  epilogue absorption -18.8%, M5 flash combine net zero, Path 3 semantic
  RMSNorm -0.9% to -1.3% wall + 110 kernels. The view-preserving boundary
  (binding producer views through symbolic) is the stated precondition for
  epilogue absorption; it crashes today in `rangeify.py:1085`
  (`symbolic+reduce_collapse+debuf`).
- **No promotion, no user files.** Nothing promotes to `dev`/`exp`/`master`;
  `docs/README.md`, `docs/beating-llama-first-principles-20260731.md`,
  `docs/what-makes-a-token-fast-20260731.md`, `extra/llm_research/microbench/
  *` binaries, and `scratchpad/t6_metal_admission_probe.py` are user files
  and must never be committed.

## 8. Assumptions to challenge

Number each one TRUE / FALSE / UNPROVEN with the evidence that would settle
it:

1. llama's ~22% graph overlap is the dominant structural advantage, and the
   ~1.2 ms sequential non-GEMV tail is ours to recover.
2. The GEMV like-for-like comparison is fair (quantize-excluded, node-filtered
   llama vs our GEMV-class census), and the ~0.6 ms cap is nearly spent.
3. The native multi-GPFIFO probe verdict (zero concurrency) is a hardware/
   driver property, not an artifact of our channel setup. Note the
   separate-ctxshare setup gap: the probe only proved that channels sharing
   one context share serialize; it did not complete an independently
   scheduled-context run.
4. Decode is 95% GPU busy, so the wall is kernel execution, not host or
   launch overhead.
5. The W==D harness methodology and the llama-bench comparison (mean vs
   median, same model/config, same session) are valid.
6. The five levers are additive and their optimistic sum reaches parity.
   Sanity arithmetic: our node-sum 6.02 ms minus the full 0.6 ms GEMV cap =
   5.42 ms; with llama-like 22% overlap that is ~4.23 ms wall = ~236 tok/s
   vs llama 251.8 = ~0.94x. Even optimistic levers do not close the gap.
   What is missing from this arithmetic?
7. `B_route` for our decode is really ~46% of peak bandwidth, and the route
   would be bandwidth-bound at parity speeds (i.e., no hidden compute or
   latency bound).
8. The depth penalty delta (-11.9% vs -9.3%) is a KV-read side problem and
   fp16 cache is the right fix.

## 9. Grounding artifacts

- `nv-decode-parity-final-20260802.md` (wall authority; amendment: M2 open for
  NV:sm_120, M3/M4/M5/Path3 closed)
- `nv-decode-gap-decomposition-record-20260803.md` (llama node trace, our
  per-layer attribution, 1.67 ms decomposition)
- `decode-replay-overlap-measurement-record-20260803.md` (0.0% overlap,
  d512/d4096; artifacts `docs/five-lever-test-20260803-overlap-d512.json`,
  `-d4096.json`)
- `five-lever-test-record-20260803.md` (L1-L5 proof runs)
- `nv-multi-compute-queue-execution-scope-20260803.md` +
  `nv-multi-compute-queue-probe-measurement-record-20260803.md` +
  `docs/five-lever-test-20260803-multiqueue-probe.json` (probe E1-E5, verdict
  FAIL)
- `extra/llm_research/decode/nv_multi_queue_probe.py` (probe source)
- `decode-kv-store-chain-fusion-scope-20260803.md` (wall-neutral fusion
  precedent, 948 -> 948)
- `like-for-like-cap-settling-record-20260803.md` (3.836 vs 3.24 ms)
- `decode-gemv-efficiency-forward-scope-20260803.md` (scopes A/B/C/D)
- `executable-taskgraph-ir-scope-20260803.md` (five-lever ranking rationale)
- `nv-parity-and-beyond-forward-scope-20260803.md` (canonical forward
  authority, lifecycle states)
- `docs/what-makes-a-token-fast-20260731.md` (the principles document the
  section 2 model is drawn from; user file, do not edit)
- Harness: `extra/llm_research/decode/decode_runtime_overhead.py`,
  `extra/llm_research/decode/gemv_class_census_nv.py`

## 10. What we want back

Findings first, ordered by severity, grounded in file/line references where
possible:

1. The single most likely clear reason we cannot see for failing to reach
   parity, stated as a falsifiable claim.
2. Verdicts on assumptions 1-8 (TRUE / FALSE / UNPROVEN).
3. Whether the five levers can close the gap at all (Q6 arithmetic), and if
   not, what the missing term is.
4. The three cheapest decisive experiments, ranked, each with the exact
   measurement that would flip our current belief.
5. Any evidence-class errors (OBSERVED claims that are really INFERRED) in
   sections 5-6.
