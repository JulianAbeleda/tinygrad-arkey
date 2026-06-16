# GPU Performance: First Principles, Measurement, and Our Campaign Mapping

Audience: anyone optimizing kernels in this repo (or reasoning about why a
decode/GEMM path is slow). This is the canonical reference for *what can limit a
GPU kernel*, *how to measure each limit*, and *which limit each problem we hit
actually was*. Written in the spirit of `structure/Development/coding-principles.md`:
centralize the model in one source of truth, define invariants explicitly, give
a machine-checkable way to measure each, and explain tradeoffs near the concept.

> Core rule: **every kernel is limited by exactly one of three things at a time —
> moving bytes, doing math, or overhead.** Find which, measure it, attack only
> that. Everything below is an elaboration of that rule.

## The Meta-Model: Roofline

Roofline is not one of the three limits; it is the model that tells you *which*
of them binds you. Plot a kernel by its **arithmetic intensity** (useful FLOPs ÷
bytes moved from memory) against the machine's **ridge point** (peak FLOP/s ÷
peak bandwidth):

- Left of the ridge (low intensity) → **memory-bound** (bucket 1).
- Right of the ridge (high intensity) → **compute-bound** (bucket 2).
- Below both lines for no architectural reason → **overhead-bound** (bucket 3).

**Measure:** compute arithmetic intensity analytically (bytes of weights+activations
÷ FLOPs), and place it against the gfx1100 ridge point. **Our mapping:** batch-1
Q4_K decode GEMV has intensity ~2 ops/byte — far left of the ridge → memory-bound.
This single number explained why four schedule-knob families plateaued: they
reshuffle compute, which is not the binding resource.

---

# Bucket 1 — Bound by Bytes (Memory)

You are waiting on data. The levers below all change *how many bytes move* or
*how efficiently they move*.

## 1.1 Memory hierarchy
**Definition:** the tiers data lives in, fastest/smallest to slowest/largest —
registers → shared memory (LDS on AMD) → L1 → L2 → HBM/global (VRAM). Each tier
differs in latency and bandwidth by orders of magnitude.
**Measure:** which tier a value lives in (read the generated code / ISA); cache
hit rates via `rocprof` (L1/L2 hit %).
**Our problem:** the 32B sidecar-vs-shared-storage work — sidecar buffers blew
the VRAM *capacity* budget (OOM at 23.78 GB). Capacity is the hierarchy's
hardest tier limit; `QK_PRIMITIVE_STORAGE=shared` (typed views over the raw GGUF
buffer) removed the duplicate, fitting 32B at `storage_bytes=0`.

## 1.2 Bandwidth
**Definition:** peak bytes/sec deliverable from a tier (usually HBM→compute).
For a memory-bound kernel, runtime ≥ bytes_moved ÷ bandwidth — a hard floor.
**Measure:** achieved GB/s = bytes_moved ÷ kernel_time, reported as **% of peak
bandwidth** (the deterministic metric — far better than noisy tok/s). On AMD use
`rocprof`/`rocm-smi` memory-throughput counters.
**Our problem:** THE central problem. We sit at ~52–62% of llama.cpp; the
literature shows naive 4-bit AMD decode at ~49% of peak vs ~91% optimized. The
gap is bandwidth *efficiency*, and quantization (4-bit vs 16-bit = 4× fewer
bytes) is our single biggest bandwidth lever.

## 1.3 Coalescing / access pattern
**Definition:** adjacent lanes (threads) reading adjacent addresses collapse
into one wide memory transaction; strided/scattered access wastes most of each
transaction. Load *width* (e.g. 128-bit `uint32x4`) is part of this.
**Measure:** inspect emitted load instructions for width (our DEBUG=4 load-width
report: `vector_u32x4` vs scalar); profiler memory-transaction / fetch
efficiency.
**Our problem:** the wide-load thread (`uint32x4` devectorizer change, the
`vector_load` vs `tile_custom` three-way). Verdict: wide loads are **necessary
plumbing but not sufficient** (−8.58% vs v1 with matched parallelism) — the
remaining cost was downstream, not load width.

## 1.4 Locality / reuse / tiling
**Definition:** reuse a byte while it is in a fast tier instead of re-reading it
from a slow one. Tiling is the technique; it is what *raises arithmetic
intensity* and can move a kernel from memory-bound toward compute-bound.
**Measure:** arithmetic intensity (FLOPs ÷ unique bytes); reuse factor; L1/L2
hit rate.
**Our problem:** the direct-output vs split-K reduction experiments, and the
Phase-B pivot to a **fused Q4_K GEMM + batching** — batching is precisely
raising reuse (one weight read serves many tokens), the only way off the
batch-1 bandwidth floor.

## 1.5 Latency hiding
**Definition:** memory access has huge *latency*; you hide it not by faster
memory but by having enough independent in-flight work (warps/wavefronts, ILP)
so the scheduler runs other work while a load is outstanding.
**Measure:** memory-stall cycles / scheduler-stall reasons (`rocprof`); achieved
vs theoretical bandwidth (poor hiding shows as bandwidth far below peak despite
a memory-bound kernel).
**Our problem:** implicit in the LOCAL:0:32 parallelism — the opaque
`tile_custom` kernel that dropped 32-lane parallelism cratered to −90%, because
with no in-flight wavefronts there was nothing to hide memory latency behind.

---

# Bucket 2 — Bound by Math (Compute)

You are waiting on the ALUs. Levers change *how much useful math per cycle*.

## 2.1 Instruction throughput / ILP
**Definition:** useful instructions retired per cycle; depends on instruction
mix, dependency chains (instruction-level parallelism), and avoiding
low-throughput ops.
**Measure:** VALU utilization / instruction mix (`rocprof`); the gap between
`device_q4_eff` (kernel-time throughput) and the roofline compute ceiling.
**Our problem:** the "inner dequant/dot cost" the wide-load result pointed at —
once loads were wide, the bottleneck was the dequant arithmetic, i.e. compute
throughput of the unpack+multiply.

## 2.2 Special-function units & precision
**Definition:** hardware paths beyond scalar FMA — matrix units (WMMA/MFMA),
integer dot (`dp4a` / `v_dot4`), and low-precision modes (FP8/FP4). Using the
right one can be a multiple, not a percent.
**Measure:** confirm the intrinsic is actually emitted (read generated source /
ISA — our `_dp4a` helper, the `q8_1_intdot`/`v_dot4` paths); compare against the
scalar baseline.
**Our problem:** the `q8_1` intdot and `v_dot4` experiments. Note: at batch-1
GEMV (M=1) matrix units (WMMA) are largely wasted — they belong to the Phase-B
*prefill/GEMM* track, not decode.

## 2.3 Divergence
**Definition:** SIMT executes a warp/wavefront in lockstep; a data-dependent
branch that splits lanes serializes both sides, wasting lanes.
**Measure:** active-lane / divergence counters (`rocprof`); inspect for
data-dependent branches in the hot loop.
**Our problem:** not a primary limiter for our straight-line GEMV kernels, but
the reason wavefront-width (32 on RDNA) decomposition matters — keep all 32
lanes doing useful work.

---

# Bucket 3 — Overhead (Neither Bytes Nor Math)

The kernel is fast but you are paying tax around it.

## 3.1 Launch / dispatch overhead
**Definition:** fixed cost to launch a kernel (dispatch, argument setup). Murder
for *many tiny kernels* — exactly batch-1 decode, where each token triggers
launches.
**Measure:** **the gap between wall-clock throughput and device throughput** —
we already capture both (`q4_eff` = wall GB/s, `device_q4_eff` = device GB/s);
the spread between them *is* the overhead bucket. Also kernel count per token.
**Our problem:** why JIT / graph capture matters; why microbench wall numbers
diverged from device numbers (the tile-lowering +7.2% wall vs the −90% device
reconciliation — different buckets).

## 3.2 Synchronization / atomics / barriers
**Definition:** cross-thread coordination cost — `__syncthreads`/barriers,
atomics, partial-reduction read/write traffic.
**Measure:** barrier/atomic counts; partial-reduction memory traffic.
**Our problem:** the split-K partial-reduction question (write partials → reduce)
— whether the reduction traffic was worth it; for some tensors (8B ffn-down)
direct-output without the reduction was catastrophically worse, meaning the
reduction was doing real work there.

---

# Cross-Cutting: Utilization (enables Buckets 1 & 2)

These are not limits themselves; they are how fully you use the machine, and
they trade off against each other.

## U.1 Occupancy
**Definition:** resident warps/wavefronts per compute unit. Enough is required
to hide latency (1.5); but **high occupancy ≠ fast** — past a point, more
occupancy costs registers/shared memory and can slow things down.
**Measure:** achieved occupancy (`rocprof`); occupancy calculator vs register/
LDS usage.

## U.2 Parallelism / work decomposition
**Definition:** enough total threads and the right block/workgroup geometry to
fill the machine. Our `parts` / `LOCAL` knobs are exactly this.
**Measure:** grid/block sizing vs CU count; idle CUs.
**Our problem:** the entire parts/LOCAL schedule-knob search — which we proved
*exhausted* for this kernel, because decomposition (a compute/utilization lever)
cannot move a memory-bound kernel.

## U.3 Register pressure / spilling
**Definition:** too many live registers → spill to (slow) local memory, and caps
occupancy. A hidden tax on aggressive unrolling/tiling.
**Measure:** compiler register count + spill warnings (ROCm/LLVM); occupancy drop.

---

# How to Use This Document

1. **Find the bucket first (roofline).** Compute arithmetic intensity; decide
   memory vs compute vs overhead. Do not optimize a primitive in the wrong
   bucket — that was the lesson of the exhausted schedule-knob search.
2. **Measure deterministically.** Prefer counter-based metrics (% peak bandwidth,
   occupancy, instruction mix) over wall-clock tok/s; the wall↔device gap is your
   overhead read.
3. **Attack only the binding limit.** For our batch-1 decode that is *bandwidth +
   coalescing/locality*; for Phase-B batching/GEMM it shifts toward *compute +
   special units + tiling*.

# Persons of Interest & Canonical Reading

- **Roofline model** — Samuel Williams, Andrew Waterman, **David Patterson**,
  "Roofline: An Insightful Visual Performance Model" (CACM 2009). The source of
  the meta-model. <https://dl.acm.org/doi/10.1145/1498765.1498785>
- **Horace He — "Making Deep Learning Go Brrr From First Principles."** The best
  single read here: it frames *everything* as compute- / memory- / overhead-bound,
  applied to DL kernels. Start with this.
  <https://horace.io/brrr_intro.html>
- **Simon Boehm — "How to Optimize a CUDA Matmul Kernel."** The legendary
  step-by-step that walks the primitives in order, each unlocking the next.
  <https://siboehm.com/articles/22/CUDA-MMM>
- **Stephen Jones (NVIDIA)** — GTC talks "How GPU Computing Works" / "How to
  Write a CUDA Program" — best first-principles video on memory hierarchy and
  latency hiding.
- **NVIDIA CUDA C++ Best Practices Guide** — canonical on coalescing, occupancy,
  bandwidth. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- **AMD — GPUOpen + ROCm/RDNA ISA & optimization guides.** Our actual hardware;
  the RDNA WMMA guides are theirs. <https://gpuopen.com/>
- **Books:** Kirk & Hwu, *Programming Massively Parallel Processors* (the GPU
  bible); Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*
  (where roofline lives).
- **Community:** **GPU MODE** (formerly CUDA MODE) — lectures + Discord, serious
  kernel practitioners on exactly these primitives.
- **Autotuning / cost-model side** (formalizing these primitives into search):
  Jonathan Ragan-Kelley (Halide), Tianqi Chen + the TVM/Ansor crew.

## Distilled Principles by Source

The actual primitives/lessons each source teaches, mapped to the buckets above.

### Horace He — "Go Brrr" (the three-bucket frame)
- Every kernel is **compute-bound, memory-bandwidth-bound, or overhead-bound** —
  diagnose *which* before optimizing (the meta-model).
- **Most DL ops are memory-bound, not compute-bound** (pointwise, norm,
  activation, and — for us — batch-1 decode). You are rarely actually hitting the
  FLOPs ceiling.
- **Operator fusion is the #1 memory-bandwidth win** — fuse pointwise/dequant ops
  so you read & write the data once (maps to 1.4 reuse; this is *why* fused Q4_K
  dequant matters).
- **Overhead (framework/dispatch) dominates for small ops** — fix with tracing /
  CUDA-graph capture / compilation (maps to 3.1).
- **Diagnostic test:** does making the op *bigger* change runtime proportionally?
  Scales with data → memory-bound; flat regardless of size → overhead-bound;
  scales with FLOPs → compute-bound.

### Simon Boehm — "Optimize a CUDA Matmul" (the primitives, in order)
Each step targets one primitive and unlocks the next:
1. **Global-memory coalescing** (1.3) — consecutive threads → consecutive
   addresses. First and biggest single win.
2. **Shared-memory cache-blocking / tiling** (1.4) — stage a tile in LDS, reuse
   it; raises arithmetic intensity.
3. **1D then 2D register blocktiling** (1.4 + U.3) — each thread computes
   multiple outputs from registers → more reuse, fewer memory trips.
4. **Vectorized memory access** (1.3) — 128-bit / `float4` loads.
5. **Autotune the tile sizes** (U.2) — and **warptiling** (U.1).
- Lesson order is the lesson: **coalesce → tile for reuse → register-reuse →
  vectorize → tune occupancy.** Don't skip ahead; each exposes the next bottleneck.

### NVIDIA CUDA C++ Best Practices (the priority hierarchy)
NVIDIA gives an explicit *order of priority* — optimize in this sequence:
1. **Maximize parallel execution / occupancy** (expose enough parallelism) (U.1/U.2).
2. **Optimize memory usage** — coalesced global access, then shared memory, then
   minimize host↔device transfers (1.2/1.3/1.1). *Memory before instructions.*
3. **Optimize instruction usage** last — intrinsics, avoid divergent branches,
   reduce low-throughput ops (2.1/2.3).
- Framing: **APOD** — Assess (profile to find the bottleneck), Parallelize,
  Optimize, Deploy. "Memory optimizations have the highest priority" is the
  through-line.

### AMD ROCm / RDNA best practices (our hardware)
- **Wavefront = 32 on RDNA** (64 on CDNA) — occupancy is counted in wavefronts;
  decompose so all 32 lanes do useful work (U.1, 2.3).
- **LDS (shared memory)** usage and **bank conflicts** — the AMD analog of
  shared-mem tiling (1.1/1.4).
- **Wide/coalesced loads** (128-bit) to saturate memory channels — the "wide-K"
  pattern (1.3); this is the lever our roofline points at.
- **`v_dot`/WMMA/MFMA** for low-precision math (2.2) — relevant to Phase-B GEMM,
  largely wasted at batch-1 decode.

### Books (the canonical primitives)
- **Kirk & Hwu, PMPP:** thread+memory hierarchy, coalescing, tiling, control
  divergence, occupancy, and the **compute-to-global-memory-access ratio**
  (= arithmetic intensity) — the same primitives, taught from scratch.
- **Hennessy & Patterson:** the **roofline**, **Amdahl's law** (speedup is capped
  by the un-optimized fraction), the **memory wall**, and the "quantitative
  approach" — *measure, don't guess*, which is the discipline this whole doc
  encodes.

## Recommended path
Read Horace He ("Go Brrr") to lock the three-bucket frame, then Simon Boehm's
matmul post to see the primitives applied in sequence. Those two give ~80% of the
working model. Use the NVIDIA Best Practices Guide and AMD RDNA ISA guide as
reference manuals for knob behavior on the target hardware.

## Note on scope for this campaign
We are (decode) almost entirely in **Bucket 1 + launch overhead**. The compute-
side primitives (2.1/2.2, WMMA) only become *our* problem once Phase-B batching
pushes the kernels compute-bound. Don't tune compute primitives while bandwidth
binds — and vice versa once batching lands.
