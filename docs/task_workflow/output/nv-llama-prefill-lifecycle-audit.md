# NV pp512 prefill lifecycle audit: tinygrad versus llama.cpp

## Verdict

The corrected tinygrad path is legitimately slower than llama.cpp at Qwen3-8B
pp512.  The historical tinygrad result near 46 ms is not a useful comparator:
it lost LDS stores and aliased output rows.  The valid authority is 84.025 ms
(6,093 prompt tok/s), versus the retained same-session llama.cpp average of
36.608 ms (14,074 prompt tok/s).

The 47.417 ms wall gap is overwhelmingly a dense-projection representation and
service-rate gap.  It is not explained by Flash Attention, launch count,
overlap, fusion, `cp.async`, or TMA.

The two projection paths have surprisingly similar large-role *algorithmic
weight-stream rates* after accounting for the four 128-token M tiles.  What
differs is the object being streamed:

- tinygrad materializes and consumes FP16 weights, 16 bits/value;
- llama keeps Q4_K/Q6_K weights compressed, quantizes activations to Q8_1, and
  consumes both through INT8 tensor-core MMA.

For gate/up, the inferred tiled weight stream is about 0.71 TB/s in tinygrad
and 0.68 TB/s in llama.  For the Q6 down projection it is about 0.76 TB/s on
both.  Llama therefore gets almost the expected 3.56x Q4 or 2.44x Q6 increase
in useful work from moving fewer weight bits through a similar service pipe.
This is stronger evidence than the vague claim that llama merely has a better
tile.

There is also a real graph-liveness advantage: llama gathers only requested
output rows after the final attention block, so the last FFN runs at M=1.
Tinygrad prunes only the LM head and still runs the last FFN at M=512.  The
final O projection remains full-batch on both sides because it is part of the
attention block before the gather.  This pruning is worth roughly 1--2 ms on
the current tinygrad profile, not the full gap.

## Scope and authority

Target and workload:

- RTX 5090, sm_120;
- `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`;
- prompt length and ubatch 512;
- dense Qwen3-8B, Flash Attention enabled;
- tinygrad corrected code at `dff21c0cb`;
- llama.cpp CUDA build at `ac4cddeb0`.

Wall authorities:

- tinygrad:
  `docs/task_workflow/evidence/nv-prefill-corrected-tile-20260828/authority-complete-r7.json`;
- llama:
  `docs/task_workflow/evidence/nv-prefill-restoration-20260828/llama-pp512-r5.json`.

Fresh checks made during this audit:

- unprofiled corrected tinygrad: `/tmp/tiny_cuda_pp512_audit.out`, 83.9 ms;
- llama CUDA Nsight Systems capture:
  `/tmp/llama_cuda_pp512_audit.nsys-rep`, 33.096 ms GPU busy and
  44.840 ms profiled wall;
- llama kernel summaries:
  `/tmp/llama_cuda_pp512_kerns.csv`,
  `/tmp/llama_cuda_pp512_gb.csv`, and
  `/tmp/llama_cuda_pp512_trace.csv`;
- tinygrad HCQ graph profile:
  `/tmp/tiny_pp512_graph_profile_audit.jsonl`, 95.810 ms summed device entries
  under `PROFILE=1`.

The profiler walls are not substituted for the authority walls.  Nsight adds
a large host-side gap to the llama run, while `PROFILE=1` slows the tinygrad
run from about 84.0 to about 96.4 ms.  Profile entries are used for category
shares, kernel counts, and per-role comparisons; the R7/R5 files remain the
wall authority.

## Apples-to-apples wall result

| implementation | valid pp512 wall | reported prompt rate | samples | status |
| --- | ---: | ---: | --- | --- |
| tinygrad corrected candidate | 84.025 ms | 6,093 tok/s | 84.025--84.328 ms, R7 | finite, four-role bit-exact, model SHA/argmax equal to safe FP16 route |
| llama.cpp CUDA MMQ | 36.608 ms | 14,074 tok/s | 34.822--42.747 ms, R5 | retained same-session comparator |
| gap | +47.417 ms | 0.433x llama throughput | | real |

`512 / 0.036608 = 13,986 tok/s` differs slightly from llama's 14,074 field
because llama averages per-sample throughput rather than taking the reciprocal
of average time.

## Device lifecycle accounting

The next table totals each *profiled* regime.  It is intentionally separate
from the authority wall table so profiler overhead is not hidden.

| lifecycle region | tinygrad `PROFILE=1` | share of 95.810 ms entries | llama Nsight GPU busy | share of 33.096 ms busy |
| --- | ---: | ---: | ---: | ---: |
| full-batch dense projections | 85.981 ms, 252 calls | 89.7% | 27.263 ms, 249 MMQs plus quant/fixups | 82.4% |
| Flash Attention main + reduction | 3.286 ms, 36 calls | 3.4% | 1.714 ms, 72 calls | 5.2% |
| vocabulary head | 2.880 ms | 3.0% | 0.311 ms MMVQ | 0.9% |
| norms, rope, residuals, KV writes, activation | 3.662 ms | 3.8% | 3.808 ms | 11.5% |
| device total | 95.810 ms, 1,018 entries | 100% | 33.096 ms, 1,186 calls | 100% |
| profiled wall / residual | about 96.4 ms / about 0.6 ms | | 44.840 ms / 11.744 ms | |

The 11.7 ms llama residual is profiler-induced: the trace contains a roughly
12 ms host gap before the final-token MMVQ tail.  The unprofiled authority has
only about 3.5 ms beyond the fresh 33.1 ms busy sum.  It must not be charged as
normal llama work.

Even though llama launches more kernels, it is much faster.  Tinygrad submits
six HCQ graph groups in the captured pass; the prior llama graph capture has
one serialized CUDA stream and essentially no overlap.  Kernel-count removal
cannot explain a 47 ms gap.

### Dense projection decomposition

Tinygrad's instrumented candidate times are:

| role | calls | shape `(M,N,K)` | CTAs/call | average/call | total | useful rate in profiled regime |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Q/O | 72 | 512,4096,4096 | 128 | 183.5 us | 13.210 ms | 46.8 TMAC/s |
| K/V | 72 | 512,1024,4096 | 32 | 177.3 us | 12.769 ms | 12.1 TMAC/s |
| gate/up | 72 | 512,12288,4096 | 384 | 569.7 us | 41.020 ms | 45.2 TMAC/s |
| down | 36 | 512,4096,12288 | 128 | 527.3 us | 18.982 ms | 48.9 TMAC/s |
| total | 252 | 3.556 TMAC | | | 85.981 ms | 41.4 TMAC/s |

The unprofiled wall is 12.4 ms lower, so 41.4 TMAC/s is not an authority
service rate.  Proportionally mapping the profile share to the authority wall
puts projection service near 47 TMAC/s.  This is an estimate, not a measured
kernel sum.  Counting one multiply and one add as two operations, those values
are about 82.7 effective TFLOP/s in the instrumented regime and 94 effective
TFLOP/s in the wall-scaled estimate.

Fresh llama dense kernels:

| component | calls | total |
| --- | ---: | ---: |
| Q4_K `mul_mat_q<12,128>` | 214 | 20.057 ms |
| Q6_K `mul_mat_q<14,128>` | 35 | 4.253 ms |
| Q4 stream-K fixup | 214 | 1.782 ms |
| Q6 stream-K fixup | 35 | 0.312 ms |
| Q8_1 activation conversion | 249 | 0.859 ms |
| total dense lifecycle | | 27.263 ms |

Llama performs about 3.479 TMAC in these full-batch MMQs because it prunes the
last layer after attention.  The raw MMQs service about 143 TMAC/s; including
activation conversion and stream-K fixups gives about 128 TMAC/s.  The prior
measured tinygrad exact FP16 issue ceiling is 127.7 TMAC/s.  Llama's raw useful
rate can exceed that FP16 ceiling because it uses INT8 tensor-core MMA.  In
two-operation convention these are about 286 effective TFLOP/s raw and 255
effective TFLOP/s for the complete quantize/MMQ/fixup lifecycle.

### Weight representation and algorithmic traffic

Q4_K stores 256 values in 144 bytes, or 4.5 bits/value.  Q6_K stores 256
values in 210 bytes, or 6.5625 bits/value.  Relative to an FP16 overlay:

- Q4_K has 3.56x fewer weight bytes;
- Q6_K has 2.44x fewer weight bytes.

For the 36-layer dense projection population alone, the resident tinygrad FP16
overlay is about 13.89 GB.  Applying the actual mixed Q4/Q6 population's block
sizes gives about 4.41 GB of packed projection weights, a 3.15x aggregate
representation ratio.  Overlay construction is a one-time model-load cost and
is not charged to the 84 ms wall; its larger payload is charged every time the
projection kernels stream their tiles.

Both kernels own 128 rows on the M axis, so an M=512 projection has four M
tiles.  Counting the weight payload once per M tile gives the following
analytic stream rates.  These are not DRAM-counter measurements; they are the
bytes the tiled algorithm must request before cache effects.

The Q4_K_M file is mixed: gate/up and the ordinary Q/K/O population are Q4;
V and down include both Q4 and Q6 tensors (18 Q6 V tensors and 18 Q6 down
tensors in the model census).  The table therefore shows both packed cases.

| role | tiny FP16 tiled weight bytes / time | tiny rate | llama packed bytes / time | llama rate |
| --- | ---: | ---: | ---: | ---: |
| Q/O | 134.2 MB / 183.5 us | 0.73 TB/s | 37.75 MB / about 61.5 us | 0.61 TB/s |
| K/V Q4 | 33.55 MB / 177.3 us | 0.19 TB/s | 9.44 MB / about 20 us | 0.47 TB/s |
| V Q6 | 33.55 MB / 177.3 us | 0.19 TB/s | 13.76 MB / about 31 us | 0.44 TB/s |
| gate/up | 402.7 MB / 569.7 us | 0.71 TB/s | 113.2 MB / about 166 us | 0.68 TB/s |
| down Q4 | 402.7 MB / 527.3 us | 0.76 TB/s | 113.2 MB / about 166 us | 0.68 TB/s |
| down Q6 | 402.7 MB / 527.3 us | 0.76 TB/s | 165.2 MB / about 217 us | 0.76 TB/s |

The near equality for gate/up and down is the cleanest causal signal in the
audit.  The current tinygrad kernel does not merely need a few instruction
cleanups: it is streaming a 2.44--3.56x larger weight representation at
roughly the same byte-service rate.

The local movement differs in the same direction.  Tinygrad stages FP16 A and
B values in shared memory at two bytes/value.  Llama stages signed-int8 Q8
activations and unpacked signed-int8 weight values at one byte/value, alongside
smaller FP16 scale/correction metadata.  Exact shared-sector counts still need
NCU, but the source-level payload is materially smaller on both the global and
shared sides of llama's MMA instruction.

K/V adds an independent topology loss.  Tinygrad launches only 32 CTAs on a
170-SM GPU, so at most 19% of SMs receive a CTA.  Llama stream-K launches 170
CTAs and splits K work, then reduces partials in a fixup kernel.

## Projection kernel pseudocode

### tinygrad corrected FP16-overlay candidate

The promoted artifact declares a 128x128x32 CTA tile, 256 threads / eight
warps, FP16 A/B, FP32 accumulation, FP16 C, 16-byte cooperative loads, padded
shared strides of 80, and one producer/consumer stage.  Its `buffer_count=2`
reservation does not make the exported epoch graph a two-stage asynchronous
pipeline: the artifact still declares a single body stage and barriers before
fragment reads and before slot reuse.

```text
once at model load:
  for every covered quantized linear:
    W_fp16 = materialize(dequantize(W_q4_or_q6), contiguous=True)

for each projection A[M,K] @ W_fp16[N,K]^T:
  grid = (ceil(M/128), ceil(N/128))
  block = 256 threads = 8 warps
  each CTA owns one complete 128x128 output tile
  accum[warp fragments] = fp32 zero

  for k0 in 0..K step 32:
    threads cooperatively issue 16-byte global loads
    store A[128,32] and B[128,32] to padded shared memory
    __syncthreads()                         # before fragment load

    for each warp-owned 16x8 output fragment:
      load FP16 A/B fragments from shared
      accum = mma.sync.m16n8k16.f32.f16.f16(accum, A, B)

    __syncthreads()                         # before slot reuse

  scalar lane-mapped epilogue:
    C_fp16 = cast(accum)
```

There is no K split across CTAs.  Grid sizes are therefore 32 for K/V, 128 for
Q/O and down, and 384 for gate/up.  No `cp.async` or TMA use is evidenced by
the promoted artifact or generated schedule.

### llama.cpp Q4_K/Q6_K x Q8_1 MMQ

Source is `ggml/src/ggml-cuda/quantize.cu` and
`ggml/src/ggml-cuda/mmq.cuh`.  `ggml_cuda_should_use_mmq` returns true whenever
Turing MMA is available, so this pp512 path is MMQ rather than cuBLAS.

```text
for each projection input X_fp32[M,K]:
  quantize kernel:
    each lane reads float4
    warp shuffle-reduce absmax (and sum when the weight format needs it)
    q = round(x * 127/absmax)
    write packed signed-int8 Q8_1 plus FP16 scale/sum metadata

  stream-K planner:
    make exactly 170 main CTAs on this 170-SM GPU
    assign each CTA an output-tile/K interval

  MMQ CTA: 256 threads = 8 warps, nominal 128x128 output tile
    int32 MMA fragment accumulators = zero
    for assigned K interval in 256-value iterations:
      cooperatively load packed Q4_K or Q6_K weight blocks
      unpack/sign-extend quant values and load block scale/min metadata
      cooperatively load packed Q8_1 activation tiles
      place packed int8 tiles and metadata in shared memory
      __syncthreads()

      load fragments; issue mma.sync.m16n8k16 signed-int8 tensor-core MMA
      apply Q8 and weight scales and Q4 zero/min correction into fp32 sums
      __syncthreads()

      load the second half of the Q8 tile
      __syncthreads()
      issue more INT8 MMA and corrections
      __syncthreads()

    if CTA owns a complete result:
      write FP32 result
    else:
      write stream-K partial to scratch

  stream-K fixup kernel:
    combine partials and publish FP32 result
```

The binary evidence is unambiguous: the Q4_K specialization contains 512
`IMMA.16832` instructions and the Q6_K specialization contains 512
`IMMA.16816`; neither contains HMMA or IDP.4A.  This is INT8 tensor-core MMA,
not DP4A.  The MMQ source uses ordinary global/shared operations and
`__syncthreads`; it does not use the repository's separate `cp-async.cuh` and
does not use TMA.

The captured main kernels launch with grid `(170,1,1)`, block `(32,8,1)`, one
CTA/SM launch bounds, about 251--255 registers/thread, and 57,856 bytes dynamic
shared memory.

## End-to-end topology and final-layer pruning

For layers 0--34 llama executes:

```text
rms -> Q8 -> Q MMQ -> qnorm/rope
    -> Q8 -> K MMQ -> knorm/rope
    -> Q8 -> V MMQ -> KV write
    -> Flash + Flash fixup
    -> Q8 -> O MMQ -> residual
    -> rms -> Q8 -> gate MMQ
           -> Q8 -> up MMQ
           -> SiLU gate multiply
           -> Q8 -> down MMQ -> residual
```

At the final layer, llama's Qwen3 graph performs `ggml_get_rows` immediately
after attention when only the last token is requested.  Q/K/V/O have already
run full-batch, but the final FFN is therefore M=1 and uses MMVQ, including the
existing gate/up fusion machinery.  The output head is also an M=1 Q6 MMVQ.

Tinygrad's `logits()` loops over every block before `[:, -1, :]` is applied.
The lazy head is correctly pruned to one token, but the last block is not.
The profile consequently contains 36 full-batch gate, up, and down calls, not
35.  Removing the last full-batch gate/up/down work is expected to save about
1.67 profiled ms before the roughly 0.07 ms replacement vector tail, likely
around 1--2 ms wall.  This is a clean transferable graph optimization.

## Ranked causal differences

1. **Compressed representation plus INT8 tensor-core consumption — high
   confidence, dominant.**  Source, SASS, payload ratios, per-role timings, and
   matched algorithmic byte rates all agree.  Q8 conversion plus fixup costs
   only 3.0 ms total; they enable a roughly 58.7 ms reduction from tinygrad's
   instrumented projection time to llama's dense lifecycle.

2. **Stream-K SM-sized work distribution — high confidence for mechanism;
   medium confidence for isolated wall recovery.**  Fresh grid captures show
   170 llama CTAs for every large MMQ.  Tinygrad's static output ownership gives
   only 32 K/V CTAs and 128 Q/O/down CTAs.  K/V is the clearest victim: about
   12 TMAC/s versus 45--49 TMAC/s for tinygrad's other roles.

3. **Tinygrad FP16 candidate reaches only about one third of its measured
   issue ceiling — high confidence for observation; medium confidence for the
   internal stall decomposition.**  Candidate metadata shows K=32 staging,
   two full barriers per epoch, scalar epilogue, and no asynchronous copy.
   The exact shares of memory throttle, barrier stall, fragment issue, and
   occupancy still require hardware counters.  Llama is also barrier-heavy,
   so “add async copy” is not yet a supported conclusion.

4. **Last-layer requested-row pruning — high confidence, small-to-medium
   recovery.**  It is directly present in llama Qwen3 source and fresh trace,
   and absent around tinygrad's block loop.  Expected tinygrad wall recovery is
   around 1--2 ms.

5. **Compressed M=1 vocabulary head — high confidence, about 2.6 profiled ms
   of visible difference.**  Tinygrad's generic lazy head takes 2.880 ms in
   the instrumented pass; llama's Q6 MMVQ takes 0.311 ms.  This is not the
   principal projection gap but is a bounded independent target.

6. **Attention — high confidence, low maximum recovery.**  Tinygrad's
   instrumented attention is 3.286 ms versus llama's 1.714 ms.  Even perfect
   parity recovers only about 1.6 profiled ms.

7. **Launch/graph topology — high confidence that it is secondary.**  Llama
   has more kernel calls and no useful overlap in the captured pp512 stream.
   Tinygrad's six graph submissions could cost a few milliseconds, but not 47.

8. **Fusion, overlap, `cp.async`, and TMA — high confidence as rejected primary
   explanations.**  Llama gate/up and epilogues are separate for the 35
   full-batch layers.  Its MMQ source has no async-copy/TMA path.  Its kernels
   serialize in one stream.  The fast result does not require these features.

9. **Different numerical contract — high confidence, design constraint.**
   Tinygrad's corrected path is bit-exact against a safe FP16 route.  Llama
   rounds each projection input to Q8_1.  A direct llama-like transfer must be
   admitted with whole-model/token quality criteria; it cannot be bit-exact to
   the current FP16 comparator by construction.

## Smallest discriminating tests

### 1. Exact-role native primitive triangle

For each of the four shapes, time with identical inputs and resident weights:

1. current tinygrad FP16 candidate;
2. a minimal native CUDA FP16 128x128x32 kernel with the same static grid;
3. llama-extracted Q4/Q6 x Q8 MMQ including conversion and fixup.

Use R9 synchronized wall, output finite/error gates, cubin/SASS capture, and
one NCU pass.  Expected signal:

- if native FP16 is close to tinygrad, representation/topology is binding;
- if native FP16 is much faster, tinygrad code generation/staging is also a
  large recoverable layer;
- llama-like MMQ should be 2.5--3.5x faster on Q4 roles and 2--2.5x on Q6.

Potential whole-wall recovery if direct packed MMQ reaches llama's dense
lifecycle: about 35--48 ms, moving 84 ms to roughly 36--49 ms, or about
10.4k--14.2k prompt tok/s.  The top of this range changes numerical semantics
and is not a promotion claim.

### 2. K/V-only stream-K geometry gate

Keep FP16 arithmetic and split K so the 32-CTA K/V shape launches about 170
main CTAs plus a fixup.  Do this before writing an all-role MMQ.

Expected signal: a 2--5x K/V primitive improvement if underfill is dominant,
with a likely whole-wall recovery around 5--10 ms after fixup overhead.  A
10 ms recovery changes 84.0 ms / 6.1k to about 74.0 ms / 6.9k.  If K/V does
not move, counters should show that the role is limited by a different shared
or issue bottleneck.

### 3. Prepacked dtype-only tensor-core A/B

Run equal 128x128xK work with prepacked FP16/HMMA and signed-INT8/IMMA operands,
then add Q4 unpack/scales/corrections.  This separates tensor-pipe capacity from
quantization grammar.

Expected signal: prepacked INT8 exceeds the measured FP16 MAC ceiling; adding
Q4/Q6 corrections retains enough advantage to pay the sub-millisecond Q8
producer per lifecycle.  Failure here would falsify a tinygrad port before
model integration.

### 4. One-pass counters on one call per role

Collect actual DRAM bytes, L2 bytes/hit rate, tensor-pipe utilization, active
warps, barriers, and long/short scoreboard stalls for current versus llama.

Expected signal:

- similar L2/global requested rates for gate/up and down;
- materially larger tinygrad weight bytes;
- low tinygrad K/V SM occupancy from 32 CTAs;
- low tensor utilization or high barrier/memory stalls explaining the
  remaining gap to tinygrad's own FP16 ceiling.

This test converts the analytic byte-rate evidence into hardware accounting.

### 5. Final-layer output-row pruning

Add an experimental graph spelling that gathers requested rows immediately
after final attention, then runs final residual/FFN/head at M=1.  First compare
the final token hidden state and logits to the unpruned path.

Expected signal: remove two full gate/up and one full down candidate, replace
them with vector work, and recover around 1--2 ms wall.

### 6. FP16 staging sweep, only if test 1 leaves FP16 headroom

Sweep K tile 32/64/128/256, one/two effective stages, shared padding/layout,
and epilogue vectorization.  Do not start with `cp.async` or TMA; first prove
that barrier or copy stalls dominate with counters.

Plausible recovery is several milliseconds, not the full 47 ms.  At 100% of
the measured 127.7 TMAC/s FP16 ceiling, tinygrad's 3.556 TMAC alone takes about
27.8 ms.  Adding even 8--10 ms of nonprojection work leaves essentially no
robust margin over llama's 36.6 ms wall.  A practical 75--85% FP16 ceiling
lands around 41--46 ms total (about 11.1k--12.4k tok/s).

### 7. Attention and vocabulary tail

After dense mechanism tests, separately A/B the final vocabulary MMVQ and
attention lifecycle.  Visible upper bounds are about 2.6 and 1.6 profiled ms.
They matter near parity but cannot create parity by themselves.

## Recovery translation

Starting from the 84.025 ms authority:

| recovered wall | resulting wall | prompt tok/s | gain |
| ---: | ---: | ---: | ---: |
| 5 ms | 79.025 ms | 6,479 | +386 |
| 10 ms | 74.025 ms | 6,917 | +824 |
| 20 ms | 64.025 ms | 7,997 | +1,904 |
| 30 ms | 54.025 ms | 9,477 | +3,384 |
| 40 ms | 44.025 ms | 11,630 | +5,537 |
| 45 ms | 39.025 ms | 13,120 | +7,027 |
| 47 ms | 37.025 ms | 13,829 | +7,736 |

Stream-K alone cannot close the gap.  Last-layer pruning, attention, and vocab
together also cannot.  A compressed tensor-core projection representation is
the only currently evidenced mechanism with enough recovery to approach the
llama wall.

## Transferable principles versus llama-specific details

Transferable:

- preserve low-bit weights through the compute kernel instead of expanding a
  persistent FP16 overlay;
- co-design the activation representation with the weight format;
- use integer tensor-core MMA for quantized prefill;
- distribute K when an output-tile grid cannot fill the GPU;
- gather only requested output rows at the earliest graph point after which
  other rows are dead;
- judge producer + main + fixup as one lifecycle;
- use actual role shapes and quality gates, not isolated square GEMMs.

Llama-specific and not suitable for blind copying:

- the exact Q8_1 DS4/D4 packed ABI;
- Q4_K min/scale and Q6_K correction grammar;
- a hard grid equal to 170 SMs;
- 128x128 tiles, 8 warps, 57,856-byte shared allocation, and 251--255 registers;
- its stream-K fixup data structure;
- its particular final-token gate/up MMVQ fusion;
- numerical acceptance implied by Q8_1 activation rounding.

A generic tinygrad substrate should express quantized block metadata,
activation quantization layout, INT8 fragment ownership, device-derived
stream-K partitioning, and a fixup epilogue.  The candidate search should then
choose geometry per target and role instead of encoding llama's constants.

## Claims this audit rejects

- **“The valid tinygrad projection kernel is already faster than llama's.”**
  False for pp512.  That was true only of the invalid lost-store/aliased path.
- **“Llama wins because it uses cuBLAS FP16.”** False; source dispatch and fresh
  capture show Q4_K/Q6_K MMQ.
- **“Llama's MMQ is DP4A.”** False on this target; SASS shows IMMA.
- **“Llama has a larger 128x128 tile.”** Misleading; tinygrad also owns a
  128x128 CTA tile.  Arithmetic representation and K-work distribution differ.
- **“Llama wins through overlap.”** False for this capture; the kernel path is
  serialized on one stream.
- **“Llama's MMQ depends on `cp.async` or TMA.”** No source or binary evidence
  supports that claim.
- **“Attention is the principal prefill loss.”** False; its visible gap is
  about 1.6 profiled ms versus tens of milliseconds in dense projections.

## Next investment decision

Run tests 1--4 before production code.  If the extracted/direct packed INT8
triangle confirms the current source and timing evidence, invest in a generic
Q4/Q6 x Q8 INT8-MMA candidate substrate with stream-K ownership.  In parallel,
take the small, low-risk final-layer row-pruning win.  Do not spend the next
cycle primarily on attention, overlap, or async copy: none has an evidenced
recovery remotely large enough to explain the current wall gap.
