# Why vectorized Q4_K loads do not pay on the NV decode wall

> **Status: superseded on 2026-08-24.** The negative wall result was caused by
> 37 uncounted fp32 boundary-copy invocations on the vector residual-add path,
> not by an immutable DRAM roofline. After fixing the scheduler's scalar-only
> kernel-name match, two depth-512 reverse brackets recover 88.36 and
> 66.56 us/token, the depth-128 bracket recovers 74.60 us/token, and cold NCU
> shows unchanged bytes with a higher achieved DRAM rate. See
> `nv-vector-load-reopen-result-20260824.md`.

Date: 2026-08-24
Repo: `/home/ubuntu/tinygrad-arkey`
Branch: `nvidia-bringup-20260731`, HEAD `6570abc025514273faa100c66b979e531585a1e1`
GPU: RTX 5090 (`sm_120`), ~1.79 TB/s GDDR7 peak
Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, single-token decode, depth 512

This is a self-contained write-up of one measured negative result. It explains
why widening the scalar global loads in the quantized GEMVs - the single
highest-leverage recommendation from the Phase 1 mechanism partition - is
bit-exact and 36% faster in isolation, but wall-neutral in the installed
decode loop.

## 1. The claim being tested

Phase 1 (`nv-phase1-mechanism-partition-result-20260823.md`) concluded the
remaining gap is body/codegen, not admission or wait, and identified this as
the top mechanism:

> NV quantized GEMVs use scalar loads. SASS shows zero `LDG.E.64`/`LDG.E.128`
> in the three largest GEMV bodies ... tinygrad issues ~4x more narrow load
> instructions than a vectorized llama kernel, lowering memory-level
> parallelism at the same occupancy.

The recommended fix was "vectorize loads", spanning a projected `~440 us/token`
of node-sum delta. We implemented exactly that for the Q4_K single-projection
GEMV (`q4k_g3_lanemap_gemv_kernel`, serving Q/K/O), the same spelling already
used for the landed gate/up kernel.

## 2. The candidate

Scalar spelling, per block of 256 activations:

- four header words read as four separate 32-bit loads;
- each qpack word read twice (once per group of a group-pair);
- the four fp16 activations per group read as four separate 16-bit loads.

Vectorized spelling, bit-identical by construction:

- four header words read as one `uint4` (`LDG.E.128`);
- each qpack word read once and reused for both groups;
- four activations read as one `half4` (`LDG.E.64`), lanes extracted with a
  `CUSTOMI` projection (`float(v.xyzw)`).

The per-lane accumulation order, shifts, and masks are unchanged, so the fp32
dot is identical. This is not an approximation: correctness is exact.

## 3. What was measured

### Isolated microgate (L2-hot)

The kernel is launched 200 times back-to-back on the same 9.44 MB weight
buffer, so after the first launch the weights are L2-resident.

- `[MEASURED]` bit-identical on non-zero data (memcmp, exact fp32 bits).
- `[MEASURED]` 4096x4096 (Q/O): 8.05 us -> 5.12 us per launch, ~36% faster.
- `[MEASURED]` 4096x12288 (down shape): 21.4 us -> 13.0 us, ~39% faster.
- `[MEASURED]` registers 61 -> 43, no spills, no stack.

This is real. The scalar spelling pays a real instruction-issue cost for its
redundant header re-reads, and the vector spelling removes it.

### Structural census

- `[MEASURED]` 83 Q4_K single-projection kernels per capture switch to `_vec_`
  names (`q4k_g3_lanemap_gemv_1024_4096` x28, `_4096_4096` x19,
  `_epi_resadd_4096_4096` x36).
- `[MEASURED]` zero new materialization/copy/cast kernels. Same node count.
- `[MEASURED]` the shared-Q8 cooperative attention route (43 kernels) and the
  landed gate/up vectorized route (36 kernels) are untouched.

### Installed wall (DRAM-cold), reverse control/candidate/control

- `[MEASURED]` Bracket 1, reps=5: control midpoint 4.70263 ms, candidate
  4.68820 ms, +14.43 us/token, token hash `f25083e5...` identical in all arms,
  `WALL_PASS`.
- `[MEASURED]` Bracket 2, reps=7 (higher SNR): control midpoint 4.70524 ms,
  candidate 4.70734 ms, -2.10 us/token, `NO_GO_WALL`.
- `[MEASURED]` the per-window token hashes are byte-identical across both
  brackets and both arms. Correctness is exact in every arm.

The two brackets disagree on sign, and the +14 us is inside the control-arm
spread of ~13.8 us. The verdict is wall-neutral, and the candidate was not
promoted.

## 4. Why the isolated win disappears: the roofline

The decode GEMVs are DRAM-cold streamers, not L2-resident kernels. This was
already measured in the FFN island (Phase 5):

```text
gate/up  reads 56.64 MB DRAM at  6.39% L2 hit   (1.501 vs 1.609 TB/s)
down q6  reads 41.34 MB DRAM at 16.59% L2 hit   (1.353 vs 1.449 TB/s)
down q4  reads 28.36 MB DRAM at 17.84% L2 hit   (1.357 vs 1.501 TB/s)
```

Every layer's full weight matrix streams from DRAM once per token. The whole
~5 GB of Q4_K weights is re-read each token, so there is no working set to
hide behind L2.

The two spellings read the same DRAM bytes:

```text
scalar DRAM traffic = qpack words + activations + header words
vector DRAM traffic = qpack words + activations + header words
```

The only traffic the vector spelling removes is the *redundant* header
re-reads. Those re-reads hit L1/L2 (the header was just loaded) and therefore
never contribute to DRAM traffic in the first place. The vector spelling
reduces instruction count and register pressure, but it does not reduce DRAM
bytes.

So the two measurements are consistent:

```text
L2-hot  (microgate):  instruction-issue bound  -> fewer loads wins  -> +36%
DRAM-cold (wall):     bandwidth bound          -> fewer loads hidden -> ~0
```

The narrow loads were real, but they were not the bottleneck. The bottleneck
is the DRAM streaming rate: tinygrad runs the GEMVs at 1.35-1.50 TB/s
(75-84% of peak) while llama runs at 1.45-1.61 TB/s (81-90% of peak). Widening
the loads does not change which side of that roofline the kernel sits on.

## 5. Why Phase 1's SASS conclusion was a mis-attribution

The Phase 1 observation was correct and the inference was not:

- Correct: the cubins contain zero vectorized loads.
- Incorrect: "therefore the ~440 us is caused by narrow loads."

The 440 us GEMV delta is a DRAM-rate delta (1.35-1.50 vs 1.45-1.61 TB/s)
plus non-folded support kernels, not a load-instruction delta. Evidence that
the body is already fine: L2-hot, tinygrad's gate/up body is 22.945 us vs
llama's 35.201 us. tinygrad is *faster* than llama in the cache-resident
regime and only slower in the DRAM-cold regime. That asymmetry points at the
memory system, not the instruction stream.

## 6. The gate/up contrast and its SNR caveat

The gate/up (w1w3) vector-load spelling landed at +31.04 us/token. Why does
the same change pay there but not on Q/K/O?

- `[MEASURED]` gate/up bracket was also reps=5, with control spread 20.9 us
  and recovery 31.0 us - above its noise floor, but only barely.
- `[MEASURED]` single-projection bracket 1 was +14.4 us against a 13.8 us
  spread; bracket 2 (reps=7) was -2.1 us.

The honest reading: both wins are small, and both sit near their own noise
floor. The gate/up one was large enough relative to its spread to pass; the
single-projection one was not. A genuine mechanism difference may also exist
(gate/up is 12288 rows and 56.64 MB, and its scalar spelling re-reads x for
both projections), but it has not been separated from SNR. This is an open
question, not a settled explanation.

## 7. What this means for the 240 campaign

- `[INFERRED]` Load-width vectorization is essentially spent for the GEMV
  pool. The remaining GEMV delta is DRAM streaming rate, which is set by
  topology/occupancy (gate/up is one warp per row, 45.6% warps active) and
  cache behavior, not by load instruction width.
- `[INFERRED]` The three levers that can still move the GEMV gap are:
  1. match llama's DRAM rate via topology (the prior 4-warp gate/up failed on
     an output-cast regression, not on the load path);
  2. overlap DRAM streaming with compute (llama runs ~1128 us overlapped vs
     tinygrad ~3-6 us; the prior two-queue/PDL attempts were wall-negative or
     neutral);
  3. reduce actual DRAM bytes (narrower intermediates / activation reuse).
- `[MEASURED]` The genuinely body-bound rows that remain are flash combine
  (+65 us, single-warp reduce over 48 splits) and vocab tail (+65 us, serial
  ~55 us wall). Those are not load-width problems and remain open.

## 8. What a reviewer should check

1. Does the L2-hot vs DRAM-cold argument survive the DRAM-rate numbers, or is
   there a way for wider loads to raise the *rate* (not the bytes) that this
   experiment failed to capture?
2. Is the gate/up +31 us a real DRAM-rate improvement or measurement noise?
   Re-bracket it at reps>=7 to find out.
3. If load width is spent, is the highest-value next move topology (4 warps
   without the cast regression), overlap, or byte reduction?

## Evidence

- `docs/task_workflow/evidence/nv-q4k-single-vector-load-microgate-20260823/`
- `docs/task_workflow/evidence/nv-q4k-single-vector-load-wall-20260823/`
- `docs/task_workflow/evidence/nv-q4k-single-vector-load-wall2-20260823/`
- `docs/task_workflow/evidence/nv-q4k-single-vector-load-census-20260823/`
- `docs/task_workflow/output/nv-installed-islands-phase5-ffn-result-20260822.md`
- `docs/task_workflow/output/nv-phase1-mechanism-partition-result-20260823.md`
