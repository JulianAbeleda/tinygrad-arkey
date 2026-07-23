# Shared Attention Handoff

Date: 2026-07-23

Repository: `tinygrad-arkey`

## Objective

Move both prefill routes closer to the AMD roofline with one centralized score-resident attention implementation:

- 8B fp16 overlay route
- 14B packed Q4 projection route, producing fp16 Q/K/V
- No duplicated K/V assets
- No materialized full score or probability tensors
- Correct fp32 online state: `m`, `l`, and `acc[Hd]`
- QK and PV tensor-core contractions in one scheduled attention body

## Current status

The implementation and correctness foundation is complete. The native path is integrated into both model route policies, proof-gated, and verified on real AMD hardware.

The remaining work is performance tuning and whole-prefill promotion. The current native kernel is substantially faster than ordinary attention, but still below the device roofline.

## Committed milestones

- `eeb0ff90b`: two-KV-tile online correction, real AMD/HIP correctness
- `badd8a81a`: grid causal/tail correctness with q-tile-aware masking
- `23c78b8e9`: runtime KV loop through KV4096, dynamic causal/tail handling
- `29210e567`: 8B/14B GQA grid geometry and direct `q_head // G` ownership
- `2dbb7d29b`: exact 8B/14B Q512/KV512 and KV4096 static compilation
- `00f95b5a2`: model semantic integration, one compute call, no K/V expansion
- `5a0d444c3`: generic composite slot and owned-vector regressions
- `37a19b4d4`: model route context reaches final ProgramInfo and WMMA role ledger
- `63d283097`: compiler-owned capture schema and fail-closed proof API
- `262744515`: four real AMD proof artifacts and aggregate proof
- `f2da3fdf5`: corrected TinyJit replay-only benchmark protocol
- `9d7b985e8`: corrected replay benchmark matrix
- `0b902aeb3`: launch/full-output/timing bottleneck audit
- `ef0b033f0`: one-wave LDS fence replacing redundant workgroup barrier
- `3e069552f`: v2 captures/proof after wave-fence change
- `1299e4531`: wave-fence replay delta, performance-neutral
- `ca05471a1`: fail-closed G2 multi-wave ABI
- `618a4edfc`: wave-id/lane provenance and disjoint multi-wave P fences
- `b53ebfebd`: G2 shared K/V staging through an 8 KiB LDS slab

All completed milestones above were pushed. Do not reset or discard them.

## Mathematical and ownership invariants

Physical online state is heterogeneous:

```text
m:       scalar fp32 per query row
l:       scalar fp32 per query row
acc:     fp32[Hd] per query row
```

For each KV tile:

```text
block_m = row_max(scores with invalid entries = -inf)
new_m   = max(old_m, block_m)
alpha   = exp(old_m - new_m), guarded for -inf/-inf
P       = valid ? exp(score - new_m) : 0
new_l   = alpha * old_l + row_sum(P)
new_acc = alpha * old_acc + P @ V
output  = new_acc / new_l
```

Never normalize `P` by `new_l` before PV. Divide only once at final output.

Current gfx1100 fragment ownership:

```text
QK C[lane,e] -> row = 2*e + (lane >> 4), col = lane & 15
PV C[lane,e] -> row = 2*e + (lane >> 4), hd  = lane & 15
```

The single-wave fixed-register ABI uses approximately:

```text
PV C:   v8..v71
m:      v72..v79
l:      v80..v87
QK C:   v88..v95
alpha:  v96..v103
A/B:    v200..v215
```

The native path is spill-free in the proven single-wave kernel, with highest fixed register near `v215` and captured resource reports around `VGPR=254`.

## Model geometries

Both routes converge on the same attention ABI:

```text
8B:  B=1, Hq=32, Hkv=8, G=4, Hd=128
14B: B=1, Hq=40, Hkv=8, G=5, Hd=128
Q:   512-token chunks
KV:  512, 1024, 1536, 2560, 4096
```

GQA ownership is direct:

```text
kv_head = q_head // G
```

Do not repeat-interleave or materialize expanded K/V tensors.

The causal rule is prefix-aware lower-right causal:

```text
q_global = q_tile * 16 + q_row
kv_global = kv_tile * 16 + kv_col
valid = kv_global < KV and kv_global <= (KV - Q) + q_global
```

## Proof and correctness status

The aggregate proof is at:

`docs/artifacts/shared-attention-m10e1-20260723/shared_attention_proof.json`

It contains compiler-owned graph/source/ISA/resource/numeric records for:

- 8B overlay first chunk
- 8B overlay prefix chunk
- 14B bounded packed first chunk
- 14B bounded packed prefix chunk

The proof is fail-closed. It requires QK/PV role attribution, final ownership, no score/probability materialization, no spills, numeric records, and hashes. Do not replace it with caller-supplied strings.

The latest synchronization proof uses a one-wave LDS wait with zero workgroup barriers. It is valid only for the single-wave workgroup. Multi-wave paths must use workgroup barriers around shared K/V overwrites.

## Corrected benchmark results

The first benchmark matrix was invalid because each timed callback rebuilt graph/schedule/output. It is marked superseded and must not be used.

The corrected TinyJit replay matrix is at:

`docs/artifacts/shared-attention-benchmark-replay-20260723/summary.json`

Candidate versus ordinary GQA baseline, replay-only medians:

| Route | KV | Candidate ms | Baseline ms | Speedup |
|---|---:|---:|---:|---:|
| 8B | 512 | 0.5445 | 2.1793 | 4.002x |
| 8B | 1024 | 0.9386 | 5.2111 | 5.552x |
| 8B | 2048 | 1.8968 | 19.5711 | 10.318x |
| 8B | 4096 | 3.4832 | 75.3762 | 21.640x |
| 14B | 512 | 0.5776 | 2.6683 | 4.619x |
| 14B | 1024 | 0.9952 | 6.4406 | 6.471x |
| 14B | 2048 | 1.9404 | 23.9272 | 12.331x |
| 14B | 4096 | 3.5523 | 91.9405 | 25.882x |

Maximum numeric error in the corrected matrix was approximately `6.1e-05`.

Approximate candidate throughput using `4 * Hq * Q * KV * Hd` FLOPs:

```text
8B:  about 7.9-9.9 TFLOP/s across KV512-4096
14B: about 9.3-12.1 TFLOP/s across KV512-4096
```

These are much better than baseline but still well below the audited device peak of about `122.8 TFLOP/s`.

## Performance theory

The measured pattern is:

1. Time scales nearly linearly with KV length, confirming the KV loop dominates.
2. Removing the one-wave workgroup barrier changed replay time by less than `0.2%`; synchronization was not the main bottleneck.
3. 8B and 14B latency is surprisingly close despite 25% more query heads, consistent with saturated dispatch/resource behavior rather than a pure arithmetic limit.
4. The candidate uses one wave per workgroup and repeatedly loads Q/K/V fragments for each query tile/head.
5. GQA is logically shared (`q_head // G`) but K/V traffic is not yet physically shared across query-head waves.
6. Captured resources show very high VGPR allocation (`VGPR=254`) even though live virtual pressure is much lower. Occupancy/register partitioning may be limiting throughput.
7. Same-grid trivial dispatch overhead is only about `0.16 ms`, so launch overhead does not explain multi-millisecond KV4096 latency.

Primary theory: the remaining roofline gap is caused by physical K/V traffic and register/occupancy pressure, not online-softmax math or the workgroup barrier.

## Active next experiment

The current next step is GQA multi-wave reuse:

- One workgroup owns `(kv_head, q_tile)`.
- It has `G` wave32 waves: `G=4` for 8B, `G=5` for 14B.
- Each wave owns one query head and its independent Q/P/state/PV accumulators.
- K/V for the KV tile is staged once into shared LDS and reused by all waves.
- P remains in disjoint per-wave LDS slices.
- Workgroup barriers surround shared K/V overwrite; wave-local waits handle P slices.

The exact G2 ABI is committed in `ca05471a1`, wave/lane fence support in `618a4edfc`, and shared K/V staging in `b53ebfebd`.

Before extending G2 to G4/G5, compare replay-only timing against the single-wave baseline and prove:

- Full-output numeric equivalence
- No K/V aliasing between waves
- No P/state/C aliasing
- Correct GQA mapping
- Correct barrier ordering on K/V overwrite
- Resource/spill limits
- Lower physical global K/V load count

If G2 does not improve, do not assume G4/G5 will. Record the negative result and reassess register partitioning.

## Whole-model status

Whole-prefill benchmarks have not been promoted as final evidence after the corrected replay protocol. Run them only after the attention candidate and multi-wave experiment are settled. Required contexts are 512, 1024, 2048, and 4096 for both 8B and 14B routes.

Default promotion remains disabled unless the proof artifact, fallback policy, full-output numeric gates, resource gates, and performance evidence all agree.

## Agent handoff instructions

Start by reading this document and checking the latest pushed commits. Preserve all superseded artifacts; do not overwrite them. Use bounded commands and commit each safe milestone.

When reporting a performance result, always include:

- profile and GQA geometry
- KV/Q dimensions
- candidate and baseline raw replay samples
- whether compile/allocation/copies were excluded
- full-output numeric error
- physical workgroup/local sizes
- WMMA/barrier/LDS counts
- VGPR/SGPR/scratch/spill resource data
- graph/source/ISA/proof hashes

Do not claim roofline progress from the superseded non-replay matrix.

## 2026-07-23 theory realignment from published attention work

The project has completed the original FlashAttention-style IO objective: the standard attention graph is rewritten into an exact score-resident online-softmax computation, and the full score/probability matrices are not materialized.

The remaining gap is best described as a FlashAttention-2-style work-partitioning and occupancy problem, not an online-softmax correctness problem:

> Long-lived PV output accumulators coexist with QK fragments, running softmax state, WMMA operands, and address temporaries. The resulting register residency appears to limit resident waves and latency hiding.

Evidence already collected:

- Removing the single-wave workgroup barrier changed replay by less than `0.2%`.
- G2 LDS K/V sharing reduced load sites but was up to `1.65%` slower because LDS, barriers, and private state offset the gain.
- Compacting the attention VGPR address range was numerically safe but `1.43%` slower at KV512 and `2.46%` slower at KV4096 because it did not reduce live-state quantity or lifetime.
- The single-wave kernel remains spill-free while reporting roughly 250 allocated VGPRs. Spill-free is not equivalent to occupancy-efficient.

Published guidance:

- FlashAttention establishes the IO-aware, score-resident foundation: <https://arxiv.org/abs/2205.14135>
- FlashAttention-2 identifies low occupancy and unnecessary shared-memory communication from poor block/warp partitioning: <https://arxiv.org/abs/2307.08691>
- The Hopper implementation case study emphasizes balancing tile geometry, register pressure, and shared memory across the fused QK/softmax/PV pipeline: <https://arxiv.org/abs/2312.11918>
- FlashAttention-3 adds overlap and specialization after useful parallel residency exists: <https://arxiv.org/abs/2407.08608>
- RegDem supports selective shared-memory demotion only when the gained occupancy exceeds access cost: <https://arxiv.org/abs/1907.02894>

### Revised execution order

1. Build a live-state/residency ledger for persistent PV, QK, softmax, WMMA fragment, address, and temporary state.
2. Test smaller query-row ownership per wave/block with disjoint outputs and more independent work.
3. If that does not reduce the persistent state, test a smaller output-dimension accumulator as a diagnostic, initially allowing score recomputation.
4. If smaller ownership proves the theory but recomputation loses, selectively stage inactive state in LDS with no compiler spill/private-memory path.
5. Retry grouped-query K/V reuse only after per-wave residency improves.
6. Then overlap loads, QK, softmax, and PV before final tile tuning.
7. Promote only after corrected whole-prefill benchmarks pass for 8B and 14B at KV512, KV1024, KV2048, and KV4096.

The next experiment is therefore not another physical register remap. It must reduce allocated live state or change independent work ownership enough to increase calculated residency before its timing result is considered meaningful.
