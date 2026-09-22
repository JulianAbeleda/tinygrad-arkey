# NV pp512 Flash vector-topology discriminator

Date: 2026-08-29  
Packet: F0  
Verdict: **PASS**

This packet scopes one exact F1 primitive. It does not modify production
routing, integrate a candidate, or claim wall recovery.

## Frozen authority

The target is the current Qwen3-8B Q4_K_M pp512/ubatch512 route on RTX 5090.
The live selected PROGRAM is
`nv_sm120_q16_grid_hd128_loop_attention`, with 36 calls, global geometry
`(1024,1,1)`, local geometry `(32,1,1)`, and FP16 buffers:

| slot | role | bytes | logical shape |
|---:|---|---:|---|
| 0 | output | 4,194,304 | `(1,32,512,128)` |
| 1 | Q | 4,194,304 | `(1,32,512,128)` |
| 2 | K | 1,048,576 | `(1,8,512,128)` |
| 3 | V | 1,048,576 | `(1,8,512,128)` |

The inputs are read-only. The mask is causal lower-right with `start_pos=0`:
for query row `t`, only KV rows `<= t` are valid. Head grouping is GQA
`Hq=32`, `Hkv=8`, group size 4. Score and reduction are FP32; output is FP16.
The independent oracle is
`docs/task_workflow/evidence/nv-prefill-flash-20260829/oracle.npz` and its
record is PASS, finite, output shape `(1,32,512,128)`.

The live qualification captured 26 graph calls, selected graph index 6, full
output coverage, replay freshness, and oracle max-absolute error `0.00930290`
under the retained allclose gate. The profile-only extractor currently reports
zero calls because an opaque graph-owned PROGRAM does not expose profile
entries under its expected name; this is an extraction limitation, not a
replacement authority. The live capture is authoritative for ABI and shape.

## Existing topology ledger

| property | tinygrad installed path | llama reference path |
|---|---|---|
| CTA ownership | one opaque fused PROGRAM call per layer; installed launch `(1024,32)` | score main grid `(340,1,1)`, block `(32,4,1)`; reduction grid `(340,16,4)`, block `(128,1,1)` |
| logical ownership | fused score/reduction over 32 Q heads, 8 KV heads, 512 rows | grouped Q/KV attention; stream-K score partitions and separate reduction/fixup |
| vector width | not exposed by opaque PROGRAM metadata; live ABI is scalar FP16 buffer storage | source/template path is `fattn-common.cuh`, with cooperative vectorized K/V movement and stream-K partitions |
| KV access order | not recoverable from profile-only artifact; candidate must make it explicit | partitioned K/V traversal by score CTA, then reduction/fixup over partitions |
| reduction owner | fused program, not separately observable | reduction/fixup CTA owns online/statistical combination for score partitions |
| output publication | direct fused write to slot 0 | reduction/fixup publishes final head/tile output |
| registers/shared memory | not exposed by current live observer; F1 must retain compiler report | llama binary counter/source evidence is the comparison, not a guessed register count |
| service time | tinygrad Flash active `3.328096 ms` over 36 calls | llama Flash active `1.657447 ms` over 36 score/reduction pairs |

The existing S6/cooperative tinygrad spellings are closed. Their source may
not be renamed or cosmetically retiled to satisfy this packet. The old trace
also shows device idle is small (`0.307232 ms` tinygrad versus `0.201442 ms`
llama), so overlap or launch placement is not the F0 mechanism.

## Exact F1 topology: `VKV_H4_T64_W4_ONLINE128`

Build an isolated vectorized KV/head-tile primitive with these fixed choices:

* One CTA owns one Q-head and one contiguous 64-query tile. Grid is
  `(32,8,1)`: `blockIdx.x = q_head`, `blockIdx.y = q_tile`, giving 256 CTAs.
* The CTA loads the four KV heads belonging to the Q head through a fixed GQA
  group mapping. K and V are read as 16-byte vectors (`half8`) along `Hd=128`;
  each warp owns a disjoint contiguous vector strip. No scalarized per-element
  KV path is permitted.
* Four warps own the four 16-query subtiles inside the 64-query tile. Each warp
  traverses KV rows in 64-row chunks, applies the causal bound before score
  contribution, and keeps FP32 online max/sum/output accumulators.
* Warp 0 is the named reduction owner for CTA-wide boundary publication. Warp
  shuffles combine the four warp partials for each query row; one elected lane
  writes the final 128-element output vector. There are no global partial
  buffers and no second fixup kernel.
* K and V tiles are staged once per CTA into shared memory with a swizzled
  16-byte layout. Q remains register/vector loaded per query subtile. The
  topology must not duplicate a KV tile once for each of the four Q heads in a
  GQA group.
* The causal tail is predicated per query row. Out-of-range scores contribute
  `-inf` before the FP32 max reduction; no post-hoc masking or row pruning is
  allowed.

Resource envelope for F1 admission: 128 threads (4 warps), shared K+V staging
no greater than 32 KiB per CTA, no local-memory traffic, and compiler-reported
register count no greater than 96 per thread. Any spill, incomplete sentinel
coverage, or inability to express the reduction owner is an immediate STOP.

This is structurally new relative to S6: ownership is query-head/query-tile,
KV is vectorized and shared once per CTA, and reduction ownership is explicit
inside the same CTA rather than a relabeled cooperative/fixup spelling.

## F1 execution contract

F1 must bind the exact saved Q/K/V fixture, compare all `(1,32,512,128)` output
elements against the oracle, verify read-only inputs and complete sentinels,
and retain source, program hash, launch geometry, registers, shared memory,
local-memory report, and timing. It must compare hot and cold R9 against the
installed primitive over the exact 36-call population. A primitive-only win is
not a composed win. The candidate must beat the installed primitive in the
complete producer/main/epilogue lifecycle; otherwise F1 is STOP.

No F1 implementation or timing was performed by F0. No wall recovery is
claimed.

