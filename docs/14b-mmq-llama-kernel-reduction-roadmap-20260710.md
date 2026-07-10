# 14B MMQ Llama-Kernel Reduction Roadmap - 2026-07-10

Goal: convert the local llama.cpp Q4_K MMQ kernel into a minimized hand-coded tinygrad atom, one bounded proof at a
time. The process stops only when the remaining clone-backed pieces cannot be converted without violating correctness,
resource, or promotion gates. This mirrors the 8B discipline: preserve working evidence, reduce the hand kernel into
machine-search-owned surfaces, and stop at an explicit blocker instead of inventing claims.

## Rule

```text
local llama clone = unreduced source kernel
tinygrad atom = reduced hand-coded translation
machine-search = proof/selection layer over converted pieces
```

Every kernel piece is in exactly one state:

| State | Meaning | Required proof |
|---|---|---|
| `source_clone` | not translated; source of truth remains the local llama clone | source path + anchors |
| `converted_searchable` | translated into tinygrad and passes bounded proof | machine-search done row + tests |
| `blocked_translation` | translated/probed but wrong or incomplete | blocked row with exact reason |
| `owned_atom` | no longer clone-backed for this slice; tinygrad atom owns it | bounded correctness + source hash + report row |

Nothing moves from `source_clone` to `owned_atom` by documentation alone.

## Source Queue

The source queue is the Q4_K MMQ path in:

| Source component | Clone anchor | Current tinygrad state |
|---|---|---|
| route/launch selection | `mmq.cu`: `GGML_TYPE_Q4_K`, `mul_mat_q_case` | source_clone |
| tile geometry | `mmq.cuh`: `MMQ_ITER_K`, `MMQ_NWARPS`, `get_mmq_y_*`, `get_mmq_x_max_*` | partially converted: report metadata only |
| Q8_1 MMQ DS4 block | `mmq.cuh`: `block_q8_1_mmq` | converted_searchable |
| Q8_1 MMQ DS4 quantizer | `quantize.cu`: `quantize_mmq_q8_1`, `MMQ_Q8_1_DS_LAYOUT_DS4` | converted_searchable in reference form |
| Q4_K block format | `ggml-common.h`: `block_q4_K` | converted enough for existing Q4_K helpers |
| Q4_K tile loader | `mmq.cuh`: `load_tiles_q4_K` | source_clone |
| Q4_K x Q8_1 MMQ dot formula | `vecdotq.cuh`: `vec_dot_q4_K_q8_1_impl_mmq` | converted_searchable |
| packed dot primitive | `vecdotq.cuh`: `ggml_cuda_dp4a`/AMD `sudot4` path | partially converted |
| packed DS4 dot4x4 lane mapping | `mmq.cuh`: callsite around `vec_dot_q4_K_q8_1_impl_mmq` | converted_searchable |
| shared/LDS tile layout | `mmq.cuh`: shared `tile_y`, `tile_x`, `mmq_get_nbytes_shared` | source_clone |
| cooperative tile loop | `mmq.cuh`: `mul_mat_q_process_tile` | source_clone |
| writeback | `mmq.cuh`: `mmq_write_back_mma` / `mmq_write_back_dp4a` | source_clone |
| full launch integration | `mmq.cuh`: `launch_mul_mat_q`, `mul_mat_q_case` | source_clone |

## Already Converted

These are machine-search-owned now:

| Converted piece | Backend/report row | Proof |
|---|---|---|
| DS4 layout | `done_components: DS4 layout` | `test_mmq_q4k_q8_reference.py` |
| DS4 reference correctness | `done_components: DS4 reference correctness` | `test_mmq_q4k_q8_reference.py` |
| Q4_K x DS4 formula | `done_components: Q4_K x DS4 formula` | `test_mmq_q4k_q8_reference.py` |
| `sudot4` primitive availability | `done_components: sudot4 primitive availability` | `test_mmq_q4k_q8_atom.py` |
| direct DS4 GPU atom | `amd_ds4_warp_direct` | `test_mmq_q4k_q8_atom.py`, `mmq_machine_search.py --run` |
| packed DS4 dot4x4 atom | `amd_ds4_dot4x4_packed` | `test_mmq_q4k_q8_atom.py`, `mmq_machine_search.py --run` |

Current executable proof shape:

```text
direct_packed_comparator      PASS
ds4_reference_formula         PASS
amd_ds4_warp_direct           PASS
staged_ds4_reference_probe    PASS
amd_ds4_dot4x4_packed         PASS/searchable
cooperative_shared_lds_tile   blocked
full_14b_prefill_route        blocked
production_dispatch_changed   false
default_route                 direct_packed
```

## Conversion Phases

### R0 - Clone Anchor Lock

Purpose: make the clone-backed source explicit and stable enough to reduce.

Work:

```text
record local clone paths and anchors in mmq_machine_search.py
record done_components and blocked_candidates
keep production_dispatch_changed=false
```

Done when:

```text
test_mmq_machine_search.py proves source policy and done rows
mmq_machine_search.py --run passes converted candidates
```

Status: done.

### R1 - Packed DS4 Dot4x4 Correctness

Purpose: convert llama's packed Q4_K x DS4 dot lane mapping into tinygrad.

Source anchors:

```text
vecdotq.cuh: vec_dot_q4_K_q8_1_impl_mmq
mmq.cuh: Q4_K vec_dot callsite near the Q4_K type trait path
mmq.cuh: load_tiles_q4_K packed q/nibble layout
```

Current tinygrad row:

```text
candidate_id = amd_ds4_dot4x4_packed
backend = q4k_q8_1_mmq_amd_ds4_dot4x4_atom_v0
state = converted_searchable
```

Work:

```text
derive lane -> output-row ownership
derive lane4/subtok -> q8 DS4 value offset
derive Q4_K group -> q nibble offset
derive where q4 scale/min is applied once per group
compare against q4k_q8_1_mmq_ds4_tile_reference at 4x5x256, 4x8x256, 8x8x256
```

Done when:

```text
remove xfail from test_q4k_q8_1_mmq_amd_ds4_dot4x4_atom_matches_reference_when_amd_available
bounded harness backend=q4k_q8_1_mmq_amd_ds4_dot4x4_atom_v0 reports PASS
mmq_machine_search marks amd_ds4_dot4x4_packed searchable
same-session direct_packed comparator is present
production_dispatch_changed remains false
default_route remains direct_packed
```

Status: done. The bug was over-applying the precomputed DS4 group sum once per packed lane; the corrected atom applies
the Q4_K min correction only on `lane4 == 0` before the 8-lane reduce.

Stop if:

```text
correct indexing requires unsupported UOp lane shuffles or impossible output ownership
packed dot correctness only holds for one shape and fails adjacent bounded shapes
```

### R2 - Q4_K Tile Loader Translation

Purpose: convert the relevant `load_tiles_q4_K` behavior into a tinygrad-owned tile-load description.

Source anchors:

```text
mmq.cuh: load_tiles_q4_K
mmq.cuh: unpack_scales_q45_K
ggml-common.h: block_q4_K
```

Work:

```text
create a tinygrad-side Q4_K tile-load spec for the bounded atom
prove q nibble layout matches direct Q4_K reference for all 8 groups per 256 block
prove scale/min unpacking matches current Q4_K helpers
expose tile loader source hash in machine-search artifacts
```

Done when:

```text
tile-loader unit test compares source-clone-derived layout expectations to tinygrad loader
packed dot4x4 uses the tile-load spec instead of ad hoc offsets
machine-search report lists Q4_K tile loader as converted_searchable
```

Stop if:

```text
tinygrad UOp cannot express the loader without duplicating large opaque instruction streams
resource use regresses direct DS4 warp candidate without a path to shared/LDS reuse
```

### R3 - Shared/LDS Layout Skeleton

Purpose: convert llama's shared memory tile allocation and padding model into a bounded tinygrad atom skeleton.

Source anchors:

```text
mmq.cuh: extern __shared__ int data_mul_mat_q[]
mmq.cuh: tile_y, tile_x partition
mmq.cuh: mmq_get_nbytes_shared
mmq.cuh: MMQ_TILE_NE_K and padding constraints
```

Work:

```text
add backend id q4k_q8_1_mmq_amd_ds4_lds_skeleton_atom_v0
allocate local/shared memory for DS4 activation tile and Q4_K tile
copy DS4 and Q4_K tile data into local memory
barrier
compute either a tiny tile or no-op checksum first
```

Done when:

```text
bounded correctness passes for a minimal output tile
lifecycle counters distinguish global loads, local stores, barriers, and output stores
machine-search marks lds_skeleton searchable but promotion_eligible=false
```

Stop if:

```text
local memory allocation exceeds gfx1100 limit
barrier/local-memory UOps cannot represent the required ownership safely
compiler output is unstable or non-deterministic across runs
```

### R4 - Cooperative Multi-Wave Output Ownership

Purpose: convert the output tile ownership pattern from llama's `mul_mat_q_process_tile`.

Source anchors:

```text
mmq.cuh: mul_mat_q_process_tile
mmq.cuh: sum[mmq_x*mmq_y / (nwarps*warp_size)]
mmq.cuh: write_back selection
```

Work:

```text
map workgroup axes to output N/M tile
map 8 warps to subtiles
define per-lane accumulator ownership
reduce only where llama requires it
write back bounded output tile
```

Done when:

```text
backend q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0 passes bounded correctness
tests cover at least 8x8x256, 16x16x256, and 16x16x512
machine-search compares it against direct_packed and amd_ds4_warp_direct
```

Stop if:

```text
VGPR pressure prevents any useful tile from compiling
correctness depends on hardcoded shape constants outside candidate metadata
```

### R5 - Geometry Search

Purpose: search the llama-like geometry after correctness exists.

Candidate dimensions:

```text
mmq_x: 8, 16, 24, 32, 64, 128 as supported
mmq_y: 64, 128
iter_k: 256 fixed for Q4_K
nwarps: 8 fixed initially
m/n bounded tiles: 4x5, 8x8, 16x16, 32x32, 64x64 if resource-safe
```

Done when:

```text
machine-search emits candidate rows with geometry, correctness, timing, source hashes, and comparator timing
best candidate beats or explains failure against direct_packed and current direct DS4 warp
```

Stop if:

```text
all correct geometries are slower than direct_packed and there is no remaining clone-backed structural piece likely to help
```

### R6 - One-Role Route Gate

Purpose: only after bounded win, expose one-role opt-in route evidence.

Scope:

```text
role = ffn_gate_up only
quant = Q4_K only
default route remains direct_packed
candidate route remains research
missing atom availability fails closed
```

Done when:

```text
one-role smoke proves no hidden direct_packed fallback
same-session bounded comparator exists
whole-prefill authority artifact exists only after bounded win
```

Stop if:

```text
bounded candidate does not win
route binding can silently fall back to direct_packed while claiming MMQ
attn_qo/attn_kv/ffn_down require separate unresolved kernels
```

### R7 - Continue Until Cannot Convert

Purpose: repeat the 8B discipline: reduce all feasible clone-backed source into owned machine-search/atom pieces.

Loop:

```text
pick highest-impact source_clone row
translate smallest bounded slice
prove correctness
add machine-search row
compare against previous best
either mark converted_searchable or blocked_translation
```

Stop condition:

```text
no source_clone row remains that can be translated without violating correctness/resource/route gates
or remaining rows are lower-value than the current bottleneck proven by bounded timing
```

Final acceptable outcomes:

```text
READY_FOR_ONE_ROLE_PROMOTION
BLOCKED_ON_PACKED_DOT_INDEXING
BLOCKED_ON_LDS_RESOURCE_LIMIT
BLOCKED_ON_COOPERATIVE_OWNERSHIP_UOP_GAP
BLOCKED_NO_BOUNDED_WIN_VS_DIRECT_PACKED
```

## Machine-Search Report Requirements

Every report must include:

```text
done_components
source_clone_components
blocked_candidates
searchable_candidates
llama_kernel_source_policy
production_dispatch_changed=false
default_route=direct_packed
candidate_route_id=prefill_14b_q4k_q8_1_hybrid_mmq_atom
public_label=hybrid_machine_search_mmq
same-session direct_packed comparator when run=true
```

R1 completion contract:

```text
default report includes amd_ds4_dot4x4_packed in searchable_candidates
default report keeps amd_ds4_dot4x4_packed out of blocked_candidates
promotion_verdict becomes BLOCKED_UNTIL_COOPERATIVE_TILE_PASS
production_dispatch_changed=false is preserved
default_route=direct_packed is preserved
```

Promotion is illegal unless:

```text
packed dot correctness passes
cooperative/shared tile candidate passes
bounded candidate wins against direct_packed
one-role ffn_gate_up smoke passes
whole-prefill authority beats the frozen baseline
```

## Non-Goals

```text
no pure-machine-search label
no vendoring llama source into tinygrad
no production default route change during reduction
no role expansion before ffn_gate_up proves out
no Q6_K path until Q4_K path proves useful
no hidden fallback to direct_packed while reporting MMQ
```
