# 14B MMQ R4-R7 Theory Matrix

Purpose: avoid converging too early on one explanation of llama MMQ. The
right path is multiple falsifiable theories, each tied to the information we
need before R4-R7 can complete.

This document complements:

```text
docs/14b-mmq-wave-process-deconstruction-20260710.md
docs/14b-mmq-llama-kernel-reduction-roadmap-20260710.md
extra/qk/research/llama_mmq/mmq.cuh
extra/qk/mmq_llama_oracle.py
```

## Current State

```text
R4: owner proof PASS; full numeric cooperative atom blocked
R5: ready_for_bounded_geometry_search only after numeric coop candidate exists; non-promotable until then
R6: blocked_until_bounded_win
R7: blocked_until_bounded_win plus remaining source-clone rows
```

What is already true:

```text
llama source is vendored for research
HIP/gfx1100 build compiles the CUDA-named MMQ source into libggml-hip.so
Q4_K DS4 layout/reference/formula are done
direct DS4 atom is done
packed DS4 dot4x4 atom is bounded-searchable
LDS skeleton exists but does not implement full cooperative numeric compute
llama cooperative owner oracle exists
lowered R4 store-owner proof passes as eight fragmented AMD ISA traces for 16x16
owner coverage proves no missing/duplicate stores for selected oracle/lowered shapes
production route remains direct_packed
```

## Completion Chain

R4-R7 are not independent:

```text
R4: prove one correct cooperative multi-wave numeric atom; lowered store ownership is already proven
R5: search geometry/resource choices only after a numeric coop candidate exists
R6: bind one role only after R5 has a same-session bounded win
R7: repeat conversion until no source-clone piece remains worth translating
```

So the dominant unknown is not "is llama fast?" It is:

```text
what exact subset of llama's wave/LDS/accumulator/writeback process must be preserved for tinygrad to get a correct and useful 14B Q4_K prefill atom?
```

## Theory A - Final-Store Ownership Is The Primary Blocker

Claim:

```text
The R4 failure is mainly that tinygrad cannot yet express llama's one-owner-per-output writeback.
If we reproduce the owner map, the rest of the current DS4 arithmetic can be reused.
```

Why plausible:

```text
R3 already proves LOCAL staging and a barrier.
The oracle proves the 8-wave/16x16 owner map.
The current blocked evidence names duplicate/missing stores as the exact risk.
```

Information needed:

```text
actual tinygrad wave/workgroup axes for the candidate kernel: proven for the lowered 16x16 fragmented trace
mapping from lidx/gidx to wave_id and lane_id: proven for the lowered 16x16 fragmented trace
store address expression emitted by tinygrad for each lane: proven by lowered AMD ISA rows
actual owner map from a store-only marker kernel: represented by lowered AMD ISA proof rows
comparison against llama_mma_writeback_coverage: PASS for the 16x16 R4 owner map
```

Test:

```text
lowered store-only cooperative proof writes each output exactly once
duplicate_store_count == 0
missing_store_count == 0
actual_owner_hash == oracle_owner_hash
```

Refutes if:

```text
owner map matches but numeric kernel is still wrong or slow for reasons unrelated to ownership
```

Unlocks:

```text
R4 owner trace: PASS as fragmented AMD ISA proof
R4 store-only proof: PASS as fragmented AMD ISA proof
first legal cooperative numeric atom attempt
```

## Theory B - Accumulator Placement Is The Primary Blocker

Claim:

```text
The hard part is not the final store alone. It is preserving llama's per-thread sum[] accumulator layout without spills or ownership drift.
```

Why plausible:

```text
For 128x128/wave64/8 waves, each thread owns 32 fp32 sum slots.
That is already a meaningful VGPR commitment before Q4/Q8 decode temporaries.
If tinygrad lowers the accumulator differently, ownership can appear correct but performance collapses or stores consume wrong slots.
```

Information needed:

```text
sum slot count per candidate geometry
VGPR count for store-only, dot-only, and full numeric kernels
whether emitted code spills scratch
mapping from sum index to output fragment
compiler resource records: vgpr, sgpr, lds_bytes, scratch_bytes
disassembly or static instruction evidence for unexpected memory traffic
```

Test:

```text
compile candidate with resource extraction
scratch_bytes == 0
VGPR below bounded threshold
sum_index_to_output map equals oracle map
```

Refutes if:

```text
correct owner map compiles with acceptable VGPR/scratch, but timing remains bad due to load/staging behavior
```

Unlocks:

```text
R4 full numeric feasibility
R5 resource-aware geometry pruning
```

## Theory C - LDS Staging/Reuse Is The Primary Performance Lever

Claim:

```text
Correct ownership is necessary but not sufficient. Speed depends on reproducing llama's shared tile reuse: Q4_K loaded once, Q8_1 panels staged twice, barriers at the same lifecycle points.
```

Why plausible:

```text
The direct DS4 atom is correct but not a full llama-style tile.
llama amortizes unpack/dequant/staging across a 128x128 tile and a 256-wide K slice.
BoltBeam says 14B gate-up is not HBM-bound; low utilization points to schedule/codegen underfill.
```

Information needed:

```text
exact tile_x bytes and layout for Q4_K on gfx1100 path
exact tile_y bytes and layout for Q8_1 DS4
barrier count and barrier placement in llama and tinygrad candidate
global load counts for Q4_K, Q8_1, output stores
LDS load/store counts
L2 hit, VALU busy, occupancy, memory busy, MFMA/WMMA util where available
```

Test:

```text
stage-only or dot-only candidate emits lifecycle trace:
load Q4_K -> load Q8 panel 0 -> barrier -> dot -> barrier -> load Q8 panel 1 -> barrier -> dot -> barrier
resource/timing improves over direct per-output loading at the same bounded shape
```

Refutes if:

```text
staging lifecycle matches but bounded timing is flat or worse and resources explain no path to win
```

Unlocks:

```text
R4 lifecycle-correct atom
R5 geometry search with real LDS/resource constraints
```

## Theory D - Geometry Selection Is The Main Difference

Claim:

```text
The exact 128x128 llama tile is not necessarily the tinygrad optimum. The transferable piece is the ownership law; geometry may need to shrink for VGPR/LDS/compiler reasons.
```

Why plausible:

```text
llama host chooses mmq_x by shared-memory fit and output-column tiling.
tinygrad's custom UOp/resource behavior may differ from llama HIP C++.
Full 128x128 implies 32 sum slots per thread; smaller mmq_x lowers accumulator pressure.
```

Information needed:

```text
compile feasibility matrix for mmq_x = 16, 32, 64, 128
mmq_y = 16, 32, 64, 128 where owner law remains valid
VGPR/LDS/scratch per geometry
bounded correctness per geometry
bounded timing per geometry
same-session direct_packed and amd_ds4_warp comparator timing
```

Test:

```text
machine-search emits candidate rows:
geometry, owner_hash, correctness, resource, timing, comparator timing
```

Refutes if:

```text
all correct resource-safe geometries lose to direct_packed and no remaining source-clone piece plausibly changes that
```

Unlocks:

```text
R5 geometry search
R6 only if a geometry wins
```

## Theory E - Q8_1 DS4 Dataflow Is Still Not Faithful Enough

Claim:

```text
The current DS4 reference is correct, but the GPU atom may not yet move Q8_1 data in the same panelized layout/lifecycle as llama.
```

Why plausible:

```text
llama's block_q8_1_mmq has packed data, scales, and partial sums arranged for contiguous shared-memory copies.
The R3 skeleton stages DS4 values, but not necessarily the exact two-panel tile_y lifecycle.
```

Information needed:

```text
byte-level Q8_1 MMQ block layout for role shape
tile_y panel offsets for panel 0 and panel 1
whether activation sums are used in the Q4_K path for the target formula
global-to-LDS copy coalescing for Q8_1
alignment/padding rules for block_q8_1_mmq
```

Test:

```text
Q8_1 tile_y copy oracle:
for bounded input, tinygrad stage buffer bytes equal llama/reference panel bytes
```

Refutes if:

```text
tile_y bytes match but numeric failure remains in Q4_K loader/dot/writeback
```

Unlocks:

```text
R4 numeric correctness confidence
R5 avoids searching geometries over a wrong activation carrier
```

## Theory F - Q4_K Loader/Scale Decode Is The Hidden Bottleneck

Claim:

```text
The owner map can be correct while Q4_K loader/scales/mins decode dominates or misindexes, especially on gfx1100 wave64.
```

Why plausible:

```text
Q4_K uses packed nibbles plus scale/min unpacking.
llama has AMD-specific comments in load_tiles_q4_K around wave64 and double work.
The dot4x4 candidate already exposed indexing/math fragility earlier.
```

Information needed:

```text
Q4_K tile_x byte/int layout after load_tiles_q4_K
per-lane txi mapping for wave64
scale/min unpack mapping
load coalescing and instruction counts
standalone tile_x correctness test against vendored-source-derived oracle
```

Test:

```text
Q4_K tile_x staging oracle:
tinygrad staged x_qs/x_dm equals reference for multiple rows and k offsets
```

Refutes if:

```text
tile_x bytes match and full numeric failure remains
```

Unlocks:

```text
R4 numeric atom correctness
better R5 search features for loader variants
```

## Theory G - Runtime Route Binding Is The Real End-To-End Risk

Claim:

```text
Even after a correct bounded atom, R6 can fail because route binding may silently fall back to direct_packed or apply to the wrong role/quant/model profile.
```

Why plausible:

```text
Prior audit found hidden dispatch authority and proxy-pass problems.
Current policy intentionally keeps default_route=direct_packed.
R6 is illegal until same-session comparator and no-fallback proof exist.
```

Information needed:

```text
route trace showing selected role, quant, backend id, candidate id
kernel-name proof that direct_packed did not execute while claiming MMQ
same-session whole-prefill artifact
fallback reason for every skipped role
model-profile facts for 14B without model-size dispatch branches
```

Test:

```text
one-role opt-in smoke:
role=ffn_gate_up, quant=Q4_K, candidate backend executes, no hidden direct_packed fallback, default remains unchanged when opt-in is off
```

Refutes if:

```text
bounded atom wins but route trace cannot prove execution authority
```

Unlocks:

```text
R6 one-role route gate
eventual 14B e2e smoke
```

## Theory H - The Correct Atom Still Cannot Beat Direct Packed

Claim:

```text
It is possible to port the structure correctly and still lose because tinygrad custom kernel overhead/resource pressure exceeds llama's HIP implementation quality.
```

Why plausible:

```text
tinygrad custom UOp lowering is not llama HIP C++.
R4 correctness says "can represent"; it does not guarantee scheduler quality.
14B direct_packed already runs end to end.
```

Information needed:

```text
bounded timing for every correct geometry
whole-prefill role movement, not only isolated microbench
resource comparison against llama trace if collectable
roofline placement for direct_packed vs coop candidate
```

Test:

```text
same-session comparator:
candidate must beat direct_packed on bounded shape before R6
candidate must move ffn_gate_up whole-prefill timing before any promotion claim
```

Refutes if:

```text
correct candidate wins same-session and moves role timing
```

Unlocks:

```text
R6 route evidence
R7 continued reduction
```

## Information Required By Phase

### R4 - Cooperative Atom

Required information:

```text
wave_id/lane_id mapping in tinygrad custom kernel
oracle owner map and per-output coverage
actual store coverage from store-only atom
sum slot mapping
tile_x Q4_K staging bytes/layout
tile_y Q8_1 staging bytes/layout
barrier lifecycle
numeric correctness for 16x16x256, 32x16x256, 32x32x256, 128x128x256 if resource-safe
resource summary: VGPR, SGPR, LDS, scratch, code hash
production_dispatch_changed=false
```

R4 complete means:

```text
q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0 passes bounded correctness
owner coverage matches oracle: PASS for the lowered 16x16 fragmented AMD ISA proof
no duplicate/missing stores: PASS for the lowered 16x16 fragmented AMD ISA proof
resource evidence exists
machine-search row changes from blocked_numeric_compute to searchable/evidence with PASS
```

### R5 - Geometry Search

Required information:

```text
candidate space: mmq_x, bounded_m, bounded_n, maybe mmq_y subset, nwarps fixed first
compile feasibility per candidate
owner coverage per candidate
correctness per candidate
resource/timing per candidate
same-session direct_packed and amd_ds4_warp timings
failure reason for every rejected candidate
```

R5 is ready to run bounded geometry search only after a cooperative numeric candidate exists. Before then, it remains
non-promotable even though the R4 owner proof passed.

R5 complete means:

```text
search report ranks correct candidates
best candidate either beats comparator or explains why none can
no promotion if no bounded win
```

### R6 - One-Role Route Gate

Required information:

```text
one-role opt-in route policy
route trace proving ffn_gate_up Q4_K selected the coop backend
negative tests for attn_qo, attn_kv, ffn_down, Q6_K
kernel execution proof: no hidden direct_packed fallback
whole-prefill artifact with candidate enabled/disabled
same-session comparator preserved
```

R6 complete means:

```text
ffn_gate_up-only opt-in route can execute the winning candidate
default route remains direct_packed
route authority is single-source and test-proven
```

### R7 - Continue Until Cannot Convert

Required information:

```text
remaining source-clone rows after R4-R6
impact ranking by role time and source dependency
for each row: convert, keep as oracle, or stop with explicit blocker
delete/reclassify converted source-clone responsibility
route/evidence docs updated
```

R7 complete means:

```text
no remaining source_clone row can be translated without violating correctness/resource/route gates
remaining clone-backed pieces are explicitly oracle-only or blocked with evidence
```

## Immediate Next Experiments

Order matters:

```text
1. Preserve the lowered R4 store-owner proof as PASS evidence.
2. Keep R5 geometry rows non-promotable until a numeric coop candidate exists.
3. Add sum-slot mapping without Q4/Q8 math.
4. Add Q8_1 tile_y byte/panel staging check.
5. Add Q4_K tile_x byte/scales staging check.
6. Combine staging + sum + writeback for 16x16x256 numeric.
7. Scale to 32x32x256, then resource-safe larger tiles.
8. Only then run R5 timing search for promotion.
```

Stop conditions:

```text
full numeric compute cannot reuse the proven owner map without spills, wrong values, or unsupported lowering
numeric coop candidate is unavailable, so R5/R6/R7 loop without a bounded win/source-row update
resource extraction shows unavoidable scratch/spill for all useful geometries
all correct resource-safe candidates lose and no remaining source-clone mechanism explains a path to win
```
