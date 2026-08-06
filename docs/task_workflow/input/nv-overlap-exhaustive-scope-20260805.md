# NV decode overlap exhaustive scope

Date: 2026-08-05
Status: **mechanism enumeration complete; in-graph co-schedule arithmetic is
CPU_NO_GO on the redirect-on authority DAG (17.952 us ceiling vs the 50 us
gate); resource-join census augmentation is the CPU-only enabling step; all
GPU arms parked**
Authority: `nv-decode-exposure-overlap-host-forward-scope-20260805.md`
Constraint in force: no GPU use. Every mechanism below is either CPU-analyzed
now or gated behind a CPU-verified forecast plus the workstream gates before
any token wall arm. This scope changes no production defaults.

## Question and answer

What mechanism, if any, can recover the **445.954 us/token** hidden-overlap
bucket (the `+17.17 tok/s` row at the composed baseline of 5.3242440 ms/token
= 187.82 tok/s) **without cross-queue waits**? Two-queue cuts are closed: at
the calibrated 3.1865 us/wait the Q cut reproduces the P4 wall at -10.474 us
and the K cut lands at +42.962 us < 50
(`nv-decode-p4-dependency-closed-cut-record-20260805.md`). The parent scope
names exactly one surviving mechanism: **in-graph co-scheduling without
cross-queue waits**, i.e. llama's driver co-scheduling of dependency-
independent nodes on one CUDA graph.

This scope exhaustively enumerates every mechanism that could touch that
bucket, with the arithmetic each mechanism permits on the redirect-on
authority DAG (`/tmp/nv_p4_redirect_on_dag_20260805.json`, 875 nodes, 4080
edges, ordered-name digest
`49838b8ab2e7118d0c384fb93d2b4c3085b3732f1fe8d5abc69d51d232a6b413`), scored by
the new CPU-only tool `nv_co_schedule_candidate_scan.py`.

The DAG's structure decides the question before any GPU time:

```text
875 nodes = 586 support (E_/r_, no metadata, 1006.112 us)
          + 217 quant hosts (180 q4k, 37 q6k; all carry semantic metadata)
          + 72 flash hosts
critical path 4661.984 us (659 nodes: 442 support / 145 quant / 72 flash)
serialized sum 5260.256 us, serialization slack 598.272 us
support-on-critical-path exposure 801.920 us
```

Only **200 dependency-independent (support, quant/flash) pairs within the same
graph launch** exist (100 support nodes with at least one partner; 54 of those
on the critical path). 486 support nodes (860.992 us) have no independent host
at all. The exact critical-path arithmetic on those pairs:

| measure | us/token |
| --- | ---: |
| llama attribution target (Q8_1 interval mass behind MMVQ) | 445.954 |
| pair-level containment (200 rows) | 290.240 |
| best-partner containment (100 supports) | 145.120 |
| co-schedule ceiling (every partnered support hidden behind its best host) | **17.952** |
| greedy realized recovery (8 pairs, 1.0 us floor) | **10.016** |
| best single pair | **1.792** |

The target exceeds the structural ceiling by **24.8x**. Overlap on the
redirect-on topology is CPU_NO_GO; the remaining paths to the 445.954 magnitude
are fusion/dataflow mechanisms (c)/(d), which belong to the parent scope's
workstream 2 (662.128 us attribution) and carry their own exact-output gates.

## Mechanism (a): in-graph topological co-scheduling on ONE CUDA graph

**What it is.** Llama's mechanism: quantize_q8_1 and support work are emitted
as graph nodes dependency-independent of the MMVQ nodes, and the CUDA driver
overlaps them on one graph launch. The tinygrad question is whether the
captured DAG has dependency-independent support nodes whose durations can sit
inside a quant/flash kernel's shadow. The tool computes exactly this: exact
critical-path recovery per pair, per node, per population, and a greedy
selection that recomputes the critical path after every pick.

**Arithmetic on the captured DAG.** Reachability closure over all 4080 edges
(660 RAW / 1952 WAR / 1468 WAW; 147 cross-group) yields 200 same-group pairs;
flash has **zero** independent support nodes (flash sits downstream of the
rope/KV support and upstream of the residual support), so only q4k (154 pairs)
and q6k (46 pairs) host candidates. Hiding every partnered support behind its
best quant partner shortens the critical path by 17.952 us; the greedy realizes
10.016 us across 8 pairs; the best pair recovers 1.792 us. The per-node
full-hide upper-bound sum (52.352 us over the 100 partnered supports) overcounts
parallel branches and is not bookable.

| population | pairs | supports | containment | ceiling | greedy |
| --- | ---: | ---: | ---: | ---: | ---: |
| q4k | 154 | 100 | 224.032 | 17.952 | 10.016 |
| q6k | 46 | 46 | 66.208 | 0.000 | 0.000 |
| flash | 0 | 0 | 0.000 | 0.000 | 0.000 |

**Why the mass is not there.** Tinygrad's quant gemv kernels already absorb
quantization in-kernel, so the 217-node llama quantize_q8_1 population
(`nv-decode-llama-d512-timeline-ledger-20260804.json`:
`quantize_q8_1.hidden_behind_mmq_us = 445.954`) has no tinygrad counterpart
node; the tinygrad hosts are the 217 semantic-metadata q4k/q6k nodes. The
support nodes that could hide are 100 of 586, and most of their duration is
dependency-bound to the quant backbone: 286 support nodes (521.120 us) sit
quant->support->quant, and the largest support kernels
(`r_32_4_1187` 39.552 us, `r_128_16_8_1187` 10.688 us, vocab/embedding-family
rows) have no independent host.

**Feasibility in tinygrad capture/runtime.** Same-group nodes are already
co-captured within one graph launch; the open question is only whether the
driver would overlap them. That is exactly the resource-complementarity
question in (e). No scheduler change is needed to test it; the capture and
the join are capture-only seams.

**Gate before a GPU arm.** A fresh duration-bearing DAG from the current closed
model graph must clear **+50 us** of exact critical-path recovery (greedy or
ceiling) at the tool's arithmetic, or (e) must name a complementary pair. The
redirect-on DAG fails by 32 us even at the ideal ceiling (17.952 < 50).

**Status.** CPU-analyzed now, **CPU_NO_GO** on this topology.

## Mechanism (b): static reorder of support nodes inside the serialized graph

**What it is.** Reorder dependency-independent support nodes so their durations
abut quant-kernel shadow in the emitted sequence, exploiting driver overlap
without a second queue.

**Arithmetic.** The critical path is invariant to reordering a fixed DAG; a
serialized stream's span is the sum, and reorder changes only adjacency, not
span. Reorder is therefore a necessary enabler of (a) (a support node must be
adjacent enough to its host for the driver to overlap it), but it adds zero
headroom of its own: its ceiling is (a)'s 17.952 us.

**Feasibility.** The graph scheduler emits program order; no reorder seam
exists, and none is justified while (a) fails the gate.

**Gate / status.** Subsumed by (a). CPU-analyzed, no arm.

## Mechanism (c): support-kernel fusion INTO quant kernels

**What it is.** Reduce node count by moving residual/rope/cast work into the
quant gemv bodies (llama's MMVQ carries its epilogue fusions). This is the
parent scope's workstream 2 territory (662.128 us fusion/dataflow
attribution), not overlap, but it is the only arithmetic that reaches the
445.954 magnitude on the tinygrad side.

**Arithmetic.** The fusion-only ceiling is the full support-on-critical-path
exposure: **801.920 us** (4661.984 -> 3860.064 with every support node gone).
The natural first population is the 286 quant->support->quant sandwiches
(521.120 us); the flash-bound population is 0-independent by construction and
must be absorbed into flash bodies (mechanism d).

**Feasibility.** Every custom-boundary epilogue measured to date is NO-GO
(attention-O +69, llama-O +21.2, RMSNorm wrapper +60.802; see
`nv-decode-exposure-overlap-host-forward-scope-20260805.md`). The acceptance
predicate is the boundary-free ordinary-UOp gate
(`nv_boundary_free_ordinary_uop_gate.py`).

**Gate.** Exact-output native A/B per fusion population, after the boundary-free
gate passes. GPU-parked; no credit is bookable from the attribution rows.

## Mechanism (d): moving support work into flash/GEMM bodies

**What it is.** Absorb rope/KV/residual work into the flash kernels. The flash
family is 72 nodes / 306.016 us, all on the critical path, and every support
node is dependency-bound to flash (0 pairs), so body absorption is the only
route. Same class as (c), with its own exact-output A/B per population
(`nv-flash-single-stage-reopen-scope-20260805.md`).

**Gate / status.** Identical to (c). GPU-parked.

## Mechanism (e): resource-complementarity-driven pairing (needs census augmentation)

**What it is.** Prove a dependency-independent pair is physically co-resident
by per-CTA resource complementarity (grid/block/registers/shared/local), then
authorize one native two-queue span A/B. This is the parent scope's item 4 and
the only CPU-doable enabling step.

**Arithmetic.** The join can only *authorize* a probe; it cannot exceed the
topology's co-schedule ceiling (17.952 us). The llama-side 445.954 us
containment is not a transferable pair: the llama manifest's mmvq/quantize rows
have grid/block/register/static/dynamic shared but no local-memory field, and
the tinygrad census has identities and durations but no compiled resource
tuple (`nv-overlap-resource-compatibility-ledger-20260805.md`). The join
currently fails closed (`INCONCLUSIVE_FAIL_CLOSED`).

**CPU-only enabling step (this scope's plan).** Augment one aligned tinygrad
capture through the fail-closed seam
`route_b3_dag_attribution.attach_compiled_descriptors`: it requires the exact
ordered `CallRecord` list plus one descriptor per occurrence carrying
`binary_sha256`, `grid`, `block`, `registers_per_thread`, `static_smem_bytes`,
`dynamic_smem_bytes`, and `local_mem_bytes`; missing/extra/partial rows raise
and names are never a surrogate for occurrence identity. Then rerun
`nv_overlap_resource_join.py`. Only a dependency-independent pair with a
positive complementary CTA residency bound authorizes the native span probe.

**Gate.** Positive complementary CTA residency bound AND exact recovery >= 50
us. The second fails arithmetically on the redirect-on DAG, so the join is
currently a falsification device (prove the ceiling wrong) rather than a
booking device.

**Status.** Census augmentation is CPU-doable now; the GPU span probe is parked
behind both gates.

## Mechanism (f): reducing support node count by algebraic elimination

**What it is.** Eliminate redundant casts/contiguous/dataflow nodes in codegen
(the 240.319 us residual/cast/contiguous ownership row; the audit's 662.128 us
is attribution, not raw cost). Elimination targets dependency-bound support,
so it does not change the co-schedule ceiling (17.952 us); it shrinks the (c)
population.

**Gate / status.** Workstream-2 gates (boundary-free gate, exact-output A/B).
GPU-parked.

## HARD STOP

No GPU arm for overlap recovery until one of these passes on a fresh
duration-bearing DAG from the current closed model graph:

1. The co-schedule forecast clears **+50 us** of exact critical-path recovery
   (`nv_co_schedule_candidate_scan.py`). The redirect-on authority DAG does
   not: ceiling 17.952 us, greedy 10.016 us, best pair 1.792 us (all < 50).
2. The resource join names a dependency-independent pair with a positive
   complementary CTA residency bound
   (`nv_overlap_resource_join.py` after
   `route_b3_dag_attribution.attach_compiled_descriptors` census augmentation).
   Its reward is still capped by the topology ceiling.
3. A fusion population clears the boundary-free gate and an exact-output native
   A/B books it (mechanisms c/d; workstream 2 of
   `nv-decode-exposure-overlap-host-forward-scope-20260805.md`).

Cross-queue cuts remain closed: any future occurrence-pinned cut must first
clear +50 us at the calibrated 3.1865 us/wait
(`nv_wait_adjusted_cut_forecast.py`); both known cuts fail and are quarantined.

## Top candidates on the redirect-on authority DAG

The tool's ranked ledger (200 same-group pairs; 108 with positive exact
recovery). Top 5 by exact critical-path recovery, all group 4 (the final graph
launch), all q4k-hosted, all from the K-rope support family that also drives
the dependency-closed-cut K branch (`nv-decode-p4-dependency-closed-cut-record-
20260805.md`):

| rank | recovery us | hideable us | support node | dS us | host node | dH us |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| 1 | 1.792 | 1.792 | `E_2_8_16_4_4_` (id 492) | 1.792 | `q4k_g3_lanemap_gemv_1024_4096` (id 486) | 4.640 |
| 2 | 1.792 | 1.792 | `E_2_8_16_4_4_` (id 492) | 1.792 | `q4k_g3_lanemap_gemv_1024_4096` (id 487) | 4.224 |
| 3 | 1.728 | 1.728 | `r_2_8_4_4_16_` (id 490) | 1.728 | `q4k_g3_lanemap_gemv_1024_4096` (id 486) | 4.640 |
| 4 | 1.728 | 1.728 | `r_2_8_4_4_16_` (id 490) | 1.728 | `q4k_g3_lanemap_gemv_1024_4096` (id 487) | 4.224 |
| 5 | 1.600 | 1.600 | `E_4_2_8_16_4_` (id 488) | 1.600 | `q4k_g3_lanemap_gemv_1024_4096` (id 486) | 4.640 |

Names are shown as program stems; full 64-hex generated hashes are in the
ledger (`/tmp/nv_co_scan_v1.json`, schema
`tinygrad.nv_co_schedule_candidate_scan.v1`). The greedy selection adds the
same three support programs across other layers: `E_2_8_16_4_4_` (id 492),
`r_2_8_4_4_16_` (ids 202/130/634), `E_4_2_8_16_4_` (ids 320/392/272/704),
8 pairs totaling 10.016 us. These are the entire overlap surface of the
redirect-on graph.

## CPU dev this turn

- `extra/llm_research/decode/nv_co_schedule_candidate_scan.py`: ranked
  co-schedule ledger (schema `tinygrad.nv_co_schedule_candidate_scan.v1`) with
  dependency-independent pair detection, per-pair/per-node/per-population exact
  critical-path recovery, best-partner and pair-level containment, greedy
  selection with critical-path recomputation, same-group default, and the
  +50 us promotion-gate verdict. Hermetic; no GPU, no /tmp dependence.
- `test/unit/test_nv_co_schedule_candidate_scan.py`: 8 tests covering pair
  detection, exact recovery and ceilings, greedy non-double-counting and
  chain recomputation, the passable 50 us gate (not vacuously closed),
  same-group semantics, the support/metadata contract, and fail-closed
  validation.
- This scope document.

No production default changes, no GPU work, no behavior changes while the GPU
ban is in force.

## References

- `nv-decode-exposure-overlap-host-forward-scope-20260805.md` (parent scope)
- `nv-decode-final-accounting-audit-20260805.md` (location PASS / recoverability FAIL)
- `nv-decode-final-composed-same-session-record-20260805.md` (authority wall)
- `nv-decode-p4-dependency-closed-cut-record-20260805.md` (wait-cost calibration)
- `nv-rank2-native-concurrency-construction-verdict-20260805.md` (two-GPFIFO PASS, economics negative)
- `nv-overlap-resource-compatibility-ledger-20260805.md` (join INCONCLUSIVE_FAIL_CLOSED)
- `nv_wait_adjusted_cut_forecast.py` / `nv_dependency_closed_cut.py` (calibrated cut gate)
- `nv-decode-llama-d512-timeline-ledger-20260804.json` (445.954 us source)
- `nv-decode-llama-tinygrad-semantic-call-manifest-20260804.json` (host manifest)
- `nv-flash-single-stage-reopen-scope-20260805.md` (flash body work)
- `route_b3_dag_attribution.py` (attach_compiled_descriptors seam)
