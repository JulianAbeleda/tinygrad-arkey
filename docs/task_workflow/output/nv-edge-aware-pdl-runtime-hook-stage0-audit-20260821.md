# Stage 0 audit: edge-aware PDL runtime hook

Date: 2026-08-21

Commit: `6570abc025514273faa100c66b979e531585a1e1`

Scope: `docs/task_workflow/input/nv-edge-aware-pdl-runtime-hook-scope-20260821.md`

Status: Stage 0 exit conditions met for the design audit. Stages 1 through 5
remain gated and unmeasured. No production code was changed and no GPU
benchmark or device test was run in this stage.

## 1. Stage 0 exit summary

| exit condition | result |
| --- | --- |
| typed dependency census prediction for Q1 | complete, with one hard caveat: edge kinds are typed, but buffer spans were not retained by the capture, so zero edges are alias-safe yet |
| QMD latch-capability answers | every question has `supported` or `named-unavailable`; no row is inferred into support |
| wait-placement plan per consumer family | complete for all families present in the census |
| correctness checklist | complete for alias, WAR/WAW, multi-consumer, multi-producer, latch reuse, and graph flush |
| production behavior | unchanged; the audit only added documentation, evidence, and probe tooling |

The Stage 0 result is a **prediction and a gate list**, not an endpoint
measurement. The census numbers below are static inferences from the captured
DAG and scheduler replay, and are labeled accordingly.

## 2. Typed RAW-edge census prediction for Q1

The capture at
`docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_a_control.json`
contains 704 RAW edges in the current-HEAD decode route. The llama reference
DAG contains 761 programmatic RAW edges and is treated as a reference size,
not as a claim that the two routes have matching edge identity.

Arm rule used by the prediction:

- same graph group;
- same compute queue;
- consecutive `active_qmd` pair in the replayed schedule;
- no encoded queue wait between the producer and consumer;
- RAW forward static edge;
- no name-prefix filter.

The classifier reconciled its replay against the prior artifacts before
writing: Phase A reproduced 108 armed pairs for one queue and 144 for two
queues; the broad Phase D census reproduced 329 and 429 armed real edges.
That reconciliation is recorded in the census JSON.

### 2.1 Reason census

| reason | 1 queue | 2 queues | meaning |
| --- | ---: | ---: | --- |
| `candidate_armed` | 148 | 247 | structurally armable under the conservative single-producer rule |
| `multi_producer_fallback` | 109 | 110 | structurally armable pair, but the consumer has more than one RAW producer |
| `adjacency` | 434 | 192 | producer and consumer are not the consecutive same-queue QMD pair |
| `queue_split` | 0 | 107 | producer and consumer landed on different compute queues |
| `encoded_wait` | 0 | 35 | an encoded queue wait sits between the pair |
| `cross_group` | 13 | 13 | the pair crosses one of the five replay-group boundaries |
| RAW total | 704 | 704 | same static edge graph under both queue placements |
| alias-safe total | 0 | 0 | no edge has proven disjoint spans |
| span-unverified total | 704 | 704 | every edge is currently span-unverified |

If the multi-producer capability were proven, the provisional ceiling is
257 candidate edges on one queue and 357 on two queues. That ceiling is a
hypothesis upper bound, not a supported arm count.

### 2.2 Bucket census

| producer -> consumer bucket | RAW edges |
| --- | ---: |
| gemv -> gemv | 107 |
| gemv -> support | 181 |
| support -> gemv | 146 |
| support -> support | 270 |

Under the conservative rule, `candidate_armed` occurs only in
`gemv -> support` (35), `support -> gemv` (73), and
`support -> support` (40) on one queue. The
`multi_producer_fallback` bucket is dominated by `gemv -> gemv` (36) and
`support -> support` (72).

### 2.3 Why the numbers are a prediction, not coverage

Two facts keep this census from claiming faithful coverage:

1. The capture records `kind: RAW` but not the overlapping buffer spans.
   Therefore alias adjudication is impossible from this artifact alone. The
   census marks all 704 edges `span-unverified` and marks none alias-safe.
2. A QMD consumer has one `WAIT_ON_LATCH_ID` field. For the 109 or 110
   multi-producer consumers, the census falls back to full completion until a
   Stage 2 probe proves same-latch aggregation or a safe merge construction.

The multi-producer geometry is:

- one queue: all 109 cases have every other RAW producer on the same queue
  and earlier in the replay;
- two queues: 109 of 110 have that same-queue-earlier shape and one has a
  cross-queue other producer.

Same-queue-earlier is still not treated as transitive latch coverage, because
introducing PDL overlap can change the ordering proof the current full
dependency provides. The fallback is the closed default.

### 2.4 Multi-consumer geometry

The same capture also shows 74 RAW producers with more than one RAW consumer,
covering 220 of the 704 edges. The maximum RAW in-degree is 2 and the maximum
RAW out-degree is 3. Whether one producer latch may be waited on by several
consumers is not documented in the QMD header and is `named-unavailable`.

## 3. QMD latch-capability audit

Field-level source is `extra/nv_gpu_driver/clcec0qmd.h`, with the same bit
locations in `tinygrad/runtime/autogen/nv_570.py` and
`tinygrad/runtime/autogen/nv_580.py`.

| field | bits | verdict |
| --- | ---: | --- |
| `PRE_EXIT_AT_LAST_CTA_LAUNCH` | 638 | supported |
| `ENABLE_PROGRAM_PRE_EXIT` | 639 | supported |
| `ARRIVE_AT_LATCH_ID` | 671:640 | supported |
| `WAIT_ON_LATCH_ID` | 703:672 | supported |
| `ARRIVE_AT_LATCH_VALID` | 730 | supported |
| `WAIT_ON_LATCH_VALID` | 731 | supported |
| `LATCH_RELEASE_INVALIDATE_ENABLE` | 732 | named-unavailable |
| `HOLD_CTA_LAUNCH_UNTIL_PARENT_LATCH_ACQUIRE_AND_CTA_COMPLETE` | 733 | named-unavailable |
| `HOLD_MEMBAR_UNTIL_LATCH_ACQUIRE` | 734 | named-unavailable |
| `PRIORITY_DEMOTE_UNTIL_LATCH_ACQUIRE` | 735 | named-unavailable |

`supported` here means the field exists in the local header/autogen layout and
the current writer or an earlier matched-grid device row exercises it.
`named-unavailable` means the field exists but has no semantics documentation
and no device exercise in this repository.

Capability questions:

| capability question | verdict | next instrument |
| --- | --- | --- |
| usable latch ID count on sm_120 | named-unavailable | Stage 2 synthetic sweep over distinct latch IDs |
| one consumer waiting on more than one latch | named-unavailable | Stage 2 two-producer/one-consumer probe |
| multiple producers arriving at the same latch | named-unavailable | Stage 2 multi-producer probe with checksums |
| non-consecutive same-queue arming | named-unavailable | Stage 2 A-B-C chain with A -> C armed |
| multiple consumers waiting on one producer latch | named-unavailable | Stage 2 one-producer/multi-consumer probe |
| latch state across replay, flush, and replay groups | named-unavailable | Stage 2 replay/flush bracket |
| pre-exit-at-last-CTA proves producer data visibility | named-unavailable | Stage 2 native wait-after-trigger probe with checksums |

One device fact is inherited from the earlier Phase C work and is labeled
observed, not generalized: latch ID 7 with the arrive/wait/pre-exit fields
produces overlap on a same-queue consecutive synthetic pair
(`phase_c_native_qmd_latch_4.json`, median overlap +99.808 us, wall
463.488 us, checksums correct). That row proves one consecutive ID-7 pair on
the matched grid; it does not prove pool size, reuse, multi-wait, or
multi-producer behavior.

The CUDA PDL guide facts used for planning are: the trigger fires implicitly
at all-CTA end; the consumer must perform a grid-dependency
wait/synchronization before dependent access; the behavior is opportunistic;
and graph edges use launch-completion and programmatic ports. Native QMD
equivalence is not inferred from those CUDA semantics.

## 4. Wait and trigger placement plan

The current hook is name-pinned and crude:

- renderer prepends `griddepcontrol.wait` at instruction zero for matched
  consumers and appends or prepends `griddepcontrol.launch_dependents` for
  matched producers
  (`tinygrad/renderer/cuda.py:23-34`);
- the runtime writes latch ID 7 into a matched consecutive QMD pair
  (`tinygrad/runtime/ops_nv.py:38-51`).

The target rule is: after index/pointer/accumulator setup and immediately
before the first access to the buffer owned by the split edge.

| consumer family | first dependent access | wait placement | trigger candidates |
| --- | --- | --- | --- |
| `E_*` elementwise/residual | first global load of the owned input | after gidx/lidx/address setup, before that load | end, start |
| `r_*` ordinary reduction | first activation load in the Ridx loop | after accumulator setup, before the loop load | end, start |
| `reduce_output_rmsnorm_*` | first `x` load in the sumsq loop | after lidx/buf/alu setup, before the `x` load | end |
| `flash_block_tiled_xlane_score_pv_tile_whole_cache_*` | first KV cache load feeding `kstore` | after register/shared setup, before `kv_load(...)` | end |
| `flash_fused_gmax_combine_f16_*` | first `pout` read in the max reduction | after `max_init`, before the `pout` access | end |
| `rmsnorm_q8_1_llama_provider_4096` | first `x` load in the sumsq reduction | after lane/warp/red/base setup, before `xv = x[base]` | end |
| `q8_1_llama_provider_4096` | first `x` load in the rounded prelude | after group setup, before the `x` load | end |

Anchor sources are the UOp emitters in
`tinygrad/codegen/late/reduce_output.py:76-162`,
`tinygrad/llm/flash_decode_attention.py:186-238` and `295-341`, and
`tinygrad/llm/shared_q8_attention.py:60-166`.

Trigger policy is planned, not chosen:

- anchor Q/K/V/G/D/vocab GEMVs: default `end`, bracket `start`, `end`, and
  `last_cta_launch` on the real route in Stage 2/4;
- support norm/reduce/elementwise/flash/quant providers: default `end`
  because their output bytes exist at body end; a start trigger would expose
  the producer-data race.

No start/end winner is promoted from the matched synthetic grid alone.

## 5. Correctness checklist

| hazard | Stage 0 status | consequence |
| --- | --- | --- |
| alias | unverified | all 704 edges are span-unverified; the capture must retain overlapping buffer spans before any edge is called alias-safe |
| WAR | excluded | WAR edges remain full-completion dependencies; only RAW edges are launch candidates |
| WAW | excluded | WAW edges remain full-completion dependencies |
| multi-consumer | named-unavailable | 74 producers have more than one RAW consumer across 220 edges; latch sharing by several waiters is unproven |
| multi-producer | fallback | 109/110 multi-producer cases stay on the full-completion path until Stage 2 proves a safe merge |
| latch pool and reuse | named-unavailable | a per-graph pool is a design requirement, not a proven capacity; no reuse is assumed |
| graph flush and replay | named-unavailable | the five replay groups reset the QMD chain; 13 edges cross group boundaries in each queue mode |
| wait fence strength | unverified | `griddepcontrol.wait` fence semantics and producer-trigger flush must be verified in Stage 2 |
| non-consecutive and split-queue pairs | fallback | adjacency, encoded-wait, and queue-split edges stay on full completion |
| off-path identity | required | any Stage 1 construction must keep the gate-off output byte-identical |

The correctness rule for the first implementation is closed and conservative:
the only edges that may be armed are same-group, same-queue, consecutive,
zero-encoded-wait, single-RAW-producer edges that later pass span-level alias
adjudication. Everything else stays on the existing full-completion
dependency.

## 6. Q1-Q8 status at this stage

| question | Stage 0 status |
| --- | --- |
| Q1 full safe RAW chain arm | static prediction complete; every unarmed edge has a named structural reason, but alias and multi-producer safety remain unverified |
| Q2 native vs CUDA QMD semantics | header fields are supported; equivalence semantics are named-unavailable |
| Q3 real decode launch-ahead | unmeasured |
| Q4 wall recovery | unmeasured |
| Q5 factor attribution | unmeasured |
| Q6 native negative vs CUDA positive | unmeasured |
| Q7 residual after best arm | unmeasured |
| Q8 graph grouping effect | structurally observed (13 cross-group edges), effect unmeasured |

No belief-flip gate in section 8 of the scope has been triggered, because no
endpoint or device result was produced in this stage.

## 7. Gate status and next-stage prerequisites

Stage 0 gate: **pass** for the design-audit outputs above, with the alias
capture gap named rather than hidden.

Before Stage 1 can claim a faithful census, the capture must additionally
retain the overlapping buffer spans that back each RAW edge. Without spans,
the Stage 1 hard gate "expected RAW coverage is armed, or every miss is
explained" cannot distinguish an alias rejection from a structural miss.

Before any real-route arm is attempted:

1. Stage 1 must reproduce the census with span-typed dependency records
   behind `NV_SPLIT_PHASE=1`, with `off` as the closed default.
2. Stage 2 must probe latch pool size, multi-wait, same-latch aggregation,
   multi-consumer sharing, non-consecutive arming, and replay/flush safety on
   a matched synthetic grid.
3. No endpoint run is authorized before the Stage 1 census gate and the
   Stage 2 synthetic semantic gate both pass.

## 8. Evidence and provenance

Generated and retained artifacts:

- `docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821/stage0_raw_edge_census.json`
- `docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821/qmd_latch_audit.json`
- `docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821/stage0_wait_placement_plan.json`

Probe tooling:

- `extra/llm_research/decode/nv_edge_aware_pdl_stage0_census.py`

No benchmarks or device tests were run in Stage 0. The only execution was a
CPU-only static replay reconciliation that reproduced the Phase A counts
(108/144) and the broad Phase D counts (329/429) before the census artifact
was accepted.
