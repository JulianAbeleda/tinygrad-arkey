# NV edge-aware PDL runtime-hook construction scope

Date: 2026-08-21

Status: **construction and decision scope**. This packet authorizes a
feature-gated, probe-first implementation of an edge-aware split-phase launch
data dependency for the native NV runtime, and the endpoint experiments needed
to decide between scheduler work and fusion/body work. It is not an
open-ended redesign and it does not authorize promoting a route.

Repository: `nvidia-bringup-20260731` at `6570abc02`.

## 1. The decision this packet must make

The remaining architectural question is:

> Does a scheduler-owned split launch/data dependency, with llama-equivalent
> edge coverage, recover the measured S1 exposure and the 717.505 us token
> wall, or is the surviving gap a fusion/body problem?

This packet must produce enough evidence to choose exactly one of:

1. adopt edge-aware split-phase scheduling as the production direction;
2. stop PDL work and spend the gap budget on bounded fusion and Q4 FFN-down;
3. adopt both, with measured attribution;
4. report that the native construction cannot be made equivalent, and name
   the exact missing hardware, scheduler, or renderer capability.

The packet is complete only when every PDL question below has a
`supported`, `refuted`, or named-unavailable verdict. It is not complete
when a missing measurement is silently converted into support for either
direction.

## 2. Knowledge ledger before this packet

Already measured and locked:

| knowledge | status | evidence |
| --- | --- | --- |
| token wall gap and S1 exposure | locked | +717.505 us wall, +634.334 us S1 |
| PDL mechanism exists on a matched synthetic grid | observed | Phase C CUDA/native rows |
| tested name-pinned arm is not equivalent | observed | 108/144 armed pairs, zero positive overlap |
| broadest name filter is not enough | observed | Phase D census, max 429/761 real edges |
| Q4 FFN-down body deficit | observed | ~30.016 us available at the corrected 19.232 us/node floor |
| legal fusion ceiling conversion | unmeasured | residual/reduce ceilings simulated |

Phase D (`phase_d_static_coverage.json`) also classified the structural
limits of the current mechanism:

| static edge block reason | 2q count |
| --- | ---: |
| armed by all-names filter | 429 |
| non-consecutive placement | 632 |
| split across queues | 107 |
| encoded wait | 35 |
| graph-group boundary | 27 |

The name filter is no longer the binding limit. The binding limits are
adjacency, queue placement, wait encoding, and graph grouping.

## 3. The unknowns and the instrument that closes each

Every unknown is paired with one named observable. If the observable cannot
be built, the row stays `unmeasured` and the scope must record the missing
capability; it must not infer an answer.

| question | distinguishing observable | instrument | completion criterion |
| --- | --- | --- | --- |
| Q1. Can the scheduler arm the full safe RAW chain? | armed-edge census per bucket and per block reason | typed dependency census from the real DAG | every unarmed safe RAW edge has a named reason; expected coverage is reproduced on device |
| Q2. Does native QMD lowering have CUDA-equivalent semantics? | trigger shadow, consumer grid start, wait exit, producer end on the same grid | matched synthetic native-vs-CUDA probe with `%globaltimer` | native candidate shows launch-ahead and wait overlap; control does not |
| Q3. Does the real decode route actually launch ahead? | positive overlap and wait-exit-before-producer-end per armed edge | probe-only timestamped real-route capture | positive overlap count > 0 and wait timestamps valid for the intended chain |
| Q4. Does launch-ahead convert to wall recovery? | S1 delta and endpoint wall in control/candidate/control | fresh-process endpoint bracket | token SHA identical; recovery attributed to the changed mechanism |
| Q5. Which factor carries the effect? | per-factor wall and overlap deltas | trigger start/end, wait entry/prologue, 1q/2q, five-graph/one-graph brackets | factors ranked by measured delta, not by guess |
| Q6. Is native negative while CUDA positive? | same-construction CUDA vs native deltas | Phase C-class matched grid plus the new hook | isolates native lowering from the scheduling idea |
| Q7. What remains if PDL wins partially? | residual S1 after the best arm | endpoint ledger | residual number becomes the budget for 8.2/8.3 fusion work |
| Q8. Does graph grouping itself matter? | one-graph control vs one-graph candidate | `JIT_BATCH_SIZE`/`GRAPH_ONE_KERNEL` bracket | grouping labeled supported or not; no cross-group arm is claimed silently |

## 4. Knowledge-completeness contract

"Full knowledge to figure out what to do" is defined narrowly and
verifiably:

- Q1-Q8 each receive a verdict or a named unavailable reason;
- the Direction A versus Direction B decision can be made without appealing
  to an unmeasured row;
- any residual gap is attached to the next named scope (fusion, body, native
  lowering, or hardware semantics);
- no claim crosses the measured body/grid boundary.

This packet deliberately does **not** close:

- bounded fusion sufficiency (H7/H8; separate scope 8.3);
- Q4 FFN-down body parity (separate scope 8.2);
- prefill, batch decode, or other workload generalization;
- whether a candidate should be promoted to the production default. That is
  a separate promotion scope after the promotion gate passes.

If a stage fails, the packet exits with the exact missing capability. An
exit with `unmeasured` is acceptable only once per question and must name the
next construction that could measure it.

## 5. Target design

The production shape is scheduler-owned and edge-aware. The implementation in
this packet may be feature-gated and incremental, but it must converge to
this shape.

### 5.1 Dependency model

Replace the binary full-completion dependency with a typed split dependency:

```text
SplitDependency(producer, consumer):
  data_buffers   # RAW access spans owned by this edge
  access_kind    # RAW | WAR | WAW
  launch_ready   # producer launch-complete event / latch
  data_ready     # producer output bytes ready (existing signal path)
  wait_position  # entry | before_first_dependent_access
  trigger_policy # start | end | last_cta_launch
  latch_id       # assigned at graph construction
```

The existing scheduler authority is:

- `tinygrad/engine/jit.py:406-415` `DepsTracker.peek_access_resources`;
- `tinygrad/engine/jit.py:472-473` `_access_resources`;
- `tinygrad/runtime/graph/hcq.py:382-407` `_resolve_deps`;
- `tinygrad/runtime/graph/hcq.py:200-230` queue selection and schedule
  construction.

Those paths currently return anonymous `(queue, value)` dependencies. The
first implementation change is to carry access kind and buffer spans through
the dependency records while keeping the feature-off output byte-identical.
RAW edges are launch candidates; WAR/WAW edges remain full-completion
dependencies unless the implementation can prove an ordered latch gate.

### 5.2 Runtime and QMD half

Current name-pinned arming is `tinygrad/runtime/ops_nv.py:38-51` and is
called from `NVComputeQueue.exec` at `tinygrad/runtime/ops_nv.py:197-207`.
The edge-aware version must:

- compute a `launch_latch` plan during graph construction, not inside
  `exec` from environment names;
- set the producer QMD fields used today:
  `arrive_at_latch_valid`, `arrive_at_latch_id`,
  `enable_program_pre_exit`, `pre_exit_at_last_cta_launch`;
- set the consumer QMD fields used today:
  `wait_on_latch_valid`, `wait_on_latch_id`;
- leave `dependent_qmd0_pointer/action/prefetch/enable` chaining intact for
  execution order;
- assign latch ids from a per-graph pool instead of hardcoding `7`.

Stage 0 must answer, with source or device evidence, not assumption:

- how many latch ids are usable on this device;
- whether one consumer can wait on more than one latch and, if not, how
  multiple RAW producers are merged or fall back;
- whether a latch can arm a non-consecutive same-queue QMD pair;
- whether latch state is safe across graph replay/flush and across the five
  replay groups.

Until each answer is pinned, the unsafe case falls back to the current full
dependency. The fallback must be the closed default.

### 5.3 Renderer and codegen half

Current injection is `tinygrad/renderer/cuda.py:23-34`, applied at
`tinygrad/renderer/cuda.py:120-123`. It is name-pinned and places the
consumer wait at instruction zero and the producer trigger at body start or
body end.

The target changes:

- renderer receives per-program split policy from graph metadata, not from
  process-wide name prefixes;
- `wait_position=entry` reproduces today's placement;
- `wait_position=before_first_dependent_access` emits the
  `griddepcontrol.wait` after index/pointer setup and immediately before the
  first access to a buffer owned by the split edge;
- `trigger_policy` is start, end, or last-CTA per program family, with a
  measured bracket instead of an assumed best value.

The interim probe may use a versioned `NV_SPLIT_PHASE_POLICY` JSON to drive
per-program placement, but that is scaffolding, not the production
interface.

### 5.4 Graph grouping

The current five replay groups reset the QMD chain (`tinygrad/engine/jit.py:244-260`
and `tinygrad/engine/jit.py:333`). Phase D shows 27 edges cross group
boundaries. The packet must test:

- edge-aware arm within the existing five groups first;
- then one continuous graph with its own control, because
  `JIT_BATCH_SIZE=1024` was previously +112.9 us without a PDL arm and
  cannot be reused as the candidate baseline.

Do not claim a continuous launch-ahead pipeline unless the one-graph
candidate arm census shows the cross-group edges actually armed.

## 6. Staged implementation and hard gates

### Stage 0: design audit, no GPU

Exit requires:

- typed dependency records and an edge census prediction for Q1;
- QMD latch-capability answers from section 5.2;
- a wait-placement plan that names the first dependent access for each
  consumer program family;
- a correctness checklist for alias, WAR/WAW, multi-consumer, multi-producer,
  latch reuse, and graph flush.

No production behavior may change in this stage.

### Stage 1: probe-only construction census

Add the feature behind `NV_SPLIT_PHASE=1` with `off` as the closed default.
Produce:

```text
expected_safe_raw_edges
armed_pairs
per_edge: reason, queue, group, latch_id, wait_position, trigger_policy
missed_edges: adjacency | queue_split | encoded_wait | cross_group |
              multi_producer_fallback | alias_rejected | unknown
```

The census must reconcile against `phase_a_control.json` and
`phase_d_static_coverage.json`. A hard gate: expected RAW coverage is armed,
or every miss is explained. If the census itself cannot be made faithful,
stop and report the missing capability; do not spend GPU time on a
non-equivalent arm.

### Stage 2: matched synthetic semantic gate

Reuse the Phase C probe shape: one producer, one consumer, checksum,
`%globaltimer`, control/candidate/control. Compare native QMD and CUDA PDL
for `trigger_policy` and `wait_position` pairs. Gate: native candidate
launches ahead and exits its wait before producer end; native control does
not; checksums pass.

### Stage 3: real-route timestamped capture

Instrument a probe-only consumer subset with `%globaltimer` and record per
edge:

```text
producer_start, trigger, consumer_grid_start, wait_exit, producer_end,
consumer_end, overlap, useful_body
```

Existing graph timestamps supply kernel start/end. The injected timer writes
supply trigger and wait-exit. Token SHA must remain identical to the
control; instrumentation is a profiling tax, so a separate unprofiled
endpoint bracket is required in Stage 4.

### Stage 4: endpoint bracket

Run fresh-process control/candidate/control under `flock
/tmp/gpu-bench.lock` for each arm in section 7. Record S1, union, overlap
mass, dead device time, wall, token SHA, and lock session id.

### Stage 5: decision and report

Write the result under
`docs/task_workflow/output/nv-edge-aware-pdl-runtime-hook-result-20260821.md`.
No promotion happens in this stage.

## 7. Experiment matrix

Ordered cheapest-first. Every candidate shares the same token, and every GPU
row uses a fresh process and the lock.

| arm | purpose | expectation |
| --- | --- | --- |
| off | baseline | reproduce locked wall |
| edge 1q, end trigger, entry wait | minimal edge-aware arm | census full; overlap positive if mechanism works |
| edge 2q, end trigger, entry wait | queue interaction | separates placement from scheduling |
| edge 2q, start trigger, entry wait | trigger position | Phase C says shadow moves |
| edge 2q, end trigger, prologue wait | wait placement | tests H3 on the real route |
| one graph, best prior arm | graph grouping | tests H5 |
| CUDA matched grid, new hook | native-vs-CUDA | isolates native lowering |

Factor order is deliberately one-at-a-time. If the construction cannot
express one factor, that factor is marked unavailable with its reason rather
than skipped.

## 8. Belief-flip gates

Define these before seeing results:

- faithful census plus S1 recovery >= ~150 us with identical tokens supports
  Direction A;
- faithful census with flat or negative wall supports Direction B;
- native negative while the matched CUDA grid positive identifies native
  lowering as the defect, not split-phase scheduling;
- both positive but native smaller identifies coverage, wait placement, or
  graph grouping as the residual lever;
- a census that cannot arm the safe RAW chain invalidates any endpoint result
  from that arm.

The recovered amount is measured on the endpoint S1/wall ledger, not a
simulated zero-cost ceiling.

## 9. Decision matrix

| evidence combination | action after the packet |
| --- | --- |
| faithful arm, native positive, wall recovery >= ~150 us | promote Direction A to production design review; rerun fusion scopes against the new baseline |
| faithful arm, native flat/negative, CUDA positive | stop endpoint PDL; open a narrow native-lowering repair scope |
| faithful arm, both flat/negative | drop Direction A; spend the gap on 8.2 Q4 FFN-down and 8.3 bounded fusion |
| partial arm, measurable recovery | promote the winning factor, keep the residual as the next fusion/body budget |
| construction cannot express faithful coverage | record the exact missing capability and hand it back as a new construction scope |

No cell permits a performance claim without the named census and endpoint
evidence.

Completeness proof: cell 1 is Q1+Q2+Q3+Q4 true; cell 2 is Q6 true with
Q4 false; cell 3 is Q1+Q2+Q4 true with flat walls; cell 4 is Q3+Q4 partial
with Q7 assigning the residual; cell 5 is Q1 false with a named blocker.
Every Q1-Q8 result therefore maps to exactly one action, and no action
depends on the H7/H8 or Q4-FFN-down questions that belong to the parallel
fusion/body packet.

## 10. Allowed paths, rollback, and ownership

Authorized implementation paths:

- `tinygrad/runtime/graph/hcq.py` typed dependency records and schedule
  metadata;
- `tinygrad/engine/jit.py` dependency tracking and optional grouping knob;
- `tinygrad/runtime/ops_nv.py` latch plan and QMD field writes;
- `tinygrad/renderer/cuda.py` wait/trigger placement;
- `extra/llm_research/decode/**` probes and drivers;
- `docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821/**`
  and the required output report.

Every production-path change must be behind `NV_SPLIT_PHASE=1` (or a
same-scoped explicit feature gate) and must make the off path byte-identical
to current `6570abc02` behavior. The rollback is: unset the gate. No change
to model files, token numerics, or route defaults.

Ownership:

- edge identity/access kind: scheduler (`jit.py` dependency tracker);
- latch allocation and QMD writes: NV runtime;
- wait and trigger placement: CUDA renderer;
- graph grouping: JIT graph construction;
- timestamps and endpoint ledger: probe tooling.

## 11. Required output and evidence

Retain at minimum:

- typed edge census JSON with schema/version, commit, date, queues, groups,
  and per-edge reasons;
- QMD latch-capability audit notes with device/source citations;
- synthetic matched-grid driver JSON for every arm;
- real-route timestamped capture and token SHA;
- endpoint bracket rows and the wall/segment ledger;
- the decision report named in Stage 5.

The report must label every number `observed`, `inferred`, or `unmeasured`,
and must answer each Q1-Q8 individually.

## 12. Acceptance criteria

The packet is accepted only if:

- the off path remains byte-identical to current behavior;
- the construction census explains every armed and missed edge;
- synthetic native/CUDA semantics are measured, not assumed equivalent;
- the real-route capture records wait-exit, not just grid start;
- endpoint recovery is taken from control/candidate/control with token SHA
  equality;
- Q1-Q8 each have a verdict or a named unavailable reason;
- the final decision is one of the four cells in section 1 and is falsifiable
  by the retained evidence;
- no missing measurement is used to support Direction B by default.

## 13. Grounding artifacts

- `docs/task_workflow/output/nv-split-phase-pdl-causal-design-review-20260820.md`
- `docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_d_static_coverage.json`
- `docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_a_control.json`
- `docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_c_driver_fixed.json`
- `docs/task_workflow/output/nv-weighted-llama-real-edge-dag-20260820.json`
- `docs/task_workflow/output/nv-weighted-inter-anchor-ledger-20260820.json`

Reusable probe code:

- `extra/llm_research/decode/nv_pdl_phase_d_static_coverage.py`
- `extra/llm_research/decode/nv_pdl_phase_c_cuda_probe.py`
- `extra/llm_research/decode/nv_pdl_phase_c_native_probe.py`
- `extra/llm_research/decode/nv_pdl_phase_b_probe.py`

## 14. Bans and hard stops

- No endpoint run before Stage 1 census passes and Stage 2 synthetic gate
  passes.
- No arm may substitute a non-equivalent census for the failed one.
- No latch reuse, alias relaxation, or wait placement may be assumed from the
  CUDA driver semantics; each must be pinned on the native path.
- No promotion, route-default change, or production interface rename in this
  packet.
- No `unmeasured` row may be converted into support for Direction B.
- No grouping experiment may reuse the pre-PDL `JIT_BATCH_SIZE=1024` result
  as the candidate baseline.

## 15. One-line job

Build the edge-aware split launch/data dependency behind a closed default,
prove its census and native semantics, then measure whether equivalent
coverage recovers the S1 wall and use that measurement to choose the
scheduler, the fusion/body direction, or a named missing capability.
