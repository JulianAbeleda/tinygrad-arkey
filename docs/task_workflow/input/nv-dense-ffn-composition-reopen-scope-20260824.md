# NV dense FFN composition reopen scope

Date: 2026-08-24
Status: measurement-first scope; instrumentation is authorized, production changes are not authorized
until a causal gate below passes

## Objective

Determine whether the current dense-decode FFN still loses locally fast kernel
time at the gate/up-to-down production boundary. If it does, identify and test
one exact producer-side construction. If it does not, close FFN composition
and attribute the remaining opportunity to cold kernel streaming rate before
moving to the attention chain.

This is a dense autoregressive decode scope. It makes no claim about MoE,
prefill, speculative decoding, or other conditional-compute routes.

## Current authority

- Principles: `docs/what-makes-inference-fast.md`, especially the kernel and
  token lifecycles.
- Current wall and device ledger:
  `docs/task_workflow/output/nv-full-ledger-audit-vs-llama-20260824.md`.
- Ranked campaign closure:
  `docs/task_workflow/output/nv-ranked-parity-campaign-closure-20260824.md`.
- Prior FFN chain facts:
  `docs/task_workflow/output/nv-ffn-chain-structure-audit-result-20260823.md`.

The current fresh endpoint is 4355.023 us/token. Reaching 240 tok/s requires
188.357 us/token; retained llama parity requires 306.699 us/token. These are
milestone distances, not projected FFN recoveries.

## Facts already established

1. One token executes 36 gate/up programs and 36 down programs.
2. Gate/up has one RAW consumer: the corresponding down projection.
3. There is no separate activation, cast, quantization, or packing kernel
   between them in the installed route.
4. The only explicit producer/consumer value is one fp16 activation buffer per
   layer, 24 KiB, beside roughly 3.29 GB/token of streamed FFN weights.
5. The residual update is already absorbed by the down epilogue.
6. Typed four-warp gate/up, Q4 down vector loads, and Q6 down packed lane
   mapping are already promoted.
7. The retained rate comparison shows a remaining cold-streaming efficiency
   difference, but does not prove that the gate/up-to-down edge costs wall.

Therefore this scope must not begin by assuming that another fusion exists.
The first deliverable is a complete-information partition of FFN body time and
the single dependency edge.

## Hypotheses

### H0: no remaining composition lever

The edge contains no material non-body delay. Gate/up finishes, down becomes
runnable, and the remaining FFN difference is body streaming rate. This is the
default null and a valid closure.

### H1: recoverable producer-to-consumer boundary delay

The installed chain contains measurable admission, dependency-release, cache,
or scheduling delay after gate/up completes and before down begins. A bounded
producer contract or runtime construction can remove that delay without
changing arithmetic or adding transport.

### H2: fixed-byte cold-rate opportunity

The edge is already efficient, but one current gate/up or down family remains
limited by load issue, instruction mapping, occupancy, or insufficient memory
parallelism before reaching the sustainable DRAM rate. A local kernel change
can raise the cold production rate without changing compulsory bytes.

H1 and H2 are separate claims. A faster L2-hot microbenchmark does not prove
H2, and a lower node count does not prove H1.

## Prohibited assumptions and closed spellings

Do not reopen any item below without a new measured fact that invalidates its
old gate:

- adding an fp16 cast or Q8 packing stage between gate/up and down;
- treating the 24 KiB activation buffer as a material DRAM-byte lever;
- Q6 multi-row-per-CTA packing;
- an untyped gate/up output that recreates hidden materializations;
- generic queue placement or PDL timestamp reordering without a newly proven
  independent edge;
- a monolithic gate/up-plus-down kernel that duplicates gate/up work or weight
  traffic;
- booking hot-cache instruction savings as production recovery;
- comparing profiled device union with an independently sampled wall run.

## Phase 0: provenance and clean control

Record:

- repository commit and local-diff manifest;
- model hash, backend, GPU identity, clocks, power state, and decode depth;
- all route-affecting environment variables;
- fresh no-override token hash, wall bracket, node census, node sum, union, and
  overlap;
- exact selected names and counts for all gate/up, Q4 down, and Q6 down
  programs.

Unrelated user changes and untracked artifacts are out of scope and must not
be staged or modified.

## Phase 1: reconstruct the physical FFN chain

For every layer, emit one machine-readable row containing:

- producer and consumer program identity;
- buffer address, dtype, shape, byte span, and declared ownership;
- predecessor and successor edges;
- producer start/end and consumer start/end timestamps from one clock domain;
- consumer runnable/admission point when the backend exposes it;
- intervening device work, if any;
- whether the edge crosses a graph segment, queue, synchronization, copy, or
  conversion boundary.

Required invariants:

```text
36 gate/up producers
36 down consumers
one gate/up -> down RAW edge per layer
zero intervening cast/copy/activation programs
zero undeclared output materializations
```

Any violation is an information wall. Fix the census or route accounting
before testing an optimization.

## Phase 2: partition edge time from body time

Measure, in the same profiled token domain:

```text
gate_up_body[layer] = gate_up_end - gate_up_start
edge_wait[layer]    = down_start - gate_up_end
down_body[layer]    = down_end - down_start
```

`edge_wait` is only a candidate pool. Attribute it to actual intervening work,
dependency release, admission, or idle residence before calling it removable.
Negative or overlapping timestamps indicate incompatible domains or a capture
error; they are not performance evidence.

Decision:

- If edge delay is consistently attributable and large enough for a bounded
  construction to rise above wall noise, H1 may proceed.
- If edge delay is absent, already overlapped, or explained by required work,
  close H1 and continue only with H2.
- Do not sum per-layer medians and present them as the token median. Retain raw
  per-token sums and report the non-commutativity explicitly.

## Phase 3A: H1 producer-side causal gate

Before production code, specify one construction with:

- the exact boundary term it removes;
- unchanged gate/up and down arithmetic order;
- unchanged compulsory weight traffic;
- explicit output dtype/layout/ownership;
- predicted node and edge changes;
- no new copy, cast, synchronization, or graph break;
- a rollback control.

The smallest acceptable gate is an exact replay or production-conditioned
subgraph showing that the named boundary shrinks while both kernel bodies stay
within noise. If no legal construction can name and remove the measured term,
H1 closes without implementation.

## Phase 3B: H2 cold-rate attribution

Capture current exact cubins under cache-cold, production-representative input
conditions for each distinct FFN family. Record at least:

- DRAM read/write bytes and sectors;
- duration and achieved DRAM rate;
- executed instructions and issued load forms;
- active warps, registers, spills, and occupancy;
- long-scoreboard, memory-throttle, instruction-throttle, and dependency
  stalls where available;
- L1/L2 hit behavior;
- per-family calls and total token contribution.

The attribution must select one binding mechanism before a candidate is built.
Permitted search dimensions include load mapping, lane mapping, warp/CTA
geometry, legal prefetch distance, accumulator organization, and instruction
scheduling. Shape, quantization, and target facts may select a route; model
identity may not.

## Phase 4: kernel lifecycle gate

For either H1 or H2, require:

1. exact elementwise output at the producer/consumer boundary;
2. identical token-stream hash in production;
3. an isolated cold or production-conditioned subgraph win tied to the stated
   mechanism;
4. the expected counter movement;
5. no spills, hidden transports, or unplanned node additions;
6. confirmation that the candidate is selected for the intended population.

If the mechanism should pass from the measured facts but the gate is flat,
promote the result to an information-wall audit: check cache regime, route
selection, typed output substitution, final scheduled cardinality,
synchronization, and composition. Do not immediately label the mechanism
exhausted.

## Phase 5: token lifecycle qualification

Run fresh-process `control / candidate / control` wall brackets at the current
production depth with at least nine repetitions. Each repetition must cover
the same number of timed tokens and preserve per-arm token hashes.

Promotion requires:

- exact outputs and the intended production census;
- a positive candidate recovery against the control midpoint;
- no unexplained displacement in node sum, union, overlap, or boundary rows;
- confirmation at higher repetitions or a second depth when the result is
  comparable to control drift;
- a conservative booking derived from same-session controls only.

A profile-only win, isolated hot-cache win, cross-session endpoint movement,
or lower node count alone is not promotable.

## Phase 6: close and re-ledger

After a promotion, rebuild the complete token ledger before opening another
FFN candidate. Report both latency recovery and its reciprocal tok/s movement;
do not add independent profile ceilings as a forecast.

Close the FFN composition campaign and move to the attention lifecycle when
either condition holds:

1. H1 closes and H2 produces no new counter-supported candidate; or
2. all counter-supported candidates fail complete-information wall gates.

Closure is a successful result. It means the ledger has localized the next
lever outside FFN rather than leaving an unexplained wall.

## Required artifacts

Write evidence beneath:

`docs/task_workflow/evidence/nv-dense-ffn-composition-reopen-20260824/`

Minimum outputs:

- `provenance.json`
- `control-wall.json`
- `control-profile.json`
- `ffn-edge-ledger.json`
- `ffn-cold-counters.json`
- `candidate-contract.json` when a candidate is admitted
- `candidate-exactness.json` when a candidate is admitted
- `candidate-profile.json` and `candidate-wall.json` when qualified
- `final-ledger.json`

The result document must distinguish measured facts, inferences, projections,
and booked wall recovery.

## Deliverable

Return exactly one of:

- `PROMOTE_FFN_PRODUCER_CONSTRUCTION`
- `PROMOTE_FFN_COLD_RATE_CONSTRUCTION`
- `CLOSE_FFN_COMPOSITION_MOVE_TO_ATTENTION`
- `FFN_INFORMATION_WALL_REQUIRES_ACCOUNTING_REPAIR`
