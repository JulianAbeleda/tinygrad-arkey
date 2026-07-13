# Direct-L2 versus LDS: canonical completion audit

Date: 2026-07-13

Status: canonical Claude handoff for the `attn_qo` comparison arc

This supersedes the completion definition in
`direct-l2-execution-bridge-scope-20260712.md`. The broader
`pure-8b-register-route-100pct-execution-scope-20260712.md` remains the project
scope after this bounded comparison closes.

## Objective

Answer this question reproducibly:

> On AMD gfx1100 for `attn_qo` `(M,N,K)=(512,4096,4096)`, is the complete
> compiler-owned register-resident/direct-L2 implementation materially faster
> than the preserved LDS implementation?

The result may be `direct_l2_wins`, `retain_lds`, or
`measurement_inconclusive`. Completion does not require direct L2 to win. A
negative result is a completed experiment and becomes a durable refutation.

The current direct path uses normal compiler-owned lowering; LDS uses the
existing raw LDS2 generator. D100 is therefore a **whole implementation
comparison**, not proof that storage location alone caused the difference. A
causal storage-only claim requires matched lowering where transport is the only
material variable.

## Completion levels

### D100: decision-complete

D100 is the required outcome here. One canonical command must produce an
offline-verifiable artifact proving:

1. both exact candidates share workload/semantic-pair identity but preserve
   distinct candidate, compile, binary, executable, run, and output identities;
2. both pass independent structural/resource admission;
3. every GPU operation occurs inside one isolated guarded executor;
4. full-output correctness passes on deterministic nonconstant inputs;
5. synchronized, pinned, randomized paired timing passes the declared noise
   protocol;
6. the deterministic authority records a final research verdict, shipping
   decision, claim class, fallback, and stop reason;
7. LDS remains the shipping fallback.

### R100: route-complete

R100 is conditional on D100 returning `direct_l2_wins`. It additionally needs:

- one centralized route authority and no alternate activation/dispatch path;
- exact workload/target admission and tested fallback elsewhere;
- whole-prefill correctness and synchronized context-set transfer evidence;
- explicit role attribution, no decode/unrelated-role regression, and rollback;
- default-off status until the whole-prefill gate passes.

If D100 returns `retain_lds`, R100 for this candidate means preserving LDS,
recording the refutation, and deleting obsolete direct-only activation code.

### P100: project-complete

P100 adds independent `attn_kv`, `ffn_down`, and `ffn_gate_up` evidence,
whole-model route purity, and generate/evaluate/prune/remember machine search.
It is owned by `pure-8b-register-route-100pct-execution-scope-20260712.md` and
cannot be claimed from this single-role result.

## Principle baseline

This contract applies `structure/Development/coding-principles.md` and
`structure/Development/performance-primitive-research-principles.md`:

- centralize identity, safety, execution, evidence, and decision authority;
- modularize transport-specific compilation/validation behind adapters;
- encode invariants and typed states instead of relying on callbacks;
- contain all dangerous GPU power behind one reviewed process boundary;
- classify evidence and claims honestly;
- test the real failure boundary;
- treat the harness as part of the primitive;
- use rows/parameters rather than dated one-off files;
- preserve refutations and remove superseded machinery.

## Current state

Reusable foundations:

- typed storage/wait/resource policy in `compiler_policies.py`;
- staged lifecycle/proofs in `kernel_pipeline.py`;
- CPU-only exact pair generation;
- real compile-only PROGRAM/evidence for both candidates;
- runtime binary-identity bridge;
- transport-neutral guarded buffers/full comparison;
- process timeout/group termination;
- randomized paired-runner shell and CPU-only decision policy;
- passing unit tests for each isolated component.

Not proven:

- no production call site composes those pieces end to end;
- no real guarded direct-L2/LDS GPU result exists;
- no hardware test exercises the composed safety boundary;
- no causal storage-only claim is supported;
- no whole-prefill route promotion follows from fake callback tests.

## Findings

`P0` can invalidate safety or the conclusion. `P1` blocks principled route
completion. `P2` is maintainability/research-system debt.

### P0-1 — dispatch evidence is false

The hardware executor and canary invoke callbacks that may dispatch, then
return `dispatch_performed=False`. Replace the boolean with typed states:
`not_attempted`, `attempted`, `submitted`, `completed`, `failed`, `timed_out`,
and `device_lost`. Only the canonical executor may create dispatch evidence;
policy consumes it.

### P0-2 — dangerous execution remains caller-owned

Guarding and process isolation exist but the executor does not compose them.
Opaque callbacks own allocation, launch, synchronization, health, and timing.
The canonical executor must internally isolate; initialize runtime, allocate,
dispatch, synchronize, read back, compare, and close inside the child; and stop
after any timeout, device loss, guard corruption, or numerical failure. No live
GPU runtime may be constructed in the parent first.

### P0-3 — pair identities are conflated

Pair generation creates distinct candidate identities. The adapter later
writes the direct identity into both benchmark rows, and the decision gate does
not require the LDS identity. The mandatory lattice is:

```text
workload -> semantic schedule -> pair
  -> direct candidate -> compile -> binary -> executable -> run/sample/output
  -> LDS candidate    -> compile -> binary -> executable -> run/sample/output
```

Only workload, semantic schedule, and pair are shared.

### P0-4 — production cannot distinguish fake evidence

Fake callback records correctly test policy logic, but production accepts the
same shape. Canonical persisted run schemas must be required for decisions.
Synthetic fixtures must carry `synthetic=True` and be rejected in production
mode. Callback-only wrappers become private test adapters or are removed.

### P0-5 — timing is not hardware-authoritative

Wall time around an opaque callback and arbitrary `samples_ms` do not prove GPU
time. Exactly one synchronized dispatch must create one sample bound to
candidate/binary/run/order/clock/output identity. Warmups must complete and be
excluded. Compile/runtime setup is excluded. Counter artifacts contain values
and windows, not only `status=live`.

### P0-6 — the comparison is confounded

Direct and LDS differ in compiler path, stage count, waits/barriers, fragment
layout, and storage. D100 must use claim class
`whole_implementation_comparison`. Do not infer L2 residency from zero LDS or
counters. `storage_only_causal_comparison` requires matched arithmetic,
lowering, work decomposition, launch geometry, and schedule.

### P0-7 — device-loss recovery is unproved

Killing a child does not establish GPU/host recovery. The parent must persist
child exit/signal/timeout, then run an independently timed health canary. Health
loss ends the session. No automatic reset, retry, or larger-stage continuation;
operator reset/reboot begins a new session/authorization.

### P1-1 — overlapping schema systems

`execution_bridge_contracts.py` is typed but used only by tests; active modules
use unrelated dictionaries. Adopt and extend one schema family or delete it.
Do not add another. One serializer/parser/digest implementation spans compile,
execution, samples, comparison, and verdict.

### P1-2 — two runtime dispatch APIs

`ExecutableHandle.dispatch()` uses PROGRAM geometry; `__call__()` invokes the
runtime directly. Remove `__call__` or make it semantically identical to
`dispatch`. Callers cannot supply or omit geometry.

### P1-3 — transport is inferred from strings

`validate_transport` searches residency for `stage_ab_register`, otherwise
defaults to LDS. Carry typed `StoragePolicy`/`TransportPlan`, dispatch through
an explicit adapter registry, and reject unknown transports.

### P1-4 — the common pipeline is still LDS-shaped

`KernelStage1PipelinePlan` requires positive `slot_bytes`/`active_lds_bytes`;
the wrapper retains a legacy callback API and string readiness modes. This does
not block D100, but later convergence needs one logical storage-neutral plan,
adapter-owned physical storage, typed readiness, and one public graph builder.

### P1-5 — result/error semantics are ambiguous

States such as `status=pass`, `decision=retain_lds`, and nonempty `blockers`
mix validity with outcome. Separate execution status, correctness, benchmark
validity, research verdict, and shipping decision. Errors need typed code,
phase, recoverability, candidate/run, and operator context. A valid negative
performance result is not blocked or failed.

### P1-6 — no reproducible lifecycle command

One CLI must accept a declarative experiment row and output path. Default mode
is compile/audit-only. GPU mode needs exact explicit opt-in and prints the
bounded work plan. It never changes route defaults. Offline verification is
CPU-only and deterministic.

### P2-1 — file/experiment sprawl

`extra/qk` has 178 Python files and about 29.9k lines. This arc has dated
generator, adapter, executor, canary, runner, and test files. Implement D100 by
consolidating existing files, not adding another dated runner. Variants become
rows; tests move to the normal tree; obsolete shells are removed after verdict.

### P2-2 — environment authority is incomplete

Broad prefix clearing and opaque environment dictionaries are not a durable
schema. Capture exact relevant variables, target, driver/runtime, GPU, clocks,
power state, git revision, and dirty state. Dirty-tree promotion evidence is
rejected.

### P2-3 — fixed workload policy leaks into generic lifecycle

Qwen profile, gfx1100, role, and shape are valid row data, not generic executor
policy. Keep execution/safety/identity/benchmark/decision workload-neutral and
reject unsupported adapter capabilities explicitly.

## Canonical architecture

```text
ExperimentRow
  -> CompileAuthority
       -> direct_l2 adapter
       -> LDS adapter
  -> IsolatedGuardedExecutor
  -> PairedBenchmarkEvaluator
  -> DecisionAuthority
       -> artifact + ledger verdict
```

The experiment row owns workload, target, semantic schedule, adapters, ABI,
canary stages, guards, health, timeout, numerical policy, timing protocol,
decision/stop rules, fallback, and claim class.

Compile authority returns no live runtime, only exact identity, PROGRAM launch
ABI, source/binary, final resources, and adapter-specific structural truth.

The isolated executor owns all dangerous behavior. Parent validates, launches,
terminates, records terminal state, performs independent health checks, and
persists failures. Child constructs runtime, verifies binary, allocates guarded
buffers, uploads deterministic inputs, dispatches with PROGRAM geometry,
synchronizes, reads full output/guards, compares, closes, and returns one typed
result.

## Authoritative measurement and decision

Protocol:

- three independent sessions;
- at least three warmups per candidate/session;
- at least twenty randomized paired timed rounds/session;
- exactly one sample per candidate per round;
- pinned clocks or an approved stable-clock protocol;
- telemetry before/during/after each session;
- synchronized device time, setup excluded;
- raw order, samples, output joins, counters, median, spread, CV, paired speedup,
  confidence interval, and session drift persisted.

Research verdict:

- `direct_l2_wins`: correctness passes and the lower bound of the 95% paired
  bootstrap speedup interval is at least +3% in every accepted session;
- `retain_lds`: correctness passes and the upper bound is below +3%, or direct
  L2 is slower;
- `measurement_inconclusive`: interval straddles +3%, telemetry/session drift
  exceeds policy, or stability is insufficient;
- `blocked`: prerequisites are missing/malformed;
- `failed`: execution, health, guard, or correctness fails.

Shipping always retains LDS until R100 passes. Run three required sessions. If
inconclusive, allow at most two additional predeclared sessions without tuning
candidate or threshold. After five, persist `measurement_inconclusive`, retain
LDS, and close this shipping attempt. New research needs a new hypothesis and
candidate identity.

## Required artifact

The offline-validating JSON contains:

- schema/version/digest, exact command, normalized row, timestamps/sessions;
- git revision/dirty state, host/kernel, driver/runtime/compiler/tool versions;
- GPU/arch/memory/clocks/power state;
- complete identity lattice and join checks;
- source/binary, target/ABI/launch/arguments, resources/occupancy;
- direct mapping/wait proof and LDS window/store/load/barrier proof;
- opt-in/authorization, child exit/signal/timeout/stdout/stderr;
- independent health, guards, immutability, reference/tolerances, full output,
  finite/shape/error/digest, truthful dispatch state;
- warmups, randomized order, every paired sample, synchronization, clocks,
  telemetry/counters, statistics/noise;
- claim class/confounders, research verdict, shipping decision, fallback,
  typed reasons, stop rule, R100 authorization, ledger/refutation link.

## Claude implementation phases

Each phase is reviewed, tested, committed, and pushed separately. NFC
consolidation and behavior changes do not share a commit.

### C0 — freeze authority

- mark this as the sole D100 definition in existing scope/index docs;
- select the one schema family;
- add failing regressions for false dispatch state and LDS identity replacement.

Exit: one authority, both bugs pinned, no GPU dispatch.

### C1 — unify identities and states

- implement the identity lattice/canonical serialization;
- preserve distinct candidates end to end;
- add typed execution/error/verdict states;
- reject synthetic production evidence; migrate tests.

Exit: every bad digest edge fails closed; deterministic round-trip; no GPU.

### C2 — centralize transport adapters

- explicit direct/LDS adapter table and typed transport policy;
- unknown transport rejection;
- workload configuration moved to the experiment row;
- inferred residency-string selection removed.

Exit: both compile-only paths pass via one adapter boundary; no GPU.

### C3 — one isolated guarded executor

- compose bridge, isolation, guards, health, full correctness, cleanup;
- runtime creation inside child; no production launch callbacks;
- remove/neutralize alternate runtime call path;
- truthful state and terminal artifact on every outcome.

Exit: fake-runtime tests cover success, ABI error, corruption, mutation,
numerical error, timeout, crash, no-result, cleanup; no real GPU yet.

### C4 — harmless safety canary

- define the independent health probe;
- run a known-safe tiny operation through the isolated executor;
- test synthetic child hang, parent survival, cleanup, and post-health.

Exit: target host safety boundary passes; unexpected instability ends the arc.

### C5 — progressive candidate correctness

- compile stage-specific direct and LDS artifacts;
- run small-to-exact stages in fresh children;
- full CPU-reference output at every stage; stop on first fault;
- exact correctness once per candidate before timing.

Exit: both exact candidates have identity-joined correctness; no speed claim.

### C6 — authoritative paired evaluator

- canonical executor supplies exactly one synchronized sample/round;
- enforce sessions/clocks/telemetry/counters/noise/confidence/stop rules;
- persist raw and derived evidence; dry-run prints exact GPU work.

Exit: fake-device tests prove pairing, counts, rejection, statistics, stop rules.

### C7 — execute D100

- start from a clean pushed commit;
- run required sessions and only declared additional sessions;
- validate artifact offline;
- commit/push result and ledger separately from code;
- label claim `whole_implementation_comparison`.

Exit: final verdict, fallback, and stop reason exist; no moving the goalpost.

### C8 — conditional R100

Only after `direct_l2_wins`: bind exact route centrally, remain default-off,
run whole-prefill correctness/context timing, prove attribution/no regressions,
test fallback/rollback, and promote only outside noise. Transfer failure means a
local win that did not promote; retain LDS.

### C9 — consolidate

- remove superseded callback shells and dead schema systems;
- move tests into standard tree; convert dated variants to rows;
- preserve result/refutation knowledge; run size/unit/integration/artifact tests.

Exit: one lifecycle entry point, schema family, adapter registry, and decision
authority; no obsolete activation path.

## Mandatory test matrix

- Identity: round-trip; shared pair/distinct candidates; every stale/mismatched
  digest; dirty commit; malformed schema; unknown transport; synthetic reject.
- Compile: direct zero LDS/scratch/spill and final mapping/waits; LDS exact
  windows/stores/loads/barriers; wrong geometry/ABI/target/resources; compile
  path cannot dispatch.
- Safety: no opt-in; runtime in child; success/exception/crash/no-result/timeout;
  group cleanup; health failures; guard corruption; mutation; nonfinite/wrong
  shape/mismatch; cleanup failure; no continuation after revocation.
- Benchmark: warmup completion/exclusion; deterministic randomization; exact
  sample counts; synchronization; clock/counter/output joins; invalid/duplicate/
  missing/accumulated timing; confidence/CV/drift/max-session rules.
- Decision/route: win/retain/inconclusive/blocked/failed/device-loss; negative
  result is valid; win does not auto-route; unsupported fallback; unrelated
  paths unchanged; rollback restores LDS.

## D100 definition-of-done

- [ ] one canonical schema and command;
- [ ] distinct, fully joined direct/LDS identity chains;
- [ ] both exact compile/resource/transport gates pass;
- [ ] all GPU power is inside the isolated guarded executor;
- [ ] truthful dispatch and independent health evidence;
- [ ] guards, immutability, finite/shape/full numerical parity pass;
- [ ] required paired sessions pass, or the maximum-session rule terminates;
- [ ] artifact contains raw evidence, provenance, statistics, claim, verdict,
      fallback, and stop reason;
- [ ] offline verification passes from a clean checkout;
- [ ] deterministic research and shipping decisions exist;
- [ ] LDS remains fallback unless R100 separately passes;
- [ ] result/refutation is committed and pushed;
- [ ] obsolete machinery is removed or has an explicit owner/reason;
- [ ] relevant unit, integration, size, and artifact tests pass.

No percentage based on LOC, phases started, or fake callback tests substitutes
for this checklist.
