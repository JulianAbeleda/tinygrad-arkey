# Generated prefill role certification method

Date: 2026-07-18

Status: working method derived from the corrected-v2 `attn_kv` result. This document defines how to reproduce that
result across roles. It is not a production-promotion claim.

## 1. Objective

Build a model-agnostic machine-search route for prefill when materializing or retaining FP16 weights is not admitted
by the model/GPU memory budget.

The intended route consumes quantized weights directly and must be selected from workload and device facts:

```text
quantization + operation role + M/N/K + physical layouts + target capabilities + memory admission
```

It must not be selected from a model-name branch, a GPU-name branch, or a fixed VRAM threshold.

Qwen3-14B at a 512-token prefill on gfx1100 is the current proving workload. Model-agnostic operation is the target
architecture, not a result already established by this evidence. Portability remains part of the whole-model gate.

The reusable product is the search, certification, and promotion process.

## 2. What the four prefill roles mean

Each row below is a policy/shape bucket over separate matrix-multiplication tensors, not a fused operator:

- `attn_kv` covers each of `attn_k.weight` and `attn_v.weight`;
- `attn_qo` covers each of `attn_q.weight` and `attn_output.weight`;
- `ffn_gate_up` covers each gate and up tensor.

The exact discovered tensor inventory, not merely a profile label such as `Q4_K_M`, determines whether an invocation
uses Q4, Q6, or another admitted quantization.

In this project:

- `M` is the number of input token rows evaluated together.
- `N` is the output width.
- `K` is the reduction or input width.
- Weights use Q4_K storage for the generated route.
- Activations use the Q8_1 representation expected by the five-buffer MMQ kernel.
- One generated PROGRAM covers a `K=256` epoch. A full role accumulates all required epochs.

| Role | Shape `(M,N,K)` | K256 epochs | Function in the transformer | Geometry consequence |
|---|---:|---:|---|---|
| `attn_kv` | `(512,1024,5120)` | 20 | Key/value projections from hidden state into the smaller grouped-KV width | Smallest output grid: `8x4`; the first corrected-v2 certification role |
| `attn_qo` | `(512,5120,5120)` | 20 | Query and attention-output projection class at hidden width | Same reduction depth as `attn_kv`, but a `40x4` output grid |
| `ffn_gate_up` | `(512,17408,5120)` | 20 | Feed-forward gate and up projections from hidden width to intermediate width | Widest output grid: `136x4` |
| `ffn_down` | `(512,5120,17408)` | 68 | Feed-forward down projection from intermediate width back to hidden width | `40x4` output grid and the longest epoch sequence |

These names describe workload roles and their contracts. They do not authorize reuse of a binary merely because two
roles share an output width. Reuse requires exact program-family, ABI, layout, grid, and execution-fixture evidence.

“Corrected-v2” means one distinct static-offset native PROGRAM per K256 ordinal, generated against full-role physical
strides. It is different from the older compact/shared donor PROGRAM plus fixed-VA staging evidence. For Q4 storage,
the K5120 physical row stride is 720 `uint32` words (2,880 bytes); K17408 uses 2,448 words (9,792 bytes).

## 3. The problem, reduced to first principles

Machine search emits an untrusted native program candidate. Successful compilation proves only that bytes were
emitted. It does not prove that:

1. the source recurrence matches the intended quantized operation;
2. the selected schedule fits physical registers without hidden scratch;
3. lowered memory instructions preserve the source buffer and address meaning;
4. the five realized GPU pointers are nonzero, correctly sized, and placed in the declared kernarg slots;
5. producer, initialization, code upload, and target work execute in a safe order;
6. repeated epochs reuse runtimes and buffers safely;
7. the result is numerically correct and the GPU remains healthy;
8. the complete role is faster than the admitted fallback.

The system must therefore certify a chain of contracts:

```text
role math
  -> physical layout
  -> generated schedule
  -> native PROGRAM
  -> realized five-buffer binding
  -> queue and runtime lifecycle
  -> numerical result and GPU health
  -> comparable full-role performance
  -> policy and whole-model promotion
```

A failure is assigned to the first unproven contract. Later gates must not be used to infer an earlier one.

## 4. Thesis derived from `attn_kv`

### Thesis

A machine-search-generated prefill role can be made reproducible when the candidate is frozen and advanced through a
fail-closed ladder that admits one new uncertainty at a time.

The corrected-v2 `attn_kv` result supports this thesis:

- exact role: `(512,1024,5120)`;
- 20 distinct static-offset K256 PROGRAMs;
- immutable family identity
  `5d862e43cbf924f5d8c9e239a4fbb3d0601517436b03707e9b6f3d5ebc10d38b`;
- zero scratch, zero VGPR spills, and zero SGPR spills for every PROGRAM;
- full-family runtime preconstruction without target MMQ dispatch;
- native PM4 prefix-3 pass with runtime-object reuse;
- native PM4 full-20 pass with 20 accepted target submissions;
- zero mismatches across 524,288 compared output values against the same-session retained producer-byte authority
  under combined `rtol=atol=0.003`;
- clean kernel-fault window and healthy pre/post probes.

The durable composition/summary is
[`qwen3-14b-prefill-attn-kv-v2-runtime-preconstruction-closeout-20260718.json`](qwen3-14b-prefill-attn-kv-v2-runtime-preconstruction-closeout-20260718.json).
It hashes and summarizes the raw results, but the underlying bundle and raw artifacts remain under `/tmp`; they are
not durable promotion assets.

This proves corrected-v2 role-level consumer correctness against that declared retained-producer authority and
lifecycle viability. The producer diagnostic has known independent-oracle drift: 205 Q-value, 3,344 raw-scale, and
218 raw-sum mismatches; after the target-half metadata round trip, scale has zero mismatches and sum has one. It
therefore does not prove independent producer correctness or whole-model llama parity. It also does not prove
performance superiority, AQL admission, remaining-role correctness, or production eligibility. Corrected-v2
performance is unmeasured and the default remains `direct_packed`.

## 5. What actually made `attn_kv` work

### 5.1 Register pressure was treated as a lifetime problem

The successful compiler work did not merely increase a limit. It used measured ownership and dependency evidence:

- chain-head A/B loads were ordered behind the preceding release frontier;
- drain ordering prevented later WMMAs from opening lifetimes too early;
- cross-subtile serialization closed overlap between subtiles;
- typed half operations were selected correctly instead of leaking malformed work into regalloc;
- zero-spill emission remained fail-closed.

The general lesson is to repair the producer/consumer lifetime contract at the node that can actually move. An
ordering edge on a value that has already been lowered away is not a resource proof.

### 5.2 The first multi-grid run was allowed to falsify static assumptions

The original Q8 address callbacks assumed a fixed 128-row physical record. That happened to fit the bounded case.
Scaling `M` exposed incorrect reads for later records. The repair derived the stride from the full physical
allocation.

The general lesson is that bounded mathematical correctness does not certify multi-grid physical addressing. Every
scaled axis needs an address-envelope and a real execution gate.

### 5.3 The complete epoch family was frozen

The full reduction was represented as 20 immutable K256 PROGRAMs with static offsets. The loader validated exact
program identity, ABI, grid, binary, and ordinal order before runtime execution.

The general lesson is that compilation and search should finish before the guarded GPU correctness run. Runtime code
mutation or surprise compilation would add another uncontrolled variable.

### 5.4 Runtime construction was separated from target dispatch

The first corrected-v2 PM4 prefix-3 attempt accepted targets 0 and 1, then surfaced the asynchronous prior failure
while constructing the next `AMDProgram`. Ordinal 2 and sequence `[1,2]` passed independently, so there was no
evidence that epoch 2 was intrinsically bad.

The existing tinygrad runtime cache was then used to preconstruct all PROGRAM runtimes in epoch order. A no-target
canary proved construction and code upload without target MMQ dispatch. Prefix 3 and full 20 subsequently passed
while proving exact runtime-object reuse.

The general lesson is that runtime/code-object construction is its own lifecycle phase. It must not accidentally
synchronize or perturb an earlier target while being mistaken for the next target's failure.

### 5.5 Escalation was evidence-driven

The accepted sequence was:

```text
CPU bundle validation
  -> zero-spill native resource audit
  -> no-target runtime-preconstruction canary
  -> short guarded prefix
  -> prefix 3
  -> full 20
```

Each GPU step required structured dispatch evidence, numerical evidence where applicable, a clean kernel-fault
window, and healthy pre/post canaries.

## 6. Reproducibility hypothesis

### Hypothesis

The same process will certify other prefill roles if role geometry and physical layout are treated as explicit inputs
to every gate, while the certification machinery remains shared.
The hypothesis has three parts:

1. **Shared kernel principle:** the Q4_K/Q8_1 tile math, five-buffer ABI, native AMD emitter, frozen-family loader,
   tinygrad PM4/AQL launcher, census, isolation, and numerical comparator can be reused.
2. **Role-specific contract principle:** `M/N/K`, epoch count, grid, Q4 stride, Q8 record extents, output coverage, and
   fixture identity must be regenerated and independently certified per role.
3. **Monotonic escalation principle:** move from the smallest certified geometry to the next geometry while changing
   one dominant stressor at a time.

The natural complexity ladder is:

```text
attn_kv
  -> attn_qo       # same K and epoch count, wider N/grid
  -> ffn_gate_up   # same K and epoch count, widest N/grid
  -> ffn_down      # narrower N than gate/up, but K and epochs grow to 17408/68
```

This is a hypothesis, not an assumption. It is falsified if a role cannot pass the shared gates without an
undeclared role-specific launcher, hidden fallback, relaxed correctness authority, or model/GPU-name branch.

## 7. The certification ladder

### Gate C0: role contract

Required input:

- role and `(M,N,K)`;
- quantization and rounding contract;
- K256 epoch count;
- complete five-buffer shapes, dtypes, byte extents, and physical strides;
- grid and local size;
- target architecture and queue mode.

Exit evidence:

- one serializable contract with no inferred or defaulted role fields;
- discovered workload/device facts separated from candidate choice.

Failure class: specification or inventory.

### Gate C0A: producer and reference semantics

Required input:

- source-pinned Q4_K/Q8_1 quantization and rounding contract;
- bounded deterministic reference vectors;
- an explicit distinction between producer correctness and target-consumer correctness.

Required work:

- prove that the Q8 producer bytes match the declared quantization/rounding contract, or record the exact known drift;
- separately prove that the generated target consumes the actual retained producer bytes correctly;
- never use target agreement with producer bytes to imply that the producer agrees with an independent llama/NumPy
  specification.

Exit evidence:

- producer-versus-spec result;
- target-versus-retained-producer result;
- named authority and tolerance for each comparison.

Failure class: quantization, rounding, producer, oracle, or authority.

### Gate C1: deterministic generation

Required work:

- generate every epoch-offset sink and PROGRAM on the CPU;
- freeze exact source, sink, PROGRAM, and binary identities;
- record the revision/toolchain fingerprint and every declared codegen-affecting configuration/environment input;
- reject duplicate ordinals, program keys, or missing epochs.

Exit evidence:

- one content-addressed role-family bundle;
- exact ordered program-key list;
- reproducible build provenance.

Failure class: search, generator, or provenance.

### Gate C2: native resource certification

Required work:

- assemble/reassemble every native program;
- verify that private/dynamic stack use is disabled and record VGPR, SGPR, LDS, scratch, spill counts, wavefront,
  workgroup, and grid;
- verify exact local/grid geometry and target architecture;
- fail closed on any scratch or spill.

Exit evidence:

- all epoch programs independently pass the resource gate.

Failure class: scheduling, selection, register allocation, or resource admission.

### Gate C3: memory-semantics certification

Required work:

- C3a source/sink layout: exhaustively or symbolically prove address envelopes across the full declared grid and prove
  output read/modify/write coverage, not only input load coverage;
- C3b final native provenance: map every native global-memory instruction back to one of the five ABI base pointers
  and prove its effective offset remains within the declared allocation;

Exit evidence:

- source/sink layout and coverage certificate;
- final-native provenance and bounds certificate;

Failure class: physical layout, lowering, or ABI provenance.

Current tooling is partial. The frozen-v2 loader validates sampled endpoint input loads. Exhaustive output coverage
and final-native effective-address provenance/bounds are strengthened CPU/static gates defined here and still require
implementation. A passing current loader must not be described as a complete C3 pass.

### Gate C4: runtime-preconstruction canary

Required work:

- use tinygrad's existing `get_runtime` path to construct all frozen runtimes in ordinal order;
- dispatch no target MMQ work;
- reject cache drift, binary drift, overlapping/invalid code ranges, unexpected compute dispatch, or dirty timelines;
- attest the actual runtime queue mode from schema-v2 DeviceFacts/device state rather than inferring it from an
  `AMD_AQL` environment request;
- require healthy pre/post probes and a clean fault window.

Exit evidence:

- exact runtime objects exist with the expected cache/program bindings and are eligible for later reuse. Actual reuse
  is cross-checked during C5/C6 execution.

Failure class: runtime construction, code upload, cache, or device lifecycle.

### Gate C5: phase-isolated prefix execution

Required work:

1. realize and synchronize producer/output initialization separately;
2. capture buffer allocations and verify all five realized kernarg qwords are nonzero, correctly ordered,
   allocation-backed, and correctly sized;
3. dispatch exactly one target under census instrumentation;
4. synchronize before removing instrumentation;
5. compare the output and check GPU health;
6. if it passes, repeat for prefix 3.

Exit evidence:

- producer and target phases are independently attributable;
- every accepted target has an exec/submit/return/synchronize record;
- prefix 1 and prefix 3 pass numerically and remain healthy.

Failure class: producer, initialization, target dispatch, repeated dispatch, queue, or delayed synchronization.

This is the strengthened prospective gate. The existing successful `attn_kv` run has target census, numerical, and
health evidence, but did not separately synchronize producer/output initialization under this exact phase boundary.
It is a strong role milestone, not a retrospective C5 phase-attribution certificate.

### Gate C6: complete role correctness

Required work:

- run the exact full epoch sequence using the same frozen family and runtime objects;
- retain the intended persistent/fixed-base buffer strategy;
- prove no recompile, fallback, hidden route, intermediate external accumulation, or missing epoch;
- compare the complete output with the declared authority;
- retain both producer-versus-spec and target-versus-retained-producer results from C0A;
- require finite results, tolerance compliance, clean logs, and healthy post-run canary.

Exit evidence:

- full-role correctness/resource/lifecycle composition artifact.

Failure class: repeated lifecycle, accumulation, full-role numerical correctness, or GPU health.

### Gate C7: memory admission

Required work:

- prove the route does not materialize or retain dense FP16 weights;
- retain exact persistent and peak bytes for weights, Q8 producer buffers, output accumulation, frozen code objects,
  runtimes, queue state, and required temporary storage;
- compare measured or conservatively bounded peak memory with the model/device admission budget;
- reuse the existing prefill memory plan, physical-memory ledger, schedule-memory evidence, and adaptive boundary
  machinery.

Exit evidence:

- a content-addressed memory ledger;
- `dense_fp16_weight_materialization=false`;
- peak route bytes no greater than the admitted budget for the discovered workload/device facts.

Failure class: memory plan, allocation lifetime, residency, or admission.

### Gate C8: performance and policy

Required work:

- time the complete role, never one K256 epoch against a full-K fallback;
- compare generated and fallback routes in matched warmed sessions;
- qualify both PM4 and AQL separately when either can be selected;
- bind the exact logical candidate to the exact executable family and queue-qualified evidence;
- retain explicit fallback and rollback behavior.

Exit evidence:

- a full-role generated winner or an explicit measured fallback decision.

Failure class: performance, policy, or promotion.

### Gate C9: whole-model promotion

Required work:

- execute the intended mixed route in the live model;
- retain route census, memory admission, correctness, health, and decode-regression evidence;
- run matched multi-context llama/tinygrad comparisons;
- use BoltBeam for attribution when a measured gate misses;
- enable autoscan only after the manually selected policy passes.

Exit evidence:

- the statistical promotion gates in
  [`qwen3-14b-generated-prefill-completion-scope-20260714.md`](qwen3-14b-generated-prefill-completion-scope-20260714.md).

Failure class: integration, memory admission, whole-model correctness, decode, or end-to-end performance.

## 8. Current corrected-v2 ledger

This ledger is intentionally narrower than historical role closeouts. Older `attn_qo` and `ffn_down` passes used a
different compact/donor PROGRAM and do not certify the new full-role fixed-stride corrected-v2 families.

| Role | Observed corrected-v2 milestone | Retrospective or open gaps under the strengthened method |
|---|---|---|
| `attn_kv` | Zero-resource family plus native-PM4 prefix-3/full-20 retained-producer correctness and lifecycle result | C0A producer/spec result, C1 provenance/durability, C3 final-native certificate, strengthened C5 phase isolation, internal C4 queue attestation, C7 memory ledger, and C8 performance |
| `attn_qo` | 20-program family generated; loader/resource audit passes with zero spills; first guarded prefix attempt blocked | C0A, C1 provenance/durability, C3, independent C4, and C5 attribution |
| `ffn_gate_up` | No corrected-v2 family certified through this ladder | C1 onward |
| `ffn_down` | No corrected-v2 family certified through this ladder | C1 onward |

The working corrected-v2 `attn_qo` bundle is
`/tmp/qk-attn-qo-v2-stridefix-bundle-20260718` (archive SHA256
`6f465e3f96ce6e439e63b3b3514d65c7cdbd723dc9fe6832291a0e6505a1f881`). Its family identity is
`9d54d197945c64c371af7bb3e86a3f46a4e312ed68520703239a4b2a4739fbc5`. It has 20 distinct programs and all
20 report zero scratch and zero VGPR/SGPR spills. Its first guarded PM4 prefix-1 attempt
(SHA256 `533acef6842bb73dd0d1460ee2686f73d48318ad74e7949542a8b795c6640522`) was blocked. The kernel log
recorded SQ type-2 compute-wave errors, gfxhub page faults, MES removal failure, and reset; tinygrad separately
reported `MMU fault: 0x0 NotPresent=1`. The child returned no structured result, and the error surfaced in allocator
finalization/teardown after asynchronous realization, so no target attribution or pointer payload survived.

A read-only, non-durable static review found:

- complete in-bounds output coverage;
- in-bounds Q4/Q8/metadata source address envelopes;
- five final scalar pointer loads from the declared 40-byte kernarg ABI;
- no literal null pointer or final global address operand using `v0`;
- the same legal native resource envelope as passing artifacts.

Those facts are not a C3 certificate and do not prove realized pointer payloads or target attribution.
Producer/output initialization and the frozen target remained asynchronous in the failed run.

## 9. Immediate `attn_qo` application

The next attempt must restore the ladder rather than guess at a kernel change.

### CPU/mock-only preparation

1. Preserve typed PM4/AQL dispatch evidence when delayed synchronization raises.
2. Synchronize while dispatch hooks and state are still installed.
3. Record target exec, submit-began, submit-returned, and synchronize boundaries.
4. Retain all five kernarg qwords and allocation extents.
5. Add exhaustive output and final-memory-provenance validation where current validators stop at the source sink.
6. Mock-test every failure path without using the GPU.

Commit `3342860ee [runtime] preserve typed evidence from isolated failures` begins item 1 by teaching generic
`tinygrad.runtime.process_isolated.run_isolated` to serialize typed exception evidence. The frozen-prefix wrapper
still uses its manual subprocess path and is not yet integrated with this mechanism. The commit does not by itself
prove phase isolation or repair the GPU fault.

### One guarded discriminator

After CPU/mock validation:

```text
producer/output initialization
  -> synchronize and health check
  -> capture realized buffers
  -> one target dispatch under census
  -> synchronize before removing census
  -> numerical comparison and health check
```

Interpretation:

| Observation | Owning category | Next action |
|---|---|---|
| Producer phase faults | Producer/initialization lifecycle | Minimize and repair the scheduler-owned operation |
| Any pointer is zero, misordered, or out of allocation | Allocation/binding | Repair five-buffer realization or kernarg construction |
| Pointers are valid and target faults | Generated memory semantics | Compare final address provenance; isolate wide-grid/stride-720 tiles |
| Prefix 1 passes but prefix 3 fails | Repeated lifecycle | Audit target completion, runtime reuse, buffer mutation, and queue ordering |
| Prefix 3 passes | Full-role escalation | Run all 20 with the same family and evidence contract |

No prefix-3 or full-20 run is authorized by a failed prefix-1 result. A recovered reset proves recovery, not safety.

## 10. Scaling rules

1. **One family per exact contract.** Never treat a logical role identity, donor fixture, or shared output width as an
   executable-family identity.
2. **One dominant stressor per step.** Use `attn_qo` to test wider N at the same K/epoch count, then
   `ffn_gate_up`, then the 68-epoch `ffn_down`.
3. **CPU before GPU.** Generation, resource checks, address envelopes, ABI checks, and mock failure retention precede
   any dispatch.
4. **Smallest admissible GPU discriminator.** Start with no-target or prefix 1 according to the open contract, then
   `1 -> 3 -> full`.
5. **Evidence survives failure.** A child crash, timeout, or delayed synchronize must retain the last accepted
   pointers and submission boundary.
6. **Queue modes are separate facts.** PM4 success does not imply AQL success. Policy admission must use the actual
   queue mode.
7. **Correctness before timing.** A fast epoch or partial grid is not a full-role performance result.
8. **Fallback remains explicit.** Failure cannot silently route to direct packed and still count as candidate proof.
9. **No new launcher without a demonstrated missing primitive.** The existing tinygrad PROGRAM emitter,
   `AMDProgram`, `AMDComputeQueue`, scheduler, isolation, census, and comparator are the owned path.
10. **No production claim from role proof.** C6 is a role-level milestone; only C8/C9 can change policy or production.

## 11. Required artifact set per role

A role is reproducibly certified only when its evidence set contains:

- role contract and physical layout;
- source revision and codegen-affecting environment;
- ordered family/program/sink/binary identities;
- native resource report for every epoch;
- source and final memory-semantics report;
- runtime-preconstruction census;
- producer and target phase census;
- prefix 1, prefix 3, and full-role results;
- producer-versus-spec and target-versus-producer authorities and tolerances;
- pre/post health and kernel-fault windows;
- no-dense-FP16 and peak-memory admission ledger;
- complete-role timing against the exact fallback;
- queue-qualified policy decision.

Temporary `/tmp` bundles are useful working material but are not durable promotion assets. A candidate cannot be
promoted until its executable family and all required evidence are content-addressed and retained.

## 12. First-principles acceleration plan

The fastest path is not four independent investigations. It is one shared certification system followed by four
role deltas.

### 12.1 Factor the work into invariants and deltas

The shared invariants are:

- Q4_K/Q8_1 recurrence and rounding;
- `128x128x256` tile math;
- five-buffer ABI and kernarg order;
- AMD PROGRAM emission and resource accounting;
- frozen-family loading and identity;
- tinygrad PM4/AQL launch path;
- runtime preconstruction;
- failure isolation, dispatch census, comparison, and health checks.

Each role then introduces one dominant new stressor:

| Role | What prior evidence already supplies | New stressor to prove |
|---|---|---|
| `attn_kv` | Base tile, 20-epoch family, full-role PM4 correctness | Durable evidence composition and comparable performance |
| `attn_qo` | Same M, K, epoch count, stride-720 layout, and tile math as the base | Wide grid: `gidx0=8..39` combined with full-role fixed-stride Q4 addressing |
| `ffn_gate_up` | Once `attn_qo` passes, the same 20-epoch and wide-grid mechanism | Maximum N/grid width (`136`) and largest output/weight footprint |
| `ffn_down` | Once `attn_qo` passes, the `40x4` output geometry is already certified | K growth from 5120 to 17408 and repeated lifecycle growth from 20 to 68 epochs |

This factorization prevents solved questions from being reopened while still requiring every role to pass its own
numerical and health gates.

### 12.2 Build shared leverage once

Complete these shared tasks before spending another GPU reset:

1. **Failure evidence retention:** carry PM4/AQL census, target boundary, and kernarg qwords through delayed
   synchronization failure.
2. **Phase isolation:** realize and synchronize producer/output initialization before installing the target-only
   execution boundary.
3. **Final memory certificate:** connect each native global-memory operation to its ABI base and allocation bounds,
   including exhaustive output coverage.
4. **Bundle provenance:** record source revision, codegen environment, role contract, ordered program identities, and
   archive hash in the generated family.
5. **One result schema:** emit the same gate/status/failure-category structure for every role and queue mode.

Items 1-4 are reusable infrastructure or validation, not role-specific launch code. The existing tinygrad emitter,
runtime, queue, and harness remain the execution path.

### 12.3 Use two execution lanes

CPU-only work and GPU work have different safety constraints:

```text
parallel CPU lane
  -> family generation
  -> loader/resource audit
  -> address/provenance audit
  -> mock failure-path tests

single GPU lane
  -> health check
  -> no-target canary when required
  -> producer-only phase
  -> target prefix 1
  -> target prefix 3
  -> full role
  -> health check
```

CPU generation and validation for the next role can proceed while the current role's fault is being diagnosed. GPU
runs remain sequential, and a failed health gate stops the lane.

### 12.4 Treat `attn_kv` as the golden control

Do not repeatedly rebuild or exhaustively rerun it after every harness-only change. Use the frozen corrected-v2
family as a positive control:

1. run focused CPU/mock tests for the changed evidence path;
2. run the smallest `attn_kv` prefix that exercises the changed runtime boundary;
3. require the same program identity, pointer order, target census, comparison, and health result;
4. run full 20 again only for a release/evidence checkpoint or a change that affects execution semantics.

In parallel, make its existing family and evidence durable, close the strengthened producer/spec, phase, and memory
gaps, and run a comparable complete-role timing gate. If it loses, record a C8 fallback decision without weakening
its C6 correctness result.

### 12.5 Resolve `attn_qo` by classification

`attn_qo` is the current frontier because it changes grid width while keeping K and the epoch count fixed.

Ordered solution:

1. complete the shared failure-retention and phase-isolation mock tests;
2. complete the final-ISA memory certificate and bundle provenance;
3. run producer/output initialization alone and synchronize;
4. if healthy, run one exact full-grid target with all five pointers captured;
5. if the target faults with valid pointers, isolate the first wide-grid boundary:
   - compare the already-certified `gidx0=0..7` region with the first newly exercised tile;
   - use a research-only bounded-grid diagnostic only if the existing manifest/launcher can express it explicitly;
   - do not relabel that bounded result as exact-role evidence;
6. repair only the classified owner: producer, binding, final addressing, or repeated lifecycle;
7. repeat `prefix 1 -> prefix 3 -> full 20`;
8. run comparable complete-role timing and select generated or fallback.

Exit artifact:

- exact corrected-v2 `attn_qo` family;
- C0-C8 evidence or an explicit failed gate and fallback decision.

### 12.6 Certify `ffn_gate_up` as N-scaling

Start its CPU family generation and static audits after the shared provenance format is fixed; this can overlap with
the `attn_qo` investigation.

Ordered solution:

1. generate all 20 exact static-offset programs for `(512,17408,5120)`;
2. require zero spills/scratch and exhaustive `136x4` memory bounds;
3. run the no-target runtime-preconstruction canary;
4. reuse the phase-isolated producer and target path proven by `attn_qo`;
5. escalate `1 -> 3 -> 20`;
6. compare the complete 8,912,896-value output under the declared authority;
7. measure the complete 20-epoch role, including required synchronization and preparation.

If `attn_qo` passes but `ffn_gate_up` fails, the initial search space is intentionally narrow: maximum grid index,
allocation extent, code/runtime footprint, or workload duration. The tile recurrence and 20-epoch lifecycle remain
positive controls.

Exit artifact:

- exact corrected-v2 `ffn_gate_up` C0-C8 result and route decision.

### 12.7 Certify `ffn_down` as epoch-scaling

Do not treat the historical shared-N5120 donor as corrected-v2 proof. Generate the exact 68-program role family.

Ordered solution:

1. generate all 68 static-offset programs for `(512,5120,17408)`;
2. verify every epoch's Q4/Q8/metadata envelope and binary resource report;
3. preconstruct all 68 runtimes without target MMQ dispatch;
4. use the already-certified `40x4` `attn_qo` geometry as the output-grid control;
5. escalate `1 -> 3`, then add bounded lifecycle checkpoints before the full 68 if evidence shows a transition
   boundary;
6. retain one persistent accumulation contract and prove all 68 accepted targets;
7. compare the complete output and run the complete-role timing gate.

If early prefixes pass and a later prefix fails, classify by the first failed ordinal and retained submit/synchronize
boundary. Do not assume the ordinal's program is bad until it passes or fails independently with the same binding.

Exit artifact:

- exact corrected-v2 `ffn_down` C0-C8 result and route decision.

### 12.8 Work ordering and concurrency

The critical path is:

```text
shared observability/provenance
  -> attn_qo classified prefix 1
  -> attn_qo full 20
  -> ffn_gate_up full 20
  -> ffn_down full 68
  -> queue-qualified candidate decisions
  -> policy and whole-model gates
```

Safe parallel work:

- one owner completes shared target/producer phase isolation;
- one owner completes final memory-provenance validation;
- one owner generates and statically audits the next exact role family;
- the coordinating owner reviews, integrates, and exclusively advances the GPU lane.

Do not run concurrent GPU benchmarks or probes. Do not generate all expensive families before the shared provenance
format is stable, because that would produce unattested bundles that require rework.

### 12.9 Decision rule for every candidate

Every role ends in one of three states:

```text
CERTIFIED_WIN
  C0-C8 pass and generated complete-role performance wins

CERTIFIED_FALLBACK
  C0-C7 pass, but complete-role performance loses at C8

BLOCKED_AT_Cn
  first failed contract and retained evidence are explicit
```

Only `CERTIFIED_WIN` can enter a performance-qualified candidate policy. `CERTIFIED_FALLBACK` is still a successful
correctness/certification result and supplies training evidence to the next search. `BLOCKED_AT_Cn` returns directly
to the owning layer rather than restarting the entire pipeline.

## 13. Success and falsification

The process succeeds when a new role can move through C0-C8 by supplying role facts, without adding a model-name
condition, alternate launcher, hidden fallback, or weakened oracle.

The thesis is weakened or falsified when:

- the same input contract does not reproduce the same family identity;
- final native memory provenance cannot be certified from the generated program;
- healthy execution requires untracked process resets or fresh-process aggregation outside the declared route;
- a role needs hand-coded dispatch behavior that cannot be expressed as workload/device facts.

Complete-role timing that loses to the admitted fallback does not falsify the certification thesis. It rejects that
candidate at C8 and feeds a measured objective back into search.
