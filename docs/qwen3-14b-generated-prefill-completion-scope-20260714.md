# Qwen3-14B generated-prefill completion scope

Date: 2026-07-14
Last reconciled: 2026-07-18

## Objective

Reach correctness-qualified Qwen3-14B pp512 parity or better against the frozen llama.cpp comparator using generated
tinygrad kernels, while leaving behind a model-agnostic system rather than a 14B route fork.

The model is the proving workload. It is not compiler policy. Runtime and compiler decisions must be functions of:

```text
quant format + operation role + M/N/K + layouts + target capability + memory budget + correctness class
```

Model profile names may select input facts and benchmark fixtures. They may not select compiler lowering, transport,
schedule, or promotion state.

## 2026-07-19 `ffn_gate_up` performance-comparability correction

This correction supersedes every `ffn_gate_up` `34.694x` win, `2.895x` loss, “full-K direct-packed
`9.351988 ms`,” and Phase-3-complete claim below. Source inspection proves the `9.351988 ms` direct observation used
the same bounded `128x128x256` logical workload as the `0.277502 ms` generated observation, but the timed regions
were different: the generated side timed a resident compiled launch, while direct packed included tensor
construction/conversion, realization, and NumPy readback. The complete-role generated observations
(`42.882262 ms` synchronized AQL, `27.076011 ms` queued AQL, and `41.861031 ms` PM4) have no matched complete-role
direct-packed observation.

The current authority is therefore: correctness/resource evidence passes; performance is
`UNKNOWN_NOT_EVALUATED`; R5 has no comparable emitted win; R6 is not ready; Phase 3 remains incomplete until a
collector measures candidate and direct packed at the exact complete role with identical workload and preparation,
allocation, readback, and synchronization descriptors.

## Current execution authority: phases 3 through 7

This section is the current progress ledger and ordering authority for the work before autoscan. It supersedes stale
progress percentages or checklist state elsewhere in this document; the detailed phase requirements below remain the
normative implementation gates.

The automatic device/model/VRAM scan, memory admission, fitting-model overlay selection, and safe non-fitting
direct-packed fallback are already connected. They are feasibility planning, not the performance autoscan. The next
objective is to produce and manually prove one complete faster non-fitting policy before any machine-selected
promotion is enabled.

| Phase | Current state | Remaining gate |
|---|---|---|
| 3 — Q4_K role completion | Incomplete for `ffn_gate_up` performance. Full-role correctness/resource/health evidence exists, but the retained bounded timing uses mismatched measurement boundaries and the exact generated full-role observations have no matched direct-packed comparator. Performance is `UNKNOWN_NOT_EVALUATED`. | Collect candidate and direct-packed full-role timing under identical workload and measurement-definition descriptors; only then decide win/loss. |
| 4 — Q6_K role completion | Complete for the declared fallback strategy. Both exact direct-packed Q6 rows (`attn_kv` and `ffn_down`) are qualified and timed under the retained policy contract. | No generated Q6 route is required unless measured whole-policy attribution shows the qualified fallbacks miss the target. |
| 5 — Central candidate policy | The default-off policy machinery is implemented, but the retained immutable six-row artifact is integration-only and is not a performance-qualified policy. It binds the now-rejected `ffn_gate_up` candidate so the live seam and census can be exercised; `production_promotion=false`. | Build a new immutable policy only after at least one comparable full-role generated candidate wins. |
| 6 — Mixed-route 14B integration | Not complete. The live research route remains explicit/default-off; no production route changed. Its current policy is an integration diagnostic, not a promotion candidate. | Diagnose the live-model route and complete correctness/census/memory/health/decode/rollback proof, but do not advance it to Phase 7 unless the bound candidate is also performance-qualified. |
| 7 — Parity and promotion | Not complete | The same immutable revision must pass the matched multi-context llama comparison and the statistical promotion gates below. |

The dependency order is strict:

```text
Phase 3 Q4 measurement closeout
  -> comparable full-role generated performance winner
  -> Phase 4 fallback/strategy qualification
  -> Phase 5 performance-qualified six-row policy
  -> Phase 6 manual end-to-end proof
  -> Phase 7 parity/beyond-parity proof
  -> performance autoscan
  -> post-proof pruning and restoration of the 30,000-line authored budget
```

Phase 6 may begin incrementally while individual Phase 3 rows are being validated, but it cannot close until every
selected invocation has one exact binding or declared fallback. Phase 7 timing cannot authorize a route unless Phase
6 proves that the intended binaries actually executed and that correctness, memory, resource, health, and decode gates
passed on the same candidate revision.

The Phase 7 authority matrix is whole-prefill synchronized tokens per second at prompt/context lengths 512, 1,024,
2,048, and 4,096. Llama and tinygrad run sequentially in at least three alternating, pinned-clock sessions with raw
samples retained. No accepted context may fall below 98% of llama, every declared context median must exceed llama,
the paired aggregate 95% lower confidence bound must exceed 1.00, and the aggregate geometric-mean target is at least
105%. Kernel TFLOP/s, role timing, roofline data, and Boltbeam traces are attribution evidence, not promotion authority.

After each matched llama/tinygrad end-to-end comparison, any missed context or aggregate target triggers a Boltbeam
timing pass on the exact tinygrad candidate revision and workload. The trace must reconcile the observed whole-prefill
wall into candidate roles, activation preparation/dequantization, attention, launch/synchronization, and residual work,
then compare that decomposition with the matched llama timing artifact. Optimization resumes from the largest measured
gap; a projected role speedup or stale trace cannot replace a fresh end-to-end rerun.

Autoscan remains a subsequent phase. It may enumerate, gate, measure, cache, and bind policies only after the complete
manual policy passes Phase 7. Until then, the ordinary non-fitting automatic route remains direct packed.

## Definition of 100%

Completion requires all of the following:

1. The exact six dominant packed 14B prefill role/quant rows are discovered from the loaded GGUF/trace, not maintained
   as compiler branches.
2. Every row binds an identity-qualified generated candidate through the same adapter, admission, compiler, executor,
   and evidence path used by other models.
3. Every selected binary passes full-output or an explicitly equivalent exhaustive correctness authority, immutable
   input checks, output guards, timeout isolation, and a post-dispatch GPU health check.
4. Evidence proves the executed binary, packed ABI, launch geometry, WMMA family, resource allocation, and candidate
   identity. A route name or source-level intention is not evidence of execution.
5. Q4_K reaches at least 60 aggregate logical TFLOP/s and Q6_K reaches at least 69 aggregate logical TFLOP/s, or a
   newer measured role budget proves an equivalent whole-model result.
6. Three alternating sequential pp512 sessions produce a candidate median at least 98% of the frozen llama median,
   with no credible correctness or stability regression. The beyond-parity target is at least 105% of llama; the
   project target is at least 2,000 tok/s.
7. The route census proves that every intended packed role fired and that no handwritten/oracle, scalar fallback, or
   unmeasured route silently supplied the result.
8. Decode remains correct and does not regress outside the declared tolerance.
9. The same machinery admits and tests 8B plus at least one non-default shape without adding a model-name condition.
10. Rollback is one explicit candidate-policy change and retains the last correctness-qualified default.

Anything less is a diagnostic or candidate result, not a shipped 14B completion.

## Current position

### 2026-07-18 Phase 3 all-Q4 measurement closeout

All four generated Q4 roles are correctness/resource/health measured, but `ffn_gate_up` performance is not
evaluated. Its emitted `0.277502 ms` and direct-packed `9.351988 ms` observations both cover bounded
`128x128x256`, but their preparation/allocation/readback/synchronization scopes differ. The exact generated role
requires 20 epochs. It passes all 8,912,896 values with zero mismatches and maximum absolute error `0.00341796875`,
and its retained raw observations are
`42.8822620306164 ms` with synchronized AQL dispatch, `27.07601070869714 ms` with queued AQL dispatch plus final
synchronization, and `41.86103114625439 ms` with PM4. No matched exact-role direct-packed observation exists, so
those samples establish neither a win nor a loss.

Therefore:

```text
phase_3_measurements_complete=false
selected_generated_roles=
performance_not_evaluated_generated_roles=ffn_gate_up
six_row_policy_scope=integration_only
performance_qualified_policy=false
production_promotion=false
```

The immutable six-row research artifact is retained unchanged for integration diagnostics and identity plumbing. Its
candidate binding must not be described as a performance selection.

The `attn_kv` Q4 row is resolved as a measured fallback; it is not a generated-candidate promotion.
`docs/qwen3-14b-prefill-attn-kv-role-closeout-20260718.json` composes the exact five retained raw artifacts:

- Commit `d3c605890` stages Q4, Q8 values, scales, and sums through fixed VAs. The resulting PM4 prefix-3 and full-20
  runs pass.
- The pre-fix AQL prefix-3 failure is retained as evidence. Its log proves an SQ memory violation plus gfxhub
  page-fault/MES-removal/reset sequence; it does not prove an instruction-page or instruction-fetch fault.
- Commit `6216e9e4e` publishes AQL packet headers last. Post-fix AQL prefix-3 and full-20 runs pass.
- Both full-20 modes use the exact frozen PROGRAM, 20 in-place accumulation launches, fixed-VA GPU-SDMA staging for
  all inputs, no intermediate readback/external add/recompile/fallback, `VGPR=256`, `LDS=57,856 B`, and
  `scratch=0`. Each checks all 524,288 final values with zero mismatches, maximum absolute error
  `0.002685546875`, clean fault windows, and healthy pre/post canaries.

The performance result is deliberately weaker than the correctness result. Each full-20 timing is one cold isolated
sample: AQL `19.601495820097625 ms`, PM4 `74.58446017699316 ms`. Neither beats the retained direct-packed
`attn_kv` baseline of `7.89 ms`. These are not matched, warmed, repeated, or statistical timing claims. They are
sufficient only for the fail-closed decision to reject this generated candidate and retain the existing immutable
policy row:

```text
attn_kv_q4_generated_candidate=REJECTED
attn_kv_q4_selected_route=direct_packed
attn_kv_q4_policy_row_changed=false
production_promotion=false
```

The shared N5120 PROGRAM then closes the other two Q4 roles:

- `attn_qo` PM4 prefixes 1 and 3 and the full 20 epochs pass. The full run checks 2,621,440 values with zero
  mismatches, maximum absolute error `0.00341796875`, clean kernel logs, healthy pre/post canaries, and the same
  `VGPR=256`, `LDS=57,856 B`, zero-scratch resources. Its only generated full-role timing is one cold isolated sample
  at `76.49636000860482 ms`, slower than the retained `11.41 ms` direct-packed baseline.
- Q4 `ffn_down` reuses the exact N5120 PROGRAM binary while retaining the `attn_qo` donor fixture separately and
  validating a distinct 68-epoch execution fixture. PM4 prefixes 1 and 3 and the full 68 epochs pass. The full run
  checks 2,621,440 values with zero mismatches; maximum absolute error `0.01123046875` is within the declared
  tolerance, kernel logs are clean, pre/post canaries pass, and resources remain unchanged. Its only generated
  full-role timing is one cold isolated sample at `114.26727805519477 ms`, slower than the retained `11.76 ms`
  direct-packed baseline.

These samples are rejection evidence, not matched/statistical performance authority. The immutable policy already
selects direct packed for both roles, so it does not change. The compact composition
`docs/qwen3-14b-prefill-q4-role-closeout-20260718.json` records all four Q4 decisions and exact evidence hashes.

The next owning performance work is a generated candidate that wins under comparable full-role timing. Phase 6 may
continue in parallel as an integration diagnostic, but the retained route cannot become a promotion candidate.

### Historical 2026-07-18 reconciled research-policy and `attn_kv` checkpoint

This checkpoint predates the fixed-all-input-VA PM4 passes, AQL header-last publication fix, full-20 AQL pass, and
measured fallback decision above.

Phase 4 and the research implementation portion of Phase 5 are now complete:

- `docs/qwen3-14b-prefill-q6-attn-kv-qualification-20260718.json` and
  `docs/qwen3-14b-prefill-q6-ffn-down-qualification-20260718.json` qualify the two declared direct-packed Q6 fallback
  rows.
- `docs/qwen3-14b-prefill-six-row-research-policy-20260718.json` is the immutable six-row, research-only policy.
  Its exact selector, runtime attachment, candidate/fallback dispatch, and actual execution-census plumbing are
  implemented and default-off. Unknown workloads, missing authority, identity drift, and candidate failure do not
  silently fall back.
- Exact frozen artifacts now exist for `attn_kv` and the shared `(512,5120,256)` PROGRAM geometry. The frozen runtime
  binding validates PROGRAM ABI/grid/key/source/binary identity and keeps donor artifact fixtures distinct from
  execution-role fixtures.

At this checkpoint Phase 3 remained open. `docs/qwen3-14b-prefill-attn-kv-fresh-epoch-isolation-20260718.json` retains the compact
fresh-process result: PM4 epochs 0, 1, and 2 each pass one isolated target dispatch with zero mismatches, clean fault
windows, and healthy pre/post canaries. This rules out a deterministic bad epoch or bad epoch offset for those three
epochs. It does not prove repeated dispatch. In
`docs/qwen3-14b-prefill-attn-kv-pm4-aql-differential-20260718.json`, both PM4 and AQL pass prefix 1, then both complete
two epochs and fault entering the third same-process dispatch with SQC/memory-violation, page-fault, MES-removal, and
reset evidence. The shared failure across launch modes is the immediate Q4 qualification blocker.

The current route boundary is:

```text
immutable_six_row_policy=true
exact_research_route_binding_implemented=true
actual_execution_census_implemented=true
exact_research_route_default_off=true
whole_model_live_census_performed=false
production_promotion=false
production_dispatch_changed=false
default_route=direct_packed
```

There is no project progress percentage. Phase gates are the authority.

### Historical 2026-07-18 one-role evidence checkpoint

This checkpoint predates the completed Q6 fallback rows, immutable policy, live research binding/census implementation,
and the retained `attn_kv` frozen/fresh-process evidence above.

Project Phase 3 had advanced, but remained incomplete at this historical checkpoint. The exact Q4_K/Q8_1
`ffn_gate_up` `512x17408x5120` role appeared evidence-ready to implement a research opt-in:

- The retained R5 artifact, `docs/qwen3-14b-prefill-r5-geometry-20260718.json`, records a zero-mismatch emitted
  full-grid result and three same-session timing rounds. Its candidate median/min
  `0.277502/0.269477 ms` and direct-packed `9.351988/9.349303 ms` both cover bounded `128x128x256`, but use
  different measurement boundaries. The historical `34.694x` label is invalid. Exact source/binary identity,
  native `VGPR=256`, `LDS=57,856 B`, `scratch=0`, timing samples, and no-fallback evidence are still useful.
- The strict frozen-PM4 full-role artifact covers all 20 K epochs in one process with in-kernel FP32 accumulation,
  stable fixed-VA GPU-SDMA metadata, no intermediate readback/external add/recompile/fallback, and zero mismatches
  across 8,912,896 outputs. Later exact full-role generated observations (`41.861031 ms` PM4, `42.882262 ms`
  synchronized AQL, and `27.076011 ms` queued AQL) have no matched exact-role direct-packed comparator.
- The independent fresh-process all-epoch artifact checks 178,257,920 epoch outputs with zero mismatches and clean
  per-epoch health.
- `docs/qwen3-14b-prefill-mmq-one-role-evidence-20260718.json` composes those proofs. Its corrected verdict is
  `BLOCKED_UNTIL_COOPERATIVE_TILE_WIN`; machine-search R6 reports
  `BLOCKED_NO_COMPARABLE_FULL_ROLE_WIN`. Machine-search R6 is an evidence gate and machine-search
  R7 is source-component reduction; neither is project Phase 7.

The current route boundary is:

```text
research_opt_in_implementation_eligible=true
one_role_opt_in_eligible=false
route_binding_implemented=false
live_route_census_performed=false
promotion_eligible=false
production_dispatch_changed=false
default_route=direct_packed
```

The frozen artifact work reused the existing tinygrad five-buffer emitter/harness and PM4/AQL launch paths. It did
not build or require a HIP launcher.

### Completed substrate

- Q4_K and Q6_K share one `PackedWeightTransform` and `dequant_tile` interface.
- Packed slot-2 weights can be decoded inside a generated fp16 LDS-to-WMMA program.
- The generated packed primitive has passed full-output real-GPU correctness at `(512,4096,4096)` for both formats.
- Guarded spawn isolation, timeout handling, immutable inputs, output guards, health checks, compile identity, final ISA,
  resource evidence, and packed ABI gates are present.
- Candidate construction, capability selection, admission, and canonical identity are centralized in `runtime_specs`.
- The current prefill adapter now consumes any exact capability-supported workload; it contains no model or exact-shape
  selector.
- Canary artifact generation is shape-driven. A profile/role request can reuse an existing exact candidate or rebind a
  same-role schedule template, after which ordinary admission must prove legality.
- The 8B and 14B profile role matrices, for Q4_K and Q6_K, pass the same CPU admission boundary.
- 14B-named policy entry points are compatibility wrappers over generic profile/quant gates.

### Not completed

- Three required Q4 roles remain unqualified: `attn_qo`, Q4 `ffn_down`, and `attn_kv`.
- The repeated-dispatch `attn_kv` fault blocks its full-role qualification even though isolated epochs 0, 1, and 2
  pass individually.
- The immutable policy and default-off live research binding/census implementation exist, but their Phase 6
  whole-model execution and genuine live census have not passed.
- The profile's family quant label does not encode the exact mixed Q4_K/Q6_K tensor inventory. Exact quant must come
  from loaded tensor facts.
- Whole-model mixed-route correctness, memory reconciliation, route census, health, decode regression, and matched
  multi-context performance have not run for the candidate policy.
- The current 14B default remains the safe direct-packed baseline, not the new primitive.

Do not replace these gates with a progress percentage. The phase ledger above is the authority.

### 2026-07-14 execution update

The exact six-row generated packed candidate now has full-output correctness and three isolated timing sessions. The
latest candidate is stable and spill-free, but its final code object remains in the 248 allocated-VGPR bucket and is
well below the promotion budget: Q4 rows measure approximately 9.5-26.6 TFLOP/s and Q6 rows approximately 9.9-24.3
TFLOP/s. The authoritative artifacts are the six-row correctness artifact and timing sessions recorded in
`bench/prefill-pure-full-kernel/`; this is a correctness-complete candidate, not a 14B promotion.

Two alternative strategies were also checked on the smallest full Q6/Q4 role (`attn_kv`, M=512, N=1024, K=5120):

- Q6 dequant-once-to-fp16 followed by generated fp16 WMMA is numerically exact, but the fresh dequant-plus-GEMM
  median was about 4.7 ms versus about 0.54 ms for the fused candidate. It is therefore not a viable fast route in
  its current implementation; keep it as an explicit diagnostic/rollback candidate.
- Q4 Tensor-level Q8_1 packing plus generated integer-WMMA is numerically correct (max absolute error about 0.016),
  but the fresh packing-plus-contraction median was about 232 ms on the same role. The graph-level packing lifecycle
  is refuted for promotion even though the integer-WMMA algebra is sound.

The next owning layer is therefore the fused MMQ-style tile producer: stage/reuse Q4_K and Q8_1 tile data inside one
bounded kernel lifecycle, then reconnect the existing correctness/admission/evidence path. The current bounded
cooperative MMQ atoms remain proof substrates only; they are not silently selected by the production route.

### Historical 2026-07-16 source-pinned llama-oracle checkpoint (superseded)

The spill/emission blocker in this historical checkpoint is closed by the 2026-07-18 evidence above.

The speculative cooperative descriptor has been removed. The current Phase 3 implementation authority is the
source-pinned llama MMQ structure adapted to the ordinary five-buffer ABI: a `128x128x256` tile, eight waves, 57,856
bytes of LDS, persistent decoded Q4 data, two K128 Q8 phases, and exact half2 metadata recurrence. The split producer,
bounded graph, full-grid ownership/writeback seam, and structural correctness tests are connected without a model-name
or VRAM branch.

The legacy five-buffer route remains a correctness reference rather than a performance candidate. Fresh exact timing
on Q4 `attn_kv` attributes approximately 0.088 ms to Q8 production and 5.916 ms of a 6.007 ms total to MMQ, so the
measured tax is inside the contraction kernel. All four Q4 roles are output-correct and report zero final scratch/spill,
but remain far below the role budgets.

At this historical checkpoint, the source-pinned generated oracle did not yet emit a spill-free final binary. Generic late-scheduler and liveness
repairs have localized constrained loads, prioritized immediate release, and removed false `END` body liveness. At the
current revision the first request is the base carrier of an A `DS_LOAD_B128` fragment constrained to `v200..v203`; its
companion B fragment is constrained to `v204..v207`, and both feed the same `V_WMMA_I8` consumer. Seven other A/B pairs
using those exact leases are already live across the same boundary. The next repair is therefore to schedule each
complete A/B load pair immediately before its matching WMMA, after the preceding pair has been consumed.

The reported 255 candidate-slot count is the union of registers that live virtuals are allowed to occupy, not an
additive hardware-VGPR requirement. Ordinary one-register values advertise most of the pool. The actionable evidence
is the assigned constrained-run overlap: multiple four-register A/B leases are open simultaneously even though the
machine requires only one A pair, one B pair, and the in-place eight-register C/D fragment at each WMMA. Replaying LDS
loads in regalloc or permitting scratch remains unsafe. Metadata-carrier removal, atomic eight-register WMMA result
spans, and a vec8 recurrence bundle were tested in isolation and rejected or reverted because none proved lower
physical pressure without regressions.

The next acceptance gate is unchanged: the complete source-pinned bounded oracle must emit with zero scratch/spills,
then the full-grid kernel must pass correctness and whole-primitive timing. Small proxy kernels or a later first spill
are diagnostic progress only and cannot close Phase 3.

## Frozen measurement authority

The current comparator pair is historical but reproducible and must remain immutable until a controlled refresh:

| implementation | commit | clean pp512 wall | prefill |
|---|---|---:|---:|
| llama.cpp | `ac4cddeb0` | 271.230 ms | 1,889.41 tok/s |
| Arkey generated scalar baseline | `05b67146a` | 1,397.840 ms | 366.28 tok/s |

The six measured packed rows account for 96.4% of the Arkey profile and about 98.5% of the absolute gap:

| role | quant | M | N | K | Arkey rate | llama practical rate |
|---|---|---:|---:|---:|---:|---:|
| `ffn_gate_up` | Q4_K | 512 | 17408 | 5120 | 12.37 | 59.90 TFLOP/s |
| `attn_qo` | Q4_K | 512 | 5120 | 5120 | 11.41 | 50.05 TFLOP/s |
| `ffn_down` | Q4_K | 512 | 5120 | 17408 | 11.76 | 51.96 TFLOP/s |
| `attn_kv` | Q4_K | 512 | 1024 | 5120 | 7.89 | 32.38 TFLOP/s |
| `ffn_down` | Q6_K | 512 | 5120 | 17408 | 5.13 | 70.21 TFLOP/s |
| `attn_kv` | Q6_K | 512 | 1024 | 5120 | 5.93 | 53.75 TFLOP/s |

These are optimization budgets, not hardcoded route rules. Refresh them only with the same model, workload, clock
policy, correctness policy, and sequential-run protocol, recording both old and new artifacts.

## Target architecture

```text
GGUF tensor facts / model trace
  -> typed workload inventory
  -> generated candidate registry
  -> capability + memory admission
  -> tinygrad schedule/compiler
  -> compile evidence
  -> isolated correctness execution
  -> isolated role timing
  -> BoltBeam ranking/policy artifact
  -> tinygrad runtime binding
  -> route census + whole-model correctness/timing
  -> promotion or rollback
```

Authority boundaries:

- Tensor/model facts own quant, role, shape, and layout.
- Candidate descriptors own mathematical/dataflow strategy and schedule parameters.
- Capability admission owns target legality, LDS, vector alignment, divisibility, and memory limits.
- tinygrad owns lowering, compilation, and execution.
- BoltBeam owns analysis, search orchestration, comparison, and policy output; it does not fabricate runtime evidence.
- The route policy owns selection. The manifest reports provenance/lifecycle and must not become a second selector.
- The whole-model harness requests a policy and reports the exact bound binaries; it does not choose kernels.

## Phase 0 — Generalize the integration boundary

Status: implemented and integration-verified.

Deliverables:

- Typed `FullKernelWorkload` parser.
- One capability resolver for candidate sets, adapters, and compatibility binding.
- Workload rebinding that changes only workload/applicability and forces a new canonical identity.
- Profile/shape-driven prefill adapter and correctness canary.
- Generic profile policy gate and generic quant-route decision gate.
- One structured harness record per model profile.
- Cross-model admission and no-model-selector tests.

Exit gate:

- Focused and broad CPU suites pass (202 integration-focused tests in the implementation change).
- Existing 8B Q4_K packed candidate binary remains
  `5821ce7e86dc14f88f3f6063134fece6af9261a5327aa2e4729c8d4087336449` with 248 allocated VGPRs, 40,960-byte
  LDS, and 32 WMMA instructions.
- Adapter source contains no model name or exact model shape decision.

## Phase 1 — Build the exact 14B workload and candidate inventory

Status: implemented and CPU-admitted from the actual GGUF.

The committed inventory contains 282 tensor facts collapsed into the six measured role/quant rows, partitioned into
four Q4_K and two Q6_K candidate sets so identical shape warmstart keys cannot alias across formats. Regeneration from
the current 8.4 GiB GGUF is byte-for-byte JSON-equivalent to the committed artifact, and all six canonical candidates
pass schema, geometry, capability, packed-layout, and collision admission.

Tasks:

1. Read the actual GGUF tensor inventory and map each packed linear to canonical roles. Profile-level `Q4_K_M` is not
   sufficient because the model contains mixed Q4_K and Q6_K tensors.
2. Emit a machine-readable inventory row containing tensor identity, quant format, role, M/N/K, layout, call count,
   source bytes, logical FLOP, and memory lifetime.
3. Reconcile inventory with the six measured trace rows. Unknown or duplicate mappings block promotion.
4. Generate exact candidate payloads by rebinding admitted schedule templates to inventory facts and deriving packed
   operand metadata centrally.
5. Store the resulting candidate set as immutable JSON with canonical identities. Do not add six Python branches.
6. Add candidate-set tests for duplicate exact keys, warmstart-key collisions, tensor/shape mismatches, and unsupported
   formats.

Exit gate:

- All six rows have exact tensor evidence and canonical candidate identities.
- Candidate payloads pass CPU schema/capability/admission.
- No model-specific compiler code was added.

## Phase 2 — Fix packed producer efficiency at the established canary shape

Status: critical compiler path.

Current diagnosis:

- WMMA geometry and the correctness contract work.
- Packed expansion remains scalar and ALU-heavy.
- Q4 uses roughly 841 final instructions and Q6 roughly 947 versus about 574 for dense.
- Q4/Q6 allocate 248 VGPRs and issue more scalar global loads than the dense program.
- The tile API did not alter final ISA because UOp CSE already commoned equivalent scalar loads.

Work packages:

1. Add a native packed-block/group carrier so the compiler sees physical packed units rather than unrelated scalar
   logical values.
2. Fold aligned adjacent same-base packed reads into b64/b128 loads in generic lowering.
3. Unpack lanes from vector carriers using generic GEP/bitcast/permute operations while preserving bounds and format
   semantics.
4. Hoist Q4 `d/dmin/scale/min` and Q6 `d/scale` once per native group.
5. Use packed half2 conversion/arithmetic where it reduces instruction count without changing required fp16 rounding.
6. Decode near the LDS store, shorten fp32 temporary lifetimes, and measure whether allocation leaves the 248-VGPR
   bucket.
7. Common fragment loads by semantic `(operand,row,k_substep)` identity where final ISA proves duplication.
8. Only after lowering improves, search tile M/N/K, wave decomposition, load width, buffering, and stage depth through
   typed candidate data.

Per-change evidence:

- Full Q4_K and Q6_K canary correctness.
- Exact binary/resource/ISA diff.
- Packed global-load, shift/mask, conversion, WMMA, LDS, scratch, spill, VGPR, and SGPR counts.
- At least three isolated timing sessions with medians and spread.

Decision rule:

- Continue the fused fp16 packed primitive if controlled changes move it above 40 TFLOP/s and toward 50-60.
- If two well-attributed lowering iterations leave it below about 40 TFLOP/s, retain it as a fallback/candidate and
  prioritize Q4 integer-WMMA and Q6 dequant-once. Do not keep tuning an uncompetitive universal route indefinitely.

Exit gate:

- One correctness-qualified generated strategy reaches practical candidate territory without spills or hidden global
  dequant materialization, or a precise refutation identifies which alternative strategy must own each quant family.

## Phase 3 — Q4_K role completion

Candidate families:

1. Fused packed-to-fp16 LDS/WMMA primitive from Phase 2.
2. Existing generated Q4_K/Q8_1 integer-WMMA substrate, including activation-pack cost.
3. Scalar direct-packed only as rollback/comparator.

Tasks for all four Q4 roles:

- Compile exact final programs and reject scalar fallback or missing WMMA.
- Prove no full global RAW/dequant tensor is materialized.
- Prove activation Q8_1 lifecycle, reuse, and packing cost for integer-WMMA candidates.
- Run full-output correctness where memory permits; otherwise use a bounded proof only temporarily and require full
  role correctness before model integration.
- Benchmark whole primitive cost, not a kernel that excludes required Q8 packing/correction work.
- Search schedules per workload descriptor without role-local environment branches.

Aggregate gate: at least 60 TFLOP/s; target at least 66 TFLOP/s. Role wall budgets are approximately gate/up 122-132
ms, q/o 43 ms, down 35 ms, and k/v 10 ms per pp512 step.

## Phase 4 — Q6_K role completion

Evaluate in order:

1. Dequant once to fp16 followed by ordinary generated WMMA. This is the shortest path to the measured llama design.
2. Per-role or ephemeral dequant-once if a full resident overlay fails memory admission.
3. Fused packed-to-fp16 LDS/WMMA from Phase 2 when materialization/lifetime is not admissible.
4. Scalar direct-packed only as rollback.

Memory authority must account for model weights, KV/cache/runtime buffers, compiled graphs, output/reference buffers,
and allocator headroom. A theoretical byte count alone is not admission.

Gates:

- `ffn_down` dequant plus GEMM no more than about 26 ms aggregate.
- `attn_kv` dequant plus GEMM no more than about 2 ms aggregate.
- Q6 aggregate at least 69 TFLOP/s; target at least 70 TFLOP/s.
- No stale captured overlay, OOM, unbounded materialization, spill, or excluded dequant cost.

## Phase 5 — Central candidate search and policy

Tasks:

- Materialize each eligible strategy as a typed `GeneratedCandidate` with quant, role, shape, target, required
  features, memory requirements, lifecycle, and authority gates.
- Keep lifecycle states explicit: diagnostic, candidate, shipped, refuted, deferred.
- Make registry plus route policy the sole selection path. Remove selection decisions from harness environment defaults.
- Have BoltBeam consume compile/correctness/timing artifacts and emit an authorized policy keyed by exact workload and
  target.
- Cache measurements by source identity, binary identity, driver/runtime identity, device, clock policy, and workload.
- Require an explicit comparator and rollback for every promoted row.

Exit gate:

- A policy file selects one exact eligible candidate for every six-row workload entry.
- Unknown workload, identity drift, or missing evidence fails closed to the safe rollback.
- Route manifest and harness cannot independently override the policy.

## Phase 6 — Mixed-route 14B integration

Sequence:

1. Load model and inventory without allocating optional overlays.
2. Admit all selected candidates against one measured memory budget.
3. Compile all candidates in isolation and record binary/resource evidence.
4. Run role correctness before whole-model timing.
5. Run one smoke prefill with route census and output/token checks.
6. Run the synchronized authority workload only after the smoke gate passes.
7. Confirm decode route identities and run decode regression separately.

Required census fields per role:

```text
tensor -> quant -> role -> M/N/K -> candidate id -> canonical identity
-> source hash -> binary hash -> launch count -> median kernel time -> correctness artifact
```

Exit gate: all intended rows fire, no fallback fires, outputs are correct, GPU remains healthy, and the measured wall is
consistent with the sum of role and residual budgets.

## Phase 7 — Parity and promotion

Protocol:

- Run llama and Arkey sequentially because concurrent residency is not admissible.
- Alternate order across at least three sessions.
- Use the same pp512 workload, model bytes, clock policy, warmup, sample count, and host synchronization.
- Report median, spread, and all individual samples.
- Re-profile the candidate after timing; do not project from stale role shares.

Post-comparison Boltbeam attribution is mandatory whenever any declared context fails to beat llama or the aggregate
geometric mean is below 105%:

- trace the exact candidate identity, binary identities, model content, context, clock policy, compiler/runtime
  revision, and route census used by the matched end-to-end session;
- capture pp512 plus every context that misses its gate, without changing policy or measurement semantics between the
  authority run and attribution run;
- map launches back to the Phase 6 census and separate Q4/Q6 kernels, activation packing or dequantization, attention,
  synchronization, launch overhead, memory movement, and non-GEMM residual work;
- retain final ISA/resource/occupancy and roofline evidence for the dominant tinygrad costs, alongside the matched
  llama phase/kernel timing artifact;
- reconcile the summed attributed time with synchronized whole-prefill wall time and record any unattributed gap
  explicitly instead of assigning it to a guessed kernel;
- emit a ranked tax ledger naming the largest measured deltas, owning layer, proposed controlled experiment, and
  acceptance/rejection result;
- after each accepted optimization, rerun correctness/resource/health gates and the matched end-to-end protocol before
  updating the comparison. Boltbeam timing alone cannot promote a candidate.

Decisions:

- Below 98% of llama: candidate remains unshipped; attribute residual by role before changing code.
- At least 98%: parity-qualified, subject to correctness and decode gates.
- At least 105%: beyond-parity-qualified.
- At least 2,000 tok/s: project target reached.

Optimize norms/elementwise residuals only if Q4 and Q6 gates pass but whole-model parity does not. At projected Q4=60
and Q6=70 TFLOP/s, the existing residual should already permit parity.

## Phase 8 — Generalization closeout

- Re-run 8B prefill and decode with the same registry/admission/policy path.
- Add a supported profile from facts without editing compiler or adapter code.
- Exercise tail/non-divisible shapes: either generate a legal tail strategy or reject them through typed admission.
- Verify Q4_K/Q6_K format code contains no model names.
- Verify runtime selection contains no model-size or exact-shape branch outside immutable candidate data/test fixtures.
- Delete superseded experimental scripts and flags only after their negative evidence is preserved in the lessons ledger.

Exit gate: adding a previously unseen supported model requires data/artifacts and search, not a tinygrad source edit.

## Failure classification and stop rules

Every failed run must classify one owning layer:

- inventory/layout
- candidate schema/admission
- packed math/rounding
- compiler vectorization/instruction selection
- register scheduling/resources
- synchronization/dispatch
- artifact/harness
- memory lifecycle
- route binding
- whole-model residual

Stop a candidate, not the project, when:

- exact correctness fails after the smallest violated invariant is isolated;
- required wide loads cannot preserve alignment/bounds;
- controlled lowering variants cannot leave an occupancy/resource cliff;
- required activation packing erases the isolated kernel win;
- memory admission rejects dequant-once and measured fused lowering is superior;
- role timing cannot translate to synchronized whole-model wall;
- correctness depends on a handwritten/oracle wrapper.

Unavailable gfx11 performance counters are not a completion blocker. They limit dynamic cache attribution; they do not
invalidate correctness, final ISA/resource evidence, controlled A/B timing, or whole-model measurement.

## Immediate execution order

1. Retain the current immutable six-row policy only as an integration diagnostic. Continue the bounded live-route
   fault/census investigation without treating its `ffn_gate_up` binding as a performance selection.
2. Use the existing machine-search, five-buffer emitter, frozen-artifact, and tinygrad PM4/AQL runtime stack to search
   for a generated candidate that beats direct packed under comparable full-K role timing.
3. Rebuild a performance-qualified immutable policy only after the winning candidate has exact full-role correctness,
   resources, health, identity, and repeated timing evidence.
4. Run Phase 6 whole-model manual mixed-route memory reconciliation, route census, correctness/output/token checks,
   GPU health, and decode correctness/performance regression on that exact policy.
5. Run Phase 7 matched llama/tinygrad contexts 512/1024/2048/4096 in at least three alternating pinned sessions and
   apply the statistical gates. If a gate misses, use Boltbeam only on that exact candidate revision/session.
6. Only after all preceding gates pass, promote production; then enable performance autoscan.

Until step 6, production dispatch remains unchanged and direct-packed remains the default.
