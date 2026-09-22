# NVIDIA pp512 parity execution scope for Luna-low agents

Date: 2026-08-29

Status: authoritative handoff scope for the next campaign. This document
supersedes the execution queue in
`docs/task_workflow/output/nv-prefill-ranked-test-invest-plan.md`; it does not
replace that document's historical evidence.

## Objective and frozen authority

Close the remaining Qwen3-8B Q4_K_M pp512/ubatch512 RTX 5090 NVIDIA prefill
gap without weakening full-logit correctness, recurrent replay, canonical
packed-weight ownership, or the default-off policy.

| runtime | R9 minimum | settled median |
|---|---:|---:|
| tinygrad composed unroll4 + Q4-V | 67.153915 ms | 67.235719 ms |
| llama.cpp | 34.680367 ms | 35.019399 ms |
| remaining difference | 32.473548 ms | 32.216320 ms |

The tinygrad authority has 198 compact-Q8 producers and 198 compiler mains:
72 gate/up, 36 K, 72 Q/O, and 18 Q4-V. It has 54 FP16 overlays: 18 Q6-V and
36 down. It has zero packed transport copies, old fixups, or partial
workspace. Full logits pass the declared tolerance, token 198 matches, and
the deep20 replay is exact.

The baseline authority is:
`docs/task_workflow/output/nv-prefill-composed-unroll4-q4v-result-20260829.md`.
An agent must not substitute an older 69.378-ms or 83-ms result as the current
control.

## Low-agent operating contract

Each assignment below is one packet. An agent receives exactly one packet ID.
It must not continue into the next packet, broaden admission, or optimize an
adjacent role. A packet ends with PASS, STOP, or BLOCKED and one output record.

Every packet must:

1. Start from the current composed route and preserve default-off selection.
2. Reuse the named harnesses and evidence; do not create a parallel framework.
3. Change one declared variable or mechanism at a time.
4. Keep candidate and control on the same commit, process protocol, inputs,
   clocks, warmup count, and R9 timing protocol.
5. Record exact commands, environment, commit, GPU, clocks, program hashes,
   launch geometry, minimum, median, all samples, and failure text.
6. Write generated measurements under a new packet-specific directory in
   `docs/task_workflow/evidence/` and the verdict under
   `docs/task_workflow/output/`.
7. Never edit an existing authority result or overwrite an existing evidence
   directory.
8. Stop on a correctness failure. Do not tune a failing spelling until a
   separate localization packet names the failure.
9. Stop on a population or whole-model performance failure. An isolated
   primitive win is not authorization to integrate.
10. Leave all new routes default-off. Do not change production defaults.

No packet may claim recovery by adding profiled region debts or scaling a
profile percentage onto the unprofiled wall. Device interval accounting ranks
work only; synchronized unprofiled R9 is the wall authority.

## Shared gates

### G0: primitive correctness

- Canonical real model weights and legal real-role activation input.
- Finite, nonzero output and complete sentinel coverage.
- Inputs and packed weights remain read-only.
- Independent oracle over the full output, not a checksum alone.
- Preserve the existing declared tolerance unless the packet explicitly
  states a stricter bit-exact requirement.
- Retain source, cubin, SASS, register count, shared memory, local-memory
  traffic, launch geometry, and program identity.

### G1: exact role population

- Exact model role count and exact Q4/Q6 format split.
- Canonical unique packed-weight base per admitted role.
- Exact compact-Q8 producer/main/output census.
- Zero unintended copies, old fixups, partial workspace, and expanded overlay
  for admitted roles.
- Same-input replay and distinct-input freshness.

### G2: composed correctness

- Full logits finite and within the existing `rtol=0.02, atol=0.5` gate.
- Greedy token matches the control.
- Deep20 records, stage outputs, all KV slices, logits, and token replay are
  exact where the current authority requires exactness.
- Queue/dependency policy is regenerated for the candidate graph; a stale
  digest is forbidden.

### G3: performance

- Fresh candidate and control processes with synchronized R9.
- Candidate minimum and median must both improve.
- Repeat in an independent confirmation process.
- The confirmed improvement must exceed both 0.25 ms and three times the
  pooled median absolute deviation; otherwise record NO-SIGNAL and STOP.
- A primitive candidate must win its complete producer/main/epilogue
  lifecycle before a model integration packet may begin.

### G4: promotion

- G0 through G3 pass on the exact composed graph.
- No exact graph digest, mutable device-global cursor, or process-environment
  cache is the sole admission mechanism.
- Concurrent captures and two-model ownership are tested.
- The route remains default-off until a separate promotion decision.

## Closed lanes: do not assign

| lane | authoritative result | reopening prerequisite |
|---|---|---|
| final-row prune | correct, +0.277480 ms slower | none; closed |
| existing vocab Q6/Q8 and top-1 assets | full-logit asset about 1.043 ms slower; top-1 violates API | new many-row full-logit design |
| current Q6-V spelling | marginal V-only signal; combined lifecycle loses | materially different lifecycle hypothesis |
| current Q4-down spelling | max_abs 2.695646 and slower | numerical root cause plus new lifecycle |
| current Q6-down spelling | exact primitive; model down-only loses about 1.61 ms | boundary attribution naming a different mechanism |
| current Flash S6/cooperative spellings | oracle-capable but service-time loss | new vectorized topology |
| unsafe Q/O default-ready placement | recurrent replay failure | never reopen without exact dependency correctness |
| generic overlap/cp.async/TMA campaign | no causal support | a counter packet must first name the corresponding stall |

## Dependency graph and concurrency

```text
A0 current low-perturbation ledger
 +--> B0 gate/up schedule discriminator --> B1 one schedule implementation
 +--> C0 Q4-down numerical localization --> C1 new Q4-down primitive --> C2 population/composition
 +--> D0 Q6-down boundary attribution ----> D1 one lifecycle implementation --> D2 population/composition
 +--> E0 vocabulary contract/design ------> E1 many-row primitive -----------> E2 composition
 +--> F0 Flash topology discriminator ----> F1 vector topology primitive ----> F2 composition
 +--> H0 support semantic census (deferred until B/C/D/E/F decisions)

Accepted composed wins, serialized in measured-value order
 +--> P0 dependency-identity queue policy
 +--> P1 binding/admission hardening
 +--> P2 final parity ledger and promotion recommendation
```

A0 is first. B0, C0, D0, E0, and F0 may run in parallel after A0. Their
implementation packets must not run until the corresponding discriminator
passes and names exactly one mechanism. Model-composition packets are
serialized because they share `tinygrad/llm/model.py`, graph inventory, and
the Q/O safe cut. P0 through P2 are serialized last.

## Packet A0: current low-perturbation HCQ ledger

Goal: allocate the current 67.154-ms composed graph without the large
`PROFILE=1` perturbation.

Inputs:

- `extra/llm_research/prefill/nv_prefill_hcq_exact_accounting.py`
- `extra/llm_research/prefill/nv_prefill_cross_runtime_accounting.py`
- `extra/llm_research/prefill/nv_post_unroll_support_attribution.py`
- `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/`
- the exact 198-role composed authority

Allowed implementation: add one HCQ-native timestamp capture helper and one
accounting script. A core runtime hook is allowed only if it is inert when no
observer is installed and cannot change queue placement or dependencies.

Required output:

- 100% of device intervals classified by role and layer; unknown count zero.
- Interval union, overlap sets, device idle, graph boundaries, and host
  residual close the measured wall.
- Observer-off and observer-on synchronized R9 brackets.
- Observer overhead no greater than 2% at the median. Otherwise STOP and
  report the perturbing boundary; do not use the ledger for ranking.
- New current-route region ledger, with old traced rows explicitly marked
  historical.

Verdict file:
`docs/task_workflow/output/nv-prefill-current-hcq-ledger-20260829.md`.

## Packet B0: gate/up register-safe schedule discriminator

Goal: identify one executed-path change that addresses the measured tensor
issue and scoreboard deficit without changing arithmetic or ownership.

Inputs:

- `extra/llm_research/prefill/nv_prefill_gateup_ncu_bridge.py`
- `extra/llm_research/prefill/nv_prefill_gateup_unroll_discriminator.py`
- `extra/llm_research/prefill/nv_compiler_q4k_gkqo_model_arm.py`
- `docs/task_workflow/output/nv-prefill-gateup-unroll4-result-20260829.md`

Fixed facts: unroll4 is the control; tinygrad and llama execute the same
6,291,456 IMMA count. The observed deficit is tensor duty 14.65% versus
31.71%, eligible warps 0.554 versus 0.807, and higher long-scoreboard stalls.

Test exactly these isolated variants against unroll4:

1. Reorder fragment loads to increase load-to-use distance without changing
   K64 arithmetic or CTA ownership.
2. Move only the scale/min metadata load relative to the fragment load.
3. One register-safe double-buffered fragment schedule, rejected immediately
   if it spills or lowers occupancy.

Do not test a new tile, Stream-K, cp.async, TMA, fusion, or queue placement in
this packet. Run G0 and matched physical counters for one real gate role, then
the exact 72-role proxy for a candidate that improves counters and lifecycle.

PASS names exactly one variant for B1. Otherwise STOP the schedule family.

Verdict file:
`docs/task_workflow/output/nv-prefill-gateup-schedule-discriminator-20260829.md`.

## Packet B1: one gate/up schedule implementation

Dependency: B0 PASS.

Implement only B0's named schedule in the compiler-owned K64 route. Preserve
unroll4 as the rollback control. Run G0, G1 for all 72 roles, G2, and G3
against the exact current composed route. Do not modify admission, Q/O queue
policy, or another role.

PASS produces a default-off composed candidate for serialized acceptance.
Failure at any gate is STOP, not authorization for another variant.

Verdict file:
`docs/task_workflow/output/nv-prefill-gateup-schedule-composed-result-20260829.md`.

## Packet C0: Q4-down numerical localization

Goal: explain the authoritative `max_abs=2.695646` failure before any new Q4
down performance work.

Inputs:

- `extra/llm_research/prefill/nv_q4down_capture_io.py`
- `extra/llm_research/prefill/nv_q4down_matched_ab.py`
- `extra/llm_research/prefill/nv_q4k_down_static_oracle.py`
- `extra/llm_research/prefill/nv_compiler_q4k_down_asset.py`
- `extra/llm_research/prefill/nv_compiler_q4k_down_pp512_binding.py`
- `docs/task_workflow/evidence/nv-q4down-matched-ab-20260829/result.json`

Use one real type-12 down role and the exact same saved-Z input for every arm.
Compare, in order: compact-Q8 record, decoded Q4 group metadata, per-K32
corrected subtotal, pre-epilogue FP32 output, post-epilogue output. Emit the
first mismatching stage, first output coordinate, expected/actual value, and
the exact source expression owning it.

No timing claim and no model edit is allowed. PASS means the first divergent
stage and one correction mechanism are proven. Ambiguous localization is
BLOCKED with retained tensors, not permission to guess.

Verdict file:
`docs/task_workflow/output/nv-prefill-q4down-numerical-localization-20260829.md`.

## Packet C1: corrected Q4-down primitive

Dependency: C0 PASS.

Implement only C0's correction in the direct compiler Q4-down primitive.
Keep `(512,4096,12288)`, canonical packed weights, compact Q8, direct output,
and zero global partials. Run G0 on all 18 Q4-down weights. Then compare the
complete producer + main + residual epilogue lifecycle with the installed
FP16 control under hot and rotated-cold R9.

Advance only if correctness passes and both lifecycle minimum and median win.
Do not integrate the model in this packet.

Verdict file:
`docs/task_workflow/output/nv-prefill-q4down-corrected-primitive-20260829.md`.

## Packet C2: Q4-down population and composition

Dependency: C1 PASS.

Integrate exactly 18 type-12 down roles into the current composed route. Run
G1, G2, and G3. Require 216 packed producers/mains if no prior accepted packet
changed the census, with 36 remaining FP16 overlays: 18 Q6-V and 18 Q6-down.
If the incoming accepted baseline differs, derive the expected census by
adding exactly 18 and state it before the run.

Verdict file:
`docs/task_workflow/output/nv-prefill-q4down-composed-result-20260829.md`.

## Packet D0: Q6-down lifecycle boundary attribution

Goal: name why an exact Q6 primitive loses about 1.61 ms at model scale.

Inputs:

- `extra/llm_research/prefill/nv_compiler_q6k_imma_gate.py`
- `extra/llm_research/prefill/nv_compiler_q6k_model_arm.py`
- `extra/llm_research/prefill/nv_compiler_q6k_pp512_binding.py`
- `docs/task_workflow/output/nv-prefill-complete-lifecycle-ledger.md`

Measure the same 18 real roles at four forced boundaries: compact-Q8 producer;
Q6 main alone; main plus output publication; main plus rank-preserving residual
epilogue. Compare hot and rotated-cold service, allocations, queue readiness,
copies, and materializations with FP16. No arithmetic redesign is allowed.

PASS names exactly one dominant boundary and a single removable mechanism.
If no boundary exceeds noise independently, STOP Q6-down rather than issuing
an implementation task.

Verdict file:
`docs/task_workflow/output/nv-prefill-q6down-boundary-attribution-20260829.md`.

## Packet D1: one Q6-down lifecycle implementation

Dependency: D0 PASS.

Implement only D0's named mechanism. The paired K16 correction contract must
remain exact; summing the two K16 corrections into an invalid K32 correction
is forbidden. Run G0 on all 18 Q6-down roles and the complete hot/cold
producer-main-publication-epilogue lifecycle. No model integration.

Verdict file:
`docs/task_workflow/output/nv-prefill-q6down-lifecycle-primitive-20260829.md`.

## Packet D2: Q6-down population and composition

Dependency: D1 PASS.

Integrate exactly 18 Q6-down roles, then run G1, G2, and G3. State the expected
census as the incoming accepted baseline plus exactly 18. Do not include Q6-V.

Verdict file:
`docs/task_workflow/output/nv-prefill-q6down-composed-result-20260829.md`.

## Packet E0: many-row full-logit vocabulary contract

Goal: freeze the legal prefill vocabulary ABI and select one new design before
implementation.

The legal input/output is one final hidden row and all 151,936 logits. A
top-1-only output is illegal. Existing decode M=1 and rejected Q6/Q8 assets
are reference evidence, not candidates.

Produce an executable oracle fixture containing the canonical Q6 vocabulary
weight, one real final hidden row, FP32/FP16 reference logits, tolerance,
sentinels, and read-only hashes. Compare two paper designs only:

1. Q8 producer + packed Q6 MMVQ writing all logits directly.
2. Fused input quantization inside a many-row Q6 MMVQ writing all logits.

Estimate no wall recovery. Select a design only from explicit ownership,
launch-population, output-write, correction, and workspace constraints. PASS
names one design and its ABI for E1.

Verdict file:
`docs/task_workflow/output/nv-prefill-vocab-manyrow-contract-20260829.md`.

## Packet E1: many-row vocabulary primitive

Dependency: E0 PASS.

Implement E0's selected ABI as an isolated default-off primitive. Run G0 over
all 151,936 logits and compare the full producer + main + output lifecycle to
the installed tinygrad vocabulary control. Retain full logits for downstream
sampling. No model integration and no top-1 shortcut.

Verdict file:
`docs/task_workflow/output/nv-prefill-vocab-manyrow-primitive-20260829.md`.

## Packet E2: vocabulary composition

Dependency: E1 PASS.

Integrate the primitive after the exact current final hidden row. Run full
logits, greedy token, recurrent state, and G3. The dense 198-role census must
not change unless earlier accepted packets changed it; vocabulary remains a
separate one-call tail.

Verdict file:
`docs/task_workflow/output/nv-prefill-vocab-manyrow-composed-result-20260829.md`.

## Packet F0: Flash vector-topology discriminator

Goal: identify a new topology, not tune the rejected S6 family.

Inputs:

- `extra/llm_research/prefill/nv_prefill_flash_program_extract.py`
- `extra/llm_research/prefill/nv_prefill_flash_oracle.py`
- installed program `nv_sm120_q16_grid_hd128_loop_attention`
- `docs/task_workflow/output/nv-prefill-ranked-campaign-closure-20260829.md`

Freeze the live 36-call shapes, Q/K/V/output ABI, causal mask, head grouping,
and oracle. Produce a topology ledger for the installed tinygrad and llama
paths: CTA-to-head/tile ownership, vector width, KV access order, reduction
ownership, output publication, registers, shared memory, and service time.

Evaluate only paper/compile candidates that differ structurally from S6:
vectorized KV loads with head/tile ownership and an explicitly named reduction
owner. Do not integrate or claim a win. PASS names one exact topology whose
resource bounds fit the target and whose generated primitive can be tested.

Verdict file:
`docs/task_workflow/output/nv-prefill-flash-vector-topology-scope-20260829.md`.

## Packet F1: Flash vector primitive

Dependency: F0 PASS.

Implement only F0's topology in an isolated harness. Run the live full-output
oracle, complete coverage, read-only checks, hot/cold R9, and counters. It must
beat the installed Flash primitive over the exact 36-call population before
composition. The rejected S6/cooperative source may not be relabeled as the
new topology.

Verdict file:
`docs/task_workflow/output/nv-prefill-flash-vector-primitive-20260829.md`.

## Packet F2: Flash composition

Dependency: F1 PASS.

Integrate the exact 36-call population, regenerate the Q/O dependency cut,
and run G1, G2, and G3. Any recurrent failure is STOP even if a one-shot R9
sample wins.

Verdict file:
`docs/task_workflow/output/nv-prefill-flash-vector-composed-result-20260829.md`.

## Packet H0: support semantic census

Dependency: A0 PASS and B/C/D/E/F decisions recorded.

Map 100% of remaining support intervals to stable semantic family, source
operation, layer, input/output bytes, dependency predecessor/successor, and
llama counterpart. Separate required math from transport, materialization,
shape-only movement, and observer overhead. Do not fuse or remove anything.

Output a ranked family ledger. A future implementation packet may be created
only for a single family whose independently measured current-route exposure
exceeds noise and whose removal preserves dependencies.

Verdict file:
`docs/task_workflow/output/nv-prefill-support-semantic-ledger-20260829.md`.

## Packet P0: dependency-identity queue policy

Dependency: all accepted performance packets composed and frozen.

Replace exact graph-digest admission with dependency-identity admission for
the Q/O-to-Flash cut. Test default-ready, minimal-cut, conservative-cut,
primary-only, and one-queue controls. The fastest deep20-correct arm is the
only admissible policy. No kernel arithmetic changes.

Verdict file:
`docs/task_workflow/output/nv-prefill-dependency-identity-policy-20260829.md`.

## Packet P1: binding and admission hardening

Dependency: P0 PASS.

Remove mutable device-global record/output cursors from standalone bindings;
separate immutable compiled assets from per-model and per-capture buffers;
prove two models and two captures cannot alias. Replace whole-AST packed-dtype
scanning with operand-scoped A/B ownership. Preserve exact Qwen3-8B pp512
admission until all ownership tests pass; broader shapes are out of scope.

Verdict file:
`docs/task_workflow/output/nv-prefill-binding-admission-hardening-20260829.md`.

## Packet P2: final parity ledger

Dependency: P1 PASS.

Run fresh tinygrad and llama.cpp minimum/median wall, low-perturbation interval
accounting, exact role/overlay/copy/workspace census, full-logit correctness,
token, deep20 replay, memory footprint, and independent confirmation. Compare
only accepted mechanisms with direct rollback controls. Report remaining gap
without extrapolation and recommend PROMOTE DEFAULT-OFF, CONTINUE, or STOP.

Verdict file:
`docs/task_workflow/output/nv-prefill-final-parity-ledger-20260829.md`.

## Dispatcher checklist

Before assigning a packet, the dispatcher must provide:

- packet ID and no other packet;
- exact starting commit and accepted-baseline result path;
- writable file allowlist and evidence/output destinations;
- GPU/model path and fixed clock/session prerequisites;
- dependencies marked PASS with their verdict paths;
- explicit instruction to stop at the packet verdict.

Before accepting a packet, the dispatcher checks:

- no closed lane was reopened;
- no unrelated core or evidence file changed;
- every command and sample is retained;
- candidate/control protocol is matched;
- correctness precedes performance;
- minimum and median both pass the declared threshold;
- no profile exposure is booked as wall recovery;
- the next packet named by the result is actually authorized by this DAG.

## Source authorities

- `docs/task_workflow/output/nv-prefill-ranked-campaign-closure-20260829.md`
- `docs/task_workflow/output/nv-prefill-composed-unroll4-q4v-result-20260829.md`
- `docs/task_workflow/output/nv-prefill-exact-cross-runtime-accounting.md`
- `docs/task_workflow/output/nv-prefill-complete-lifecycle-ledger.md`
- `docs/task_workflow/output/nv-prefill-gateup-unroll4-result-20260829.md`
- `docs/task_workflow/output/nv-final-row-prune-result-20260829.md`
- `docs/task_workflow/output/nv-prefill-q4v-result-20260829.md`
- `docs/task_workflow/evidence/nv-q4down-matched-ab-20260829/result.json`

