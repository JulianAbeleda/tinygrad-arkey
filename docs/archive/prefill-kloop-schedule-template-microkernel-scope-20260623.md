# Prefill K-Loop Schedule-Template Microkernel — Exhaustive Scope / Claude Prompt

Date: 2026-06-23

## Mission

Build the first **emitter-side** prototype for the prefill schedule-template
representation.

The previous task built the detector:

```text
extra/qk_schedule_interleave_detector.py
```

It can classify a GEMM kernel as:

- `PHASED`: load/stage first, barrier, then WMMA compute;
- `PIPELINED`: next-tile loads / LDS ops appear inside the WMMA span.

The detector proved:

- `build_gemm_lds2(down)` is `PHASED`;
- Tensile is `PIPELINED`;
- the remaining ~4-5% prefill gap is K-loop software pipelining plus register
  lifetime / VGPR pooling, not tile config.

This scope is the next step:

```text
emit a tiny local microkernel that the detector classifies as PIPELINED
and that remains numerically correct with an acceptable resource envelope.
```

This is **not** a full prefill replacement and not a model route. It is the
minimal proof that the new representation can generate the missing primitive.

## Core Question

Can we express a bounded K-loop schedule template that interleaves next-tile
global/LDS work with current-tile WMMA compute, while staying correct and
not spilling?

If yes, machine search gets a real new representation level:

```text
schedule_template + register_lifetime
```

If no, the prefill residual remains a hand-asm / renderer capability, not a
searchable surface.

## Required Reading

Read these first:

1. `docs/machine-search-representation-expansion-decode-prefill-result-20260623.md`
2. `docs/prefill-schedule-diff-oracle-and-search-reduction-result-20260623.md`
3. `docs/prefill-search-result-20260623.md`
4. `docs/prefill-amd-gemm-leanaddr-result-20260620.md`
5. `docs/prefill-tensile-schedule-template-extraction-result-20260620.md`
6. `docs/prefill-tensile-lds-tile-map-sketch-20260620.md`
7. `docs/prefill-primitive-pmc-result-20260619.md`
8. `docs/native-codegen-microprimitive-search-result-20260623.md`
9. `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`
10. `docs/project-search-ledger-contract-20260623.md`
11. `bench/qk-decode-eval/HARNESS_GUIDE.md`
12. `structure/Development/performance-primitive-research-principles.md`
13. `structure/Development/session-handoff.md`

Inspect code:

- `extra/qk_schedule_interleave_detector.py`
- `extra/gemm/rdna3_wmma_matmul.py`
- `extra/qk_prefill_graph_gemm_route.py`
- `extra/qk_prefill_search_execute.py`
- `extra/qk_amdgpu_isa_primitive_audit.py`
- `extra/qk_isa_primitive_audit.py`
- `extra/qk_project_search_ledger.py`

Inspect artifacts:

- `bench/qk-machine-search-representation-expansion/`
- `bench/qk-prefill-schedule-diff-oracle/`
- `bench/qk-prefill-search/`
- `bench/qk-native-codegen-microsearch/`

## Non-Goals

- Do not change `tinygrad/` source.
- Do not change defaults.
- Do not route this into the model.
- Do not claim whole-prefill speedup.
- Do not run broad tile-config search.
- Do not promote vendored Tensile.
- Do not build a full `ffn_down` / `qo_proj` replacement.
- Do not bypass correctness, ISA, VGPR, or spill gates.
- Do not use no-sync/profile/raw-dispatch timing as authority.

## Authority For This Scope

This is a **microprimitive representation proof**, not a speed promotion.

Required gates:

1. build/compile succeeds;
2. local numeric correctness versus a reference;
3. `schedule_interleave_gate` says `PIPELINED`;
4. ISA/resource audit:
   - WMMA present;
   - LDS present;
   - vector/global loads present;
   - no scratch/spill;
   - VGPR within declared envelope;
5. artifact and ledger entry.

Optional diagnostics:

- local kernel timing;
- PMC counters;
- comparison to `PHASED` microkernel.

Timing is diagnostic only. No W==D / whole-prefill claim in this scope.

## Phase 0 — Authority Lock / Baseline Detector Recheck

Re-run the detector on known references.

Tasks:

1. Run `extra/qk_schedule_interleave_detector.py --builder down`.
2. Run detector on the known Tensile code object if available.
3. Confirm:
   - `build_gemm_lds2(down)` -> `PHASED`;
   - Tensile -> `PIPELINED`.
4. Record tool versions, git SHA, GPU/ROCm.

Deliverable:

```text
bench/qk-prefill-kloop-schedule-template/authority.json
```

Verdicts:

- `KLOOP_TEMPLATE_AUTHORITY_LOCKED`
- `KLOOP_TEMPLATE_DETECTOR_DRIFT_STOP`

Stop if the detector does not reproduce.

## Phase 1 — Define The Minimal Schedule Template

Define a small local GEMM-like microkernel, not full prefill.

Recommended shape constraints:

- small enough to iterate quickly;
- still uses RDNA3 WMMA;
- at least two K-tiles, otherwise no pipeline exists;
- half inputs, fp32 accumulate or matching existing builder behavior;
- one or few output tiles;
- deterministic random input and numpy/reference output.

Template fields:

```json
{
  "template_id": "kloop_pipeline_v0",
  "k_tiles": 2,
  "prefetch_distance": 1,
  "prefetch_A": true,
  "prefetch_B": false,
  "ds_store_next_inside_wmma_span": true,
  "ds_load_current_grouping": "before_wmma_microgroup",
  "wmma_group_size": 4,
  "waitcnt_policy": "before_consumer_only",
  "barrier_policy": "stage_boundary",
  "register_budget": {
    "max_vgpr": 240,
    "reject_spill": true
  }
}
```

Start conservative:

- prefetch A only if B-prefetch exceeds VGPR envelope;
- then try A+B only if resource gates pass;
- keep correctness first.

Deliverable:

```text
bench/qk-prefill-kloop-schedule-template/template_spec.json
```

Verdict:

- `KLOOP_TEMPLATE_SPEC_READY`

## Phase 2 — Implement Candidate Emitter

Implement a new isolated tool/module.

Suggested file:

```text
extra/qk_prefill_kloop_template_microkernel.py
```

Acceptable implementation routes:

1. Extend / wrap `extra.gemm.rdna3_wmma_matmul.build_gemm_lds2` with a
   schedule-template mode.
2. Emit a small hand-owned AMDGCN/HIP microkernel if that is much simpler.
3. Use existing compile/run helpers if present.

Requirements:

- keep it outside model routes;
- do not alter shipped graph-GEMM defaults;
- accept CLI knobs for the template:
  - `--prefetch-a`;
  - `--prefetch-b`;
  - `--wmma-group-size`;
  - `--max-vgpr`;
  - `--out`;
- output:
  - source or instruction list;
  - compiled code object if applicable;
  - run result;
  - correctness metrics;
  - detector output;
  - ISA/resource output.

Deliverable:

```text
extra/qk_prefill_kloop_template_microkernel.py
```

Verdicts:

- `KLOOP_TEMPLATE_EMITTER_BUILT`
- `KLOOP_TEMPLATE_EMITTER_BLOCKED`

If implementation expands into broad renderer surgery, stop and classify:

```text
KLOOP_TEMPLATE_REQUIRES_RENDERER_WORK
```

## Phase 3 — Local Correctness Gate

Run the candidate against a deterministic reference.

Tasks:

1. Generate fixed-seed fp16 input matrices.
2. Run candidate.
3. Compare to numpy or existing graph-GEMM reference.
4. Report:
   - `rel_max`;
   - `rel_rmse`;
   - absolute max;
   - pass/fail threshold.

Suggested threshold:

- use the existing prefill graph-GEMM tolerance if known;
- otherwise start with `rel_rmse <= 3e-4` for fp16/WMMA path and record if this
  differs from local convention.

Deliverable:

```text
bench/qk-prefill-kloop-schedule-template/correctness.json
```

Verdicts:

- `KLOOP_TEMPLATE_CORRECTNESS_PASS`
- `KLOOP_TEMPLATE_CORRECTNESS_FAIL`

Stop on correctness failure unless a trivial bug is fixed in-scope.

## Phase 4 — Schedule-Interleave Gate

Use the detector as the defining gate.

Tasks:

1. Run `extra/qk_schedule_interleave_detector.py` on the candidate code object
   or instruction list.
2. Require:
   - classification `PIPELINED`;
   - global loads and/or LDS ops inside WMMA span;
   - evidence that the interleaving corresponds to next-tile work, not dead
     unreachable code.
3. Compare against:
   - `build_gemm_lds2` baseline (`PHASED`);
   - Tensile oracle (`PIPELINED`) if available.

Deliverable:

```text
bench/qk-prefill-kloop-schedule-template/interleave_gate.json
```

Verdicts:

- `KLOOP_TEMPLATE_INTERLEAVE_PASS`
- `KLOOP_TEMPLATE_STILL_PHASED`
- `KLOOP_TEMPLATE_INTERLEAVE_UNDETERMINED`

If still `PHASED`, do not proceed to timing. Classify what representation is
missing.

## Phase 5 — ISA / Resource Gate

Audit the candidate code object.

Required fields:

- has WMMA;
- has LDS;
- has vector/global loads;
- VGPR count;
- SGPR count;
- scratch bytes;
- spill yes/no;
- LDS bytes;
- kernel arg layout;
- occupancy proxy if available.

Hard reject:

- scratch/spill appears;
- VGPR exceeds declared envelope unless explicitly recorded as the failure;
- no WMMA;
- no LDS;
- detector pass was fake/unrelated.

Deliverable:

```text
bench/qk-prefill-kloop-schedule-template/isa_resource_gate.json
```

Verdicts:

- `KLOOP_TEMPLATE_ISA_RESOURCE_PASS`
- `KLOOP_TEMPLATE_VGPR_WALL`
- `KLOOP_TEMPLATE_SPILL_REJECT`
- `KLOOP_TEMPLATE_MISSING_ISA_PRIMITIVE`

## Phase 6 — Optional Local Timing / Counter Diagnostic

Only after correctness + interleave + ISA pass.

Purpose:

- sanity check that the pipelined template is not catastrophically slower;
- do not claim whole-prefill speedup.

Tasks:

1. Run a small repeated local timing against the phased microkernel.
2. Optionally collect counters if cheap:
   - wait/stall proxy;
   - LDS activity;
   - global load count;
   - active cycles.
3. Mark all timing as diagnostic.

Deliverable:

```text
bench/qk-prefill-kloop-schedule-template/local_diagnostic.json
```

Verdicts:

- `KLOOP_TEMPLATE_LOCAL_DIAGNOSTIC_READY`
- `KLOOP_TEMPLATE_LOCAL_DIAGNOSTIC_SKIPPED`

## Phase 7 — Candidate Classification

Classify the result.

Possible final classifications:

### Best outcome

```text
KLOOP_SCHEDULE_TEMPLATE_MICROKERNEL_PASS
```

Meaning:

- candidate is correct;
- detector classifies it `PIPELINED`;
- no spill;
- VGPR acceptable;
- representation can emit the missing primitive.

Next step:

- create a bounded search over schedule-template knobs on the microkernel;
- only later consider a role-specific prefill route.

### Representation emits interleave but resource fails

```text
KLOOP_TEMPLATE_REP_PASS_REGISTER_POOL_REQUIRED
```

Meaning:

- schedule-template concept works;
- static register allocation/VGPR wall blocks practical use;
- next representation is register-pool/lifetime allocator.

### Cannot emit interleave

```text
KLOOP_TEMPLATE_EMISSION_BLOCKED
```

Meaning:

- current builder/renderer cannot express interleaving;
- next work is renderer/hand-asm capability, not search.

### Correctness fails

```text
KLOOP_TEMPLATE_CORRECTNESS_FAIL
```

Meaning:

- do not search;
- fix semantics first.

Deliverable:

```text
bench/qk-prefill-kloop-schedule-template/decision.json
```

## Phase 8 — Project Ledger And Explorer Integration

Append one project ledger entry.

Fields must include:

- lane: `prefill_codegen_microprimitive`;
- primitive_class: `kloop_software_pipeline`;
- knobs/template fields;
- oracle: Tensile pipelined schedule;
- correctness;
- route_identity: n/a / local only;
- materialization_abi: n/a;
- ISA;
- local_diagnostic;
- authority_benchmark: local-only, no promotion;
- verdict;
- stop_reason;
- artifact links;
- learned_rule.

Update oracle explorer artifacts only if cheap:

- add schedule-template microkernel as a learning-only prototype;
- do not change speed-search readiness unless the microkernel passes.

Deliverables:

```text
bench/qk-project-search-ledger/ledger.jsonl
bench/qk-prefill-kloop-schedule-template/ledger_entry.json
```

Verdict:

- `KLOOP_TEMPLATE_LEDGER_RECORDED`

## Phase 9 — Result Doc

Write:

```text
docs/prefill-kloop-schedule-template-microkernel-result-20260623.md
```

Required answers:

1. Did the detector baseline reproduce?
2. What microkernel/template was built?
3. Did it compile?
4. Is it numerically correct?
5. Does the detector classify it as `PIPELINED`?
6. What ISA/resource envelope did it use?
7. Did it hit the VGPR/register wall?
8. Does this make prefill machine-searchable now?
9. What is the next step?
10. Were any defaults/model routes changed?

## Expected Final Verdicts

Expected best-case:

```text
KLOOP_SCHEDULE_TEMPLATE_MICROKERNEL_PASS
PREFILL_SCHEDULE_TEMPLATE_REPRESENTATION_EMITTABLE
PREFILL_FULL_SPEED_SEARCH_STILL_DEFERRED
```

Expected blocker if the prior analysis is right:

```text
KLOOP_TEMPLATE_REP_PASS_REGISTER_POOL_REQUIRED
```

or:

```text
KLOOP_TEMPLATE_EMISSION_BLOCKED
```

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch
`qk-prefill-flag-leak-resolution`.

Task: execute the prefill K-loop schedule-template microkernel scope.

The goal is not to optimize whole prefill yet. The goal is to prove whether the
new `schedule_template` representation can **emit** the missing K-loop software
pipelining primitive that the detector identifies in Tensile.

Read first:

- `docs/prefill-kloop-schedule-template-microkernel-scope-20260623.md`
- `docs/machine-search-representation-expansion-decode-prefill-result-20260623.md`
- `docs/prefill-schedule-diff-oracle-and-search-reduction-result-20260623.md`
- `docs/prefill-amd-gemm-leanaddr-result-20260620.md`
- `docs/prefill-tensile-schedule-template-extraction-result-20260620.md`
- `extra/qk_schedule_interleave_detector.py`
- `extra/gemm/rdna3_wmma_matmul.py`
- `bench/qk-decode-eval/HARNESS_GUIDE.md`
- `structure/Development/performance-primitive-research-principles.md`
- `structure/Development/session-handoff.md`

Execute phases:

1. Recheck detector authority: baseline phased vs Tensile pipelined.
2. Define a minimal K-loop schedule-template microkernel.
3. Build the isolated emitter/tool.
4. Run local numeric correctness.
5. Run schedule-interleave gate.
6. Run ISA/resource gate.
7. Optionally run local timing/counter diagnostic.
8. Classify the result.
9. Record ledger entry and result doc.

Boundaries:

- no tinygrad source changes;
- no model route;
- no default flip;
- no whole-prefill speed claim;
- no broad search;
- no vendored Tensile promotion;
- no RL/LoRA/training.

Final response must include:

- verdict labels;
- whether the microkernel is PIPELINED;
- correctness result;
- ISA/resource result;
- whether VGPR/register-pool blocked it;
- whether prefill machine search is now reopened or still deferred;
- artifacts written;
- files changed;
- git status.
