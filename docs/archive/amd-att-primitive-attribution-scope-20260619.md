# AMD ATT Primitive Attribution Scope

Date: 2026-06-19

## Purpose

Use the newly working AQLprofile ATT replay path on real tinygrad inference primitives.

This is not more ROCprofiler packet plumbing. R1-P2 already proved the tooling can import AQLprofile's thread-trace
lifecycle into tinygrad HCQ and recover decodable body packets. This phase uses that instrument to answer the remaining
primitive questions that PMC/timing can localize but not fully explain.

Primary questions:

1. **Decode:** why does tinygrad's standalone MMVQ/GEMV surface reach `~76%` HBM, while in-model weight-GEMV is only
   `~44%`, versus llama retaining `~54%`?
2. **Prefill:** after Tensile and transpose-free routing were refuted as e2e wins, which non-matmul primitive class
   explains the remaining pp512 gap?

## Authority

Current measured map:

- `docs/inference-perf-measured-map-20260619.md`
- `docs/decode-integration-diagnostic-result-20260619.md`
- `docs/prefill-tensile-transpose-free-result-20260619.md`
- `docs/amd-rocprofiler-r1p2-hcq-replay-result-20260619.md`

Important constraints:

- `rocprofv3` still does not trace tinygrad HCQ directly.
- tinygrad HCQ and same-process HSA/HIP runtime initialization are mutually exclusive.
- ATT is sampled by WGP/SIMD target; short kernels can miss body attribution unless enlarged/repeated.
- ATT is for root cause attribution, not timing authority. Timing authority remains clock-controlled in-model A/B and
  native PMC.

## Non-Goals

- Do not change model defaults.
- Do not start another standalone kernel-search pass.
- Do not reopen SQTT register sweeps; R1-P2 proved packet import works.
- Do not use ATT as a broad profiler for every kernel in the model. Use it only where a primitive claim needs
  instruction/resource evidence.
- Do not treat one ATT trace as a speed result. It must explain an already measured timing/PMC gap.

## Deliverables

New probe:

```text
extra/qk_att_primitive_atlas.py
```

Artifacts:

```text
bench/qk-att-primitive-atlas/decode_mmvq.json
bench/qk-att-primitive-atlas/prefill_nonmatmul.json
bench/qk-att-primitive-atlas/result.json
bench/qk-att-primitive-atlas/summary.md
```

Result doc:

```text
docs/amd-att-primitive-attribution-result-20260619.md
```

The probe should reuse the R1-P2 packet factory/replay mechanics rather than duplicating new profiler plumbing. If it
needs a shared helper, promote the minimal pieces from `extra/amd_rocprofiler_r1p2_hcq_replay.py` into a probe-local
utility first; do not touch tinygrad runtime code in this phase.

## Phase A - Instrument Adapter

Goal: make R1-P2 reusable for one named tinygrad `Program` dispatch.

Work:

1. factor a probe-local ATT wrapper:
   - export AQLprofile start/stop packets in a separate HSA helper process;
   - allocate tinygrad-owned control, command, and trace buffers;
   - patch raw `uint64` VAs;
   - patch PM4 page-address fields (`VA >> 12`);
   - submit `start -> target dispatch -> stop` through HCQ.
2. add a decode step that reports:
   - packet counts by SQTT packet class;
   - body-like instruction packet count;
   - top instruction classes if available from tinygrad's SQTT decoder;
   - wave start/end counts;
   - trace miss state: lifecycle-only, empty, or body-attributed.

Gate:

- reproduce R1-P2 on the smoke body kernel from the adapter path:
  - sync pass;
  - nonzero trace;
  - `>=10,000` body-like packets.

Kill:

- if the adapter regresses R1-P2, stop and fix the adapter before touching model primitives.

## Phase B - Decode MMVQ Contract Attribution

Goal: decide whether the `76% -> 44%` in-model loss is visible as a scheduler/resource contract difference in the real
MMVQ roles.

Targets:

| Target | Why |
|---|---|
| standalone fastest Q4_K/Q6_K GEMV surface | positive control: known high HBM efficiency |
| in-model `ffn_gate/up` Q4_K role | activation lifecycle + occupancy suspect |
| in-model `ffn_down` Q6_K role | high-share occupancy/coverage suspect |
| in-model `lm_head` Q6_K role | high-share low-efficiency role |

For each target, collect:

- native PMC summary already used by the decode atlas: time, GL2 hit/miss, VALU util, effective `%HBM`;
- ATT summary: instruction packet mix, wave count, wave duration distribution if decodable, body/lifecycle ratio;
- static resource metadata if available: VGPR, LDS, workgroup, grid;
- role context: standalone, eager in-model, or JIT/graph replay.

Questions to answer:

1. Does in-model MMVQ actually launch the same compiled contract as standalone?
2. If not, is the delta visible as:
   - fewer waves / less coverage;
   - different instruction mix;
   - more reduction/control packets;
   - shorter waves with worse memory-level parallelism;
   - graph/runtime scheduling gaps?
3. Does ATT support the current conclusion that decode is an integration/contract-preservation problem rather than a
   dot-product codegen problem?

Pass:

- at least one high-share role gets a concrete ATT-backed label that explains the PMC/timing loss, such as
  `low_wave_coverage`, `extra_stage2_reduce`, `different_program_identity`, or `same_kernel_not_scheduler_visible`.

Kill:

- if ATT shows no meaningful difference between standalone and in-model for the target roles, close ATT for decode and
  keep the current conclusion: large decode win requires project-level MMVQ integration or spec-decode, not another
  primitive-local codegen edit.

Decision after Phase B:

| Finding | Next |
|---|---|
| standalone/in-model program identity differs | fund runtime/cache identity fix before renderer work |
| same program, lower wave coverage/resource behavior | fund AMD scheduler/resource project |
| stage2/reduce dominates a high-share role | scope a reduce-fusion route, but only if projected W==D `>=5%` |
| no actionable difference | stop decode primitive work; prioritize spec-decode/project-level route |

## Phase C - Prefill Non-Matmul Attribution

Goal: map the remaining pp512 gap after matmul routes were refuted as e2e levers.

Targets:

| Target | Why |
|---|---|
| prefill attention / SDPA | likely non-matmul high-share component |
| RMSNorm / residual / elementwise bands | suspected dilution around fast matmuls |
| activation layout/cast/transposes still present in PREFILL_V2 | prior route-specific transpose was refuted, but broader layout tax remains possible |
| lm_head prefill surface | can be large and easy to misclassify |

For each target:

- use native PMC/timing as the authority for share;
- use ATT only if the component is high enough share or ambiguous after PMC;
- classify as compute issue, memory issue, launch/fragmentation issue, or negligible.

Pass:

- produce a component ledger where `>=80%` of pp512 GPU time is assigned to one of:
  - fast matmul already-close;
  - attention;
  - normalization/elementwise/residual;
  - layout/cast;
  - lm_head;
  - runtime/fragmentation.

Kill:

- if ATT cannot body-attribute short non-matmul kernels even with enlargement/repetition, fall back to PMC/timing only
  and mark ATT as unsuitable for that component class.

Decision after Phase C:

| Finding | Next |
|---|---|
| attention dominates residual | scope attention-specific prefill route |
| norm/elementwise fragmentation dominates | scope fusion/graph-level route |
| layout/cast still dominates | scope layout-lifecycle route |
| no single component is large | close prefill as distributed overhead unless a model-level redesign is funded |

## Phase D - Llama Cross-Check

Goal: keep the tinygrad story grounded against llama without overfitting to unavailable counters.

Work:

- use `rocprofv3 --kernel-trace` for llama timing/launch metadata;
- do not require ATT on llama unless the existing ROCprofiler path already emits a clean `.att` for the same surface;
- compare only contract-level properties that are observable on both sides:
  - number of kernels per primitive;
  - grid/workgroup;
  - VGPR/LDS;
  - instruction-family if available;
  - time share.

Pass:

- each tinygrad primitive label has an equivalent llama comparison row or is explicitly marked `tinygrad-only`.

## Phase E - Consolidation

Goal: update the source-of-truth map.

Work:

1. write `docs/amd-att-primitive-attribution-result-20260619.md`;
2. update `docs/inference-perf-measured-map-20260619.md` only if ATT changes a conclusion;
3. update `docs/README.md`;
4. preserve all claims with `[M]` measured, `[I]` inferred, or `[H]` hypothesis labels.

Final acceptable outcomes:

| Outcome | Meaning |
|---|---|
| `DECODE_LABEL_FOUND` | ATT identifies the concrete in-model MMVQ contract loss |
| `PREFILL_LABEL_FOUND` | ATT/PMC identifies the non-matmul residual |
| `ATT_NOT_DECISIVE` | ATT works, but does not add actionable explanation beyond PMC/timing |
| `NO_PRIMITIVE_LEFT` | remaining gap is distributed/project-level, not a bounded primitive |

## Expected Impact

This phase may not directly improve tok/s. Its value is narrowing the next build decision:

- if decode gets an ATT-backed contract label, fund the matching runtime/scheduler/MMVQ integration route;
- if prefill gets a high-share non-matmul label, fund that component;
- if neither gets a bounded label, stop primitive-local implementation and keep the source-of-truth at project-level
  integration/spec-decode.

