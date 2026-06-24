# Decode FFN-GEMV Scheduler Diagnostic — Scope

Date: 2026-06-22

Status: `FFN_GEMV_SCHEDULER_DIAGNOSTIC_SCOPE_READY`

Follow-on to `docs/decode-time-tax-audit-result-20260622.md`.

The Route B attention lane is now closed for default-promotion work:

- B3 owned decode-attention tile: local pass.
- B4 graph-node route: integration pass.
- B4 split policy: no policy clears gate.
- B5 combine: local combine pass, W==D saturates around +5.7%@ctx4096.

The measured time-tax audit names the next primitive:

```text
NEXT_PRIMITIVE_Q4K_GEMV_SCHEDULER
```

This scope is **diagnostic only**. It must explain the FFN Q4_K/Q6_K GEMV tax before any scheduler/codegen work is
funded.

## Objective

Determine why the FFN weight GEMVs remain the dominant decode tax and whether there is a **bounded, lossless**
scheduler/codegen lever before committing to deep AMD backend work.

Specifically:

- Attribute the flat FFN gate/up `q4k_gemv_partial_12288_4096` cost.
- Attribute the FFN down `q4k_gemv_partial_4096_12288` / `q6k_coop_partial_4096_12288` cost.
- Compare in-graph tinygrad behavior to llama/reference behavior where available.
- Decide whether the next action is a bounded GEMV schedule/codegen patch, q8 lifecycle hardening, or rest.

## Why This Is Next

From `docs/decode-time-tax-audit-result-20260622.md`:

| bucket | ctx1024 | ctx4096 | meaning |
|---|---:|---:|---|
| FFN gate/up | 24% | 22% | single biggest primitive kernel group |
| FFN down | 14% | 12% | same GEMV family |
| FFN activation | 9% | 8% | work-conserved / fusion previously closed |
| attention compute | 15% | 24% | Route B attention closed by B5 saturation |
| lm_head | 4% | 4% | too small |

Aggregate FFN is about **48%@ctx1024 / 42%@ctx4096**. The FFN weight GEMVs alone are about **38%@ctx1024**.

The decisive transfer test:

- `Q8_FFN_HANDWRITTEN=1` gives about **+6% W==D** and therefore proves FFN gate/up is on the critical path.
- B5 attention improves local combine substantially but W==D saturates, proving attention is no longer the next
  default-promotable primitive.

## Must-Read Inputs

Read these first:

| file | why |
|---|---|
| `docs/current-project-state-handoff-20260621.md` | canonical current state |
| `docs/README.md` | doc map and supersession |
| `structure/Development/session-handoff.md` | compact current handoff |
| `structure/Development/performance-primitive-research-principles.md` | primitive methodology and split-KV lesson |
| `docs/decode-time-tax-audit-result-20260622.md` | source of this scope's verdict |
| `bench/qk-decode-time-tax-audit/latest.json` | exact bucket timings and toggles |
| `docs/b4-cheaper-combine-result-20260622.md` | attention closure / B5 saturation |
| `docs/q8-ffn-handwritten-a4-decode-result-20260619.md` | q8 FFN measured W==D transfer |
| `docs/q8-ffn-dynamic-scheduler-observability-result-20260619.md` | prior q8 scheduler attribution |
| `docs/q8-ffn-route-a-scheduler-codegen-result-20260619.md` | prior native scheduler/codegen closure |
| `docs/qk-mmvq-int-dot-closeout-20260618.md` | Q4_K int-dot path closure |
| `docs/llama-q4k-mmvq-scheduler-audit-20260618.md` | llama scheduler decomposition |
| `docs/llama-rocm-gemv-primitive-audit-20260617.md` | llama ROCm GEMV primitive reference |
| `docs/llama-kernel-residual-primitive-audit-20260619.md` | llama residual headroom |
| `extra/qk_decode_time_tax_audit.py` | current time-tax harness |
| `extra/qk_decode_eval.py` | evaluator ladder |
| `extra/qk_b4_decode_eval.py` | example of route-firing and W==D timing discipline |
| `extra/q4_k_gemv_primitive.py` | Q4_K GEMV primitive implementation |
| `extra/q6_k_gemv_primitive.py` | Q6_K GEMV primitive implementation |
| `tinygrad/llm/model.py` | decode routing and FFN callsites |

## Boundaries

Do **not**:

- optimize a kernel in this task;
- build a new GEMV primitive in this task;
- start deep Route-A/AMD backend codegen;
- reopen Q4_K int-dot MMVQ without new bounded evidence;
- promote q8 or make it default;
- change defaults;
- run broad search before naming the failing layer.

Allowed:

- add diagnostic harnesses;
- disassemble existing kernels;
- compare resource/ISA/launch geometry;
- run targeted microbenchmarks for current primitives;
- produce artifacts and a result doc;
- scope a follow-on implementation only if the diagnostic names a bounded lever.

## Primary Questions

1. Is the Q4_K FFN GEMV gap caused by memory bandwidth, q4 unpack/dequant, reduction shape, occupancy, instruction
   mix, wait/scheduling, or graph lifecycle?
2. Why does q8 FFN transfer to W==D while the lossless Q4_K path remains below llama/reference quality?
3. Is the residual ~57% to ~70% peak gap a bounded scheduling issue or a broad backend/codegen project?
4. Is FFN down the same issue as gate/up, or a different Q4_K/Q6_K blend?
5. Is there a lossless bounded lever with projected W==D >= +5%@ctx1024?

## Phase G0 — Baseline And Role Inventory

Reproduce the time-tax audit and extract FFN role kernels.

Commands:

```sh
cd /home/ubuntu/tinygrad-arkey
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_time_tax_audit.py
```

Record:

- default tok/s by ctx;
- `q4k_gemv_partial_12288_4096` timing;
- `q4k_gemv_partial_4096_12288` timing;
- `q6k_coop_partial_4096_12288` timing;
- `E_49152` / FFN activation timing;
- q8 toggle W==D if available from artifact.

Output a role table:

| role | kernel name | shape | ms/token | share | dtype/quant | existing alternative |
|---|---|---:|---:|---:|---|---|
| gate/up | `q4k_gemv_partial_12288_4096` | 12288x4096 | | | Q4_K | q8 route |
| down | `q4k_gemv_partial_4096_12288` | 4096x12288 | | | Q4_K | q8 route / q6 mix |
| down/q6 | `q6k_coop_partial_4096_12288` | 4096x12288 | | | Q6_K | none |

Stop if baseline drifts materially from the time-tax audit.

## Phase G1 — Existing Primitive Anatomy

Audit current Q4_K/Q6_K GEMV primitive structure.

For each role, record:

- rows/cols;
- workgroup geometry;
- threads per row;
- rows per block;
- K split;
- reduction location;
- output write shape;
- q4/q6 unpack path;
- scale/min decode path;
- vector/dot instruction use;
- VGPR/SGPR/LDS/private segment;
- occupancy proxy;
- static instruction mix;
- memory access coalescing hypothesis.

Use:

- source inspection;
- generated kernel names;
- `llvm-objdump` / descriptor metadata where possible;
- existing artifact metadata if already produced.

Required artifact fields:

```text
role
kernel_name
shape
grid
block
workgroups
vgpr
sgpr
lds
spill
dot_instruction_count
memory_instruction_count
branch_or_wait_markers
estimated_bytes
measured_us
effective_bandwidth_or_ops
```

## Phase G2 — Llama/Reference Gap Audit

Use existing llama docs/artifacts first. Do not port new llama code unless the existing references are insufficient.

Questions:

- What is llama's corresponding FFN gate/up/down primitive for Qwen3-8B Q4_K_M?
- What shape decomposition does llama use?
- Does llama use q4 MMVQ, q8 activation, different scheduling, or different row grouping?
- What measured throughput/percentage of peak does llama achieve at these roles?
- Is tinygrad's ~57% vs llama/reference ~70% gap role-specific or global?

Output:

| role | tinygrad timing | llama/reference timing | gap | likely cause | evidence |
|---|---:|---:|---:|---|---|

If llama reference is not precise enough, classify `REFERENCE_GAP_INSUFFICIENT` and scope a separate bounded reference
capture. Do not guess.

## Phase G3 — Controlled Variant Ladder

Run only diagnostic variants that already exist or are trivial controls.

Minimum controls:

- default Q4_K route;
- q8 FFN route (`Q8_FFN_HANDWRITTEN=1`);
- any existing q4/q6 primitive debug mode that isolates gate/up/down;
- storage/layout controls if already implemented.

For each:

- local role timing;
- W==D if already available or cheap;
- correctness/dNLL where required;
- Amdahl projection vs observed W==D.

Purpose:

- prove whether a local FFN GEMV role speedup transfers;
- avoid repeating the attention mistake where local speedup saturated in graph.

## Phase G4 — Failure Classification

Classify the Q4_K/Q6_K FFN GEMV gap into exactly one primary class:

| class | meaning | next action |
|---|---|---|
| `GEMV_SCHEDULE_BOUND` | work decomposition / rows/block / K split / occupancy is the visible issue | scope bounded scheduler variant |
| `GEMV_MEMORY_COALESCING_BOUND` | loads/unpack layout underuse memory path | scope layout/load-shape variant |
| `GEMV_INSTRUCTION_SELECTION_BOUND` | missing useful dot/vector form, but prior int-dot closure must be respected | scope only if new evidence differs from closed MMVQ path |
| `GEMV_WAIT_SCHEDULER_BOUND` | waits/issue ordering dominate like prior q8 scheduler evidence | likely deep backend; no bounded patch unless narrow |
| `GEMV_ACTIVATION_LIFECYCLE_BOUND` | q8/activation side-channel lifecycle dominates | scope q8 lifecycle hardening |
| `GEMV_BACKEND_PROJECT_LEVEL` | no bounded lever; deep codegen/backend project only | recommend q8 opt-in as practical cap |
| `GEMV_REFERENCE_GAP_INSUFFICIENT` | llama/reference data is not enough | scope reference capture |

## Phase G5 — W==D Headroom And Recommendation

For each plausible lever, compute:

```text
projected_WD_gain = bucket_share * (1 - 1 / local_speedup)
```

Use both:

- gate/up-only;
- gate/up + down combined.

Required recommendation table:

| lever | local target | affected share | projected W==D | bounded? | verdict |
|---|---:|---:|---:|---|---|
| q8 FFN hardening | measured | gate/up | ~+6% measured | yes/no | |
| lossless gate/up schedule | 1.2x | 24% | ~+4% | TBD | |
| lossless gate/up+down schedule | 1.2x | 38% | ~+6% | TBD | |

## Deliverables

Required:

- `extra/qk_ffn_gemv_scheduler_diagnostic.py`
- `bench/qk-ffn-gemv-scheduler-diagnostic/latest.json`
- `docs/decode-ffn-gemv-scheduler-diagnostic-result-20260622.md`

Optional if a follow-on implementation is justified:

- `docs/decode-ffn-gemv-scheduler-implementation-scope-20260622.md`

Update current handoff docs after result:

- `structure/Development/session-handoff.md`
- `docs/README.md`

## Result Doc Requirements

The result doc must answer:

1. Which exact FFN GEMV role is the top tax?
2. Is the gap schedule, memory/coalescing, instruction, wait/scheduler, activation lifecycle, or project-level backend?
3. Does the q8 transfer test imply the lossless GEMV lever will transfer?
4. Is there a bounded implementation scope?
5. What should not be pursued?

Final verdict must be one of:

- `FFN_GEMV_DIAGNOSTIC_BOUNDED_SCHEDULE_SCOPE_READY`
- `FFN_GEMV_DIAGNOSTIC_Q8_LIFECYCLE_SCOPE_READY`
- `FFN_GEMV_DIAGNOSTIC_REFERENCE_CAPTURE_NEEDED`
- `FFN_GEMV_DIAGNOSTIC_BACKEND_PROJECT_LEVEL`
- `FFN_GEMV_DIAGNOSTIC_BLOCKED`

## Expected Outcome

Do not end with "deep codegen" by default. The useful result is a named failure layer.

If a bounded FFN GEMV scheduler lever exists, scope it next. If not, bank the diagnosis and treat q8 FFN as the
practical opt-in cap while the lossless path becomes project-level AMD backend/codegen work.
