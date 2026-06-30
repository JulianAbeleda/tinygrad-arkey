# Claude Handoff: Active-Work Audit + Agnostic Search Scope (2026-06-30)

Status: consolidated current handoff for Claude. This supersedes and consolidates:

- `docs/amd-isa-active-surface-principles-audit-20260629.md`
- the speed-loss certainty tooling scope originally carried on `q6k-direct-route`
- the quant/shape/target agnostic search future-work scope originally carried on `q6k-direct-route`

Use this file first. The older docs remain as provenance.

## Executive Summary

The project is no longer blocked on the old "missing primitives" story. The active system already proved the core loop:

```text
route proposal -> route attribution -> correctness -> W==D/whole-prefill authority
-> attribution/ceiling -> promote/refute/rollback
```

Current actuals:

| area | actual state | decision |
|---|---|---|
| Decode Q4_K GEMV | Generated G3 LaneMap is speed-equivalent to owned at all measured contexts. | Keep/promote G3 as the generated route; owned stays rollback/reference. |
| Decode Q6_K direct route | Half-warp direct route is token-correct and route-bound, but regresses W==D by about 5-6%. | Keep default-off; do not re-chase without a new route premise. |
| Decode attention native ISA | Native generated tile is correct/route-bound and useful as infrastructure, but remains correct-not-fast and low-leverage for whole decode. | Do not make attention the active max-out target under current ceiling. |
| Prefill global pipe | `pipe_tm2_tn2` was promoted to default and is TIER_A through ctx8192. | Keep default; rollback with `PREFILL_GEMM_PIPELINE=0`. |
| Prefill role-selective pipe | Excluding saturated `ffn_gate_up` beats global pipe by 2.9-3.7% and old default by 11.7-23.4%; master commit `8278565c0` flips it to default. | Treat role-selective as current default; rollback with `PREFILL_PIPE_ROLE_SELECTIVE=0`. |
| Search architecture | Strong evidence discipline exists, but route flags and phase scripts are too one-off. | Consolidate into profile/manifest-driven search. |

Primary next task for Claude:

1. Do not reopen Q4_K G3, Q6_K direct, or broad decode-attention tuning.
2. Make the live search surface manifest/profile-driven.
3. Start the quant/shape/target agnostic search substrate so future work is not Qwen/gfx1100/Q4_K hardcoded.

## Source Citations

Load-bearing current artifacts and files:

| claim | citation |
|---|---|
| G3 speed-equivalent Q4_K route | `bench/amd-isa-backend-g3-weight-promotion/summary.md`, `bench/amd-isa-backend-g3-weight-promotion/latest.json`, scope `docs/amd-isa-g3-weight-promotion-hardening-scope-20260629.md` |
| Q6_K direct route refuted | `bench/amd-isa-backend-q6k-direct-speed/summary.md`, `bench/amd-isa-backend-q6k-direct-speed/latest.json`, scope `docs/amd-isa-q6k-direct-route-full-scope-20260629.md` |
| Attention two-kernel combine exhausted | `docs/decode-two-kernel-problem-audit-result-20260625.md`, `bench/qk-decode-two-kernel-problem-audit-20260625/decision.json` |
| Attention ceiling moved search away from attention | `bench/amd-isa-backend-decode-attention-ceiling/latest.json`, `bench/amd-isa-backend-decode-attention-ceiling/summary.md`, scope `docs/amd-isa-decode-attention-ceiling-audit-scope-20260629.md` |
| Native attention route/candidate ledger | `bench/qk-pure-search-loop/decode_attention_loop_ledger.jsonl`, `bench/amd-isa-backend-phase-n6/latest.json`, `bench/amd-isa-backend-phase-n7/latest.json` |
| Prefill global pipe promoted | `bench/qk-prefill-pipe-promotion/summary.md`, `bench/qk-prefill-pipe-promotion/latest.json`, `extra/qk_prefill_graph_gemm_route.py` |
| Prefill role-selective beats global | `bench/qk-prefill-pipe-role-selective/summary.md`, `bench/qk-prefill-pipe-role-selective/latest.json`, `extra/qk_prefill_pipe_role_selective.py` |
| Prefill role attribution and mechanism | `bench/qk-prefill-whole-role-attribution/summary.md`, `bench/qk-prefill-whole-role-attribution/latest.json`, `bench/qk-prefill-pipe-tm2-tn2-hardening/role_mechanism.json` |
| Active-surface principles audit provenance | `docs/amd-isa-active-surface-principles-audit-20260629.md` |
| Role-selective default flip | commit `8278565c0`, `extra/qk_prefill_graph_gemm_route.py`, `bench/qk-prefill-pipe-role-selective/latest.json` |

## Current Performance Actuals

### Decode

G3 promotion gate:

| ctx | owned tok/s | G3 BubbleBeam tok/s | result |
|---:|---:|---:|---|
| 512 | 103.79 | 103.93 | speed-equivalent |
| 1024 | 101.98 | 102.04 | speed-equivalent |
| 2048 | 99.56 | 99.74 | speed-equivalent |
| 4096 | 94.83 | 94.44 | speed-equivalent |

Source: `bench/amd-isa-backend-g3-weight-promotion/summary.md`.

Q6_K direct route:

| ctx | shipped coop baseline | direct candidate | delta |
|---:|---:|---:|---:|
| 512 | 103.63 | 97.35 | -6.06% |
| 1024 | 101.68 | 95.76 | -5.82% |
| 2048 | 99.21 | 94.19 | -5.06% |
| 4096 | 94.50 | 89.99 | -4.77% |

Token match and route-binding both passed. Speed failed. Source: `bench/amd-isa-backend-q6k-direct-speed/summary.md`.

Decode conclusion:

- Q4_K generated route is promoted/speed-equivalent.
- Q6_K direct route is refuted/default-off.
- Attention native route is correct-but-not-fast and low-leverage at whole-decode level.
- Practical decode route/kernel tuning is closed unless the model, quant mix, target GPU, or representation changes.

### Prefill

Global `pipe_tm2_tn2` default promotion:

| ctx | old default | global pipe | delta |
|---:|---:|---:|---:|
| 512 | 3598 | 4289 | +19.2% |
| 1024 | 3506 | 4095 | +16.8% |
| 2048 | 3253 | 3708 | +14.0% |
| 4096 | 2821 | 3137 | +11.2% |
| 8192 | 2234 | 2423 | +8.5% |

Source: `bench/qk-prefill-pipe-promotion/summary.md`.

Role-selective pipe, now default on master:

| ctx | old default | global pipe | role-selective | role-selective vs global |
|---:|---:|---:|---:|---:|
| 512 | 3593 | 4292 | 4434 | +3.3% |
| 1024 | 3492 | 4092 | 4236 | +3.5% |
| 2048 | 3259 | 3708 | 3846 | +3.7% |
| 4096 | 2779 | 3083 | 3192 | +3.5% |
| 8192 | 2266 | 2461 | 2532 | +2.9% |

Correctness-equivalent; max run spread 0.3%. Source: `bench/qk-prefill-pipe-role-selective/summary.md`.

Prefill conclusion:

- Global pipe is already a promoted TIER_A default.
- Role-selective pipe is now the promoted default because global pipe helped under-saturated roles but hurt saturated
  `ffn_gate_up`; role-selective excludes that role and beats global.

## Active-Work Principles Audit

The active path is mostly principled:

| principle | current grade | current read |
|---|---:|---|
| Whole-primitive measurement | A- | N4 and prefill role attribution prevent optimizing isolated kernels blindly. |
| Audit-before-build | A | N1B, Q6K-3, and attention ceiling stopped bad directions after measurement. |
| Correctness/fallback discipline | A- | Token/logit gates and route-bound checks are consistent; candidate fallbacks must stay strict. |
| Centralized authority | C+ | Results are authority-backed, but route/env contracts are copied across scripts. |
| Modularity/orthogonality | B- | Kernel work is mostly contained; compile/cache/grid/profiling policy is still intertwined. |
| Tiny / anti-sprawl | C | There are many one-off phase tools; useful as provenance, weak as durable interface. |
| Invariant encoding | B- | Artifacts are strong; manifests/profiles are still not first-class enough. |
| Profiling containment | B | PMC fallback is usable; ATT/SQTT per-PC remains a known infrastructure wall. |

High-value findings to preserve:

| id | finding | action |
|---|---|---|
| A1 | Proven levers are still partly encoded as flags/scripts instead of a route manifest. | Build a route/candidate manifest and make gates import it. |
| A2 | Do not keep treating register accumulators, scalar address N1B, or scheduler-only changes as open wins. | Put them in `do_not_search` / refuted-axis ledger unless new evidence changes. |
| A3 | Dynamic-S and other route choices should be positive candidate names, not negative/absence flag contracts. | Replace "absence of fixed-S" style semantics with named routes. |
| A4 | Per-kernel owner taxonomy and token/fallback/W==D gates are duplicated. | Consolidate into a table-driven candidate evaluator. |
| A5 | PMC category attribution is the active profiler; per-PC ATT/SQTT is deferred. | Do not block future search on per-PC tracing. Add differential probes as fallback proof. |
| A6 | Role-selective prefill is now the live high-value surface. | Harden and promote it before broader speculative work. |

## Required Decoupling

### 1. Route Manifest

Add or extend a manifest that describes route candidates independent of ad hoc env maps.

Minimum schema:

```json
{
  "route_id": "prefill_pipe_role_selective",
  "workload": "prefill",
  "profile_id": "qwen3_8b_q4_k_m_gfx1100",
  "env": {"PREFILL_GEMM_PIPELINE": "1", "PREFILL_PIPE_ROLE_SELECTIVE": "1"},
  "strict_fallback": true,
  "rollback": {"PREFILL_PIPE_ROLE_SELECTIVE": "0"},
  "expected_roles": ["attn_qo", "attn_kv", "ffn_down"],
  "excluded_roles": ["ffn_gate_up"],
  "authority_gate": "extra/qk_prefill_whole_synced.py"
}
```

Minimum route IDs:

- `decode_q4k_g3_generated`
- `decode_q6k_coop_shipped`
- `decode_q6k_direct_refuted`
- `decode_attention_owned_two_kernel`
- `decode_attention_native_dynamic_s_correct_not_fast`
- `prefill_pipe_global_default`
- `prefill_pipe_role_selective_default`

### 2. Candidate Evaluator

Collapse duplicated gates into one table-driven evaluator.

Required fields per candidate:

| field | meaning |
|---|---|
| `candidate_id` | Stable route id. |
| `profile_id` | Model/quant/GPU/workload profile. |
| `correctness_gate` | Token/logit exactness, with sampling noise handled explicitly. |
| `route_gate` | Expected kernels/roles fire; forbidden fallbacks absent. |
| `authority_gate` | Decode W==D or whole-prefill synced harness. |
| `threshold_policy` | TIER_A/TIER_B/TIER_C thresholds. |
| `counter_gate` | Optional PMC/PC/source attribution. |
| `rollback` | Exact env or code rollback. |
| `verdict` | Promote/refute/defer/inconclusive. |

### 3. Refutation Ledger

Add durable rows for these closed/refuted paths:

| candidate/axis | disposition | citation |
|---|---|---|
| Q6_K direct half-warp route | refuted: W==D regression | `bench/amd-isa-backend-q6k-direct-speed/latest.json` |
| Q4_K offline layout reshuffle | deprioritized: G3 matches owned | `bench/amd-isa-backend-g3-weight-promotion/search_space_update.json` |
| Attention combine/fused lifecycle | exhausted/low-leverage | `docs/decode-two-kernel-problem-audit-result-20260625.md` |
| Native attention as default | correct-not-fast | `bench/amd-isa-backend-phase-n7/latest.json`, `bench/qk-pure-search-loop/decode_attention_loop_ledger.jsonl` |
| N1B scalar address path | refuted/dead/faulting | `bench/amd-isa-backend-phase-n1b/latest.json` |
| Occupancy/LDS-only attention tuning | refuted/no W==D movement | `bench/amd-isa-backend-phase-m/latest.json` if present; otherwise Phase M notes in current handoff |
| Scheduler-only attention tuning | small/no movement | `bench/amd-isa-backend-phase-k/latest.json` |

### 4. Profiler Boundary

Current supported profiler path:

- PMC/category counters and route/kernel attribution are usable.
- Static PC/source estimate exists.
- True per-PC ATT/SQTT instruction timing is not reliable under the current HCQ path.

Do not block promotion or candidate selection on per-PC ATT/SQTT. Use it only if repaired. Use differential probes for
certainty when per-PC is blocked.

## Speed-Loss Certainty Tooling Scope

Keep two complementary tracks.

### Track A: Per-PC Trace Repair

Goal:

```text
PC -> asm -> source_group -> measured cycles/stalls/events
```

Add or maintain:

```text
extra/amd_isa_per_pc_trace_repair.py
bench/amd-isa-backend-per-pc-trace-repair/latest.json
bench/amd-isa-backend-per-pc-trace-repair/summary.md
bench/amd-isa-backend-per-pc-trace-repair/capability_matrix.json
```

Verdicts:

```text
AMD_ISA_PER_PC_TRACE_PASS_PC_ROWS
AMD_ISA_PER_PC_TRACE_BLOCKED_HCQ_NO_PROFILED_AQL
AMD_ISA_PER_PC_TRACE_BLOCKED_DECODER_NO_INSTRUCTIONS
AMD_ISA_PER_PC_TRACE_BLOCKED_NO_EXTERNAL_ROCPROF_PATH
AMD_ISA_PER_PC_TRACE_PASS_BLOCKER_PINNED_WITH_PMC_FALLBACK
```

This track is useful, but it is not on the critical path while PMC and differential probes are enough to choose
candidate families.

### Track B: Differential Source Probes

Goal:

```text
If this source group is removed/reduced/replaced, does W==D or kernel GPU time move?
```

Add or maintain:

```text
extra/amd_isa_differential_source_probes.py
bench/amd-isa-backend-differential-source-probes/latest.json
bench/amd-isa-backend-differential-source-probes/summary.md
bench/amd-isa-backend-differential-source-probes/probe_matrix.json
```

Probe rows must include:

| field | meaning |
|---|---|
| `probe_id` | Stable id. |
| `source_group` | The suspected source group. |
| `probe_type` | measurement-only / semantic-preserving / semantic-masking / microkernel proxy. |
| `env_or_patch` | Exact env or patch. |
| `correctness_required` | Whether token/logit match is required. |
| `baseline_metrics` | W==D, kernel time, PMC, PC/source rows. |
| `probe_metrics` | Same metrics under probe. |
| `movement` | Delta. |
| `interpretation` | live lever / refuted / inconclusive. |
| `next_action` | optimize / defer / fix tooling. |

Initial probes:

| probe | purpose | disposition rule |
|---|---|---|
| `PREFILL_ROLE_SELECTIVE` | global pipe vs role-selective pipe | already live and default on master; keep as canonical prefill route |
| `ATTN_REG_ACCUM_LDS` | register accumulator / LDS traffic | already banked but not enough for attention promotion |
| `ATTN_WAITCNT_THRESHOLDS` | waits as speed loss | continue only if >=2% W==D |
| `ATTN_ADDRESS_IV` | induction-variable address update | defer if <2% wall |
| `ATTN_LOCAL_FMA_MOV` | local arithmetic/move cleanup | refute if <1% |
| `Q6K_DIRECT_VARIANTS` | alternative Q6_K route topology | only reopen with a different topology than refuted half-warp |

## Agnostic Search Architecture

The next abstraction is:

```text
QuantSpec + ShapeSpec + TargetSpec + RuntimeContext
  -> ProblemInstance
  -> RouteCandidate[]
  -> Cost/Floor Model
  -> Measurement Gate
  -> Amdahl Residual Model
  -> Search Decision
  -> Promotion Gate
```

### QuantSpec

Defines legal math and storage, not the winning route.

Required fields:

```text
name
family
bits_per_weight
values_per_block
block_bytes_total
weight_payload_bytes
metadata_bytes
scale_layout
zero_or_min_layout
group_size
packing_order
signedness
unpack_ops_per_value
dequant_ops_per_value
legal_accum_dtypes
preferred_dot_dtype
quality_class
requires_exact_semantics
```

Examples:

- `Q4_K`: generated G3 currently wins on gfx1100/Qwen3-8B decode.
- `Q6_K`: shipped coop currently beats the tested half-warp direct route.
- Future: `Q4_0`, `Q5_K`, AWQ/GPTQ int4, fp8, q8 variants.

### ShapeSpec

Defines the role and dimensions.

Required fields:

```text
role
M
N
K
batch
ctx
heads
kv_heads
head_dim
vocab
ffn_dim
input_dtype
output_dtype
residual_dtype
is_decode
is_prefill
is_lm_head
```

Known roles:

- `ffn_gate_up`
- `ffn_down`
- `attn_qo`
- `attn_kv`
- `lm_head`
- `attention_tile`
- `attention_reduce_combine`
- `norm_rope`

### TargetSpec

Defines the GPU/backend capabilities.

Required fields:

```text
vendor
arch
backend
wave_or_warp_size
num_cu_or_sm
peak_hbm_bw
measured_stream_bw
shared_mem_or_lds_per_cu
register_file_limits
max_waves_or_warps
dot_primitives
shuffle_primitives
barrier_primitives
global_load_shapes
vector_load_shapes
supported_native_dtypes
compiler_control_level
profiler_available
```

Examples:

- `gfx1100`: wave/lane routes, `ds_bpermute`, `v_dot2`, native AMD ISA backend, PMC available, ATT per-PC blocked.
- NVIDIA: warp 32, shuffle, tensor cores/PTX; SASS ownership is harder.
- Metal: SIMD-group primitives and MSL compiler control; different profiling/tooling constraints.

### RouteCandidate

Required fields:

```text
candidate_id
quant_spec_id
shape_spec_id
target_spec_id
route_family
kernel_or_codegen_path
env
expected_kernels
forbidden_kernels
rollback
known_refutations
expected_floor
promotion_threshold
```

Route families:

- lane-map GEMV
- coop partial + reduce
- direct warp/wave reduce
- graph-GEMM pipeline
- role-selective graph-GEMM pipeline
- owned-oracle fallback
- native ISA attention tile

### PromotionDecision

Required outputs:

```text
PROMOTE
PROMOTE_TIER_B
CORRECT_BUT_NOT_FAST
REFUTED_REGRESSION
REFUTED_LOW_CEILING
DEFER_TOOLING
DEFER_PROFILE_NOT_SUPPORTED
```

## Claude Execution Plan

### Phase 1: Make The Current Live Surface Manifest-Driven

Deliverables:

- route manifest with the route IDs listed above;
- helper to materialize env maps and strict fallback policy;
- evaluator helper shared by decode and prefill gates;
- refutation ledger seeded from current artifacts.

Acceptance:

- no default behavior changes;
- existing G3, Q6K, prefill pipe, and role-selective gates can read the manifest;
- policy check passes;
- docs cite the manifest as the source of route truth.

### Phase 2: Validate The Role-Selective Default In The Manifest

Role-selective is already default on master. This phase should make the manifest/evaluator represent that fact, not
re-decide the promotion.

- run synced whole-prefill authority across 512/1024/2048/4096/8192;
- prove logit/token equivalence;
- prove route attribution excludes `ffn_gate_up` from pipe and keeps pipe for `attn_qo`, `attn_kv`, `ffn_down`;
- verify rollback with `PREFILL_PIPE_ROLE_SELECTIVE=0`.

Acceptance:

- >=2% over global pipe at every context or clearly tiered residual pass;
- no protected-context regression >1%;
- rollback flag documented.

### Phase 3: Build The Agnostic Profile Schema

Deliverables:

```text
bench/qk-audit-profiles/profiles.json
extra/qk_audit_profile.py
extra/qk_route_manifest.py
extra/qk_candidate_evaluator.py
```

First profiles:

- `qwen3_8b_q4_k_m_gfx1100_decode`
- `qwen3_8b_q4_k_m_gfx1100_prefill`

Acceptance:

- existing hardcoded context ladders and thresholds can be loaded from profile;
- route candidates can declare quant/shape/target assumptions;
- no benchmark values change.

### Phase 4: Optional Profiling Tooling

Do this after Phase 1-3 unless a candidate is ambiguous:

- per-PC trace repair, if possible;
- differential probes, if per-PC remains blocked.

Acceptance:

- blocker matrix or usable PC rows;
- PMC fallback preserved;
- probe matrix classifies each suspected speed-loss row.

## Non-Actions

- Do not reopen Q4_K layout reshuffle while G3 parity holds.
- Do not promote Q6_K direct half-warp route; it regressed.
- Do not chase decode-attention combine/fused lifecycle as current max-out work.
- Do not require ATT/SQTT per-PC tracing for promotion.
- Do not add another one-off phase script if a manifest/evaluator row can express the gate.
- Do not hide fallback under candidate routes; shipped runtime fallback is fine, candidate fallback must be fail-loud.

## Claude Prompt

Use this prompt when handing off:

```text
Read docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md completely.

Task: implement Phase 1 only unless explicitly told otherwise:
make the current live route/search surface manifest-driven without changing default behavior.

Required citations:
- bench/amd-isa-backend-g3-weight-promotion/summary.md
- bench/amd-isa-backend-q6k-direct-speed/summary.md
- bench/qk-prefill-pipe-promotion/summary.md
- bench/qk-prefill-pipe-role-selective/summary.md
- docs/decode-two-kernel-problem-audit-result-20260625.md
- bench/amd-isa-backend-decode-attention-ceiling/latest.json

Constraints:
- no default flips in Phase 1; role-selective prefill is already default on master;
- no new optimization kernels;
- no broad retuning;
- no hidden fallback in candidate gates;
- do not reopen refuted routes;
- preserve rollback flags;
- keep the implementation tiny and table-driven.

Deliver:
1. route manifest/helper;
2. evaluator helper or first consolidation point;
3. seeded refutation ledger;
4. updated docs/README.md link;
5. policy/compile checks;
6. commit with artifacts.
```
