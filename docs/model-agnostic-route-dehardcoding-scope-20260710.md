# Model-Agnostic Route Dehardcoding Scope

Date: 2026-07-10.

## Decision

Prepare 14B by removing model-size constants from live route logic and moving them into reusable shape/profile data.

Do not build a new harness family. Reuse the existing route assets:

- `extra/qk/runtime_specs.py` for operation/shape/role descriptors.
- `extra/qk/generated_candidates.py` for generated candidate selection.
- `extra/qk/route_manifest.py` for route inventory, provenance, and current authority status.
- `tinygrad/llm/route_policy.py` for policy loading and per-shape route selection.
- existing gates: `prefill_14b_model_authority_gate.py`, `prefill_14b_q6_decision_gate.py`,
  `q4k_wmma_tiled_*_gate.py`, `prefill_whole_synced.py`, and `hybrid_machine_search.py`.

The goal is not generic performance for every model immediately. The goal is that 8B and 14B use the same route/spec
machinery, with model-specific facts supplied as data rather than code constants.

## Current Classification

| Surface | Current state | Problem | Target |
|---|---|---|---|
| GGUF/model load | Mostly config-driven from GGUF metadata | Some architecture aliases are expected and OK | Keep as-is unless a route reads model names |
| Decode attention route | Structural class `B=1,Hd=128,Hkv=8,Hq%Hkv==0` | Manifest still has separate 8B/G4 and 14B/G5 rows | Keep one implementation; make shape-profile rows data |
| Q4_K decode G3 | Has broad structural path plus older 8B shape pin | Mixed old exact shapes with generic eligibility | Retain generic eligibility and move exact positive controls to policy/tests |
| Q6_K decode | Mostly generated/spec-driven, but coop decision has exact 8B down shape | Exact `4096x12288` in live route | Convert to role/shape profile rules |
| Direct-packed prefill | Mostly runtime `m,n,k,role,quant` driven | Default opts carry 14B comments and global assumptions | Keep route generic; move tuned opts to policy/profile rows |
| Q4_K/Q8_1 tiled WMMA 14B | Explicit `QWEN3_14B_Q4K_ROLE_SHAPES` constant | Research gate cannot generalize to 8B/32B or policy rows | Replace with profile-derived role-shape provider |
| `hybrid_machine_search` | Hardcoded 8B `ffn_gate_up = 512x12288x4096` | Good 8B candidate serializer, not reusable for 14B | Parameterize candidate rows over a role-shape profile |
| codegen extensions | Uses shape constants `{1024,4096}` and `12288` | Route behavior hidden in codegen policy | Feed decisions from route/spec metadata or named shape profile |

## Principles

1. Runtime route code must decide from quant, role, shape, device, and explicit policy. It must not infer "8B" or "14B"
   from literal dimensions except through a named compatibility table.
2. Research gates may pin a model profile, but that profile must be a data row consumed by the same generic code path.
3. Route authority must stay single-sourced through `route_manifest.py` plus policy/gate artifacts. Do not add another
   manifest.
4. Existing harnesses remain canonical. If a missing input is needed, add a data adapter, not a replacement harness.
5. Exact old shape constants should survive as regression fixtures, not as route logic.

## Reused Assets

| Need | Reuse |
|---|---|
| Runtime operation descriptor | `RuntimeOpSpec` |
| Candidate inventory | `GeneratedCandidateRegistry` / `builtin_registry()` |
| Route status/provenance | `route_manifest.ROUTES` |
| Per-shape selection | `QK_ROUTE_POLICY` rows loaded by `route_policy.load_qk_route_policy()` |
| 14B Q4 role-shape validation | `prefill_14b_model_authority_gate.py` and `q4k_wmma_tiled_*_gate.py` |
| 14B Q6 residual decision | `prefill_14b_q6_decision_gate.py` |
| 8B hybrid backend-atom search | `hybrid_machine_search.py` |
| E2E timing | `prefill_whole_synced.py` |

## New Shared Data Shape

Add one small data module under `extra/qk`, not under `tinygrad/`:

```text
extra/qk/model_profiles.py
```

It should expose data-only helpers:

```text
ModelProfile(id, family, size_label, quant, device_profile, roles, attention)
LinearRoleShape(role, phase, quant, M, N, K, tensor_patterns)
AttentionShape(B, Hq, Hkv, Hd)
profile_from_transformer_config(config, *, quant, device_profile)
prefill_role_shapes(profile)
attention_shape(profile)
```

Initial rows:

- Qwen3 8B Q4_K_M gfx1100: current 8B role shapes.
- Qwen3 14B Q4_K_M gfx1100: current `QWEN3_14B_Q4K_ROLE_SHAPES`.
- Qwen3 32B Q4_K_M gfx1100: only add if already evidenced by manifest/artifacts; otherwise leave as derived/admitted
  structural attention only.

Rows must be data, not behavior. Route logic may consume them only through policy/gate interfaces.

## Work Slices

### A. Profile Data Extraction

Move model role shapes out of ad hoc constants into `extra/qk/model_profiles.py`.

Convert:

- `extra/qk/q4k_wmma_tile_lowering.py::QWEN3_14B_Q4K_ROLE_SHAPES`
- `extra/qk/prefill/hybrid_machine_search.py::SHAPE`
- test fixtures that duplicate these exact role shapes

Done when those callers accept a profile/role-shape iterator and the old constants are compatibility aliases only.

### B. Route Policy Shape Rows

Extend existing `QK_ROUTE_POLICY` support to represent prefill direct/tiled candidates by route id and shape.

Do not add a second policy file format. Use existing `boltbeam.route_policy.v1` rows and `_SUPPORTED_QK_ROUTE_IDS`.

Target route ids:

- `prefill_q4k_direct_tile4x4_default`
- `prefill_q6k_direct_generated`
- `prefill_q4k_int8_wmma_tiled_research`
- later, a Q6_K MMQ route only if the Q6 decision gate proves it matters

Done when `prefill_14b_model_authority_gate.py` can check loaded policy rows without importing a Qwen-named constant.

### C. Hybrid Search Parameterization

Generalize `hybrid_machine_search.py` from one 8B candidate into a candidate serializer over `LinearRoleShape`.

Do not change the current 8B default candidate behavior. The default CLI may still write the 8B ffn_gate_up artifact,
but it should do so by selecting the 8B profile row.

Add a `--profile` and `--role` path only if existing tests can cover it without GPU.

Done when:

- 8B output is byte/stable except schema path/name changes already committed.
- 14B candidate rows can be serialized without pretending authority passed.
- Unsupported 14B backend-atom candidates say blocked with exact reason.

### D. Q4_K/Q8_1 Tiled WMMA 14B Gates

Replace `describe_qwen3_14b_q4k_full_role_lowering()` with a profile-driven function:

```text
describe_q4k_full_role_lowering(profile, *, wmma_surface)
```

Keep `describe_qwen3_14b_q4k_full_role_lowering()` as a compatibility wrapper until tests and docs stop importing it.

Done when the existing gates still report:

- tiled lowering feasible,
- lifecycle pass,
- role-shape execution blocked on `scheduler_owned_tile_loop_missing`,
- no hand-kernel gate pass.

### E. Decode Route Shape Cleanup

Keep the live-split attention implementation structural. Clean the naming around route ids and docs so 8B/G4 and 14B/G5
are profile rows over one implementation, not separate hidden systems.

Do not rename promoted route ids in this pass unless the manifest and policy compatibility layer preserve old ids.

For Q4/Q6 decode:

- Preserve exact 8B/14B positive controls as tests.
- Make the runtime branch use structural predicates or policy profile rows.
- Any route-policy strict fallback must name the selected row and the structural predicate that failed.

### F. Codegen Extension Hooks

Move shape-specific decisions in `extra/qk/codegen_extensions.py` behind a profile/spec query.

Current literals:

- `{1024, 4096}` pipe dimensions
- `12288` FFN dimension
- implicit `M=512` ubatch shape

Done when codegen hooks receive a named route/spec context or consume a data helper. If that plumbing is too large, add a
single data helper in `extra/qk/model_profiles.py` and mark it as transitional.

## Stop Conditions

Stop and report if:

- a runtime path needs model name or filename to select a kernel,
- a new policy format is being invented,
- an existing authority gate is bypassed instead of reused,
- a 14B candidate is promoted without same-regime comparator and quality evidence,
- exact shape constants move from one code file to another without becoming profile data.

## Acceptance Gates

Required CPU/no-GPU checks:

```bash
python3 -m pytest test/unit/test_runtime_specs.py
python3 -m pytest test/unit/test_prefill_hybrid_machine_search.py
python3 -m pytest test/unit/test_prefill_14b_policy_gates.py
python3 -m pytest test/unit/test_pure_search_guard_boundary.py
python3 extra/tools/check_doc_links.py
MAX_LINE_COUNT=28000 python3 sz.py
```

Required route/gate checks after implementation slices:

```bash
python3 extra/qk/q4k_wmma_full_role_contract_gate.py
python3 extra/qk/q4k_wmma_tiled_role_shape_exec_gate.py
python3 extra/qk/prefill_14b_model_authority_gate.py
python3 extra/qk/prefill_14b_q6_decision_gate.py
```

GPU authority remains deferred until a 14B policy row selects candidates and CPU gates classify them as runnable.

## Parallelization Plan

These can run in parallel after the profile data shape is agreed:

| Lane | Scope | Files |
|---|---|---|
| profile-data | add `model_profiles.py`, tests, compatibility aliases | `extra/qk/model_profiles.py`, runtime spec tests |
| 14B tiled gates | consume profile role shapes | `q4k_wmma_tile_lowering.py`, `q4k_wmma_*_gate.py`, 14B tests |
| hybrid search | parameterize candidate serializer | `prefill/hybrid_machine_search.py`, its tests |
| route policy | add prefill route ids and policy validation | `tinygrad/llm/route_policy.py`, 14B policy gate tests |
| codegen hooks | replace literals with transitional helper | `extra/qk/codegen_extensions.py`, compile-capture tests |

Sequence constraint: profile-data lands first; route-policy and gate lanes can follow independently.
