# Model Route Plan Implementation Scope

## Problem

QK dispatch is now candidate-based at execution time, but model load still has a legacy policy layer:

- primitive install calls `q4k_policy(name)` / `q6k_policy(name)` when no generated policy is supplied;
- those policy functions infer route choices from tensor-name substrings;
- role inference also appears in prefill/decode helpers;
- model profiles exist for search and gates, but runtime loading does not yet produce a single route-facts object.

The remaining failure mode is not an 8B/14B branch in the hot route anymore. It is a hidden name-policy authority in
the loader.

## Target

Loading a GGUF should produce route facts once, then every QK install/dispatch step should consume those facts:

1. `ModelFacts`: normalized metadata from GGUF headers and tensor rows.
2. `TensorRouteFact`: tensor name, module path, quant, rows/cols, role.
3. `ModelRoutePlan`: route-install entry per tensor/module, including parts/opts/kernel mode for primitive install.
4. Primitive install uses the plan. Generated policy is an override/input to the plan, not a separate name-policy path.
5. Runtime candidates bind on quant, role, shape, arch capability, env, and explicit route policy.

## Non-Goals

- Do not remove Qwen/Llama CLI aliases.
- Do not make 14B WMMA promoted; the scheduler-owned tile loop remains the blocker.
- Do not delete `q4k_policy` / `q6k_policy` in the first integration commit. They can remain as compatibility adapters
  while the plan proves parity.
- Do not import `extra.qk` directly from `tinygrad/llm` core modules other than the existing `route_ops` adapter.

## Design

### ModelFacts

`tinygrad/llm/model_facts.py` owns model metadata normalization. It should not depend on `extra.qk`.

Inputs:

- GGUF `kv` metadata;
- `meta["tensor_infos"]` rows `(name, dims, typ, offset)`.

Outputs:

- architecture fields: `arch`, `hidden_size`, `intermediate_size`, `n_heads`, `n_kv_heads`, `head_dim`;
- tensor facts for 2D weight tensors:
  - `name`;
  - `module_path` (`name` without `.weight`);
  - `ggml_type`;
  - `quant` (`Q4_K`, `Q6_K`, or `unknown`);
  - `rows`, `cols` using runtime linear orientation;
  - `role`.

Role registry:

- `ffn_gate_up`: `ffn_gate` / `ffn_up`;
- `ffn_down`: `ffn_down`;
- `attn_qo`: `attn_q` / `attn_output`;
- `attn_kv`: `attn_k` / `attn_v`;
- `lm_head`: `output.weight`;
- otherwise `unknown`.

This centralizes name heuristics. It does not authorize routes by model size.

### ModelRoutePlan

`tinygrad/llm/model_route_plan.py` owns primitive-install decisions.

Entries:

- tensor name and module path;
- quant;
- role;
- rows/cols;
- install `parts`;
- install `opts`;
- `kernel_mode` for Q4 direct/partial;
- optional generated-policy provenance fields.

Default builder:

- consumes `ModelFacts`;
- preserves current default choices by translating existing `q4k_policy` / `q6k_policy` compatibility output into plan
  entries;
- does not let install loops call those policies directly once a plan is supplied.

Generated-policy builder/overlay:

- if `QK_GENERATED_POLICY` is supplied, generated entries remain authoritative exactly as today;
- unsupported generated families still skip with current counters.

### Primitive Install

`_install_q4k_primitives` and `_install_q6k_primitives` gain an optional `route_plan`.

Expected behavior:

- with `generated_policy`: current generated-policy behavior is preserved;
- with `route_plan` and no generated policy: route entries come from the plan;
- with neither: compatibility fallback may call old policy functions temporarily;
- debug counters remain explainable (`policy_missing`, `policy_fallback`, etc.).

### Runtime Wiring

`Transformer.from_gguf` should build facts/plan after metadata is available and before primitive install.

The load path remains:

1. metadata/admission;
2. load GGUF with metadata;
3. build config/model;
4. build route facts/plan;
5. install Q4/Q6 primitive linears from plan;
6. decode/prefill candidates bind at call time.

## Verification

Minimum required tests:

- `test_model_facts.py`: Qwen3 8B-like and 14B-like metadata fixtures produce the same roles by structure/name, without
  a model-size branch.
- `test_model_route_plan.py`: default plan entries match legacy policy decisions for representative Q4/Q6 tensor names.
- primitive install test/seam: when a route plan is passed, install does not require direct `q4k_policy` / `q6k_policy`
  calls.
- generated policy override remains accepted.
- boundary test still passes: no direct `extra.qk` imports from tinygrad core outside adapters.

Final gates:

- focused route/model tests;
- `python3 -m pytest`;
- `git diff --check`;
- `MAX_LINE_COUNT=28000 python3 sz.py`.

## Follow-Up After This Slice

Once parity is proven:

- delete install-loop compatibility fallback;
- move remaining role inference in decode/prefill helpers to facts carried on primitive linears;
- extend role registry beyond Qwen dense models;
- convert demotion logic to use route-plan roles instead of name substrings.
