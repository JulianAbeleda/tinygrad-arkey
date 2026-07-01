# Model-Agnostic Runtime Route Planner Scope

Date: 2026-06-30

Status: scope for Claude/Codex implementation. No route defaults should be flipped blindly; the output of this work is an explicit, auditable route plan chosen from model shape, quant, backend, context, and available VRAM.

## Problem

The current benchmark/runtime path still relies too much on ambient flags and 8B-specific intuition:

- Decode routes are mostly default-on, but shape guards mean larger Qwen models can silently miss the optimized 8B route and fall back.
- Prefill fast paths are intentionally not global default-on because they realize fp16 weights next to quantized weights and can OOM larger models.
- The benchmark currently reports one "default" path, which can make tinygrad look slow when the tuned/server prefill profile was simply not selected.

We need a model-agnostic route planner:

```text
model file + backend + max_context + VRAM budget
  -> inspect metadata and shapes
  -> compute memory feasibility
  -> choose validated decode routes
  -> choose best safe prefill profile
  -> emit route plan + reasons + fallback causes
```

The user-facing workflow should become:

```text
select model -> load -> runtime computes route plan -> inference
```

not:

```text
select model -> remember a set of env flags manually
```

## Principles

- Use calculations, not hand-picked model names.
- Separate **fit** from **speed**:
  - fit: can this route fit in VRAM with margin?
  - speed: among routes that fit, which validated route is fastest for this shape/backend?
- Do not pretend an unvalidated shape is covered.
- If auto disables a route, return the reason in `/runtime/status` and benchmark artifacts.
- Keep flags as internal rollback/diagnostic controls, not the public UX.

## Inputs

The planner should consume:

- GGUF metadata and tensors:
  - architecture;
  - quant type per tensor where available;
  - layer count;
  - dim / hidden dim;
  - head count, kv-head count, head dim;
  - vocab size;
  - model file bytes;
  - tensor shapes for all linears.
- Runtime config:
  - `max_context`;
  - requested mode: `interactive`, `server`, `benchmark_default`, `benchmark_tuned`;
  - backend target: `AMD:gfx1100`, future `METAL`, future `CUDA`.
- Hardware:
  - total VRAM;
  - optionally free VRAM if available;
  - backend feature descriptor: wave size, tensor core availability, supported custom kernels.

## Core Memory Equations

Use decimal GB for user-facing logs and raw bytes in JSON.

### Quantized Weight Footprint

Prefer actual file/tensor storage when available:

```text
W_quant_bytes = model_file_bytes
```

For role/tensor-level accounting:

```text
W_quant_role_bytes = sum(storage_bytes(tensor) for tensor in role_tensors)
```

If tensor storage bytes are unavailable, estimate from quant block metadata:

```text
W_quant_role_bytes ~= num_weights * bytes_per_weight(quant_type)
```

### Fast Prefill FP16 Realization Footprint

Fast prefill realizes fp16 copies for covered dense linears:

```text
W_prefill_fp16_bytes = sum(out_features_i * in_features_i * 2 for each covered prefill linear i)
```

Covered linears should come from the existing single source:

```text
Transformer._PREFILL_V2_LINEARS
```

Do not hardcode Qwen3-8B sizes.

### KV Cache Footprint

For decoder-only dense attention:

```text
KV_bytes = 2 * num_layers * n_kv_heads * max_context * head_dim * kv_dtype_bytes
```

Where:

```text
2 = K and V
kv_dtype_bytes = 2 for fp16 cache, 4 for fp32 cache
```

The current code uses the fp16 estimate in `Transformer.from_gguf`; make the dtype explicit in the plan because decode attention route can imply fp16 cache.

### Runtime Margin

Planner must reserve headroom for activations, temporary buffers, JIT allocations, allocator fragmentation, graph captures, and output logits:

```text
required_bytes(route) =
  W_quant_bytes
  + route.extra_weight_bytes
  + KV_bytes
  + route.precompile_bytes
  + safety_margin_bytes
```

Initial safety policy:

```text
safety_margin_bytes = max(3GB, 0.10 * total_vram_bytes)
```

This should be configurable per backend profile later.

### Fit Decision

```text
fits(route) = required_bytes(route) <= total_vram_bytes
```

If free VRAM is reliable:

```text
fits_now(route) = required_bytes(route) <= free_vram_bytes + already_owned_runtime_bytes
```

But do not rely only on free VRAM because process-local allocation and driver accounting can be noisy.

## Route Families To Plan

### Decode Q4_K

Current optimized route:

- generated G3 LaneMap route;
- validates 8B-style shapes such as:
  - `4096 -> 12288`;
  - `12288 -> 4096`;
  - `4096 -> 4096`.

Planner behavior:

- enumerate all Q4_K decode linears by `(out_features, in_features, role)`;
- for each linear, ask a route-capability function whether G3 covers that shape;
- if covered, plan `q4k_g3_lanemap`;
- if not covered, plan fallback and emit `reason=shape_not_covered`.

This must make the 14B/32B gap obvious:

```text
q4k_g3_lanemap: disabled for ffn_gate because shape 5120x13824 not in validated topology set
```

or whatever the actual model shapes are.

### Decode Q6_K

Current shipped route:

- coop route for Q6_K lm_head / selected down roles;
- direct half-warp route was refuted.

Planner behavior:

- keep shipped coop route for covered Q6_K roles;
- never auto-enable refuted `Q6K_DIRECT_ROUTE`;
- emit explicit `refuted_route=q6k_direct_halfwarp` in debug plan if useful.

### Decode Attention

Current shipped route:

- owned AMDGCN decode attention for validated AMD/gfx1100 Qwen3-8B style shape;
- fallback to generated/gqa path for unsupported shapes.

Planner behavior:

- gate by backend, head shape, head dim, batch, dtype, and context threshold;
- emit whether fp16 KV cache is implied;
- expose fallback reason when not route-bound.

### Prefill Universal

Always available:

```text
prefill_universal
extra_weight_bytes = 0
expected = slow long-prompt path
```

### Prefill V2 / Graph GEMM / Role-Selective Pipe

Fast path:

```text
prefill_v2_graph_gemm_role_selective
extra_weight_bytes = W_prefill_fp16_bytes
requires: backend feature supports route, ubatch validated, route shapes covered
```

Planner behavior:

- If mode is `interactive` and no explicit server/tuned intent, it may choose universal to avoid load-time precompile and memory tax.
- If mode is `server` or `benchmark_tuned`, evaluate fit and enable the fastest validated prefill route that fits.
- If it does not fit, return:

```json
{
  "route": "prefill_v2_graph_gemm_role_selective",
  "enabled": false,
  "reason": "requires 31.4GB including 3GB margin; total_vram=24.0GB"
}
```

Expected examples on RX 7900 XTX 24GB:

- 8B Q4_K_M: may fit depending on max_context and margin; plan should calculate it.
- 14B Q4_K_M: likely OFF because fp16 realization is too large.
- 32B Q4_K_M: OFF.

## Deliverable: RoutePlan Object

Add a small structured object, preferably in `tinygrad/llm/route_planner.py` or equivalent:

```python
@dataclass
class RouteDecision:
  area: str                 # decode_q4k, decode_q6k, decode_attention, prefill
  role: str                 # ffn_gate, ffn_down, attn_qo, lm_head, etc.
  route: str
  enabled: bool
  reason: str
  env: dict[str, str]
  estimated_required_bytes: int | None
  estimated_extra_bytes: int | None
  fallback_route: str | None
  validation_status: str    # validated, shape_not_covered, refuted, backend_not_supported, memory_not_fit
```

```python
@dataclass
class RuntimeRoutePlan:
  model_id: str
  backend: str
  target: str
  max_context: int
  total_vram_bytes: int | None
  quant_weight_bytes: int
  kv_cache_bytes: int
  safety_margin_bytes: int
  decisions: list[RouteDecision]
  selected_env: dict[str, str]
```

The route plan must be serializable to JSON.

## Phase P0 - Audit Current Coverage

Build an audit-only tool:

```bash
PYTHONPATH=. python3 extra/model_route_plan_audit.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --max_context 2048 \
  --mode benchmark_tuned
```

It should emit:

```text
bench/models/qwen/route-plans/amd-gfx1100/qwen3-8b.json
```

Run it for:

- qwen3-0.6b Q8_0;
- qwen3-8b Q4_K_M;
- qwen3-14b Q4_K_M;
- qwen3-32b Q4_K_M;
- qwen3.5-27b when downloaded.

Acceptance:

- No route changes yet.
- Plan explains why prefill fast path is ON/OFF per model.
- Plan explains which decode Q4_K shapes are covered by G3 and which fall back.
- 14B/32B route gaps are visible without reading code.

## Phase P1 - Runtime Integration

Expose route plan in runtime:

- `/runtime/status` includes `route_plan_summary`.
- `/runtime/cache` or new `/runtime/route-plan` returns full JSON.
- Server load can accept:

```json
{"model": "qwen3:8b", "profile": "auto"}
```

Supported profiles:

- `default`: current conservative behavior;
- `server`: enable fast routes when calculated fit/validated;
- `benchmark_default`: no hidden tuned profile;
- `benchmark_tuned`: best validated route that fits.

Acceptance:

- Existing behavior remains unchanged for `profile=default`.
- `profile=server` does not require users to set raw prefill flags.
- Failed route fits do not crash; they return plan reasons and choose fallback.

## Phase P2 - Benchmark Harness Integration

Update `extra/model_e2e_bench.py`:

- add `--profile default|server|benchmark_default|benchmark_tuned`;
- compute route plan before loading or immediately after metadata load;
- apply selected route env internally for the subprocess/harness;
- record route plan in artifact;
- split prefill columns:
  - `default_ttft_pp512`;
  - `tuned_prefill_pp512` when profile supports it;
  - mark `not_fit` when fast prefill is infeasible.

Acceptance:

- 8B benchmark can show:

```text
decode: default route, route-attributed
prefill default: universal TTFT
prefill tuned: server/graph-gemm route
```

- 14B/32B benchmark should not show a fake tuned prefill number if the planner says it cannot fit.
- Rendered docs clearly distinguish default vs tuned route.

## Phase P3 - Route Attribution Gate

Add route attribution to the model benchmark:

- Q4_K route counts:
  - `q4k_g3_lanemap`;
  - `q4k_owned_warp`;
  - `q4k_coop`;
  - fallback graph.
- Q6_K route counts:
  - coop;
  - fallback;
  - refuted direct disabled.
- attention route:
  - owned AMDGCN;
  - gqa/generated fallback.
- prefill route:
  - universal;
  - prefill_v2;
  - graph_gemm;
  - role_selective.

Acceptance:

- 8B decode table is not accepted unless G3 route counts show eligible Q4_K roles firing.
- 14B/32B decode gaps are tagged with route coverage, not guessed.
- A benchmark row with low tok/s must say whether it is `route_missed`, `memory_not_fit`, or `validated_but_slow`.

## Phase P4 - Shape-Generalization Queue

Once route plans show gaps, produce a queue for TG/PMS:

```json
{
  "missing_decode_shapes": [
    {"quant": "Q4_K", "role": "ffn_gate", "shape": [out, in], "model": "qwen3-14b", "reason": "g3_shape_not_covered"}
  ],
  "candidate_generation_priority": "largest wall-share first"
}
```

Acceptance:

- 14B/32B are not treated as benchmark anomalies; they become new profile targets.
- No new route is promoted without the usual correctness/W==D gates.

## Phase P5 - Docs

Update:

- `bench/models/qwen/amd-rx7900xtx-gfx1100.md`;
- `docs/current-project-state-handoff-20260624.md`;
- runtime/client docs if profile field is added.

Docs must say:

- default is conservative;
- server/tuned profile is route-planned;
- prefill fast path is VRAM-gated;
- decode optimization is shape-gated;
- flags remain rollback/debug controls.

## Regression Gates

Static:

```bash
PYTHONPATH=. python3 extra/qk_policy_consistency_check.py
PYTHONPATH=. python3 extra/qk_search_space_manifest_check.py
PYTHONPATH=. python3 extra/qk_candidate_evaluator.py
PYTHONPATH=. .venv/bin/python -m pytest test/unit/test_verdict_ssot.py -q
```

Runtime smoke:

```bash
PYTHONPATH=. python3 extra/model_route_plan_audit.py --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --max_context 2048 --mode benchmark_tuned
PYTHONPATH=. python3 extra/model_route_plan_audit.py --model /home/ubuntu/models/Qwen3-32B-Q4_K_M.gguf --max_context 2048 --mode benchmark_tuned
```

Benchmark:

```bash
PYTHONPATH=. python3 extra/model_e2e_bench.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --id qwen3-8b \
  --max_context 2048 \
  --profile benchmark_tuned \
  --decode-tokens 512 \
  --warmup-skip 96 \
  --prefill 512 \
  --out bench/models/qwen/data/amd-gfx1100/qwen3-8b.tuned.json
```

## Non-Goals

- Do not make fast prefill global default-on.
- Do not force 14B/32B into fp16-prefill if memory math says no.
- Do not promote new 14B/32B decode kernels in this scope.
- Do not hide raw flags entirely; keep them as debug/rollback.
- Do not claim benchmark parity without route attribution.

## Expected Outcome

After this scope, the system should answer, for any model:

```text
What routes did tinygrad choose?
Why did it choose them?
What did not fit?
What shape was not covered?
What should TG/PMS search next?
```

That is the model-agnostic version of the current 8B-specific knowledge.

