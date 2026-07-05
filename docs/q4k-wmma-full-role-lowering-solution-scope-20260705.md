# Q4_K WMMA Full-Role Lowering Solution Scope - 2026-07-05

## Objective

Finish `PREFILL_Q4K_Q8=wmma_tiled` by replacing the current one-tile proof with a direct full-role lowering that:

- maps 14B role shapes to bounded tiles,
- keeps RAW tile-local,
- uses `iu8` WMMA through tinygrad-owned lowering,
- applies Q4_K/Q8_1 scale/min correction inside the tile lifecycle,
- writes final `[M,N]` output directly,
- never falls back silently to `prefill_q4k_direct_tile4x4_default`.

This is the missing piece after:

- `q4k_wmma_tiled_lowering_feasibility`: pass,
- `q4k_wmma_tiled_microgate`: pass,
- `q4k_wmma_tiled_role_shape`: `blocked.full_route_lowering_missing`,
- `generated_q4k_prefill_e2e`: `GENERATED_Q4K_PREFILL_E2E_TILED_BLOCKED_FULL_ROUTE`.

## Problem Statement

The current Tensor oracle is correct but global:

```text
raw = [groups, M, N]
```

For 14B `attn_qo`, that is:

```text
160 * 512 * 5120 = 419,430,400 RAW elements
```

For `ffn_gate_up` and `ffn_down`, it reaches:

```text
1,426,063,360 RAW elements
```

The solution must instead run:

```text
for M tile:
  for N tile:
    acc_fp32[M_TILE,N_TILE] = 0
    for Q4_K group tile:
      raw_i32[M_TILE,N_TILE] = iu8_wmma(q8_tile, q4_tile)
      qsum_i32[M_TILE] = sum(q8_tile)
      acc_fp32 += xscale * (D*SC*raw_i32 - DMIN*MN*qsum_i32)
    store acc_fp32
```

The only live RAW allowed is tile-local:

```text
live_raw_elems <= m_tile * n_tile * group_tile
```

## Existing Opt-In Scan

The repo already has several nearby opt-ins. None are the missing full-role Q4_K/Q8_1 tiled WMMA lowering.

| Opt-in | Existing owner | What it does | Reuse / disposition |
| --- | --- | --- | --- |
| `PREFILL_QK_GENERATED_TILE=1` | `tinygrad/llm/prefill_routes.py` + `extra/qk/prefill_packed_tile_spec.py` | Generated packed/dequant tile route with `lane_partials` and `direct_warp` modes. | Reuse shape/route discipline only. It is not Q8_1 int8 WMMA and was refuted on 14B ffn_gate_up (`lane_partials` 0.99 GB/s, `direct_warp` 1.29 GB/s). |
| `PREFILL_Q4K_Q8=mmq_direct` | `tinygrad/llm/prefill_routes.py` + `q4k_q8_1_sdot4_coop_direct_out_kernel` | Existing full-role bounded Q4_K/Q8_1 direct-output route using scalar/generated-UOp dot4 plus in-kernel 8-lane reduction. | Reuse algebra/direct-output lessons only. It is not WMMA and measured `85 tok/s`, so it is not a promotion candidate. |
| `PREFILL_Q4K_Q8=wmma` | `extra/qk/prefill_int8_wmma_spec.py::Q4KInt8WMMAPrefillSpec` | Correct Tensor-expression Q4_K/Q8_1 WMMA oracle. | Keep as graph-explosion oracle. It materializes/globalizes RAW shape and trips the full-model guard. |
| `PREFILL_Q4K_Q8=wmma_tiled` | `Q4KInt8WMMATiledPrefillSpec` + one-tile emitter | Correct one-tile proof and route branch. | Reuse descriptor/gates. Full role shapes still raise and are classified `blocked.full_route_lowering_missing`. |
| `Q4K_UNFUSE` | `tinygrad/llm/model.py` | Runs FFN matmuls in fp16 so ordinary dense WMMA can apply. | Not a quantized Q4_K/Q8_1 prefill route. |
| `extra/qk/prefill/wmma.py` / `prefill_schedule_spec.py` | generated fp16 prefill schedule and hand-assembly history | RDNA3 WMMA schedule generator for fp16 GEMM path. | May inform scheduling, but direct use for Q4_K route is out of scope if it becomes hand assembly or fixed source. |

Conclusion: there is no existing opt-in that already gives full-role bounded Q4_K/Q8_1 WMMA prefill. The closest reusable
pieces are:

- `PREFILL_QK_GENERATED_TILE` for generated tile route structure,
- `wmma_tiled` one-tile gates for Q4_K/Q8_1 correctness,
- `SHAPED_WMMA` / TC infrastructure for legal WMMA lowering.

## Chosen Strategy

Build a reusable generated tile-lowering substrate first, then use it from Q4_K.

Do not implement full-role `wmma_tiled` by:

- looping Python over Tensor matmul tiles and `cat`/summing them,
- materializing `[groups,M,N]`,
- adding a route-local HIP/CUDA source body,
- adding route-local inline asm,
- direct route-local `__builtin_amdgcn_wmma`,
- defaulting to scalar `sdot4` under the WMMA route name.

The preferred implementation path is:

```text
Q4KInt8WMMATiledPrefillSpec
  -> generated tile lowering descriptor
  -> scheduler/codegen-owned SHAPED_WMMA or TC-owned WMMA lowering
  -> one generated kernel family over role shapes
```

Use `Ops.SHAPED_WMMA` only if the construction is moved into a reusable generated lowering layer and audited. A
route-local hand-authored WMMA UOp body is not acceptable.

## Reuse Targets

Reuse directly:

- `extra/qk/prefill_int8_wmma_spec.py`
  - `Q4KInt8WMMATiledPrefillSpec`
  - Q4_K Tensor helper algebra
  - one-tile microgate oracle
- `extra/qk/q4k_wmma_tiled_lowering_feasibility.py`
  - exact RAW tile codegen proof
- `extra/qk/q4k_wmma_tiled_microgate.py`
  - one-tile Q4_K/Q8_1 numeric proof
- `extra/qk/q4k_wmma_tiled_role_shape_gate.py`
  - 14B role-shape classification and RAW bounds
- `tinygrad/schedule/rangeify.py`
  - `Ops.SHAPED_WMMA` lowers to `Ops.WMMA`
- `tinygrad/codegen/opt/postrange.py`
  - existing TensorCore matching constraints and `Ops.WMMA` construction
- `tinygrad/renderer/cstyle.py`
  - existing HIP `iu8` WMMA wrapper emission
- `tinygrad/llm/prefill_routes.py`
  - default-off `PREFILL_Q4K_Q8=wmma_tiled` route branch
- `extra/qk/gate_registry.py`
  - all gates must remain registry-owned

Do not duplicate:

- Q8_1 quantization,
- Q4_K reference math,
- scalar `_sdot4` route,
- direct-packed route,
- throughput harnesses.

## Implementation Phases

### Phase A - Generated Tile IR Contract

Add a data-only lowering contract, likely in a new module:

```text
extra/qk/q4k_wmma_tile_lowering.py
```

Core types:

```text
Int8WMMATileLoweringSpec
Q4KWMMAFullRoleLoweringSpec
```

Required fields:

- `m`, `n`, `k`, `role`
- `m_tile`, `n_tile`, `group_tile`
- `wmma_m=16`, `wmma_n=16`, `wmma_k=16`
- `groups`
- `waves_per_block`
- `output_layout=direct`
- `wmma_surface=tc_matcher | shaped_wmma`
- `live_raw_elems`
- `forbidden_full_raw_elems`

Exit criteria:

- Unit tests cover all four 14B role shapes.
- Spec rejects non-aligned shapes.
- Spec computes grid shape and bounded RAW size.
- No emitted kernel yet.

### Phase B - WMMA Surface Decision Gate

Before building Q4_K full-role logic, decide which generated WMMA surface is viable.

Test two micro emitters:

1. `tc_matcher_tile`
   - uses ordinary int8 `Tensor.matmul(..., dtype=int)` for one tile.
   - already proven feasible.
   - likely not sufficient for full-role loop ownership.

2. `shaped_wmma_tile`
   - constructs a declarative `Ops.SHAPED_WMMA` tile in a reusable lowering module.
   - relies on `rangeify.py::lower_shaped_wmma`.
   - must not contain HIP strings, inline asm, or `__builtin_amdgcn_wmma`.

This phase is mandatory discovery, not plumbing. The repo has `SHAPED_WMMA` lowering support, RDNA3 int8 TC metadata,
and HIP `iu8` WMMA rendering, but there is no existing Q4_K tile-lifecycle producer. Passing the current
`q4k_wmma_tiled_lowering_feasibility` gate does not prove `SHAPED_WMMA` can own full-role loops.

Gate:

```text
q4k_wmma_tiled_surface_gate
```

Required artifact fields:

- `surface`
- `has_ops_shaped_wmma`
- `has_ops_wmma_after_rangeify` where observable
- `has_iu8_wmma_isa_or_source`
- `numeric_ok`
- `no_route_local_builtin`
- `no_route_local_asm`
- `live_raw_elems`

Exit criteria:

- Choose exactly one surface for full-role implementation.
- If `tc_matcher_tile` cannot own loops without graph explosion, choose `shaped_wmma`.
- If `shaped_wmma` needs scheduler changes, classify that explicitly before Q4_K work continues.

### Phase C - One-Kernel Tile Lifecycle

Build a generated tile kernel over a small full-output shape that has multiple output tiles:

```text
M=32, N=32, K=256
```

The kernel must:

- cover four `16x16` output tiles,
- loop over Q4_K groups inside the kernel lifecycle,
- keep `acc_fp32[m_tile,n_tile]` tile-local,
- keep `raw_i32[m_tile,n_tile]` tile-local,
- apply scale/min correction before leaving the group loop,
- write final `[M,N]` output.

Gate:

```text
q4k_wmma_tiled_lifecycle_gate
```

Exit criteria:

- numeric parity vs q8-dequant reference,
- `wmma_i32_16x16x16_iu8` visible on AMD,
- no `[groups,M,N]` Tensor,
- `live_raw_elems <= m_tile*n_tile*group_tile`,
- kernel count bounded and reported,
- compile time reported.

### Phase D - Synthetic Role-Shape Execution

Scale to real 14B dimensions without loading the model.

Gate:

```text
q4k_wmma_tiled_role_shape_exec_gate
```

Do not reuse the current blocked classifier verdict as the execution gate. The current `q4k_wmma_tiled_role_shape`
gate is allowed to pass only because it classifies `blocked.full_route_lowering_missing`. Phase D needs a distinct
execution gate/verdict so CI cannot treat the old blocker as progress.

Run shapes:

- `attn_qo`: `M=512,N=5120,K=5120`
- `attn_kv`: `M=512,N=1024,K=5120`
- `ffn_gate_up`: `M=512,N=17408,K=5120`
- `ffn_down`: `M=512,N=5120,K=17408`

Suggested order:

1. `attn_kv`
2. `attn_qo`
3. `ffn_down`
4. `ffn_gate_up`

For each role, artifact must report:

- `compile_ms`
- `runtime_ms`
- `kernel_count`
- `graph_node_count` where available
- `live_raw_elems`
- `forbidden_full_raw_elems`
- `wmma_present`
- `numeric_sample_ok`
- `fallback_absent`

Exit criteria:

- all role shapes run or fail with a precise blocker,
- no graph explosion,
- no OOM,
- no fallback to direct-packed default.

### Phase E - Runtime Route Binding

Only after Phase D passes, wire full-role execution into:

```text
tinygrad/llm/prefill_routes.py
```

The current `wmma_tiled` branch raises on full route shapes. Replace that raise only when Phase D proves the full-role
lowering.

Strict behavior:

- `PREFILL_Q4K_Q8=wmma_tiled` must never fall back.
- Unsupported shape must raise in strict and non-strict modes because this is an explicit experimental route.
- `auto` remains unchanged.
- `PREFILL_Q4K_Q8=wmma` remains the old graph-explosion oracle until explicitly retired.

Exit criteria:

- route tests prove `wmma_tiled` calls the full-role emitter,
- unknown Q4K_Q8 modes still raise,
- route-clean evidence is available.

### Phase F - Canonical 14B Smoke And Authority

Run canonical smoke:

```bash
TC=1 TC_OPT=1 PREFILL_Q4K_Q8=wmma_tiled DEVICE_IN_FUNCTION_BUG=1 ALLOW_DEVICE_USAGE=1 \
  PYTHONPATH=. .venv/bin/python extra/qk/bench.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --prefill --prefill-mode smoke
```

If the selected surface is `SHAPED_WMMA` and does not require heuristic TC matching, the smoke artifact may omit
`TC=1 TC_OPT=1` only if it explicitly proves the selected path bypasses the TC heuristic and still emits
`wmma_i32_16x16x16_iu8`.

Classify:

- `blocked.surface`: selected WMMA surface cannot lower legally,
- `blocked.lifecycle`: tile lifecycle fails parity or graph bounds,
- `blocked.role_shape`: synthetic role shape fails,
- `blocked.route_binding`: model route falls back or misses kernels,
- `correct_not_fast`: route-clean but slower than default,
- `promotion_candidate`: route-clean and faster than default.

Promotion requires:

- canonical smoke completion,
- route-clean trace,
- `wmma_i32_16x16x16_iu8` evidence,
- no fallback kernels on selected roles,
- throughput faster than `prefill_q4k_direct_tile4x4_default`,
- route manifest updated to `promotion_candidate`.

## Proposed Kernel Lifecycle

The target generated lowering should behave like:

```text
grid_m = ceil(M / m_tile)
grid_n = ceil(N / n_tile)

for tile_m, tile_n in grid:
  acc_fp32[m_tile,n_tile] = 0

  for group in groups:
    q8_frag_a = load xq[tile_m, group]
    q4_frag_b = unpack q4[tile_n, group]
    raw_i32 = wmma_i32_16x16x16_iu8(q8_frag_a, q4_frag_b)
    qsum_i32 = sum(q8_frag_a)
    scale = xscale[tile_m, group] * D[tile_n, block] * SC[tile_n, group]
    mincorr = xscale[tile_m, group] * DMIN[tile_n, block] * MN[tile_n, group] * qsum_i32
    acc_fp32 += scale * raw_i32 - mincorr

  store out[tile_m,tile_n] = acc_fp32
```

The important property is not the exact loop spelling. The important property is that `raw_i32` is born, corrected, and
discarded inside the tile lifecycle.

## Code Ownership Boundary

Acceptable files to touch:

- `extra/qk/prefill_int8_wmma_spec.py`
- `extra/qk/q4k_wmma_tile_lowering.py` or similarly named generated lowering module
- `extra/qk/q4k_wmma_tiled_*gate.py`
- `extra/qk/gate_registry.py`
- `tinygrad/llm/prefill_routes.py`
- `tinygrad/llm/route_ops.py`
- `tinygrad/llm/generated_candidates.py`
- `tinygrad/llm/runtime_specs.py`
- targeted scheduler/codegen files only if the chosen WMMA surface requires it:
  - `tinygrad/schedule/rangeify.py`
  - `tinygrad/codegen/opt/postrange.py`
  - `tinygrad/renderer/cstyle.py`

Avoid:

- `extra/qk/prefill/wmma.py` hand-assembly path,
- new `extra/qk/prefill/*.py` assembly emitters,
- new benchmark harnesses,
- default route changes before promotion.

## No-Hand-Kernel Audit

Add a hard gate, separate from the current broad binding audit, to check the new lowering files for:

- no `asm volatile`,
- no `__builtin_amdgcn_wmma` string in route-local files,
- no HIP/CUDA source string builder,
- no direct `Ops.WMMA` in route-local Q4_K files unless the code is the reusable scheduler/codegen-owned lowering
  selected by Phase B,
- no calls into `extra/qk/prefill/wmma.py`.

The existing `generated_quant_binding_audit` is not enough by itself: it scans a fixed file list and its pass verdict
does not fail on all binding findings. The Q4_K WMMA tiled route needs a route-specific no-hand-kernel gate that fails
hard on new route-local handwritten surfaces.

Renderer-owned `__builtin_amdgcn_wmma` remains allowed in `tinygrad/renderer/cstyle.py`.

## Risks

1. `SHAPED_WMMA` may not currently support the exact int8 operand layout needed for Q4_K unpacked fragments.
   - Mitigation: Phase B isolates this before role-shape work.

2. Q4_K unpacking may dominate VALU and erase the expected win.
   - Mitigation: classify `correct_not_fast`, do not promote.

3. Register pressure may be too high for `16x16` direct output plus correction.
   - Mitigation: tune `n_tile`, `group_tile`, or split correction without full RAW.

4. A two-stage RAW materialization may look tempting.
   - Mitigation: allow only as a diagnostic bridge; promotion still requires bounded full-role behavior.

5. The generated route may pass small shapes but compile too slowly at full roles.
   - Mitigation: Phase D requires `compile_ms`, `kernel_count`, and role-by-role artifacts before model smoke.

## Definition Of Done

This is complete only when:

- `q4k_wmma_tiled_surface_gate` chooses a legal WMMA surface,
- `q4k_wmma_tiled_lifecycle_gate` passes,
- `q4k_wmma_tiled_role_shape_exec_gate` executes all four 14B role shapes instead of classifying missing lowering,
- `PREFILL_Q4K_Q8=wmma_tiled` binds full model roles without fallback,
- canonical 14B smoke completes,
- throughput beats the current default,
- route manifest moves from `research` to `promotion_candidate`.

Until then, the honest status is:

```text
one-tile correct; full-role lowering missing
```

## Review Disposition

An xhigh review independently scanned for existing opt-ins and did not find a full-role bounded Q4_K/Q8_1 WMMA prefill
route. The review confirmed:

- `PREFILL_QK_GENERATED_TILE` is generated packed/dequant tile work, not Q8_1 WMMA.
- `PREFILL_Q4K_Q8=wmma` is the global RAW Tensor oracle and remains graph-explosion blocked.
- `PREFILL_Q4K_Q8=wmma_tiled` is one-tile only and raises for full role shapes.
- `PREFILL_Q4K_Q8=mmq_direct` is full-role and bounded, but scalar dot4, not WMMA, and too slow to promote.

The review corrections folded into this scope:

- Phase B is mandatory surface discovery because no existing Q4_K `SHAPED_WMMA` tile-lifecycle producer exists.
- A hard no-hand-kernel gate is required; the broad binding audit alone is insufficient.
- Phase D must use a distinct execution gate/verdict, not the current blocked-classifier pass verdict.
- Canonical smoke must pin `TC=1 TC_OPT=1` unless the selected `SHAPED_WMMA` path proves TC heuristics are unnecessary.
