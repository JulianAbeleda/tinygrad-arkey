# Q4_K WMMA Fused/Tiled Prefill Execution Scope - 2026-07-05

## Decision

Build the next Q4_K/Q8_1 prefill candidate as a bounded fused/tiled generated route. Do not clone the scalar
`sdot4` direct-output topology and do not add a fixed HIP/assembly kernel. The implementation must reuse the existing
descriptor, route, gate, and algebra surfaces already in tree.

The current blocker is not math, candidate selection, or AMD `iu8` WMMA codegen. The blocker is the full-model lowering
shape in `group_tensor_matmul_v0`: it builds too many Tensor matmul graph fragments and keeps too much RAW/correction
state live for 14B authority.

## Current Evidence

Already proven:

- `Tensor.matmul(..., dtype=dtypes.int)` can lower to `wmma_i32_16x16x16_iu8` on RDNA3.
- `prefill_mmq_parity_gate.py` validates `mmq`, `sdot4`, and `wmma_generated` against the same Q8-dequant reference.
- `PREFILL_Q4K_Q8=wmma` reaches the generated candidate and then stops at the intended graph-explosion guard:
  `RAW groups*m*n=419430400 > limit=67108864` for the 14B `attn_qo` shape.
- `PREFILL_Q4K_Q8=mmq_direct` combines the existing Q4_K/Q8_1 dot4 algebra with the existing in-kernel 8-lane
  direct-output pattern and is correct, bounded, and route-clean.

Measured outcomes:

- Current promoted default remains `prefill_q4k_direct_tile4x4_default`.
- `mmq_direct` 14B pp512 smoke: `85 tok/s`; useful topology evidence, not a replacement.
- `wmma_generated` full 14B smoke: blocked by graph explosion before timing.

## Reuse Inventory

Use these directly:

- `extra/qk/prefill_int8_wmma_spec.py`
  - Owns `Q4KInt8WMMAPrefillSpec`.
  - Owns the current Tensor-expression generated WMMA oracle.
  - Owns the `group_tensor_matmul_v0` implementation label that must be superseded, not deleted.
- `extra/qk/quant/q4_k_gemv_primitive.py`
  - Owns Q4_K metadata helpers and the scalar/generated-UOp Q4_K/Q8_1 algebra.
  - Owns `q4k_q8_1_sdot4_coop_direct_out_kernel`, the bounded direct-output evidence path.
- `extra/qk/prefill_packed_tile_spec.py`
  - Owns the generated tile spec pattern and the `direct_warp` lane-combine shape.
  - Reuse the shape discipline; do not reuse the fp16 dequant dot math for the WMMA route.
- `tinygrad/llm/prefill_routes.py`
  - Owns default-off route selection under `PREFILL_Q4K_Q8`.
  - The next route must enter here behind a new explicit flag, not behind `auto`.
- `tinygrad/llm/route_ops.py`
  - Owns lazy import shims for model-facing route calls.
- `tinygrad/llm/generated_candidates.py`
  - Already registers `quant_linear_prefill.q4k_int8_wmma_tensor_substrate`.
  - Add a distinct candidate if `wmma_tiled` gets a distinct route id or lowering strategy. A new env/route id is
    material; do not hide it behind the old `prefill_q4k_int8_wmma_generated_research` row without documenting
    supersession.
- `extra/qk/route_manifest.py`
  - Owns public route status, rollback, expected kernels, and provenance.
- `extra/qk/gate_registry.py`
  - Owns canonical gate execution and artifact IO.
- `extra/qk/bench.py`
  - Owns throughput entry; do not add another throughput harness.

Do not rebuild:

- Q8_1 activation quantization.
- Q4_K reference decode.
- `iu8` WMMA renderer/codegen support.
- Direct-packed default route.
- `mmq_direct` scalar direct-output topology.
- Canonical throughput harness.

## Non-Kernel Contract

Allowed:

- Spec/dataclass rows.
- Generated UOp/Tensor expressions.
- Small scheduler/codegen changes that make a clean int8 dot visible to existing TC matching.
- A declarative spec-to-`SHAPED_WMMA` lowering only if it is owned by scheduler/codegen infrastructure and audited as
  generated, not route-local hand WMMA.
- Default-off route wiring.
- Gates and docs.

Not allowed:

- New fixed HIP source bodies for the Q4_K WMMA route.
- Inline assembly for the Q4_K WMMA route.
- Shape-specific 14B-only kernels.
- A route that just copies `sdot4/mmq_direct` under a WMMA name.
- A route that only passes by setting `PREFILL_Q4K_WMMA_ALLOW_GRAPH_EXPLOSION=1`.
- Route-local HIP/source strings, inline asm, direct `__builtin_amdgcn_wmma`, or ad hoc route-local `Ops.WMMA`
  construction.

## Target Route Contract

Add a new default-off route, separate from the current graph-explosion oracle:

```text
PREFILL_Q4K_Q8=wmma_tiled
```

Expected output:

```text
Tensor shape: [1, M, N]
dtype: fp32
phase: prefill
quant: Q4_K weights, Q8_1 activations
roles: attn_qo, attn_kv, ffn_gate_up, ffn_down
```

Expected kernel naming:

```text
prefill_q4k_q8_1_wmma_tiled_generated_gemm_<role>_<n>_<k>_<m>_*
```

Rollback:

```text
PREFILL_Q4K_Q8=0
```

Promotion rule:

- Stay research/default-off until canonical 14B pp512 smoke is route-clean and faster than
  `prefill_q4k_direct_tile4x4_default`.

## Algebra Boundary

For group `j` of 32 K elements:

```text
RAW[m,n,j]  = sum_k xq[m,k] * q4[n,k]
QSUM[m,j]   = sum_k xq[m,k]
out[m,n] += XSC[m,j] * (D[n,blk] * SC[n,blk,g] * RAW[m,n,j]
                      - DMIN[n,blk] * MN[n,blk,g] * QSUM[m,j])
```

Hard requirement:

- `RAW` is the only part that should require `iu8` WMMA.
- `QSUM` and Q4_K scale/min correction are ordinary generated integer/fp operations.
- Live RAW storage must be tile-local or otherwise bounded. Full `[groups, M, N]` RAW is not acceptable for 14B.
- Every gate artifact must report the live RAW bound. Minimum fields:
  - `m_tile`
  - `n_tile`
  - `group_tile`
  - `live_raw_elems`
  - `forbidden_full_raw_shape`
  - `graph_node_count` where available
  - `kernel_count` where available
  - `compile_ms` where available

Hard bound:

```text
live_raw_elems <= m_tile * n_tile * group_tile
```

Any graph tensor shaped like `[groups, M, N]` for full role dimensions is a blocker, even if the run does not OOM.

## Phases

### Phase 0 - Baseline Locks

Goal: prevent accidental movement while changing the route surface.

Work:

- Add/confirm a route-manifest row for the new `wmma_tiled` route with `status=planned` or `research`.
- Add a distinct route-manifest row:
  - route id: `prefill_q4k_int8_wmma_tiled_research`
  - env: `{"PREFILL_Q4K_Q8": "wmma_tiled"}`
  - baseline: `prefill_q4k_direct_tile4x4_default`
  - status: `research`
- Add a distinct generated candidate if the implementation has a distinct lowering strategy. If a new lowering strategy
  token is introduced, update `tinygrad/llm/runtime_specs.py::LOWERING_STRATEGIES` and its unit tests.
- Add explicit Q4K_Q8 mode validation. Unknown non-empty modes must not silently fall through to
  `q4k_q8_1_gemm_kernel`.
- Add strict-mode tests proving:
  - `wmma_tiled` reaches the new route shim.
  - unknown Q4K_Q8 modes raise.
  - no direct/default kernel is selected when `wmma_tiled` is requested.
- Keep `PREFILL_Q4K_Q8=wmma` as the graph-explosion oracle until `wmma_tiled` supersedes it.

Exit criteria:

- Unit tests pass.
- No throughput behavior changes.
- Manifest shows this is default-off.
- Unknown `PREFILL_Q4K_Q8` modes are rejected or explicitly classified.

### Phase 1 - Tiled Spec, No Math Yet

Goal: define the shape contract before implementation.

Work:

- Add a new spec, likely beside `Q4KInt8WMMAPrefillSpec`, for the tiled lowering:
  `Q4KInt8WMMATiledPrefillSpec`.
- Fields:
  - `m`, `n`, `k`, `role`
  - `group_elems=32`
  - `wmma_m=16`, `wmma_n=16`, `wmma_k=16`
  - `m_tile`, `n_tile`, `group_tile`
  - `output_layout=direct`
  - `implementation=direct_tiled_wmma_v0`
- Validation:
  - `k % 256 == 0`
  - `m % 16 == 0`
  - `n % 16 == 0`
  - `group_elems == 32`
  - tile dimensions fit a known route profile
  - route rejects unsupported shapes explicitly instead of falling through silently.

Exit criteria:

- Spec tests cover 14B role shapes:
  - `attn_qo`: `M=512,N=5120,K=5120`
  - `attn_kv`: `M=512,N=1024,K=5120`
  - `ffn_gate_up`: `M=512,N=17408,K=5120`
  - `ffn_down`: `M=512,N=5120,K=17408`
- No kernel emission yet.

### Phase 2 - Minimal Tile Microgate

Goal: prove one bounded tile can compute the same Q4_K/Q8_1 result without full RAW materialization.

Work:

- Add a microgate that runs one or two small aligned shapes:
  - `M=16,N=16,K=256`
  - `M=32,N=32,K=512`
- Compare to the existing Q8-dequant reference.
- The implementation may use a generated UOp/Tensor expression as long as it keeps live RAW bounded to the tile.
- AMD gate must inspect generated code/ISA for `wmma_i32_16x16x16_iu8` before claiming WMMA success.
- Add a separate lowering-feasibility check before the numeric microgate:
  - run with `TC=1 TC_OPT=1`
  - assert `Ops.WMMA` or emitted `wmma_i32_16x16x16_iu8`
  - record applied opts/order
  - prove the exact intended tile graph keeps RAW bounded.

Important constraint:

- Do not start with the 14B model. The first proof is tile correctness and WMMA presence.
- Do not fold QSUM/scale correction into the int dot until the clean int8 `ADD` reduce over `MUL` survives TC matching.
  The current TC path is narrow; grouped correction can hide the matmul from the matcher.

Exit criteria:

- `DEV=PYTHON` parity passes for math where possible.
- `DEV=AMD` parity passes and proves `iu8` WMMA in emitted code.
- Artifact records tile shape, live RAW shape, route id, and kernel names.
- Artifact records whether TC was selected before any correction fusion.

### Phase 3 - Role-Shape Synthetic Gate

Goal: scale the tile path to real 14B dimensions without loading the model.

Work:

- Add a synthetic gate that constructs random Q4_K/Q8_1 tensors for the four role shapes.
- Run one role at a time to keep failures attributable.
- Start with smaller `M` if needed:
  - `M=16`, `M=64`, `M=512`
- For `M=16` and `M=64`, call the spec/emitter directly or set `PREFILL_DIRECT_REQUIRE_UBATCH=0`; the real route path
  defaults to ubatch-shaped prefill and should be route-bound only at `M=512`.
- Measure:
  - compile/capture time
  - runtime
  - peak or estimated live RAW
  - route-clean kernel names
  - WMMA presence
- Compare against reference on subtiles or sampled rows/columns to avoid making the reference the bottleneck.

Exit criteria:

- No graph-explosion guard.
- No OOM.
- No full `[groups,M,N]` RAW.
- `live_raw_elems <= m_tile * n_tile * group_tile`.
- All four 14B role shapes either pass or fail with a classified artifact.

### Phase 4 - Route Wiring

Goal: wire `wmma_tiled` into `route_direct_packed_prefill` without touching default behavior.

Work:

- Add route op shim in `tinygrad/llm/route_ops.py`.
- Add the `PREFILL_Q4K_Q8=wmma_tiled` branch in `tinygrad/llm/prefill_routes.py`.
- Keep `PREFILL_Q4K_Q8=wmma` as the existing Tensor-substrate oracle.
- Add strict fallback behavior:
  - unsupported tiled shape returns `None` only when route strictness allows fallback.
  - with strict route mode, unsupported tiled shape raises a clear route error.

Exit criteria:

- Unit route tests pass.
- Route-clean evidence is defined before the smoke:
  - if the route emits named kernels, add trace regex/tests for `prefill_q4k_q8_1_wmma_tiled_generated_gemm_*`;
  - if the route remains pure Tensor and descriptor names do not become `KernelInfo` names, route-clean means emitted
    WMMA/codegen artifacts are present and forbidden default kernels are absent.
- Default `auto` remains unchanged.

### Phase 5 - Canonical 14B Smoke

Goal: determine whether the route is viable as a prefill candidate.

Command shape:

```bash
PREFILL_Q4K_Q8=wmma_tiled DEVICE_IN_FUNCTION_BUG=1 ALLOW_DEVICE_USAGE=1 \
  PYTHONPATH=. .venv/bin/python extra/qk/bench.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --prefill --prefill-mode smoke
```

Exit criteria:

- Completes within the existing smoke timeout.
- Reports `WHOLE-PREFILL@512`.
- Route-clean trace shows the new route.
- No graph-explosion guard.
- No fallback to `prefill_q4k_direct_tile4x4_default`.

Classification:

- `blocked.codegen`: no WMMA visible.
- `blocked.numeric`: parity/reference mismatch.
- `blocked.graph_shape`: still builds unbounded graph/state.
- `correct_not_fast`: correct and route-clean, but slower than default.
- `promotion_candidate`: correct, route-clean, and faster than default.

### Phase 6 - Authority and Promotion

Goal: only promote if the route actually beats the current default under canonical measurement.

Work:

- Run canonical smoke first.
- Run canonical authority only after smoke is clean.
- Compare against current default in the same session where practical.
- Update route manifest status:
  - `research`
  - `correct_not_fast`
  - `promotion_candidate`
  - `default`
- Update binding audit counts if new custom/generated surfaces are added.

Promotion blockers:

- Missing `wmma_i32_16x16x16_iu8` evidence.
- Any hidden fallback to default.
- Any reliance on `PREFILL_Q4K_WMMA_ALLOW_GRAPH_EXPLOSION=1`.
- Any handwritten HIP/assembly source for the route.
- Any regression to 8B generated prefill closure.

## Gate Plan

Add or extend gates in this order:

1. Unit tests for spec validation and route env.
2. `prefill_mmq_parity_gate.py` adds `wmma_tiled` small-shape parity.
3. New gate: `q4k_wmma_tiled_microgate`.
4. New gate: `q4k_wmma_tiled_lowering_feasibility`.
5. New gate: `q4k_wmma_tiled_role_shape_gate`.
6. Extend or replace `generated_q4k_prefill_e2e` so it recognizes the new route verdicts while preserving the old
   graph-explosion classification for `PREFILL_Q4K_Q8=wmma`.
7. `generated_quant_binding_audit` updated only after route/manifest surfaces change.

All gates should run through `extra/qk/gate_registry.py`.

## Implementation Options

Option A: scheduler-owned tiled Tensor path.

- Express each tile as a clean int8 matmul so TC matching sees `M x N x K`.
- Keep QSUM/correction separate until TC selection is proven. Fuse only after the lowering-feasibility gate proves the
  clean int8 dot is still visible.
- Best match to the non-kernel rule.
- Risk: current graph shape may still expand before lowering.

Option B: declarative scheduler/codegen WMMA path.

- Build a generated emitter that owns tile loops and direct output, but routes WMMA through scheduler/codegen-owned TC
  lowering or a declarative `SHAPED_WMMA` path.
- Add an audit gate forbidding route-local source strings, inline asm, direct `__builtin_amdgcn_wmma`, and ad hoc
  route-local `Ops.WMMA` construction.
- Best chance of bounding live state.
- Risk: this may require real scheduler/lowering work before Q4_K route work can continue.

Option C: two-stage debug path.

- Stage 1 materializes bounded RAW tiles.
- Stage 2 applies scale/min correction and group reduction.
- Acceptable as a diagnostic bridge only.
- Not promotable unless it is still faster than default and bounded at 14B role shapes.

Recommended order:

1. Try Option A only to the point of proving whether tiling is enough.
2. If Option A still expands graph state, move to Option B.
3. Use Option C only to isolate numeric/codegen bugs.

## Acceptance Definition

The work is not solved when the code compiles. It is solved when:

- `PREFILL_Q4K_Q8=wmma_tiled` is route-clean.
- Unknown Q4K_Q8 modes do not silently fall through.
- Small-shape parity passes.
- Lowering-feasibility gate proves TC/WMMA on the exact intended tile graph before role-shape gates.
- AMD codegen evidence shows `wmma_i32_16x16x16_iu8`.
- Synthetic 14B role-shape gate does not hit graph explosion or OOM.
- Synthetic gates report `live_raw_elems`, graph node count, kernel count, and compile time where available.
- Canonical 14B smoke completes.
- Throughput is faster than `prefill_q4k_direct_tile4x4_default`.
- The route manifest marks it as `promotion_candidate` or `default`.
- The old graph-explosion route remains classified or is retired with a documented replacement.

## Out Of Scope

- Changing decode routes.
- Promoting `mmq_direct`.
- Retuning fp16 prefill.
- Deleting the direct-packed default before a better route exists.
- New throughput harnesses.
- Branch work.

## Open Questions For Review

- Can the current tinygrad TC matcher be reached from a bounded UOp/custom-kernel emitter, or does the route need a
  scheduler/lowering change first?
- Should `wmma_tiled` reuse `Q4KInt8WMMAPrefillSpec` with a new `implementation`, or use a separate spec class and
  distinct generated candidate?
- What is the smallest role-shape synthetic gate that catches the current graph-explosion failure without excessive
  runtime?
- Should `generated_q4k_prefill_e2e` remain a blocker-classification gate, or become a multi-verdict promotion gate?
- What evidence is sufficient to prove no fallback to the default route during canonical smoke?

## Independent Review Disposition

An xhigh review pass was run against this scope. The review did not edit files. Its high-severity findings were folded
into the requirements above:

- Shallow env tests are insufficient because unknown `PREFILL_Q4K_Q8` modes can fall through. The scope now requires
  explicit mode validation, strict-mode dispatch tests, and no-default-kernel evidence.
- TC matching is narrow. The scope now requires a lowering-feasibility gate for the exact intended tile graph with
  `TC=1 TC_OPT=1` before role-shape gates.
- A generated UOp path can cross into handwritten WMMA if it constructs WMMA directly. The scope now limits Option B to
  scheduler/codegen-owned lowering or a declarative `SHAPED_WMMA` path with an audit gate forbidding route-local source,
  inline asm, direct builtins, and ad hoc route-local `Ops.WMMA`.

Medium findings were also incorporated:

- Bounded RAW now has a hard artifact bound: `live_raw_elems <= m_tile * n_tile * group_tile`.
- `wmma_tiled` now requires a distinct manifest route row, and a distinct generated candidate if the route/lowering id
  changes.
- Route-clean evidence must either add trace regex/tests for named kernels or use emitted WMMA artifacts plus forbidden
  default-kernel absence when pure Tensor descriptors do not become `KernelInfo` names.
- Small-M synthetic tests are emitter gates, not real route-binding tests unless `PREFILL_DIRECT_REQUIRE_UBATCH=0` is set.
