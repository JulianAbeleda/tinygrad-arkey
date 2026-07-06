# HANDOFF - 14B Qwen3 Q4_K prefill, WMMA full-role lowering

Date: 2026-07-05. Branch: `master`. Last pushed commit:

```text
a5f7c1589 [prefill] enable shaped wmma scheduler surface
```

Audience: Claude or any agent picking up the 14B Q4_K/Q8_1 prefill WMMA track.

Hard constraints from the user:

- Work on `master`; do not create feature branches unless explicitly asked.
- No hand-written GPU kernels for this route.
- No route-local HIP/CUDA source strings.
- No route-local inline asm.
- No route-local `__builtin_amdgcn_wmma`.
- No route-local direct `Ops.WMMA` construction.
- Route code may use tinygrad-owned scheduler/codegen substrate, especially `tinygrad.schedule.wmma.shaped_wmma`.
- Commit subjects must have a bracketed prefix, for example `[prefill] ...`.
- Use the canonical prefill harnesses/gates, not generate-TTFT numbers, for prefill claims.

## Big Picture

The target is 14B Qwen3 Q4_K prefill at pp512. The important comparison is:

- tinygrad current Q4_K prefill baseline: about 365 tok/s.
- llama.cpp reference: about 1849 tok/s.

The baseline bottleneck is not "tensor cores are too slow." It is the fp16 dequant path. The current default route
expands Q4_K weights into fp16-ish values and pays VALU work plus extra memory bandwidth before/during GEMM.
llama.cpp's MMQ-style path keeps weights quantized, does int dot products, then applies per-group scale/min correction.

For RDNA3, a key result is already established:

- RDNA3 `iu8` WMMA works through tinygrad codegen.
- But `iu8` WMMA is not a raw 2x throughput win over fp16 WMMA on RDNA3.
- Therefore the win is not "int8 tensor cores are faster"; the win must come from avoiding fp16 dequant and keeping the
  Q4_K/Q8_1 algebra fused/tiled.

The current WMMA track is trying to build that fused/tiled path without hand kernels:

```text
Q4_K packed weights + Q8_1 activations
  -> bounded per-output-tile int8 dot
  -> RDNA3 iu8 WMMA
  -> Q4_K scale/min + Q8 scale correction
  -> direct fp32 output
```

The crucial architectural requirement is that RAW int32 dot products must stay tile-local. For full 14B shapes, the
global RAW tensor is enormous:

```text
attn_kv     groups*M*N = 160*512*1024   =    83,886,080
attn_qo     groups*M*N = 160*512*5120   =   419,430,400
ffn_down    groups*M*N = 544*512*5120   = 1,426,063,360
ffn_gate_up groups*M*N = 160*512*17408  = 1,426,063,360
```

Materializing or graph-building that shape is the graph explosion we are trying to remove. The allowed live RAW scope is
only:

```text
live_raw_elems = m_tile * n_tile * group_tile
```

For the current 16x16x1 tile choice:

```text
live_raw_elems = 16 * 16 * 1 = 256
```

So the real missing thing is not another Tensor oracle. It is a scheduler/codegen-owned loop nest that executes the
full role while keeping each RAW tile bounded.

## Current Status

### What is solved

1. `iu8` WMMA codegen works on gfx1100.

Plain int8 `Tensor.matmul(..., dtype=dtypes.int)` can lower to:

```text
__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32
```

The existing gates prove this with max_abs 0 on synthetic int8 GEMM tiles.

2. The one-tile Q4_K/Q8_1 WMMA algebra is correct.

`extra/qk/q4k_wmma_tiled_microgate.py` validates one bounded Q4_K/Q8_1 output tile with scale/min correction. It passes.

3. A bounded multi-output-tile Tensor lifecycle is correct for a small shape.

`extra/qk/q4k_wmma_tiled_lifecycle_gate.py` runs:

```text
M=32, N=32, K=256
tile=16x16x1
output_tiles=4
```

It passes numeric parity and sees `iu8` WMMA. Latest observed:

```text
verdict: Q4K_WMMA_TILED_LIFECYCLE_PASS
rel_rmse: 1.3815123622862302e-07
kernel_count: 59
live_raw_elems: 256
forbidden_full_raw_elems: 8192
```

This is useful as a correctness lifecycle, but it is not the final full-role implementation. It still composes many
Tensor fragments.

4. The declarative `SHAPED_WMMA` scheduler surface is now executable.

Commit `a5f7c1589` fixed the shaped scheduler surface. The gate:

```bash
PYTHONPATH=. .venv/bin/python -m extra.qk.gate_registry run q4k_wmma_scheduler_surface
```

now returns:

```text
verdict: Q4K_WMMA_SCHEDULER_SURFACE_SHAPED_READY
selected_surface: shaped_wmma_tile
has_iu8_wmma: true
```

This is important. Before `a5f7c1589`, the shaped path was blocked by program verifier/render issues around fragment
assembly. Now a tiny scheduler-owned `SHAPED_WMMA` probe lowers through `rangeify` to `Ops.WMMA`, renders HIP, compiles,
and executes on AMD.

5. The no-hand-kernel audit passes.

```bash
PYTHONPATH=. .venv/bin/python -m extra.qk.gate_registry run q4k_wmma_tiled_no_hand_kernel
```

returns:

```text
verdict: Q4K_WMMA_TILED_NO_HAND_KERNEL_PASS
```

The scanned route files do not contain route-local inline asm, route-local WMMA builtins, route-local `Ops.WMMA`, or
route-local custom kernels for this path.

### What is not solved

`q4k_wmma_tiled_role_shape_exec` does not genuinely execute any 14B role shape yet.

The gate currently reports:

```text
Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_FULL_ROLE_LOWERING
```

This is not a failure in the one-tile math. It is an honest blocker: the full-role scheduler-owned loop has not been
implemented.

The gate code makes this explicit in:

```text
extra/qk/q4k_wmma_tiled_role_shape_exec_gate.py
```

It enumerates the 14B role shapes, reports the tile plan, and marks:

```text
exec.attempted = false
class = blocked.scheduler_owned_tile_loop_missing
```

Do not mark this gate as passed by relaxing the verdict. It must pass only after actual synthetic role-shape execution
with evidence: compile/runtime metrics, kernel count, graph node count or equivalent, WMMA evidence, and bounded RAW.

## Why extending the current Tensor lifecycle is the wrong path

The current lifecycle implementation is:

```text
extra/qk/prefill_int8_wmma_spec.py::emit_q4k_int8_wmma_tiled_lifecycle_tensor
```

It loops in Python/Tensor graph space:

```text
for ms in range(0, M, m_tile):
  for ns in range(0, N, n_tile):
    acc = Tensor.zeros(...)
    for blk in range(k_blocks):
      for grp in range(groups_per_block):
        q4_g = ...
        q8_g = ...
        raw = int8 Tensor.matmul(...)
        scale/min correction
        acc = acc + ...
    cols.append(acc.contiguous())
  rows.append(cat(cols))
return cat(rows)
```

This is acceptable for `M=32,N=32,K=256`; it is not acceptable for 14B roles. Full-role counts from
`extra/qk/q4k_wmma_tile_lowering.py` are:

```text
attn_kv:
  output_tiles: 2048
  raw_tile_steps: 327680
  wmma_fragment_ops: 655360

attn_qo:
  output_tiles: 10240
  raw_tile_steps: 1638400
  wmma_fragment_ops: 3276800

ffn_down:
  output_tiles: 10240
  raw_tile_steps: 5570560
  wmma_fragment_ops: 11141120

ffn_gate_up:
  output_tiles: 34816
  raw_tile_steps: 5570560
  wmma_fragment_ops: 11141120
```

If those loops remain Python/Tensor graph loops, the route explodes into hundreds of thousands to millions of graph
fragments. That is exactly why the full route currently raises instead of silently falling through.

## The correct full-role shape

The desired generated loop is:

```text
grid over output tiles:
  tile_m = program_id_m
  tile_n = program_id_n

  acc_fp32[16,16] = 0

  for group in range(groups):
    q8_tile[16,32] = load Q8_1 activation group for tile_m
    q4_tile[16,32] = decode Q4_K group for tile_n

    raw_i32[16,16] = 0
    raw_i32 += shaped_wmma(q8_tile[:, 0:16], q4_tile[:, 0:16], raw_i32)
    raw_i32 += shaped_wmma(q8_tile[:,16:32], q4_tile[:,16:32], raw_i32)

    qsum_i32[16] = sum(q8_tile, axis=1)
    d, dmin, sc, mn = Q4_K params for tile_n/group
    xscale = Q8_1 scale for tile_m/group

    acc_fp32 += xscale * (d*sc*raw_i32 - dmin*mn*qsum_i32)

  store acc_fp32[16,16] to output[M,N]
```

This loop has many iterations, but they are runtime loop iterations inside generated lowering, not separate Tensor graph
fragments. The only live RAW tile remains 256 elements.

## Existing Files and Their Roles

### Route/spec/oracle files

`extra/qk/prefill_int8_wmma_spec.py`

- Owns `Q4KInt8WMMAPrefillSpec`.
- Owns `Q4KInt8WMMATiledPrefillSpec`.
- Owns Tensor oracles/helpers for Q4_K/Q8_1 algebra.
- Current `emit_q4k_int8_wmma_tiled_prefill_tensor` is one-tile only.
- Current `emit_q4k_int8_wmma_tiled_lifecycle_tensor` is bounded small-shape lifecycle only.
- Do not scale this function to full 14B by Python loops.

`extra/qk/q4k_wmma_tile_lowering.py`

- Data-only full-role lowering contract.
- Computes grid counts, raw tile steps, WMMA fragment counts, live RAW, forbidden full RAW.
- Does not emit kernels.
- Current default `wmma_surface` may still say `tc_matcher_tile` in some JSON; conceptually the viable scheduler surface is
  now `shaped_wmma_tile` after `a5f7c1589`.

`tinygrad/llm/prefill_routes.py`

- Contains `PREFILL_Q4K_Q8=wmma` and `PREFILL_Q4K_Q8=wmma_tiled`.
- `wmma` is guarded against graph explosion.
- `wmma_tiled` reaches the route and raises a deliberate "full route shape not implemented" error for 14B.
- Do not add silent fallback to the default packed route.

### Scheduler/codegen substrate

`tinygrad/schedule/wmma.py`

- Exposes:

```python
shaped_wmma(a_frag, b_frag, acc_frag, dims, device, threads, dtype_out=None)
```

- This constructs `Ops.SHAPED_WMMA`.
- Route code should use this helper if it needs explicit WMMA. Route code should not construct `Ops.WMMA`.

`tinygrad/schedule/rangeify.py`

- Owns `lower_shaped_wmma`.
- `Ops.SHAPED_WMMA` lowers to `Ops.WMMA` here.
- After `a5f7c1589`, it supports already-vectorized per-thread fragments and returns an ordered vector result from
  register-backed WMMA stores.

`tinygrad/codegen/late/devectorizer.py`

- After `a5f7c1589`, `pm_render` contains a late rewrite that pushes `GEP` through ordered `AFTER` values so
  `GEP(AFTER(STACK(...), stores...))` renders as scalar ordered reads instead of pointer arithmetic on a vector
  expression.

`tinygrad/uop/spec.py`

- Contains verifier support for `Ops.SHAPED_WMMA` in tensor graph.
- After `a5f7c1589`, shared spec allows ordered `AFTER` over `STACK` and `INDEX`, needed by shaped WMMA lowering.

`tinygrad/codegen/opt/tc.py`

- Contains RDNA3 int8 tensor-core metadata:

```text
dtypes.char -> dtypes.int
wmma_i32_16x16x16_iu8
```

`tinygrad/renderer/cstyle.py`

- Emits HIP WMMA wrappers. This is an allowed external owner.

### Gates

Run gates through `extra.qk.gate_registry` when possible.

Important gates:

```bash
PYTHONPATH=. .venv/bin/python -m extra.qk.gate_registry run q4k_wmma_scheduler_surface
PYTHONPATH=. .venv/bin/python -m extra.qk.gate_registry run q4k_wmma_tiled_no_hand_kernel
PYTHONPATH=. .venv/bin/python -m extra.qk.gate_registry run q4k_wmma_tiled_lifecycle
PYTHONPATH=. .venv/bin/python -m extra.qk.gate_registry run q4k_wmma_tiled_role_shape_exec
PYTHONPATH=. .venv/bin/python -m extra.qk.gate_registry run generated_q4k_prefill_e2e
```

Current expected state:

```text
q4k_wmma_scheduler_surface:
  Q4K_WMMA_SCHEDULER_SURFACE_SHAPED_READY

q4k_wmma_tiled_no_hand_kernel:
  Q4K_WMMA_TILED_NO_HAND_KERNEL_PASS

q4k_wmma_tiled_lifecycle:
  Q4K_WMMA_TILED_LIFECYCLE_PASS

q4k_wmma_tiled_role_shape_exec:
  Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_FULL_ROLE_LOWERING

generated_q4k_prefill_e2e:
  GENERATED_Q4K_PREFILL_E2E_TILED_BLOCKED_FULL_ROUTE
```

That is a coherent state. Do not force the blocked gates green without implementation evidence.

## What changed in the latest commit

Commit:

```text
a5f7c1589 [prefill] enable shaped wmma scheduler surface
```

Files changed:

```text
extra/qk/q4k_wmma_scheduler_surface_gate.py
tinygrad/codegen/late/devectorizer.py
tinygrad/schedule/rangeify.py
tinygrad/uop/spec.py
```

Details:

1. `q4k_wmma_scheduler_surface_gate.py`

The shaped probe now uses `UOp.vectorize`, not tensor-style `.stack()`, to construct per-thread fragments.

Why:

- `.stack()` is tensor concat semantics.
- It lowers scalar fragments into pad/sum movement ops.
- Program verifier rejects those movement ops.
- `vectorize()` constructs the fragment value directly.

2. `rangeify.py::lower_shaped_wmma`

Now accepts already-vectorized fragment operands:

```text
src if src.dtype.count == src.shape[-1] else src[u].contract(u)
```

It also returns an ordered vector value:

```text
vals[0].vectorize(*vals[1:]).after(stores)
```

rather than returning a register buffer view that external callers then index awkwardly.

3. `devectorizer.py::pm_render`

Adds a late render rewrite:

```text
GEP(AFTER(value, stores...)) -> AFTER(GEP(value), stores...)
```

This must be late. When tried too early in symbolic rewriting, it exposed an `INDEX` pattern before load insertion and
failed with:

```text
AssertionError: param is not PtrDType dtypes.int
```

Late render placement lets loads/stores be materialized first and only fixes source rendering.

4. `uop/spec.py`

Allows `AFTER` over `STACK` and `INDEX`.

This is needed because the lowered shaped WMMA result is an ordered vector/scalar read from register-backed WMMA stores.

Risk: this is a global verifier allowance. It is small, but it is not Q4_K-local. Keep the regression gates green.

## Verification Performed After Latest Commit

Unit tests:

```bash
.venv/bin/python -m pytest \
  test/unit/test_prefill_int8_wmma_spec.py \
  test/unit/test_llm_prefill_routes.py \
  test/unit/test_qk_route_purity.py
```

Result:

```text
35 passed in 0.71s
```

Static checks:

```bash
git diff --check
python3 sz.py
```

Result: pass. Budgeted lines:

```text
26223 / 50000
```

GPU gates:

```bash
PYTHONPATH=. .venv/bin/python -m extra.qk.gate_registry run \
  q4k_wmma_scheduler_surface \
  q4k_wmma_tiled_no_hand_kernel \
  q4k_wmma_tiled_lifecycle
```

Results:

```text
q4k_wmma_scheduler_surface: Q4K_WMMA_SCHEDULER_SURFACE_SHAPED_READY
q4k_wmma_tiled_no_hand_kernel: Q4K_WMMA_TILED_NO_HAND_KERNEL_PASS
q4k_wmma_tiled_lifecycle: Q4K_WMMA_TILED_LIFECYCLE_PASS
```

`generated_q4k_prefill_e2e` was also run before the latest commit and confirmed the full-route blocker:

```text
GENERATED_Q4K_PREFILL_E2E_TILED_BLOCKED_FULL_ROUTE
```

The smoke classifications were:

```text
wmma:
  blocked.graph_explosion

wmma_tiled:
  blocked.full_route_lowering_missing
```

## The Next Work Item

Make `q4k_wmma_tiled_role_shape_exec` genuinely pass.

That means:

1. Add a scheduler-owned full-role loop producer/lowering.
2. It must consume the shape contract from `Q4KWMMAFullRoleLoweringSpec`.
3. It must keep live RAW bounded to tile scope.
4. It must use `tinygrad.schedule.wmma.shaped_wmma` or a reusable scheduler helper that calls it.
5. It must not add route-local GPU source or direct route-local `Ops.WMMA`.
6. It must update `q4k_wmma_tiled_role_shape_exec_gate.py` to actually execute synthetic role shapes, not just classify them.

The minimum execution proof should include, per role:

```text
attempted: true
compile_ms: ...
runtime_ms: ...
kernel_count: ...
graph_node_count: ... or another bounded-graph proxy
wmma_present: true
live_raw_elems: 256
forbidden_full_raw_elems: huge but not materialized
```

For synthetic role-shape execution, do not load the whole 14B model. Use synthetic tensors shaped like the roles:

```text
attn_kv:     M=512, N=1024,  K=5120
attn_qo:     M=512, N=5120,  K=5120
ffn_down:    M=512, N=5120,  K=17408
ffn_gate_up: M=512, N=17408, K=5120
```

Numeric validation can be staged:

- Small shapes: compare full output vs Q8-dequant reference.
- Full synthetic role shapes: if full reference is too expensive, validate slices/checksums plus bounded execution
  evidence, but the gate should be explicit about what is numerically checked.

## Suggested Implementation Phases

### Phase 1 - Update surface metadata to prefer shaped WMMA

Some older JSON still says:

```text
wmma_surface: tc_matcher_tile
```

because earlier work selected TC matcher while shaped WMMA was blocked. After `a5f7c1589`, shaped WMMA is available.

Cleanups:

- Change `extra/qk/q4k_wmma_tile_lowering.py` default `wmma_surface` to `shaped_wmma_tile`, if that is now the chosen
  full-role surface.
- Update tests expecting `tc_matcher_tile`.
- Consider updating `q4k_wmma_tiled_surface_gate.py`, which still has old selection language for TC matcher. There is
  now also `q4k_wmma_scheduler_surface_gate.py`, which is the authority for shaped readiness.

Do not let this become just a paper change. It should reflect the implementation route.

### Phase 2 - Build a reusable scheduler helper for Q4_K/Q8_1 tile fragments

Possible owner:

```text
tinygrad/schedule/wmma.py
```

or a new tinygrad-owned scheduler module. Avoid putting executable WMMA UOp construction in `extra/qk` route files.

Needed primitive shape:

```text
q4k_q8_tile_wmma(
  q8_frag_16,
  q4_frag_16,
  acc_frag_8,
  ...
) -> acc_frag_8
```

or a more general shaped int8 tile dot helper.

It should internally call:

```python
tinygrad.schedule.wmma.shaped_wmma(...)
```

and never construct `Ops.WMMA` directly.

### Phase 3 - Build a generated full-role loop/lowering

This is the real work.

The lowering must own:

- output tile grid,
- group loop,
- Q4_K decode for the current tile/group,
- Q8_1 loads for the current tile/group,
- two 16-wide WMMA fragments per 32-wide Q4_K group,
- qsum,
- scale/min correction,
- direct output store.

The lowering should not generate a Python Tensor graph fragment per raw tile. The raw tile loop must live in the
generated program/lowering.

### Phase 4 - Replace the role-shape exec classifier with execution

Update:

```text
extra/qk/q4k_wmma_tiled_role_shape_exec_gate.py
```

from "not attempted" to actual synthetic execution. The pass verdict should be:

```text
Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_PASS
```

Only emit that when all role rows execute and evidence is present.

### Phase 5 - Re-run e2e smoke

Expected final e2e progression:

Before full-role lowering:

```text
GENERATED_Q4K_PREFILL_E2E_TILED_BLOCKED_FULL_ROUTE
```

After full-role lowering:

```text
generated_q4k_prefill_e2e should stop classifying wmma_tiled as blocked.full_route_lowering_missing
```

The exact final pass verdict may need a gate update, but do not weaken the gate. Make it reflect real route execution.

## Things Not To Do

Do not:

- Extend `emit_q4k_int8_wmma_tiled_lifecycle_tensor` to 14B with Python loops.
- Silence `NotImplementedError` in `wmma_tiled` and fall back to the default packed route.
- Mark `q4k_wmma_tiled_role_shape_exec` pass while `exec.attempted` is false.
- Add route-local `.custom_kernel(...)` for the full WMMA route.
- Add HIP source strings under `extra/qk`.
- Add direct route-local `Ops.WMMA`.
- Re-investigate the "int8 WMMA is 2x faster on RDNA3" theory. It was refuted.
- Report generate-TTFT throughput as prefill throughput.

## Useful Commands

Quick current-state validation:

```bash
.venv/bin/python -m pytest \
  test/unit/test_prefill_int8_wmma_spec.py \
  test/unit/test_llm_prefill_routes.py \
  test/unit/test_qk_route_purity.py

PYTHONPATH=. .venv/bin/python -m extra.qk.gate_registry run \
  q4k_wmma_scheduler_surface \
  q4k_wmma_tiled_no_hand_kernel \
  q4k_wmma_tiled_lifecycle

git diff --check
python3 sz.py
```

Full classification:

```bash
PYTHONPATH=. .venv/bin/python -m extra.qk.gate_registry run generated_q4k_prefill_e2e
```

Role-shape blocker:

```bash
PYTHONPATH=. .venv/bin/python -m extra.qk.gate_registry run q4k_wmma_tiled_role_shape_exec
```

14B smoke route:

```bash
PREFILL_Q4K_Q8=wmma_tiled DEVICE_IN_FUNCTION_BUG=1 ALLOW_DEVICE_USAGE=1 \
  .venv/bin/python extra/qk/bench.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --prefill \
  --prefill-mode smoke
```

Current expected smoke error:

```text
RuntimeError: PREFILL_Q4K_Q8=wmma_tiled is not implemented for full route shape ...
This explicit stop prevents fallthrough to the default Q4_K/Q8_1 GEMM route.
```

That error is correct until the full-role lowering exists.

## Final Mental Model

There are three different levels. Keep them separate:

1. Tensor oracle:
   Correct algebra, easy to validate, but graph-explodes at 14B.

2. Bounded small lifecycle:
   Proves the tiled algebra over multiple small output tiles, but still uses Tensor graph composition.

3. Full-role scheduler lowering:
   The missing production path. Runtime loops own tile_m/tile_n/group. RAW is tile-local. WMMA is emitted by tinygrad
   scheduler/codegen substrate. This is what must be built next.

Current repo state has levels 1 and 2. Commit `a5f7c1589` made the `SHAPED_WMMA` substrate viable for level 3.
The next agent should build level 3, not stretch level 2.
