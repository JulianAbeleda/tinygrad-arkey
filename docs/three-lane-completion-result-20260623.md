# Three-Lane Completion (Runtime-KV + ISA + Native Codegen) — Result (2026-06-23)

## Verdict: lanes taken to completion — Lane 3 & 6 **DONE**, Lane 2 **CORE-ENGINE-BLOCKED** (decisively isolated)
Owner authorized "do all lanes until completed." Lane 3 (ISA wrapper) and Lane 6 (native-codegen first experiment)
are **complete with evidence**. Lane 2 (runtime-KV core persistence) was **implemented and driven to its decisive
blocker**: the model-reachable form bakes even at one layer, and the true vLLM-style form is fundamentally
incompatible with a transformer — it requires a **core tinygrad TinyJit/HCQ engine capability** that is out of
model-task scope. No source/default changes (diagnostic reverted; default decode byte-identical).

## Lane 3 — ISA audit wrapper → `ISA_WRAPPER_AMD_ONLY_READY` (DONE)
`extra/qk_isa_primitive_audit.py` built + validated: AMD backend → `AMD_ISA_PRIMITIVE_CONFIRMED` on the owned tile
(56 VGPR, 0 spill, v_dot2/LDS/cross-lane), unsupported vendor → graceful `ISA_BACKEND_TOOLING_LIMITED`. Now the
mandatory evidence guard. (Committed previously; used as the oracle for Lane 6.)

## Lane 6 — Native codegen first experiment → `NATIVE_CODEGEN_FIRST_EXPERIMENT_DONE` (DONE)
Rendered a tinygrad-native fp16 workgroup reduction → captured its compiled AMDGPU code object → disassembled. ISA
evidence (`bench/qk-native-codegen-experiment/lds_cross_lane_result.json`):
- **LDS staging: NATIVE** — tinygrad emits `ds_load`×4 / `ds_store`×1 (LDS tree reduction). `has_lds=True`.
- **Cross-lane reduction: NOT native** — no `ds_bpermute`/`ds_swizzle`/DPP; tinygrad uses an LDS tree-reduce, not a
  warp shuffle. `has_cross_lane=False`.
- **`v_dot2` fused dot: NOT native** — the fp16 MAC lowers to `v_pk_add_f16` (packed add) + separate multiply, not a
  fused `v_dot2`. `has_vector_dot=False`.

**Conclusion (concrete + ISA-evidenced)**: the owned tile's two distinguishing primitives — **`v_dot2`** and
**`ds_bpermute` (cross-lane warp reduce)** — are *exactly* what tinygrad-native codegen does **not** emit (LDS
staging already is native). So the native-codegen learning target is now precise: teach the renderer **cross-lane
reductions** and **`v_dot2` lowering**. This is expressibility evidence, **not** a W==D claim (none required for
this charter lane).

## Lane 2 — Runtime-KV core persistence → `RUNTIME_KV_CORE_CAPABILITY_BLOCKED` (implemented to the decisive blocker)
Design A (recommended) was implemented in its model-reachable form: in-graph opaque append into the fp16 cache +
`@function` bypass + owned tile reading the opaque-appended cache (no full-MAXC materialization).
(`bench/qk-runtime-kv-core-attempt/blocker.json`.)
- **Standalone microbench: PASS** (rel_rmse e-7, persistence across replays).
- **Full model: BAKES** (151936 from decode step 1).
- **New decisive isolation — bakes even at NL=1 (one layer)** with a real canonical-store prefill, while the
  materialized path gives finite distinct tokens at NL=1/2/4. **The failure is NOT multi-layer composition** — it is
  the in-graph opaque append on a canonical-store-prefilled cache with in-graph-computed k,v (vs the microbench's
  assign-filled cache + external k,v).
- **Why the "pre-graph side-effect" form also fails**: it is **fundamentally incompatible with a transformer** —
  layer N's k,v depend on layer N−1's attention *read* of the cache, so appends and reads are **interleaved per
  layer**; you cannot precompute all appends before the captured graph replays. vLLM works only because CUDA graphs
  permit in-place buffer mutation that persists *within* a replay; tinygrad's pure-`@function` graph requires
  materialization for that persistence.

**Classification**: the implementation needs a **core TinyJit/HCQ engine capability** — an in-place cache
store + subsequent load *within* the captured graph that persists across replays **without** the pure-`@function`
full-buffer materialization. This is not reachable from the model/route layer (proven: bakes at 1 layer), and per
the design-scope stop rule ("do not rewrite all TinyJit buffer semantics → `TOO_BROAD`") it is a **core-engine
project**, not completable as a model/route task. The Design A spec
(`docs/runtime-kv-core-persistence-capability-scope-20260623.md`) stands as the target; its implementation requires
tinygrad-engine work and is the only path to the remaining ~+11% (→ llama parity).

## What this means
- **Bounded 8B speed is exhausted at the model/route layer.** The single remaining ~+11% lever (KV materialization)
  is confirmed to require **core tinygrad engine** work — not a kernel, route, or schedule.
- The native-codegen gap is now ISA-precise (cross-lane + `v_dot2`).
- The ISA wrapper is the standing guard for any future work.

## Recommended next action
Two real options, both **outside** the model-task layer:
1. **Core tinygrad TinyJit/HCQ capability** for non-materializing persistent in-graph cache mutation (the ~+11%
   prize) — a substantial engine project with uncertain success (it is the pure-graph-persistence limitation).
2. **Native-codegen**: add renderer support for cross-lane reductions / `v_dot2` lowering (long-term capability, no
   immediate W==D).
Otherwise the 8B decode work is **complete at ~88–89% of llama** with attention + GEMV at parity and the residual
explained.

## Files changed
New: `bench/qk-native-codegen-experiment/lds_cross_lane_result.json`,
`bench/qk-runtime-kv-core-attempt/blocker.json`, this result doc. The Lane 3 wrapper was committed previously. **No
`tinygrad/` source or default changes** (the RUNTIME_KV diagnostic route was re-applied then reverted; default
decode byte-identical `[279,1156,22148,…]`).

## Git status
`model.py` clean (diagnostic reverted). New artifacts + result doc only. No default flip, no machine search, no
14B/32B, no production runtime-KV (it is core-engine-blocked).
