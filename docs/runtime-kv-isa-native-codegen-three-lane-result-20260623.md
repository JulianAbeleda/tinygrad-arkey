# Runtime-KV + ISA Audit + Native Codegen — Three-Lane Result (2026-06-23)

## 1. Verdict: `ISA_READY_RUNTIME_KV_SCOPE_READY_NATIVE_SCOPE_READY` (= `THREE_LANE_SCOPES_READY`)
All three lanes delivered: Lane 3 **built + validated** (vendor-neutral ISA wrapper, AMD active), Lane 2
**design-scoped** (`RUNTIME_KV_CORE_CAPABILITY_SCOPE_READY_DESIGN_A`), Lane 6 **chartered**
(`NATIVE_CODEGEN_LEARNING_CHARTER_READY` + first experiment scoped). No production runtime-KV, no
attention/GEMV reopen, no machine search, no default flip, no source/default changes.

## 2. Authority / current repo state
HEAD `334dea8c0`, gfx1100, Qwen3-8B-Q4_K_M. Owned attention default-on; Q4K GEMV warp default-eligible; tinygrad
~88–89% of llama decode; attention + FFN-GEMV closed at/near parity. AMD ISA tooling present
(`/opt/rocm/llvm/bin`). NVIDIA/Intel tooling absent (scoped-only).

## 3. Lane 3 — ISA audit infrastructure → `ISA_WRAPPER_AMD_ONLY_READY`
**Built** `extra/qk_isa_primitive_audit.py` — a vendor-neutral wrapper that delegates to the AMD backend
(`extra/qk_amdgpu_isa_primitive_audit.py`), emits the normalized cross-vendor contract, and gracefully reports
unsupported vendors. Validated:
- `--vendor amd --candidate owned_decode_attention` → `AMD_ISA_PRIMITIVE_CONFIRMED`: symbols
  `[owned_flash_tile_gqa]`, vgpr 56, sgpr 26, scratch 0, flags `has_vector_dot/has_lds/has_cross_lane/
  has_vector_global_load = true`, `has_spill = false`. Links the W==D artifact. →
  `bench/qk-isa-primitive-audit/owned_decode_attention.json`.
- `--vendor nvidia` → `ISA_BACKEND_TOOLING_LIMITED` (graceful scoped-only note; no NVIDIA/Intel impl).
- Partial: LDS-byte metadata is `null` for some auto-discovered bundles (the disasm-based `has_lds` flag is
  reliable; the readelf `.note` byte parse is partial). Flags + VGPR/SGPR/scratch are reliable.
**Policy**: this wrapper is now the **mandatory evidence guard** for every future candidate (`ISA_AUDIT_GUARD_ACTIVE`).

## 4. Lane 2 — Runtime-KV core persistence → `RUNTIME_KV_CORE_CAPABILITY_SCOPE_READY_DESIGN_A`
Scope: `docs/runtime-kv-core-persistence-capability-scope-20260623.md`. The blocker is precisely that
`cache_kv.after(store)` does **two** jobs — ordering **and** `@function` cross-replay persistence (via
materialization). Removing the materialization removes persistence → bakes. **Recommended Design A**:
runtime-managed KV object with the **append run as a pre-graph runtime side-effect** (the vLLM "update KV before
graph replay" model) and the pure attention graph reading the persistent buffer as a stable input — the prior
RUNTIME_KV attempt put the append *inside* the captured graph and baked; moving it *outside* (untried) is the
fix. Design D (two-graph split) is the fallback; B (state-token) is a sub-primitive; **C (bounded alias rule) is
rejected `TOO_BROAD`** (prior REDUCE-hazard / symbolic-alias wall). Includes a 4-rung minimal proof ladder (toy
buffer → one-layer → full-model shadow → W==D ≥+5%/parity-class), correctness/graph/W==D/ISA gates, and stop
rules. **Implementation needs explicit owner authorization + a design review** (core-runtime work).

## 5. Lane 6 — Native codegen learning → `NATIVE_CODEGEN_LEARNING_CHARTER_READY` + `NATIVE_CODEGEN_FIRST_EXPERIMENT_SCOPED`
Charter: `docs/native-codegen-learning-from-owned-primitives-scope-20260623.md`. Existence proof: Q4K GEMV warp is
already a tinygrad-native schedule that won at parity → work-decomposition is natively expressible. The gap: the
owned attention tile (LDS staging / `v_dot2` / cross-lane reduce / split-KV) is still escape-hatch-only. First
bounded experiment: a **tinygrad-native LDS + cross-lane reduction microkernel, ISA-audited** (prove
expressibility via `has_lds`/`has_cross_lane` in the renderer's output; **no requirement to beat the owned
route**). Explicitly NOT a near-term W==D lane; lower priority than runtime-KV.

## 6. Recommended execution order
1. **Lane 3 (done)** — ISA wrapper is the standing guard; use it on every future candidate.
2. **Lane 2** — the only parity-class speed prize (~+11% → llama parity). Pursue **only with owner authorization**
   for core-runtime work; start at proof-ladder rung 1 (toy buffer), Design A.
3. **Lane 6** — longer-term codegen capability; run the one LDS+cross-lane microkernel experiment only when codegen
   capability (not decode speed) is the goal, and never let it block Lane 2.

## 7. Explicitly out of scope (this task)
Production runtime-KV implementation; NVIDIA/Intel ISA backends; machine search; attention/GEMV optimization; 14B/
32B; default flips; native flash attention; any tinygrad source change.

## 8. Files changed
New: `extra/qk_isa_primitive_audit.py` (wrapper, tooling); `docs/runtime-kv-core-persistence-capability-scope-20260623.md`;
`docs/native-codegen-learning-from-owned-primitives-scope-20260623.md`; this result doc; refreshed
`bench/qk-isa-primitive-audit/owned_decode_attention.json` (via the wrapper). Updated `docs/README.md`,
`structure/Development/session-handoff.md`. **No `tinygrad/` source or default changes.**

## 9. Git status
Clean before this task. Adds one tool + three docs + one artifact + two doc updates. No source/default changes.
