# Native-Codegen Microprimitive Search — Execution Result (2026-06-23)

## Verdict: `NATIVE_CODEGEN_MICROSEARCH_EXECUTED_TARGET_FOUND` (2/4 targets native; `v_dot2` + cross-lane are the precise gaps)
Executed the Step-4 native-codegen microsearch (`extra/qk_native_codegen_microsearch.py`) — the safe, non-W==D lane.
5 bounded tinygrad-native microkernel candidates, each compiled → **local correctness vs numpy** → **ISA-audited**.
**All 5 correct.** LDS staging and vector global loads are **natively emittable**; `v_dot2` (fused fp16 dot) and
cross-lane reduction (`ds_bpermute`) are **confirmed renderer gaps**, even at warp-sized reductions. No decode/prefill
behavior touched; no W==D / speed claim (this lane cannot promote a default).

## Exhaustive candidate grid + gates (the executed scope)
Each candidate is a bounded tinygrad expression that *should* map to a target primitive; gates run cost-ordered
(compile → local correctness rel_rmse ≤ 1e-2 → ISA audit → target-flag present → no-spill envelope):

| candidate | tinygrad expr | target | rel_rmse | target ISA present | flags emitted |
|---|---|---|---:|:---:|---|
| `cross_lane_n32` | `(a*b).sum(axis=1)`, axis=32 | `ds_bpermute`/cross-lane | 1e-3 ✓ | **No** | LDS, vector-load |
| `cross_lane_n64` | same, axis=64 | cross-lane | 1e-3 ✓ | **No** | LDS, vector-load |
| `fp16_dot` | fp16 `(a*b).sum(-1)`, 128 | `v_dot2` | 1e-3 ✓ | **No** | LDS, vector-load |
| `lds_reduce` | `a.sum(axis=1)`, 4096 | `ds_load`/`ds_store` | ✓ | **Yes** | LDS, vector-load |
| `vector_load` | `a*2` contiguous, 8192 | `global_load_dwordx*` | ✓ | **Yes** | vector-load |

## Findings (the precise codegen map)
- **LDS staging — NATIVE.** Every reduction emits `ds_load`/`ds_store` (LDS tree-reduce). The owned tile's LDS
  staging is already expressible by the renderer.
- **Vector global loads — NATIVE.** All candidates emit vectorized global loads.
- **`v_dot2` — GAP.** The fp16 multiply-accumulate lowers to an LDS reduction (`v_pk_*` + reduce), **never a fused
  `v_dot2`**. The owned tile's `__builtin_amdgcn_fdot2` has no native lowering path.
- **Cross-lane reduction — GAP.** Reductions over a warp-sized axis (n=32) and n=64 still use the **LDS tree**, never
  `ds_bpermute`/`ds_swizzle`/`v_permlane`. The owned tile's `__shfl_xor` cross-lane reduce is not natively emittable.

This confirms and *hardens* the earlier single-experiment finding (`machine-code-translation-roadmap-result`) with a
multi-candidate, correctness-gated, ISA-audited run. The result is durable and recorded in the project ledger.

## What it means (the actionable bound)
The owned decode tile's two distinguishing primitives — **fused fp16 dot (`v_dot2`)** and **cross-lane warp reduce
(`ds_bpermute`)** — are *exactly* the renderer features that would have to be added for tinygrad-native codegen to
reproduce the hand-HIP owned tile. LDS + vector loads are already there. So the gap between "hand-owned escape hatch"
and "native codegen" is two specific, named lowering features — not a broad codegen rewrite. That is the precise
upstream ask, and it is now ISA-evidenced.

## Ledger
5 entries appended to `bench/qk-project-search-ledger/ledger.jsonl` (lane=`codegen`, authority=non-promotion).
`learned_rule` for the two gaps: "*`{target}` is NOT natively emittable — the precise renderer feature to add.*"

## Files changed
New: `extra/qk_native_codegen_microsearch.py`, `bench/qk-native-codegen-microsearch/result.json`, this doc; 5 ledger
entries. **No `tinygrad/` source, no default flips, no decode/prefill change, no W==D claim.**

## Git status
Clean before; adds 1 tool + 1 artifact + 1 doc + 5 ledger lines. Decode/prefill defaults unchanged.
