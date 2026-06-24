# Native-Codegen Microprimitive Search — Scope (2026-06-23)

## Mission
The **safest** machine-search lane: make tinygrad-**native** codegen reproduce the proven machine-code primitives the
owned decode tile uses — **without any decode-tok/s promotion claim**. This is a *capability* search (does the
renderer emit the target ISA?), not a speed search. Authority = ISA evidence + local numerical correctness, never
W==D. Status: `NATIVE_CODEGEN_MICROSEARCH_READY_TO_SCOPE` → this scopes it; execution is a separate run.

## Why this is safe and useful
- No default touched, no model path changed — pure microkernels.
- The targets are **already ISA-evidenced** (the native-codegen experiment, `machine-code-translation-roadmap-result`):
  tinygrad emits **LDS natively** but **not** `v_dot2` or `ds_bpermute` (cross-lane). Those two are the known gaps.
- A success here is a reusable codegen capability that makes future owned-tile/GEMM work expressible natively
  (retiring hand-HIP escape hatches), and feeds cross-shape/portability.

## Search targets (oracle = owned tile ISA)
| target | desired machine code | current native status |
|---|---|---|
| fp16 dot lowering | `v_dot2` (`__builtin_amdgcn_fdot2`) | NOT native (lowers to `v_pk_add_f16` + mul) |
| LDS staging | `ds_load`/`ds_store` at expected LDS bytes | **already native** (tree-reduce) |
| cross-lane reduction | `ds_bpermute` / `__shfl_xor` | NOT native (uses LDS tree, not cross-lane) |
| vector global loads | `global_load_dwordx*` | partially native (audit per candidate) |
| no-spill envelope | scratch/spill = 0 | invariant to enforce |

## Harness (consume the existing ISA wrapper)
- **Candidate generator**: small tinygrad-native microkernels (a fp16 dot reduction; an LDS-staged tile; a cross-lane
  reduction expressed via the renderer's reduce ops / any available primitive) — bounded variants only.
- **ISA audit**: `extra/qk_isa_primitive_audit.py` per candidate → JSON with `has_vector_dot`/`has_lds`/
  `has_cross_lane`/`has_vector_global_load`/`has_spill` + VGPR/LDS/scratch.
- **Local numerical correctness**: rel_rmse vs numpy reference (≤ 1e-2), required before recording a "found".
- **Ledger**: append each candidate to `bench/qk-project-search-ledger/ledger.jsonl` (lane=`codegen`,
  primitive_class=`codegen_microprimitive`, authority_benchmark = explicit non-promotion note).

## Gates (cost-ordered)
1. compiles + local correctness (rel_rmse ≤ 1e-2);
2. ISA audit emits a JSON;
3. target ISA present (`has_vector_dot` for the dot target, `has_cross_lane` for the reduce target, etc.);
4. resource envelope acceptable (no spill, VGPR/LDS sane).

## Boundaries
- No W==D / decode-speed promotion claim (this lane cannot promote a decode default).
- No hand-HIP/escape-hatch kernels — the point is *native* emission.
- Bounded candidate variants only; no broad/random generation.
- Decode/prefill defaults untouched.

## Success criteria / verdicts
- `NATIVE_CODEGEN_MICROSEARCH_EXECUTED_TARGET_FOUND` — a native microkernel emits the target ISA, locally correct,
  envelope OK, recorded in the ledger.
- `NATIVE_CODEGEN_MICROSEARCH_NO_TARGET_FOUND` — the renderer cannot be coaxed to emit `v_dot2`/cross-lane via the
  available ops (records the precise codegen gap as a `learned_rule` — itself valuable: it bounds what a renderer
  change would need to add).

## Recommended first run
The **cross-lane reduction** target: it's the clearest gap (LDS is already native; the dot is a builtin). Generate
2–3 native reduction microkernels, ISA-audit each, and record whether any path emits `ds_bpermute` / a warp-shuffle.
If none do, the `learned_rule` is the exact renderer feature to request upstream.
