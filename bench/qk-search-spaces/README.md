# Search-space manifests (Milestone 1 of the generic-low-level-search goal)

Roadmap: `../../docs/pure-machine-search-roadmap.md`.

`docs/generic-low-level-search-goal-scope.md` Milestone 1. Each manifest declares **what a search over that
primitive family can and cannot express**, so a search result is self-classifying: a failure is only a *true
performance wall* if the winning oracle/owned kernel used **no excluded primitive**; otherwise it is
`SEARCH_SPACE_INCOMPLETE` / `SEARCH_BLOCKED_BY_CODEGEN` (see the label set in the goal scope).

This is the machine-evidenced answer to "can we replace the hand-tuned kernels with search?":
- **No, not yet, for the two hand-written decode kernels** — the owned AMDGCN attention tile and the warp GEMV.
  Both depend on `v_dot2` (fused fp16 dot) and **cross-lane reduction** (`ds_bpermute`), which the tinygrad renderer
  has **no lowering for** (the opcodes exist in `tinygrad/runtime/autogen/amd/rdna3/ins.py` but there is no rule in
  `tinygrad/renderer/{cstyle,llvmir}.py` — `grep` = 0 hits). Their other needs (LDS staging, vector global loads) ARE
  natively emittable. ISA evidence: `docs/archive/native-codegen-microprimitive-search-result-20260623.md:24-30`.
- **Yes, for everything else** — the search over `env_policy` + `tile_config` (split/combine policy, KV identity,
  route thresholds, flash variant, tile constants) is real and runs today; within it, the hand oracle remains best
  (`docs/archive/decode-mode-b-search-result-20260623.md`).

## Provenance binding (Milestone 2)

Every candidate in `bench/qk-decode-eval/candidates.json` should carry a `search_space_id`:
- a real space id (e.g. `decode_attention_gfx1100_v1`) → the candidate is a point a search could sample;
- `manual_oracle_not_search_generated` → the candidate is hand-owned / an oracle target and is **not** evidence that
  the generic search space can express it (goal scope, "Relationship To Owned Kernels").

## Files

| file | family | searchable today? |
|---|---|---|
| `decode_attention_gfx1100_v1.json` | decode split-KV attention (tile + combine) | config-only; the winning tile is `SEARCH_BLOCKED_BY_CODEGEN` (v_dot2 + cross-lane) |
| `decode_ffn_gemv_gfx1100_v1.json` | decode Q4_K FFN/proj weight-GEMV | config-only; the winning warp GEMV is `SEARCH_BLOCKED_BY_CODEGEN` (cross-lane) |
| `manual_oracle_not_search_generated.json` | provenance marker | n/a — declares a candidate is hand-owned, not search-generated |

## Path to making these searchable (Milestone 5)

Expose the two missing renderer lowerings — **cross-lane reduction first** (`ds_bpermute`), then `v_dot2`. Once a
lowering exists, the corresponding `excluded_primitives` entry moves to `exposed_*`, the space becomes searchable at
that level, and Milestone 6 can compare a search-generated candidate against the owned oracle under the existing
W==D + byte-exact gate. Caveat (DNR arc): exposing the *instruction* makes the primitive *searchable* but may not
alone reach owned-kernel perf — the *schedule* (waitcnt/clause/live-range) is a separate, deeper gap.
