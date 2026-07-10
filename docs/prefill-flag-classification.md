# Prefill Flag Classification (flag-collapse map)

Read-only classification of all 131 `PREFILL_*` / `AMD_ISA_*` flags (audit 2026-07-10), the map that drives the
two-wrapper flag-collapse. Deleted-flag records live in [prefill-flag-graveyard.md](prefill-flag-graveyard.md).

## Target design: two wrappers, one resolver

- **Canonical route wrapper** — `select(hybrid|pure|mixed|pipe_mvp)`. No raw-flag parameter; invalid combos unrepresentable. Production + authority runs. A partial stub already exists: `tinygrad/codegen/opt/operand_staging.py::operand_staging_policy` (keeps `PREFILL_TC_LOCAL_STAGE` as its `override` hatch; "imported by nothing yet").
- **Debug wrapper** — the single home for the surviving raw flags (grouping the ~110 scattered `getenv` reads). Explicit debug opt-in; production code cannot reach a raw flag.
- Both produce one `PrefillRouteSpec`; downstream (`prefill_graph_gemm_route`, lowerer, renderer) is unchanged.

## Route → flags map (canonical selector encodes this)

| route | route_id | defining env | pp512 |
|---|---|---|---|
| hybrid | `prefill_pipe_role_selective_generated` | `GRAPH_GEMM=1` only | ~4413 |
| pure | `prefill_wmma_pipe_lds_dbuf_primitive_generated` | `GRAPH_GEMM+PIPE+LDS+DBUF` | ~1332 |
| mixed | `prefill_wmma_lds_dbuf_primitive_mixed` | `GRAPH_GEMM+LDS+DBUF` (PIPE off) | — |
| pipe_mvp | `prefill_wmma_pipe_primitive_generated` | `GRAPH_GEMM+PIPE` | — |
| default | `prefill_v2_scheduler_matmul_default` | `{}` (GRAPH_GEMM=0) | ~2688 |

Structural split (no flag): `out_f==12288` (ffn_gate_up) → lds family, else → pipe family (`_resolve_schedule`).

## Buckets

### SELECTOR_OWNED (4) — consumed by the canonical wrapper, not user toggles
`PREFILL_GRAPH_GEMM`, `PREFILL_WMMA_PIPE_PRIMITIVE`, `PREFILL_WMMA_LDS_PRIMITIVE`, `PREFILL_DBUF` (bare).

### PROMOTE → bake into route spec (~25)
Byte-identical bakes (route already always sets, or already default-on): `PREFILL_DBUF_NBUF`, `PREFILL_DBUF_D3A_STAGE_B`, `PREFILL_WMMA_CHAIN_RESIDENT_A`, `PREFILL_WMMA_CHAIN_RESIDENT_B`, `AMD_ISA_WMMA_B128_FRAG`, `PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE`, `PREFILL_REMAINDER_FIX`, `PREFILL_TC_ATTN`.
Baked-into-route (byte-identical *for that route*, real off-default flip globally — bake into the pure/kmajor/lds spec, NOT the global default): `PREFILL_DBUF_LDS_CONST_IMM`, `_LDS_INDEX_SPLIT`, `_LDS_STORE_BASE_SPLIT`, `_DIRECT_B128_CHAIN`, `_LDS_ADDR_USE_DEP`, `_D3A_POST`, `_D3A_STAGE_A`; `PREFILL_WMMA_KMAJOR_PHASE`, `_CHAIN_AB_RESIDENT`, `_AB_PROOF_META`, `_AB_PROOF_KEY`, `_AB_PROOF_FROM_LDS_STORES`, `_AB_PHASE_SCOPED_KEY`; `PREFILL_LDS_PACK_WITHLOCAL_B128`, `AMD_ISA_REG_ACCUM`, `AMD_ISA_WAITCNT_TARGETED`.
Real global default change (needs correctness+perf gate, own commit): `PREFILL_V2` (bake `auto`, keep small-VRAM override).

### DELETE (49) — see graveyard doc for the record
DBUF (8), WMMA destructive-suppression + dumps (13), TC_LOCAL_STAGE coop/refuted (16), LDS_PACK dumps/refuted/verifier-dead (12).

### KEEP_DEBUG (~53) — survive into the debug wrapper
Pure-machine-path knobs (`PREFILL_WMMA_KMAJOR_STAGE_STEAL` + additive half, `_CLUSTERED_LDS_CONSUME`, `_PHASE_EXACT_WINDOW`, `_RESIDENT_PACK_DEP`, `_PHASE_TILE_BYTES_A`, `_AB_PROOF_FROM_LDS_DESC`, `_KMAJOR_ROOTS`, `_KMAJOR_D3A_MARKER`, `_STAGE_STEAL_MEMO`); DBUF LDS-addressing knobs (`_LDS_REGION_BASE_SPLIT/_MEMO`, `_LDS_STORE_IMM_FOLD`, `_LDS_BASE_REMAT[_DEEP]`, `_LDS_RELOAD_ANCHOR`, `_LDS_LOAD_SERIAL`, `_LDS_ADDR_SERIAL`, `_DIRECT_B128_ADDR_REMAT`, `_REDUCE_RANGE_STRICT`, `_GLOBAL_ADDR_INLOOP`, `_OWNED_*_STAGE_*` scaffolding); TC_LOCAL_STAGE base ladder (`TC_LOCAL_STAGE`, `_WITH_LOCAL`, `_POST`, `_B_TILEKEY`, `_DUMP[_LIMIT]`); AMD_ISA policy knobs (`_SCHED`, `_WMMA_LOW_SCRATCH`, `_WAITCNT_CONSERVATIVE`, `_WAITCNT_D3A_SKIP_STORE_LOAD`, `_WMMA_CLUSTER_LGKM_WAIT`, `_N1B[_GIDX]`); V2/serving knobs (`_V2_MAX_REALIZE_GB`, `_V2_FORCE_REALIZE`, `_UBATCH`, `_CONCRETE_KV`, `_SERVER_PROFILE`, `_PACKED_STREAM`, `_STAGE_PRESERVE_TAGS`, `_GRAPH_GEMM_ROLES`, `_ALLOW_PARKED_4X4`, chunked family).

### KEEP_DEBUG — HAZARD (never promote, gates preserved)
`PREFILL_DBUF_LDS_CONST_IMM_UNSAFE` (+ override `_BOTH_U0_GT2_PROVEN`), `PREFILL_TC_LOCAL_STAGE_A_MULTIDIM_UNSAFE`, `PREFILL_LDS_PACK_WITHLOCAL_MULTIDIM_UNSAFE`, `AMD_ISA_N1B` (SGPR-datapath fault on decode tile).

## Execution order (after wrappers exist)
1. Delete the 49 dead flags (NFC; banking each into the graveyard as removed). Re-confirm no live-path read per flag.
2. Bake the byte-identical promotes into route specs (NFC). Then the off-default-flip promotes into the pure/kmajor/lds route spec (byte-identical for that route; prove with remu + a model run).
3. `PREFILL_V2` default-`auto` promotion: separate commit, correctness + perf gate.
4. Relocate the ~53 survivors into the debug wrapper.
