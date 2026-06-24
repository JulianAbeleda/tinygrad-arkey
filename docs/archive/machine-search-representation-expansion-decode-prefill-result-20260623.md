# Machine-Search Representation Expansion (Decode + Prefill) — Result (2026-06-23)

## Verdict: `MACHINE_SEARCH_REPRESENTATION_GAP_EXPLAINED` + `DECODE_SPEED_SEARCH_CLOSED_BUT_CODEGEN_REP_NEEDED` + `PREFILL_TILE_CONFIG_CLOSED_SCHEDULE_REP_NEEDED` + `DEEP_SEARCH_GATES_SCOPED` + `PURE_MACHINE_SEARCH_NOT_READY_UNTIL_REPRESENTATION_EXPANDS`
Designed the missing **representation layer** so machine search can operate on the primitives that currently only the
oracle/human path provides. Extends (does not replace) the existing SSOT (`extra/qk_search_spec.py` SearchRow + the
project ledger). One cheap prototype built and validated: the **schedule-interleave detector**. No defaults, no
kernels, no training, no speed claims.

## 1. Why current machine search lost to the oracle
**Not as an evaluator** — the gates, oracle, and W==D authority all work (Mode A/B and prefill tile-config ran
cleanly). It lost because **the oracle's winning primitives are outside the current SearchSpace**. Today's search
expresses `env_policy` + `tile_config` (+ schedule/lds_blocking enum stubs); the actual remaining wins live one or two
representation levels deeper:
- decode's biggest win was an **ABI/materialization transform** (whole-buffer identity) — a *human-discovered*
  boundary the search couldn't propose;
- the remaining prefill ~4–5% is a **K-loop software-pipeline + register-pool lifetime** schedule, not a tile config.

## 2. Oracle-only primitives outside current representation (`gap_inventory.json`)
| lane | primitive | search cannot express | upside |
|---|---|---|---|
| decode | buffer-identity ABI | ABI/materialization transform | solved (default-on) |
| decode | v_dot2 / cross-lane lowering | renderer lowering | learning |
| decode | strided whole-cache coalescing | cache-read schedule | <2% (long-ctx) |
| prefill | K-loop software pipeline | instruction-interleave / prefetch schedule template | ~4–5% |
| prefill | register-pool lifetime | dynamic VGPR lifetime/pool | enables the pipeline |

## 3. Representation taxonomy (`search_representation_taxonomy.json`) — 10 levels
`env_policy` · `tile_config` · `kernel_template` · **`abi_layout_transform`** · `isa_microprimitive` ·
**`schedule_template`** · **`register_lifetime`** · **`renderer_lowering`** · `cross_shape_policy` ·
`learned_primitive_spec`. The **bold** four are the new SearchSpace members the oracle wins require. Each level
carries: what it expresses, examples, gates+authority, current tools, missing tools, risk.

## 4. Decode plan (`decode_representation_plan.json`)
- **Solved/closed:** env_policy, tile_config, abi_layout_transform (buffer-identity, the +13–19% win).
- **Low-priority searchable (needs cache-read rep):** strided whole-cache coalescing — gates (route/no-E_49152/no-
  spill/byte-identical/W==D@2048-4096); stop unless >1 %@ctx4096 + no ctx512 regression (likely closes).
- **Learning/codegen (no W==D):** v_dot2 lowering, cross-lane-reduce lowering.
→ `DECODE_SPEED_SEARCH_REMAINS_CLOSED` + `DECODE_CODEGEN_SEARCH_REPRESENTATION_NEEDED`.

## 5. Prefill plan (`prefill_representation_plan.json`)
- **Searched/closed:** BK/PAD/DBUF/waves, occupancy, LEANADDR/VALU, generic tile-config.
- **Needs `schedule_template` rep:** the K-loop software pipeline — a stage list expressing prefetch distance, A/B
  prefetch separately, global-load placement vs the WMMA span, ds store/load placement, waitcnt/barrier placement,
  WMMA group size, operand liveness, VGPR budget.
- **Needs `register_lifetime` rep:** accumulator / current-tile / next-tile-prefetch regions, lifetimes by K-loop
  stage, max-live VGPR, spill-reject, dynamic pool assignment (the enabler — PLRAB hit the 256-VGPR static wall).
→ `PREFILL_TILE_CONFIG_SEARCH_CLOSED` + `PREFILL_SCHEDULE_TEMPLATE_REPRESENTATION_NEEDED` + `PREFILL_REGISTER_LIFETIME_REPRESENTATION_NEEDED`.

## 6. New gate plugins (`gate_plugin_plan.json`)
- `abi_identity_gate` (extend the materialization checker) — no slice/reshape across the precompiled boundary.
- **`schedule_interleave_gate` — PROTOTYPED** (`extra/qk_schedule_interleave_detector.py`): classifies a kernel
  PHASED vs PIPELINED by loads/ds inside the WMMA span. Validated: `build_gemm_lds2`(down) → **PHASED** (0/8 loads in
  span); Tensile → **PIPELINED** (3/4 loads, 76/76 ds in span).
- `register_lifetime_gate` (VGPR/spill from the ISA audit + a missing liveness analyzer).
- `renderer_lowering_gate` (native-codegen microsearch + ISA target).
- `whole_path_authority_gate` (existing W==D / synced whole-prefill — the only promotion authority).

## 7. SearchRow extension (`search_spec_extension.json`)
Add SearchSpace members above + SearchRow fields: `representation_level`, `oracle_primitive_id`,
`candidate_template_id`, `schedule_template`, `register_budget`, `abi_transform`, `isa_targets`,
`reject_before_compile`, `reject_after_isa`, `authority_kind`, `learning_only`. Schema-only (no production edit here).

## 8. What can be machine-searched next vs deterministic hand-work
- **Searchable now:** still only env_policy/tile_config (both closed). Nothing new is *searchable* until a generator
  for the deeper levels exists.
- **Deterministic engineering (not search):** the prefill software-pipeline + register pool (hand-asm or a renderer
  software-pipelining pass), and the v_dot2/cross-lane renderer lowerings. These are *capability builds* that, once
  they exist, **turn the corresponding representation level searchable.**

## 9. What the LoRA primitive-space proposer should emit (`learning_loop_integration.json`)
A `SearchRow` with `representation_level` + `primitive_id` + bounded knobs + required gates + stop rules +
`authority_kind` (speed-search vs learning-only). It must **not** produce unbounded assembly, declare a speedup, or
bypass the deterministic gates (`PRIMITIVE_SPACE_PROPOSER_NOT_KERNEL_JUDGE`).

## 10. Next executable prototype (`prototype_recommendation.json`)
- **Prefill (recommended first):** `prefill_schedule_interleave_detector` — **done** (this task). Then, if
  authorized, a tiny `kloop_schedule_template_microkernel` (local correctness only) to prove a template can *emit*
  interleaving before any in-model attempt. Do **not** jump to a full hand-asm kernel.
- **Decode:** `decode_codegen_cross_lane_microsearch` (learning-only; extends the existing native-codegen microsearch).

## Files changed
New: `extra/qk_schedule_interleave_detector.py` (validated prototype) + this doc + 8 artifacts under
`bench/qk-machine-search-representation-expansion/` + 1 ledger entry. **No `tinygrad/` source, no default change, no
kernel implementation, no adapter training, no speed claim.** History preserved (superseding notes only).

## Git status
Clean before; adds 1 tool + 1 doc + 8 artifacts + 1 ledger line. Defaults unchanged.
