# Cut list — what the refactor made removable

Branch `refactor/lowering-architecture` @ `bffbc256c`. Master retains everything listed here, so deletion is
recoverable by `git show master:<path>` — that is the stated safety net for this phase.

Derived from an import-graph walk over the whole tree, including the lazy `_attr("extra.qk.X", "name")` string
form, so "no importer" here means no *dynamic* reference either. Line counts are raw file lines; the `sz.py`
budget counts tokens and excludes docstrings, so budget impact is smaller than LOC in the prose-heavy modules.

Current: **35,536 / 50,000 budgeted**. Master: 34,313. Tier 1 alone returns the branch to ~34,750 — under the
old 35,000 cap, which is what makes the cap raise safely reversible at merge.

---

## Tier 1 — dead in production. Delete.

Nothing outside `test/` imports these, statically or dynamically.

| path | lines | evidence |
|---|---|---|
| `tinygrad/codegen/passes.py` | 389 | zero prod importers. Its own §LR-032b comment explains it *cannot* be made load-bearing — the registry describes passes at function granularity, `graph_rewrite` names them at call granularity, and only 7 of 64 join. It is a findings document living in budgeted core source. |
| `tinygrad/llm/kernel_specs.py` | 197 | zero prod importers. Its docstring opens "This module is INERT". |
| `tinygrad/codegen/plan.py` **lines 129-327** | ~199 | `TargetCapabilities`, `ResourceBudget`, `OptimizationPlan`, `validate()`, `PlanReapplied`, `PlanRejected`, `_gate_truthy`. `TargetCapabilities`' only prod consumer is `kernel_specs.py`, which is also in this tier. |

**`tinygrad/codegen/plan.py` lines 1-128 MUST SURVIVE.** `PLAN_GATES`, `GATE_READERS`, `observed_gate_value(s)`
build the `to_program` cache key (`codegen/__init__.py:26,519`). This is the one part of the LR-051/LR-019 work
that became load-bearing, and deleting the file wholesale would silently revert the cache-key fix.

Accompanying tests: `test_pass_registry.py` (206), `test_kernel_specs.py` (97), and the plan-machinery portion of
`test_optimization_plan.py` (~220 of 340). **Keep** that file's LR-019 block — `test_gate_readers_match_the_real_call_sites`,
`test_no_codegen_gate_is_missing_from_the_inventory`, `test_the_excuse_lists_are_not_a_dumping_ground`,
`test_observed_values_are_what_a_pass_sees_not_what_environ_says`,
`test_the_cache_key_no_longer_moves_without_the_program_moving` — those gate live behaviour.

**Core budget freed: ~785 lines.**

---

## Tier 2 — gated recorders, inert on the default path. Judgment call.

Each is wired to a real hook but returns immediately unless its env gate is set; the default-path cost is one
module-level bool test. They were built to answer a Phase-4 ownership question that is now answered.

| path | lines | gate | note |
|---|---|---|---|
| `tinygrad/schedule/scopes.py` | 204 | `composite_scopes.ENABLED` | called at `indexing.py:221,236` |
| `tinygrad/schedule/buffer_plan.py` | 133 | `BUFFER_PLAN` | **`BufferizeOpts` (~10 lines) is live** — `indexing.py:18` re-exports it and `rangeify.py` uses it. Move it back to `indexing.py` before deleting the rest. |
| `tinygrad/schedule/plan.py` | 125 | `REALIZE_PLAN` | `rangeify.py:465`, "inert unless REALIZE_PLAN is set" |

**Recommendation:** cut, after the GPU e2e run confirms the branch is sound — an inert observer is exactly the
"obsolete probe retained after its verdict is recorded" the principles name. ~452 further lines.

---

## Tier 3 — shims and resurrections

- **`extra/qk/coalesced_load_lowering.py`** (12 lines) — a pure re-export of `coalesce_loads`. This is the same
  shape as `extra/qk/kernel_pipeline.py`, which this branch deleted for being exactly that. Repoint its three
  callers (`cooperative_stage_lanemap.py`, `layout_coalesce_check.py`, `test_coalesced_load_lowering.py`) at
  `tinygrad.codegen.late.coalesced_load` and delete. Leaving it is an inconsistency, not a risk.
- **Four test files master deliberately retired** and this branch re-added along with the now-deleted
  `kernel_pipeline.py` shim: `test_scheduler_output_tile_loop.py` (115), `test_grouped_dot_update_pipeline.py`
  (94), `test_hierarchical_kernel_pipeline.py` (73), `test_kernel_pipeline_resource_plan.py` (40) — 322 lines.
  They now import from core and pass. **Decision needed:** they test real core contracts, so re-retiring them
  removes coverage master chose to drop; keeping them partly reverses that decision. Not a correctness issue
  either way, but it should be deliberate rather than inherited.

---

## Tier 4 — pre-existing orphans this audit surfaced

Not caused by the refactor; zero references anywhere in the tree. Listed because the cut phase is the moment to
deal with them (338 lines):

`extra/qk/prefill/prefill_long_context_numerics.py` (76), `extra/qk/phase_abi_v1_resource_probe.py` (72),
`extra/qk/q4k_fused_mmq_contract.py` (65), `extra/qk/prefill/prefill_flash_perf.py` (55),
`extra/qk/benchmark_split_shared_attention.py` (42), `extra/qk/packed_wmma_canary_evidence.py` (28).

`extra/` is unbudgeted, so this frees no budget — it is legibility only. Check each against the organization
manifest before deleting; `codebase_organization_audit.py` will fail on a stale record, as it did for
`kernel_pipeline.py`.

---

## Do NOT cut — these became live

| path | lines | why |
|---|---|---|
| `tinygrad/schedule/realize.py` | 46 | pure move of the real realization-map pass; `indexing.py` runs it |
| `tinygrad/codegen/late/coalesced_load.py` | 98 | the promoted pass itself, called from `codegen/__init__.py` |
| `tinygrad/uop/trace.py` | 174 | powers the pass-order gate; wired at `ops.py:2656` |
| `tinygrad/uop/invariants.py` | 100 | wired at `ops.py:2658`; the codegen-stage check now fires (255 checks/`to_program`) |
| `tinygrad/codegen/plan.py:1-128` | 128 | builds the `to_program` cache key |
| `extra/audit/*.py` | ~1,000 | the four gates; unbudgeted, and they are how any cut gets verified |

---

## Order of operations

1. **GPU e2e first** — accuracy and speed, 8B then 14B. Everything green so far is compile-only; nothing on this
   branch has executed a kernel or produced a token. Cutting before that would mean debugging two changes at once.
2. Tier 1, then re-run all four gates + unit suite. Expect byte-identical codegen: nothing here executes.
3. Tier 2 and 3, same verification.
4. Re-run GPU e2e and compare against step 1 — not against remembered numbers.
5. Re-decide `sz.py`. After Tier 1 the branch is under 35,000 and the 50,000 cap can drop back rather than being
   inherited by a merge.
