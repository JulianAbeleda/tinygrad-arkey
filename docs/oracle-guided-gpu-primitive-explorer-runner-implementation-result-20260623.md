# Oracle-Guided GPU Primitive Explorer — Runner Implementation Result (2026-06-23)

## Verdict: `EXPLORER_RUNNER_IMPLEMENTED`

The generic spec-driven runner — the connective tissue that was previously design-only — now exists at
`extra/qk_oracle_gpu_primitive_explorer.py`. It makes a bounded search spec (or a `qk_search_spec.SearchRow`) actually
drive the existing decode backend end-to-end, with a learned-proposal safety guard. **No tinygrad source, no kernel,
no default flip; W==D remains the only decode promotion authority; the runner only recommends.**

## What was built

1. **Spec-driven candidate generation.** The runner consumes the explorer spec JSON (or a `SearchRow`) and enumerates
   candidates from the spec's declared `knobs_ranges` (cartesian product, dedup, `--max-candidates` cap) — replacing
   the old inline-literal grids in the executors.
2. **`SearchRow` → candidate adapter.** `searchrow_to_spec()` maps a `qk_search_spec.SearchRow`
   (`op_scope=attention`, `search_space=primitive_policy`) to the decode policy spec (S / combine / min_ctx); other
   rows route to the gated lanes. Semantic/LoRA knob names (`split_S`, `combine_variant`, `min_ctx`) alias onto the
   real env vars (`DECODE_ATTN_AMDGCN_*`).
3. **Knob/value validation (the proposer-safety guard).** Every candidate's knobs are checked against a per-lane
   allow-list; an unknown knob or out-of-range value marks the candidate **structurally invalid** (recorded, never
   benchmarked). This enforces the "no hallucinated tool/unsupported knob" rule a learned proposer needs — e.g. the
   scope's own example value `combine=hw128` is correctly rejected (`value_out_of_range:DECODE_ATTN_AMDGCN_COMBINE=hw128`).
4. **Per-lane gate/authority registry.** `decode_policy` is runnable (W==D); `native_codegen_microprimitive` is
   non-promotion (its own tool, ISA + local correctness); `prefill_role_policy` and `cross_shape` are gated and the
   runner refuses them with the honest lane verdict instead of silently doing nothing.
5. **Delegation, not duplication.** For decode it calls `qk_decode_search_runner.run_candidate` (which spawns the real
   cost-ordered gate: route-fire → E_49152 → buffer-identity → byte-identical tokens → ISA → W==D → ctx512
   regression), then ranks vs the frozen oracle and writes one project-ledger entry per candidate.
6. **`--dry-run` / `--no-ledger` / `--selftest`.** Dry-run enumerates + validates with no benchmark (the cheap gate a
   proposer/CI uses before spending GPU); `--no-ledger` suppresses the durable ledger append for proof/CI runs;
   `--selftest` exercises the adapter + validation with no GPU.

## Proof (run today)

- **Selftest (no GPU):** decode policy spec → 16 candidates enumerated; `combine=hw128` rejected as out-of-range;
  `SearchRow{op_scope=attention, search_space=primitive_policy}` maps to `decode_policy`.
- **Dry-run:** decode policy spec → `EXPLORER_DRY_RUN_OK` (16/16 structurally valid). Prefill placeholder →
  `EXPLORER_LANE_GATED` (`PREFILL_SEARCH_GATED_OFF_AT_REST`).
- **Real capped run (`--max-candidates 2 --no-ledger`, W==D):**
  `S32_COMBINEbase_CTX512` → **PASS** (d1024 −0.4 %, within spread); `S32_COMBINEbase_CTX1024` (min_ctx=1024) →
  **`REJECTED:route_not_firing`** (route correctly does not fire at the ctx512 W==D point — gate short-circuit working).
  Verdict `EXPLORER_DECODE_ORACLE_REMAINS_BEST`, `default_flipped=false`, ledger untouched. End-to-end path confirmed.

## What this does and does not change about "thoroughness"

It closes the **mechanism** gap: specs (including future LoRA-proposed `SearchRow`s) now drive real, gated, ledgered
searches through one runner, and bad proposals are rejected cheaply at `--dry-run`. It does **not** create new
searchable headroom: the decode policy/tile spaces remain oracle-best (this run reconfirms it), prefill stays gated,
and cross-shape still needs targets. Genuinely new search signal requires a new lane (e.g. small-op fusion) or new
targets (owner-gated), not more runner plumbing.

## Files changed

- New: `extra/qk_oracle_gpu_primitive_explorer.py` (the generic runner + adapter + validation + lane registry).
- Updated: `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md` (verdict → `EXPLORER_RUNNER_IMPLEMENTED`).
- This result doc.
- Per-run artifacts under `bench/qk-oracle-gpu-primitive-explorer/runs/` are reproducible proof outputs (timing-bearing)
  and are not committed.

**No `tinygrad/` changes, no kernel changes, no default flips, no ledger pollution (proof run used `--no-ledger`).**

## Next executable step

The runner is ready to drive a LoRA proposer's specs in shadow mode (`--dry-run` validation first, then a gated real
run). But per the readiness assessment, building the proposer is only worthwhile once there is searchable headroom —
so the higher-value next step is opening a new lane/target, not the proposer. The runner is the prerequisite that now
exists either way.
