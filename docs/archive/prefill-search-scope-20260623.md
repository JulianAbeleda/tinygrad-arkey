# Prefill Search Lane — Full Scope (attribution gate → per-shape GEMM config search) (2026-06-23)

## Mission
Define the prefill machine-search lane **and its unlock gate**. Prefill is currently at rest (graph-GEMM whole-prefill
~96–99.5 % of vendored Tensile, at/above llama after the kv_proj de-WG-starve fix). Per the project rules, **prefill
kernel search stays blocked until attribution proves a real, searchable, transferable role gap.** This scope is
therefore **two-phase**: Phase A is the attribution gate (cheap, must pass); Phase B is the actual per-shape GEMM
config search (runs only if Phase A unlocks it). The authority is **synced whole-prefill** — never isolated GEMM
TFLOPS (host-bound) and never nosync `qk_prefill_v2_measure`.

## Current state (the entry condition)
- Synced whole-prefill (`extra/qk_prefill_whole_synced.py`): graph-GEMM 3554/3468/3221/2796 @512/1024/2048/4096 =
  ~99.5 % of Tensile, ~91–116 % of llama (after `out_f≤1024 → BN64` kv_proj fix).
- Per-role attribution (`extra/qk_prefill_per_role_time_tax.py`, in-model PROFILE GPU-busy): ffn_gate_up at parity
  (63 TFLOPS, beats Tensile), ffn_down 89 % (Tensile +14 % deep-K), qo_proj 87 %, kv_proj fixed (21→32 TFLOPS).
- The only kernel residual on the tuned shape is **+23 % VALU address arithmetic** (deterministic leanness, not a
  tuning knob). Coverage is complete (all roles fire graph-GEMM).
- Stale "66 %" headline retired; isolated GEMM TFLOPS confirmed host-bound (diagnostic only).

## Required reading
`docs/prefill-per-role-transfer-attribution-result-20260623.md`, `docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`,
`docs/prefill-post-decode-parity-frontier-result-20260623.md`, `bench/qk-decode-eval/HARNESS_GUIDE.md`,
`docs/project-search-ledger-contract-20260623.md`, `structure/Development/performance-primitive-research-principles.md`.
Inspect: `extra/qk_prefill_whole_synced.py`, `extra/qk_prefill_per_role_time_tax.py`,
`extra/qk_prefill_graph_gemm_route.py`, `extra/gemm/rdna3_wmma_matmul.py` (`build_gemm_lds2`), `extra/qk_amd_gemm_*`.

## The unlock condition (ALL must hold to enter Phase B)
```text
1. a prefill role has MATERIAL residual whole-prefill time (> noise band, after the kv_proj fix);
2. that role's graph-GEMM kernel is actually active in-model (not a fallback);
3. a candidate config change TRANSFERS to whole-prefill synced (not just isolated/in-model GPU-busy);
4. a correctness harness exists (rel_rmse <= 2.08e-4 + byte-identical greedy);
5. ISA audit can reject bad variants (no spill, VGPR <= 256, LDS <= 65536);
6. expected whole-prefill gain exceeds the synced spread band.
```
If any fails → `PREFILL_SEARCH_REMAINS_NOT_READY` / `PREFILL_AT_REST_AFTER_KV_PROJ_FIX`.

## Phase A — Attribution gate (cheap; must pass before any search)
**A0 Authority + oracle.** Freeze the current prefill default as the oracle: synced whole-prefill
(3 reps + spread) @512/1024/2048/4096 + per-role in-model GPU-busy table, stamped via `qk_harness_contract`. Artifact
`bench/qk-prefill-search/authority.json`, `prefill_oracle.json`.

**A1 Post-fix per-role re-attribution.** Re-run `qk_prefill_per_role_time_tax.py` (graph-GEMM and Tensile routes) and
build the **role × shape × achieved-TFLOPS × %-of-parity** table. For each below-parity role (candidate: ffn_down
deep-K, qo_proj), estimate the **whole-prefill** time recoverable if it reached parity (convert GPU-busy delta → a
whole-prefill projection, then *verify* the projection is meaningful vs the synced spread). Artifact
`prefill_role_residual.json`.

**A2 Searchability decision.** Apply the unlock condition. Verdicts:
- `PREFILL_SEARCH_READY_ROLE_SPECIFIC` (which roles, expected synced gain) → enter Phase B for those roles only;
- `PREFILL_SEARCH_REMAINS_NOT_READY` / `PREFILL_AT_REST_AFTER_KV_PROJ_FIX` → stop, record why (likely: residual is
  deterministic VALU leanness or below the synced spread);
- `PREFILL_NEEDS_INTEGRATION_FIX_NOT_SEARCH` (if the gap is integration/policy, not a kernel knob).
Artifact `prefill_search_readiness.json`. **Stop here unless `PREFILL_SEARCH_READY_ROLE_SPECIFIC`.**

## Phase B — Per-shape GEMM config search (only if A2 unlocks)
The natural prefill search space: a **per-shape tile-config map** (the kv_proj fix generalized). For each unlocked
role shape `(M=512, N, K)`, search `build_gemm_lds2` configs.

**Bounded config knobs:**
| knob | range | constraint |
|---|---|---|
| `WAVES_M × WAVES_N` | {1,2,4}² | `BM=WAVES_M·WM·16`, `BN=WAVES_N·WN·16`; M%BM==0, N%BN==0 |
| `WM × WN` (WMMA tiles/wave) | {2,4} | VGPR `SCR+2 ≤ 256` |
| `BK` (K-block depth) | {16, 32} | K%BK==0; BK64 overflows 256-VGPR (excluded) |
| `PAD` (LDS row pad) | {0, 16} | mult-of-16 (b128 align); `BUFSZ·NBUF ≤ 65536` |
| `PLRA` (A-prefetch) | {0, 1} | KT==2 single-buffer only |
| `DBUF` (double-buffer) | {0, 1} | LDS ≤ 65536 |
Workgroup-fill heuristic seeds the grid: target ~96–192 workgroups (`grid = (N/BN)·(M/BM)`), so small-N roles get
smaller BN (the kv_proj lesson). ≤ ~16 configs per role.

**B-gates (cost-ordered, per config):**
1. **Build** — `build_gemm_lds2` asserts pass (VGPR/LDS bounds); reject `REJECT_BUILD`.
2. **Numerical correctness** — `rel_rmse ≤ 2.08e-4` vs numpy (`test_lds_gemm2` harness); reject `REJECT_CORRECTNESS`.
3. **ISA audit** — JSON; WMMA `v_dot` present, **0 spill/scratch**, VGPR ≤ 256, LDS ≤ 65536; reject `REJECT_ISA`.
4. **In-model role attribution (diagnostic, NOT promotion)** — wire the config into the role via the graph-GEMM
   route, re-profile per-role GPU-busy; the config must improve *that role's* in-model GPU-busy. Reject
   `REJECT_NO_INMODEL_GAIN` (a config faster only in isolation is host-bound noise — recall the trap).
5. **Whole-prefill W==P authority** — synced whole-prefill (`qk_prefill_whole_synced.py`, 3 reps + spread) vs the
   prefill oracle. The ONLY promotion authority. Reject `REJECT_WP_NO_TRANSFER` / `REJECT_WP_REGRESSION`.
6. **Full correctness** — byte-identical greedy on 2 prompts (the route's standing gate).

**B-rank.** Rank PASS configs by whole-prefill synced Δ vs oracle; secondary = per-role GPU-busy, ISA/VGPR quality,
config simplicity. A winner is a recommend-only **per-shape config entry** (additive to the route's `_kernel` map),
**not a default flip**.

## Harness authority (strict)
| authority | use |
|---|---|
| **synced whole-prefill** (`qk_prefill_whole_synced.py`) | **promotion/ranking — the only authority** |
| in-model per-role PROFILE GPU-busy (`qk_prefill_per_role_time_tax.py`) | attribution / config selection only |
| isolated GEMM TFLOPS (`test_lds_gemm2`, host-bound) | local correctness only — **never** promotion |
| nosync `qk_prefill_v2_measure` | **forbidden** (documented inflation trap) |

## Boundaries / stop rules
- Synced whole-prefill is the only promotion authority; isolated/in-model GPU-busy are diagnostic.
- No default flip; a winning per-shape config is a recommendation (additive map in `_kernel`, gated on owner go).
- No prefill behavior change for non-searched roles; no attention-compute (QK/PV) kernel work (out of GEMM scope).
- No 14B/32B (cross-shape is a separate lane).
- Stop if: Phase A doesn't unlock (the expected case → at rest); a config only wins in isolation (host-bound); the
  search would need BK>32 (VGPR overflow) or a non-tile-divisible shape; or whole-prefill shows no transfer.
- 13-field contract on every artifact; append every config to the project ledger (lane=`prefill`, class=`GEMM`).

## Required artifacts
`bench/qk-prefill-search/{authority, prefill_oracle, prefill_role_residual, prefill_search_readiness}.json`; if Phase
B runs: `{search_plan, candidate_manifest, results.jsonl, leaderboard, decision}.json`. Result doc
`docs/prefill-search-result-20260623.md`.

## Final verdicts
Phase A: `PREFILL_SEARCH_READY_ROLE_SPECIFIC` · `PREFILL_SEARCH_REMAINS_NOT_READY` · `PREFILL_AT_REST_AFTER_KV_PROJ_FIX`
· `PREFILL_NEEDS_INTEGRATION_FIX_NOT_SEARCH`. Phase B: `PREFILL_SEARCH_EXECUTED_ORACLE_REMAINS_BEST` ·
`PREFILL_SEARCH_EXECUTED_WINNER_FOUND_RECOMMEND_ONLY` · `PREFILL_SEARCH_EXECUTED_NO_PASSING_CANDIDATES`.

## Expected outcome (honest)
Phase A most likely returns `PREFILL_AT_REST_AFTER_KV_PROJ_FIX`: whole-prefill is already ~99.5 % of Tensile, and the
ffn_down/qo residual is small and partly deterministic VALU leanness, so the recoverable **whole-prefill** time is
likely below the synced spread band — i.e. not searchable. That is a valid result and the gate's purpose: it prevents
burning a GEMM config search on a non-transferring metric (the project's recurring lesson). Phase B runs only if the
attribution surprises us with a material, transferable role gap.

## Claude prompt
You are in `/home/ubuntu/tinygrad-arkey` on `qk-prefill-flag-leak-resolution`. Prefill is at parity after the kv_proj
fix. Read+execute this scope + `bench/qk-decode-eval/HARNESS_GUIDE.md` + `docs/prefill-per-role-transfer-attribution-result-20260623.md`.
**Run Phase A (attribution gate) first**; only enter Phase B (per-shape GEMM config search) if the unlock condition
holds. Synced whole-prefill is the ONLY promotion authority (isolated GEMM TFLOPS is host-bound — diagnostic only;
nosync `qk_prefill_v2_measure` is forbidden). Correctness (rel_rmse ≤ 2.08e-4 + byte-identical greedy) and ISA (0
spill) before whole-prefill. Append every candidate to the project search ledger; stamp every artifact via
`qk_harness_contract`; reject at first failed gate; no default flip; no attention-compute; no 14B/32B. Final response:
Phase A verdict, role residual table, searchability decision, Phase B result if entered, recommendation, files, git
status.
