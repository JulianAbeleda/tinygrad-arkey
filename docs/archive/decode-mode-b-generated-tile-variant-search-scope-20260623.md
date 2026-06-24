# Decode Mode B — Generated Owned-Tile Variant Search — Full Scope (2026-06-23)

## Mission
Search **generated owned-tile kernel variants** (tile constants, not just env policy) against the frozen oracle,
under the readiness package's cost-ordered gates with **clean synced W==D as the only promotion authority**. Higher
risk than Mode A (it emits new kernels), so every variant must pass correctness + route-fire + materialization + ISA
**before** W==D, and the variant space is a closed enumerated grid — **no free-form kernel generation**.

Non-goal: chase 8B speed blindly. Decode is already at/above llama (oracle 90.6/89.3 tok/s; Mode A found the default
policy-optimal). Mode B exists to (a) prove the *generated-kernel* search loop end-to-end, and (b) confirm the shipped
tile constants are optimal — or surface a recommend-only winner. A negative result is success.

## Current state
- Oracle frozen: `bench/qk-decode-search-readiness/baseline_oracle.json` (`owned_flash_tile_gqa_whole`, W==D
  90.6@512 / 89.3@1024, byte-identical tokens, E_49152 absent, ISA 60 VGPR / 0 spill / v_dot2+LDS+cross-lane).
- Mode A (policy: S, min_ctx, combine) → `DECODE_SEARCH_EXECUTED_ORACLE_REMAINS_BEST`.
- The tile is `extra/qk_owned_flash_decode.hip` (`owned_flash_tile_gqa_whole`), built by
  `extra/qk_owned_flash_decode_graph_node.py` (`_specialize_tile` bakes `S`, `MAXC`, `#define TK 16`; geometry =
  4 warps × wave32 = 128 threads, GQA group `G=4`, `Hd=128`, `Hkv=8`).

## Required reading
`docs/decode-machine-search-readiness-package-result-20260623.md`, `docs/decode-machine-search-execution-result-20260623.md`,
`bench/qk-decode-eval/HARNESS_GUIDE.md`, `docs/project-search-ledger-contract-20260623.md`,
`docs/owned-tile-buffer-identity-kv-read-result-20260623.md` (the ABI invariant), `structure/Development/performance-primitive-research-principles.md` (esp. #12).
Inspect: `extra/qk_owned_flash_decode_graph_node.py`, `extra/qk_owned_flash_decode.hip`, `extra/qk_decode_search_gate.py`,
`extra/qk_isa_primitive_audit.py`, `extra/qk_project_search_ledger.py`.

## Bounded variant space (the ONLY search space)
A variant is a closed tuple of tile constants. Enumerated grid (≈24–36 variants max):
| knob | range | effect | constraint |
|---|---|---|---|
| `TK` (LDS position tile) | {8, 16, 32} | LDS staging depth + cooperative load loop | LDS bytes = `2·TK·Hd·2` ≤ kernel design; no occupancy collapse |
| `S` (split count, baked) | {32, 48, 64, 96} | KV-splits across workgroups | combine must cover S |
| `combine` variant | {base, hd64, hd128, sr64x2} | partial-reduction geometry | from `_combine_spec` registry |
| LDS-load vector width | {half2, half4} | `ds_load_b64`/`b128` staging | must stay 16-aligned (b128) |
| position-loop unroll | {1, 2, 4} | inner `tt` loop unroll | VGPR ≤ envelope |
**Held fixed (coupled to correctness):** workgroup = 4 warps (the GQA `G=4`→warp→q-head map), `Hd=128`, `Hkv=8`,
`v_dot2` use, cross-lane reduce, whole-cache buffer-identity ABI. **Disallowed:** changing the GQA mapping, the ABI,
or adding new buffer inputs.

## Generation mechanism
Extend `_specialize_tile` to accept `(TK, vec_width, unroll)` and bake them, emitting a **uniquely-named** symbol per
variant — `owned_flash_tile_gqa_whole_tk{TK}_v{W}_u{U}` — so route-fire and ISA attribution bind to the *specific*
variant (not a cached default). Each variant → its own `.co`, **hashed**; the candidate records the hash. The default
route is untouched (variants live only behind the candidate env/flag, in the variant `.co` cache). No edit to the
shipped `owned_flash_tile_gqa_whole` symbol or the default constants.

## Phases
**P0 — Authority + oracle recheck.** Reuse the package: re-run the gate on the oracle; confirm W==D within the frozen
3% band, tokens byte-identical, route fires, E_49152 absent, ISA confirmed. `SEARCH_ORACLE_RECHECK_PASS` or stop.
Artifacts: `bench/qk-decode-mode-b-search/{authority,oracle_recheck}.json` (stamped via `qk_harness_contract`).

**P1 — Search plan.** Record mode=B, the enumerated grid, comparator=oracle, contexts (512/1024 first pass;
2048/4096 only for first-pass winners), gates, thresholds, runtime budget, and `generated_code_objects=true`.
`SEARCH_PLAN_READY` / `SEARCH_PLAN_TOO_BROAD_STOP` (stop if the grid > ~40).

**P2 — Candidate manifest.** One id per tuple: id, knobs, expected variant symbol, expected `.co` hash (after build),
expected ISA requirements (v_dot2/LDS/cross-lane present, no spill, VGPR ≤ 96), comparator=oracle, reason.
`CANDIDATE_MANIFEST_READY`.

**P3 — Cost-ordered evaluation (per variant, stop at first reject).**
1. **Build + standalone correctness** — compile the variant, run the graph-node's numpy reference (`_numpy_ref`),
   `rel_rmse ≤ 1e-3`. Reject `REJECT_CORRECTNESS` else. (Cheapest first — never W==D a wrong kernel.)
2. **Code-object integrity** — variant `.co` built + hash recorded; reject if build fails (`REJECT_BUILD`).
3. **Route-fire** — the variant symbol fires in the captured decode graph (and the firing `.co` hash matches the
   built one). `REJECT_ROUTE_NOT_FIRING` else.
4. **Materialization/ABI** — E_49152 absent, buffer-identity inputs (the variant must keep the whole-cache ABI).
   `REJECT_MATERIALIZATION` else.
5. **ISA audit** — JSON required; v_dot2 + LDS + cross-lane present, **0 spill/scratch**, VGPR ≤ 96, LDS ≤ envelope.
   `REJECT_ISA` else.
6. **Full correctness** — 64-token two-prompt byte-identical greedy vs oracle; ctx512 correctness. `REJECT_CORRECTNESS`.
7. **W==D authority** — only now: clean synced W==D (`PROFILE=0`, .item()/tok, 30 repeats), ctx512+1024; 2048/4096
   only if it beats oracle@1024 outside spread. ctx512 below oracle−spread → `REJECT_WD_REGRESSION`.
Append one stamped result per variant to `bench/qk-decode-mode-b-search/results.jsonl`.

**P4 — Leaderboard.** Rank PASS variants by W==D Δ vs oracle @ctx1024; secondary = worst-context regression,
spread-adjusted Δ, ctx512 safety, ISA/VGPR quality, knob-distance from default. `SEARCH_ORACLE_REMAINS_BEST` /
`SEARCH_LEADERBOARD_READY` / `SEARCH_NO_PASSING_CANDIDATES`.

**P5 — Winner recheck** (only if a variant beats oracle outside spread): full W==D 512/1024/2048/4096 × 3+ repeats,
byte-identical correctness, route/materialization/ISA re-audit, fallback sanity. `WINNER_RECHECK_PASS/FAIL`. **No
default flip** — a winner is a recommendation only.

**P6 — Decision + ledger.** `bench/qk-decode-mode-b-search/decision.json` + `docs/decode-mode-b-search-result-20260623.md`;
append every variant to the project ledger (lane=`decode`, primitive_class=`attention`).

## Gates → reject reasons (reuse the encoded set)
`REJECT_CORRECTNESS` · `REJECT_BUILD` · `REJECT_ROUTE_NOT_FIRING` · `REJECT_MATERIALIZATION` · `REJECT_ISA` ·
`REJECT_WD_REGRESSION` · `REJECT_WD_NO_TRANSFER`. W==D is the only authority; ISA/correctness/route are pre-W==D gates.

## Boundaries / stop rules
- No default flip; no edit to the shipped tile symbol or default constants; variants are additive `.co`s only.
- No GQA-mapping / ABI / buffer-input changes; any variant that drops v_dot2, LDS, cross-lane, or buffer-identity is
  rejected by design.
- Stop if: oracle drifts; grid > ~40; a variant requires changing the held-fixed geometry; build/correctness can't be
  made cheap; or W==D shows no transfer for the whole grid (record `SEARCH_ORACLE_REMAINS_BEST`).
- 13-field contract on every artifact; ledger append per variant; rejects stop at first failed gate.

## Final verdicts
`DECODE_MODE_B_EXECUTED_ORACLE_REMAINS_BEST` · `DECODE_MODE_B_EXECUTED_WINNER_FOUND_RECOMMEND_ONLY` ·
`DECODE_MODE_B_EXECUTED_NO_PASSING_CANDIDATES` · `DECODE_MODE_B_BLOCKED_*`.

## Claude prompt
You are in `/home/ubuntu/tinygrad-arkey` on `qk-prefill-flag-leak-resolution`. Decode is at/above llama and Mode A
found the default policy-optimal. Read+execute this scope + `bench/qk-decode-eval/HARNESS_GUIDE.md` +
`docs/decode-machine-search-readiness-package-result-20260623.md`. Run **Mode B generated owned-tile variant search**
over the bounded tile-constant grid only. Generate uniquely-named variant kernels (additive `.co`s; default symbol
untouched); gate cost-ordered (build+standalone-correctness → route-fire → materialization → ISA → full correctness →
W==D); clean synced W==D is the ONLY promotion authority; ISA JSON + 0-spill + v_dot2/LDS/cross-lane + buffer-identity
required per variant; append every variant to the project search ledger; stamp every artifact via
`qk_harness_contract`; reject at first failed gate; no default flip; no prefill; no 14B/32B. Final response: verdict,
harness compliance, oracle recheck, grid size, reject summary, leaderboard, winner recheck if any, recommendation,
files, git status.
