# Project-Wide Machine-Search Roadmap — Result (2026-06-23)

## 1. Verdict: `PROJECT_MACHINE_SEARCH_ROADMAP_READY`
The project is now a **search-capable system** with the discipline intact. Of the five steps: **Step 1 executed**,
**Step 2 built**, **Step 3 determined**, **Steps 4–5 scoped**. Every lane has (or has a defined path to) the loop:
`bounded candidate space → cheap structural/ISA prune → correctness → authoritative whole-path benchmark → remember`.
No broad/random search started; no defaults flipped.

## 2. Step 1 — first real bounded decode search → DONE (`DECODE_SEARCH_EXECUTED_ORACLE_REMAINS_BEST`)
Mode A policy search ran (`extra/qk_decode_search_execute.py`, commit `fd5506c9c`): 6 candidates, 5 PASS all
cost-ordered gates, the `min_ctx=1024` probe correctly rejected at route-fire, **no candidate beats the oracle outside
spread** (S48-control +0.1% validates fidelity; S96 −1.1%). W==D-only authority; 8 artifacts CONFORMS 13/13. The
full search lifecycle is proven on a real run. (`docs/decode-machine-search-execution-result-20260623.md`.)

## 3. Step 2 — project-level search ledger → BUILT (`PROJECT_SEARCH_LEDGER_READY`)
One schema for all lanes: `extra/qk_project_search_ledger.py` + `bench/qk-project-search-ledger/{schema,ledger}.json[l]`
+ `docs/project-search-ledger-contract-20260623.md`. **Seeded with 9 real entries** consolidating the previously
fragmented results: decode Mode-A ×6, the buffer-identity decode win, the kv_proj prefill win, the native-codegen
expressibility experiment. The ledger enforces the inviolable rule that `authority_benchmark` is always a whole-path
synced metric (or an explicit non-promotion note), never a local timing — and carries the durable `learned_rule` per
candidate (principle #12, "isolated-doesn't-transfer", "default is policy-optimal").

## 4. Step 3 — prefill attribution gate → DETERMINED (`PREFILL_AT_REST_AFTER_KV_PROJ_FIX` / `PREFILL_SEARCH_REMAINS_NOT_READY`)
The synced whole-prefill authority shows graph-GEMM at **~96–99.5% of Tensile and at/above llama** after the kv_proj
de-WG-starve fix; the stale "66%" was retired; coverage is complete; the only kernel residual (+23% VALU) is a
deterministic leanness item, not a search knob. **Prefill search stays gated.** Its **unlock condition** is now
explicit (a role with material residual time, active in-model, local→whole-prefill transfer, oracle + ISA reject +
gain > noise). None hold today → prefill is at rest, not searchable. (`docs/prefill-per-role-transfer-attribution-result-20260623.md`,
`docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`.)

## 5. Step 4 — native-codegen microprimitive search → SCOPED (`NATIVE_CODEGEN_MICROSEARCH_READY_TO_SCOPE`)
`docs/native-codegen-microprimitive-search-scope-20260623.md`. The safest non-W==D lane: make tinygrad-native codegen
emit the owned tile's proven primitives. ISA-evidenced targets: **LDS is already native; `v_dot2` and `ds_bpermute`
(cross-lane) are the gaps**. Authority = ISA + local correctness, never decode speed. Recommended first run: the
cross-lane reduction target. Ready to execute as a separate bounded run.

## 6. Step 5 — cross-shape / generalization targets → SCOPED (`CROSS_SHAPE_SEARCH_NEEDS_TARGETS` → `CROSS_SHAPE_DEFERRED`)
`docs/cross-shape-generalization-search-targets-scope-20260623.md`. Target axes mapped (14B/32B, longer ctx, other
GPU, other quant) each with an explicit unlock condition (baseline oracle + correctness harness before any search).
Recommended first cross-shape search = the **prefill per-shape GEMM config map** at 14B (the kv_proj fix generalized).
**Deferred** pending owner authorization — standing rule: no 14B/32B without an explicit ask.

## 7. Global search rules (enforced across all lanes)
W==D / whole-prefill synced is the only promotion authority; PROFILE/DEBUG/raw/no-sync/local are diagnostic only;
correctness before speed; route-identity + materialization/ABI + ISA before W==D; stop at first failed gate; every
performance artifact satisfies the 13-field contract; no default flip from a search harness; no broad/random kernel
generation; no stale baselines.

## 8. Recommended execution order (and what's allowed now)
| step | state | allowed now? |
|---|---|---|
| 1 decode search | DONE | re-runnable any time (regression-safe) |
| 2 ledger | BUILT | append after every search |
| 3 prefill search | **GATED** (at rest) | **blocked** until the unlock condition is met |
| 4 native-codegen microsearch | SCOPED | **ALLOWED** (safe, non-promotion) — the recommended next execution |
| 5 cross-shape | DEFERRED | **blocked** pending owner target selection + oracle build |

**Allowed now:** decode policy/variant search (Mode A done; Mode B available), and native-codegen microprimitive
search (Step 4). **Blocked:** prefill kernel search (attribution gate), cross-shape/14B (owner-gated), any broad
autonomous search.

## 9. Files changed
New: `extra/qk_project_search_ledger.py`; `bench/qk-project-search-ledger/{schema.json,ledger.jsonl}`;
`docs/project-search-ledger-contract-20260623.md`; `docs/native-codegen-microprimitive-search-scope-20260623.md`;
`docs/cross-shape-generalization-search-targets-scope-20260623.md`; this result doc; the roadmap scope. README/handoff
updated. **No `tinygrad/` source, no default flips, no prefill/decode behavior change, no 14B/32B, no broad search.**

## 10. Git status
Clean before; adds 1 tool + 2 ledger artifacts + 4 docs + doc updates. Decode/prefill defaults unchanged.
