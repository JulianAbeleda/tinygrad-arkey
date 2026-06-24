# Project Search Ledger Contract (2026-06-23)

## Purpose
One project-wide memory for every machine-search lane (decode, prefill, codegen, cross-shape, small-op), so search
results stop being fragmented per-lane JSON and become a single durable, queryable record:

```text
candidate -> lane -> knobs -> gates -> authority benchmark -> verdict -> artifact links -> learned rule
```

Tool: `extra/qk_project_search_ledger.py` (append-only JSONL). Store: `bench/qk-project-search-ledger/ledger.jsonl`
+ `schema.json`. Status: **`PROJECT_SEARCH_LEDGER_READY`** — seeded with 9 real entries (decode Mode-A ×6, the
buffer-identity decode win, the kv_proj prefill win, the native-codegen expressibility experiment).

## Entry schema (15 fields)
| field | meaning |
|---|---|
| candidate_id | stable id (`lane/family/name`) |
| lane | `decode` / `prefill` / `codegen` / `cross-shape` / `small-op` |
| primitive_class | `attention` / `GEMM` / `ABI` / `fusion` / `route_policy` / `codegen_microprimitive` |
| knobs | the bounded knobs varied |
| oracle | comparator id + artifact |
| correctness | pass/fail + method (byte-identical greedy, rel_rmse, or ISA-evidence) — **checked before speed** |
| route_identity | candidate kernel present / absent (if applicable) |
| materialization_abi | E_49152 absent + buffer-identity (if applicable) |
| isa | audit verdict + key flags |
| local_diagnostic | optional, **never** an authority |
| authority_benchmark | **W==D / whole-prefill synced**, or an explicit non-promotion note (codegen) |
| verdict | enum final state |
| stop_reason | the first failed cost-ordered gate, or "passed all gates" |
| artifact_links | result/doc paths |
| learned_rule | the durable transferable lesson (promote to principles when general) |

## Inviolable rules (the discipline that made the wins transfer)
1. `authority_benchmark` MUST be a whole-path synced metric (W==D for decode, whole-prefill for prefill) **or** an
   explicit non-promotion microprimitive note. **Never** a local / PROFILE / no-sync / isolated-kernel timing.
2. Correctness before speed; route-identity + materialization/ABI before W==D; ISA before W==D when code changes.
3. Stop at the first failed gate; record it in `stop_reason`.
4. No default flip from a search harness; a win is a recommendation.
5. `learned_rule` is the point — a search that finds nothing but records *why* is still valuable (e.g. "default
   S=48 is the policy optimum"; "isolated GEMM is host-bound — use in-model GPU-busy").

## Usage
- Append after any search run: build a `qk_project_search_ledger.entry(...)` and append (or re-seed from result
  files via `--seed`).
- The ledger is the cross-lane index the per-lane result docs link into; principle #12 (buffer-identity ABI) and the
  "isolated-doesn't-transfer" lesson both originate as ledger `learned_rule`s.
