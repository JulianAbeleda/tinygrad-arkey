# Runtime-KV Core-Runtime Blocker — Result (2026-06-23)

## Verdict: `RUNTIME_GRAPH_LIFECYCLE_GAP` — runtime-KV is impact-justified but core-runtime-blocked

The full-MAXC KV materialization is **on the W==D critical path** (MAXC-shrink +11.8%@ctx1024 → ~llama parity at
MAXC=1280), so eliminating it is worth ~+11% decode. But the copy-free **opaque append fails in the full model
even with the now-fixed fp16 owned tile** — the blocker is the **core TinyJit/HCQ persistence lifecycle**, not a
bounded model-route primitive.

## Classification
| layer | finding |
|---|---|
| algorithm | KV append/read semantics are valid (microbench rel_rmse e-7) |
| work decomposition | not the blocker |
| memory movement | the full-MAXC materialization tax is REAL and on the critical path (MAXC-shrink +11.8%) |
| ISA/codegen | not the blocker (the opaque append kernel is byte-correct standalone) |
| **runtime/graph lifecycle** | **BLOCKER** — the canonical `cache_kv.after(store)` materialization provides BOTH the copy AND `@function` cross-replay persistence; removing the copy (opaque append) loses persistence and the model bakes from decode step 1 |
| W==D | potential transfer ~+11%, blocked by the lifecycle |

## Evidence the blocker is NOT what we previously suspected
- **Not the owned tile**: the fp32→fp16 dtype bug is fixed; the owned route is default-on and byte-identical. The
  opaque-append re-test **still bakes** with the fixed fp16 tile.
- **Not GraphRunner arg-patching**: previously proven correct (start_pos advances per replay).
- **Not buffer identity / cache rebase**: previously refuted.
- **Microbench passes**: opaque append + owned tile + fp16 cache persists across replays in isolation.
- The failure appears **only in the full-model decode** → it is the `@function`/replay persistence coupling.

## Stop rule (honored)
Per the scope: **do not attempt a broad TinyJit/HCQ alias/persistence redesign in this task.** Removing the
materialization without losing cross-replay persistence requires a **core tinygrad capability** (a runtime-owned,
mutable-across-replay cache buffer that the pure `@function` graph can read without materializing) — that is core
runtime work, to be scoped separately only if the owner explicitly asks.

## Bottom line
Runtime-KV stays **deferred as core-runtime work** (`RUNTIME_GRAPH_LIFECYCLE_GAP`). It is the single biggest
remaining decode lever (~+11%, → llama parity), but it is not a bounded kernel/route primitive. Next practical
(bounded) lane = small-ops/activation fusion (overlapped, uncertain transfer).
