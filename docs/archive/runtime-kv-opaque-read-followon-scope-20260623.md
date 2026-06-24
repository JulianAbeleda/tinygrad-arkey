# Runtime-KV Opaque-Read Follow-On — Scope Stub (2026-06-23)

**Origin:** `docs/fused-flash-single-tile-result-20260622.md` → `FUSED_FLASH_SINGLE_TILE_SCOPE_REDUCED_TO_OPAQUE_READ`.
The single fused tile is infeasible (cross-split combine needs grid-wide sync, unsupported in HCQ) and is not
the lever anyway (llama uses two kernels). But the **existing** two-node owned tile already reads the cache
natively, so it is the opaque read needed to remove the ~1.4 ms KV copy (measured +1.5 ms / +8 tok/s). This
stub scopes that follow-on. **Not yet authorized to implement — this is the scope only.**

## Mission
Remove the full-MAXC KV-copy tax by pairing the **proven opaque KV append**
(`extra/qk_kv_cache_state_token.py`, microprobe PASS) with the **existing owned attention tile**
(`extra/qk_owned_flash_decode_graph_node.py::amdgcn_flash_decode`) reading the **persistent cache** directly —
so neither the append (write) nor the attention (read) goes through the functional `.after()` full-buffer copy.
Both are opaque `custom_kernel` nodes → a clean buffer write→read dependency, **no same-graph functional reduce
hazard** (the wall that blocked `KV_CACHE_STATE_TOKEN` with the default `gqa_coop_vec`).

## Why this is the bounded path (recap of the established facts)
- KV copy `E_49152` ~1.4 ms/token, MAXC-bound, **measured** wall transfer +1.5 ms / +8 tok/s (`docs/ffn-activation-gap-audit-result-20260622.md`).
- Opaque append works (symbolic-offset in-place write + capture/replay with changing `start_pos`) but in-model
  the **default functional `gqa_coop_vec` reduce** over the mutated cache re-hits the read-after-write hazard
  (`docs/kv-cache-stateful-jit-capability-result-20260622.md`).
- The owned tile is an **opaque** read (not a tinygrad reduce) → it does not trigger that hazard, and it already
  reads `[Hkv,MAXC,Hd]` native cache via `custom_kernel`.

## Target route (default-off, strict 8B/gfx1100 guard)
Env: `KV_OPAQUE_READ=1` (requires the owned-tile path; pairs with the opaque append). In `_attention`:
1. opaque append writes K/V into the persistent `cache_kv` slice (existing `kv_append_node`).
2. the **owned tile** reads `cache_kv[0,0]` / `cache_kv[1,0]` (the persistent buffer, ordered after the append)
   **instead of** `assigned_kv[0,0]` / `[1,0]` (the functional copy).
3. fallback to the canonical copy + `gqa_coop_vec` on any unsupported shape/device/exception.

Strict guard: AMD/gfx1100, B=1, T=1, Hq=32/Hkv=8/Hd=128, symbolic `start_pos`, decode-only.

## Gates (first funded gate)
1. **Graph identity:** captured TinyJit graph has the opaque append node + the owned tile reading `cache_kv`;
   **no `E_49152`/full-MAXC copy**; no `assigned_kv` materialization on this route.
2. **Append-graph ordering:** the tile reads the post-append cache (device-side ordering, **no host sync**).
3. **Correctness:** greedy **byte-identical** vs the post-warp baseline over ≥64 tokens; no stale KV across two
   prompts in one process; capture/replay with changing `start_pos`.
4. **W==D:** `>= +5 %@ctx1024` (the copy is flat, so the relative win is largest at short/medium ctx; expect
   ~74→~83 tok/s @1024), no ctx512 regression, tight spread.

## Open risks to resolve in the probe
- The append (write) and the owned tile (read) target the **same** `cache_kv` buffer in one graph — confirm the
  scheduler orders write→read for two opaque nodes **without** materializing a copy (the microprobe proved the
  append; this adds the opaque read in the same graph).
- `@function(precompile=True)` must track the append mutation so the cache persists across calls **without**
  the `cache_kv.uop` repoint that reintroduced the reduce hazard — here the consumer is an **opaque** node, so
  the hazard mechanism should not fire; verify empirically.
- The owned tile's W==D attention contribution is unchanged (still ~+5.7 % ceiling); the **win here is the copy
  removal**, independent of attention speed. Do not conflate.

## Verdicts
- `KV_OPAQUE_READ_WD_PASS` (copy gone + W==D ≥ +5 %@1024, byte-identical) — promote to default-off candidate.
- `KV_OPAQUE_READ_LOCAL_PASS_WD_FAIL` (copy gone, transfer < gate) — record and rest.
- `KV_OPAQUE_READ_HAZARD_PERSISTS` (the two-opaque-node same-graph read-after-write still materializes/ fails) — the copy is unavoidable without runtime-managed (two-graph) KV; escalate to Design C.
- `KV_OPAQUE_READ_CORRECTNESS_FAIL` — byte mismatch.

## Non-goals
No default change; no 14B/32B; no native linearizer/renderer; no paged attention; no new attention tile (use
the existing two-node owned tile); no combine-only tuning; no claim from graph-identity alone (W==D required).

## First deliverables if funded
`extra/qk_kv_opaque_read_probe.py`; `bench/qk-kv-opaque-read/{graph_route,wd}.json`;
`docs/runtime-kv-opaque-read-result-20260623.md`; the gated `KV_OPAQUE_READ` route in `model.py` (`[nn]`) only
if all gates pass.
