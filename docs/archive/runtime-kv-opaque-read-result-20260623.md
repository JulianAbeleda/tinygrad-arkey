# Runtime-KV Opaque-Read — Result (2026-06-23)

## Verdict: **KV_OPAQUE_READ_CORRECTNESS_FAIL**

The opaque-read mechanism is **proven correct in isolation** (Phase-1 probe PASS) and **removes the full-MAXC
copy in-model** (no `E_49152` when it fires; the **first** decode step is byte-identical). But **multi-step
JIT decode loses KV persistence** (garbage after step 1), because the canonical `cache_kv.after(store)`
materialization provides **both** the copy **and** the `@function` cross-replay persistence linkage — removing
the copy breaks persistence. The owned tile (the only viable opaque read) is additionally **ctx-restricted to
≥2048**. The broken model route was **reverted** (default decode byte-identical; `model.py` clean). This
re-confirms the architecture conclusion: removing the copy requires **runtime-managed KV**, not a single-graph
route.

## 1. What changed
Nothing shipped. A default-off `KV_OPAQUE_READ` route (opaque/canonical write + owned-tile read of the
persistent cache) was added to `_attention`/`_init_state`, probed, and **reverted** (the multi-step replay is
not correct, so it is not fallback-safe when it fires). New (kept): `extra/qk_kv_opaque_read_probe.py`
(Phase-1 probe, PASS). `model.py` and `extra/qk_kv_cache_state_token.py` are byte-clean (reverted).

## 2. Why this route is different from the failed `.assign()` / slice-`.after()`
The prior task's `.assign()` and slice-`.after()` failed because the **read** (the default `gqa_coop_vec`
*functional reduce* over the mutated cache) re-hit the read-after-write hazard. This route's premise: the owned
AMDGCN tile is an **opaque** read (a `custom_kernel`, not a tinygrad reduce), so it should not trigger that
hazard. **That premise is correct** — the Phase-1 probe and the first in-model step both pass with no crash.
The new wall is on the **write/persistence** side under `@function(precompile=True)` (below).

## 3. Probe result (Phase 1) — `KV_OPAQUE_READ_PROBE_PASS`
`extra/qk_kv_opaque_read_probe.py` (standalone, no `@function`): opaque append into the persistent cache +
owned-tile read of `cache_kv[0,0]/[1,0]`:
- **Correctness** vs numpy flash-decode: rel_rmse ~2–5e-7 at start_pos 511/1023/2047/4095 (exact).
- **No full-MAXC copy**; captured graph = exactly `[kv_append, owned_flash_tile_gqa, owned_flash_combine]`.
- **JIT capture/replay** with changing start_pos: all finite, **append before tile**, no crash.

The mechanism works. The failure is purely the in-model `@function` lifecycle.

## 4. Graph-identity result — copy IS removed
When the route fires (decode JIT captured at ctx≥2048): `E_49152` is **absent** from the captured decode
graph, `owned_flash` fires (38 nodes), no `gqa_coop_vec` fallback. **The full-MAXC copy is removed.** (Profile
artifact: `bench/qk-kv-opaque-read/graph_route.json`.)

## 5. Correctness result — first step correct, multi-step garbage
| | KV_OPAQUE_READ=1 | baseline |
|---|---|---|
| decode step 1 (captured, start_pos 2050) | **34208** | 34208 ✓ |
| steps 2–8 (replays, start_pos 2051+) | 151936, 151936, … (garbage) | 13, 279, 3974, … |

The **first** (captured) step is byte-identical. **Replays lose the KV write** → 151936 (out-of-range/NaN).

Two additional findings:
- **Owned tile is ctx-restricted to ≥2048.** The *canonical* owned tile (materialized) also returns 151936 at
  ctx~600 — its ctx-gate is for **correctness** (S=48 over-splits short KV to NaN), not just performance. So
  the opaque read cannot apply at ctx<2048.
- **JIT capture-ctx subtlety.** A single prefill+decode `TinyJit` captures at the prefill (short) ctx → the
  captured graph is canonical and the route never fires; only a fresh decode-jit captured at ctx≥2048 fires it.

## 6. Why (root cause)
The canonical `assigned_kv = Tensor(cache_kv.after(store))` materialization provides **both** the full-MAXC
copy **and** the `@function(precompile=True)` **cross-replay persistence** (it makes `cache_kv` a tracked
stateful in/out buffer). Every attempt to remove the copy breaks persistence:
- **opaque append + repoint `cache_kv.uop`** → REDUCE read-after-write hazard (`KeyError` at schedule) — crash.
- **opaque append, no repoint** → `@function` treats the opaque write as read-only → KV doesn't persist (only
  3/6 prefill positions written) → garbage.
- **canonical store + un-materialized owned-tile read** → persists for the captured step (first token correct)
  but **not across replays** → garbage after step 1.

The copy and cross-replay persistence are **coupled** in the functional model. This is the same
`@function` stateful-mutation wall the prior task hit (`KV_RUNTIME_MANAGED_CACHE_REQUIRED`), now confirmed from
the read side too.

## 7. W==D result
**Not reached.** Correctness (multi-step byte-identical) failed at Phase 4, so the W==D gate did not run.
Even had it persisted, the owned tile's ctx≥2048 restriction means the route could not apply at ctx1024 →
could not clear **+5%@ctx1024** regardless.

## 8. Candidate / default decision
**Not registered.** Correctness failed (multi-step), so per the scope `decode_attention_owned_amdgcn_single_tile`/
`runtime_kv_opaque_read_*` is **not** added to `bench/qk-decode-eval/candidates.json`. Defaults unchanged.

## 9. Remaining 8B gap after result
Unchanged. The ~1.4 ms KV copy stays. Removing it is re-confirmed to require **runtime-managed KV** (cache out
of the `@function` pure graph: a two-graph/runtime-owned cache where the append realizes separately and the
attention reads a runtime-owned buffer — the `KV_RUNTIME_MANAGED_CACHE_REQUIRED` / vLLM-TRT-LLM model). A
single-graph in-`@function` route cannot decouple the copy from persistence. The owned tile would also need a
short-ctx-correct variant (or a non-split single-workgroup read) to apply below ctx2048.

## 10. Artifacts and commands
- `extra/qk_kv_opaque_read_probe.py` → `bench/qk-kv-opaque-read/probe.json` (Phase-1 PASS);
  `bench/qk-kv-opaque-read/graph_route.json` (in-model findings).
- Probe: `DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_kv_opaque_read_probe.py`.
- In-model repro requires re-applying the reverted `KV_OPAQUE_READ` route to `_attention`/`_init_state`, then
  `... KV_OPAQUE_READ=1 ...` with a fresh decode-jit captured at ctx≥2048.

## 11. Files changed
Source: **none shipped** (the `KV_OPAQUE_READ` route in `model.py` `_attention`/`_init_state` was added and
**reverted**; `model.py` byte-clean, default decode byte-identical `[279, 1156, 22148, …]`). New:
`extra/qk_kv_opaque_read_probe.py`, `docs/runtime-kv-opaque-read-result-20260623.md`,
`bench/qk-kv-opaque-read/{probe,graph_route}.json`.

## 12. Working tree status
`model.py` and `extra/qk_kv_cache_state_token.py` clean (reverted). New probe + docs + bench artifacts only.
No default change; no kernel built; no 14B/32B; no new attention tile; no native codegen; closed lanes untouched.
