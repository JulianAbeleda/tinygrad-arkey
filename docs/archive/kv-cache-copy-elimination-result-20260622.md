# KV-Cache Copy Elimination — Result (2026-06-22)

## Verdict: **KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED**

The KV-cache rematerialization copy (`E_49152`, ~1.4 ms/token, O(`max_context`)) is **real and
transferable** (audit-confirmed +1.5 ms / +8 tok/s), but **eliminating it requires core-JIT/scheduler
support that does not exist today**: in-place mutation of a captured buffer inside the
`@function(precompile=True)` pure decode function (variant 1), or a symbolic-sized `.after()` buffer
(variant 2). Both local branches fail at schedule/realize time with core-scheduler errors. Per the scope,
the broad core-JIT redesign is an **explicit stop/classify/defer** outcome — **not undertaken here**. No
default change; the gated experiment was reverted (`model.py` is byte-clean).

## 1. What changed
Nothing shipped. Two default-off gated variants (`KV_CACHE_INPLACE=1`/`2`) were added to
`TransformerBlock._attention` (model.py:951), probed, and **reverted**. The canonical full-buffer
`.after()` path is unchanged; post-revert greedy decode is byte-identical
(`[279, 1156, 22148, 18495, 1033, 5798, 304, 279]`). New: `extra/qk_kv_cache_copy_probe.py`,
`bench/qk-kv-cache-copy-elimination/latest.json`.

## 2. Why the original copy exists
`model.py:952`: `assigned_kv = Tensor(self.cache_kv.uop.after(<slice>.store(stack(k,v))))`, consumed by the
flash-decode path (`flash_decode_attention(..., assigned_kv[0,0], assigned_kv[1,0])`, :1014). The store
writes only the new `[start_pos:start_pos+T]` slice, but `.after()` is taken on the **full** `cache_kv`, so
the scheduler **materializes the whole `[2,B,KvH,MAXC,Hd]` buffer** to honor the after-ordering as a pure
(non-aliasing) value — kernel `E_49152` (one K/V layer-slice = `8·4608·128 = 4,718,592` elems). The copy is
the price of functional purity: there is no in-place mutation, so the post-store buffer must be a fresh
materialized value, not an alias of `cache_kv`. This is the upstream idiom (#15780); the in-place
`.assign()` was already commented out for this reason ("the function API doesn't support this well").

## 3. Branches tested (both LOCAL to the decode path; the diff that was reverted)
```python
_kv_inplace = getenv("KV_CACHE_INPLACE", 0)
if _kv_inplace and isinstance(start_pos, UOp) and T == 1:
  try:
    if _kv_inplace == 2:   # slice-scoped .after(): order on the [0:start_pos+T] prefix, not the full buffer
      pre = self.cache_kv[:, :, :, 0:start_pos+T, :]
      assigned_kv = Tensor(pre.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
      k = assigned_kv[0, :, :, 0:start_pos+T, :]; v = assigned_kv[1, :, :, 0:start_pos+T, :]
    else:                  # in-place .assign(): O(1) write; assigned_kv/k/v are VIEWS of cache_kv (no copy)
      self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
      assigned_kv = self.cache_kv
      k = self.cache_kv[0, :, :, 0:start_pos+T, :]; v = self.cache_kv[1, :, :, 0:start_pos+T, :]
  except Exception: <canonical fallback>
else: <canonical>
```

## 4. Correctness result
Baseline (canonical) greedy decode over a real prompt prefill + 8 tokens =
`[279, 1156, 22148, 18495, 1033, 5798, 304, 279]`. **Neither variant produced tokens** — both crashed at
schedule/realize time (before any output), so correctness is moot; the blocker is upstream of token
generation. (The decode-path `try/except` does **not** catch these — they fire during `realize()`, not
during the `_attention` python call.)

## 5. Kernel identity / copy-shrink result
Not reached — the variants never scheduled, so no copy-shrink could be measured. The target copy identity
(`E_49152`, pure float→float move, O(MAXC), flat across ctx) is established in
`docs/ffn-activation-gap-audit-result-20260622.md`.

## 6. W==D result
Not reached — the W==D wall sweep only runs for a correct, JIT-stable variant. Gates failed at Phase 1
(tractability), so Phases 3–4 did not execute.

### The two core-scheduler blockers (the actual finding)
| variant | branch | error at realize | meaning |
|---|---|---|---|
| 1 | in-place `.assign()` + `assigned_kv = cache_kv` view | `KeyError: UOp(Ops.REDUCE, half, (Ops.ADD,(4,)) ...)` | read-after-write hazard: mutating `cache_kv` while the same graph reads it as a view breaks the pure-function schedule (the flash reduce uop can't be mapped). |
| 2 | slice-scoped `.after()` on `[0:start_pos+T]` | `ValueError: eval failed to be a single number ... ((start_pos*1024+1024)!=4718592)` | symbolic-sized `.after()` target: the scheduler can't decide whether the symbolic-length prefix aliases the full `4718592`-elem buffer. |

Both are **core scheduler/JIT** limitations, not decode-path bugs (`assigned_kv` is correctly defined in both
branches). Eliminating the copy needs either (a) in-place buffer mutation with read-after-write inside a
`@function(precompile=True)` pure function, or (b) symbolic-sized after-buffers — each a **broad core-JIT
change**.

## 7. Default decision
**No change.** Variants reverted; the canonical full-buffer copy remains the default. The copy is a known
~1.4 ms/token (O(MAXC)) tax that **cannot be removed by a bounded local change**.

## 8. Corrected remaining-gap table (lanes restated)
| lane | status |
|---|---|
| Weight-GEMV (gate/up, down, proj, lm_head) | **closed** — at/below llama after `Q4K_GEMV_WARP`. |
| FFN activation | **closed** — mapping artifact (silu fused into gate/up GEMV; the bytes are the KV-copy). Not disproved. |
| Norm/rope/small-ops | **closed** — genuine norm at parity (−0.21 ms). Not disproved. |
| Attention | **not reopened** — B5 overlap / bounded-lever-exhausted, codegen-blocked deeper. |
| **KV-cache copy** | **`JIT_BLOCKED`** — real & transferable, but elimination needs a core-JIT/scheduler capability (deferred, not redesigned here). |

## 9. Remaining 8B gap after this result
The largest bounded *local* lever (KV-copy) is blocked at the core-JIT layer. **No bounded 8B decode
implementation remains** at the model/primitive level: weight-GEMV is closed; activation/norm are
artifacts/parity; attention's bounded lever is exhausted; the KV-copy and deeper attention both require
core-JIT/codegen capability work, not bounded primitives. The next decision is a **new decision doc** about
whether to fund a core-JIT capability (in-place KV mutation / symbolic after-buffers, which would also help
the flash-decode lane), **not** opportunistic scope expansion into the closed lanes.

## 10. Artifacts and commands
- `extra/qk_kv_cache_copy_probe.py` (subprocess-isolated; needs the §3 gated variants re-applied to reproduce — they were reverted).
- `bench/qk-kv-cache-copy-elimination/latest.json` (baseline tokens + both blocker error signatures).
- Repro: re-apply §3 to `model.py:951`, then
  `DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 PYTHONPATH=. .venv/bin/python extra/qk_kv_cache_copy_probe.py`.
- Scope: `docs/kv-cache-copy-elimination-implementation-scope-20260622.md`; audit: `docs/8b-exhaustion-next-implementation-decision-20260622.md`.

## Verdict (restated): `KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED`
The bounded local probe proved a core-JIT change is necessary; that broad redesign is intentionally **out of
scope** (stop/classify/defer), to be re-scoped separately on its own cost/benefit.
