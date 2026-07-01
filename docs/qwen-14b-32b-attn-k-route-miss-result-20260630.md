# Reduce Source Resolution (RSR0-RSR5) — attn_k route miss fixed, 14B +60%

Date: 2026-06-30

Status: the dominant 14B decode cost was resolved to a **route miss** (attn_k on the slow generic-dequant path) and
fixed. 14B decode **+60% W==D, byte-identical**. Default-off (DECODE_ROUTE_ATTN_K) pending the 8B protected-context
result. Scope: `docs/qwen-14b-32b-reduce-source-resolution-scope-20260630.md`. Hardware: gfx1100.

## RSR0/RSR1 — resolve the hottest reduce (`r_8_32_4_20_4_2_32`, 38%)

Built `extra/qk_decode_reduce_source_trace.py` (ordered `Compiled.profile_events`, per-reduce prev/next non-reduce
windows). The hot row fires **40 calls/step (once per layer)**, sits between the input-norm elementwise and the
attn_q GEMV. Micro-isolation was decisive:

- `attn_k` (5120→1024) is a **plain `nn.Linear`** (`in_prim_registry=False`), while `attn_q`, `attn_v`,
  `attn_output` are all quant primitives (Q4K g3 / Q6K partial routes).
- Isolating `attn_k(+k_norm)` emits exactly `r_8_32_4_20_4_2_32n1`.
- `attn_k` is GGUF **type 12 = Q4_K** — identical quant to `attn_q` (which routes via g3).

Root cause: `_q4k_policy` in `tinygrad/llm/model.py` covers `ffn_gate/up`, `ffn_down`, `attn_q`, `attn_output` — but
**omits `attn_k`**. So `attn_k` fell through to `policy_fallback` → plain `nn.Linear` → the slow lazy Q4_K→fp16
dequant + generic reduce (`r_8_32_4_20_4_2_32`), measured **38% of 14B decode**. A one-line oversight, not a
topology/split-K/kernel problem — consistent with KT (topology exhausted) and SK4A (FFN already efficient).

`RSR1_PASS_HOT_REDUCE_SOURCE_RESOLVED` (firm: route-missed attn_k GEMV).

## RSR2/RSR3 — fix (default-off, Branch C: route to the generated GEMV)

Add `attn_k` to `_q4k_policy` behind `DECODE_ROUTE_ATTN_K` (default 0, rollback). It then takes the same
primitive/g3 route as `attn_q` instead of the plain-Linear generic dequant.

## RSR4/RSR5 — correctness + W==D

Token-identical (14B and 8B greedy, MD5 match, flag on vs off). Clean W==D (`qk_decode_runtime_overhead`, synced,
host-sync 0%):

| model | ctx | attn_k OFF | **attn_k ON** | delta |
|---|---|---|---|---|
| 14B (g3-anyshape) | 128 | 27.8 | **44.5** | **+60%** |
| 14B (g3-anyshape) | 512 | 27.1 | **42.8** | **+58%** |
| 8B (default route) | 512 | 103.6 | 99.6 | **−4%** |

- **14B: `RSR5_PASS_TIER_A`** — +60% W==D, byte-identical, host-sync 0%. From the shipped baseline (25.5 tok/s)
  that is **1.75×**; **43% → ~67% of llama.cpp** (~65-66 tok/s). This is the single biggest win of the whole
  14B/32B track, and it is a route fix, not a new kernel.
- **8B: −4% regression** — but here `attn_k` routed via the `coop_partial` primitive (not g3, since 8B ran without
  g3-anyshape), and 8B's plain `attn_k` was already fast. This trips the protected-context regression gate, so the
  fix must NOT be global default-on.

## Disposition

- Keep `DECODE_ROUTE_ATTN_K` **default-off**. The winning large-model config is
  `DECODE_Q4K_G3_ANYSHAPE=1 DECODE_ROUTE_ATTN_K=1` (attn_k via g3), which gives 14B +60% byte-identical.
- Promote via a **profile-scoped route policy** (large dense Q4_K models), not a global flag — since default-on
  regresses 8B by routing attn_k through coop_partial. The clean promotion is: attn_k → g3 when the model is in the
  g3-anyshape class; leave 8B's shipped attn_k path unchanged.
- Memory: attn_k as a primitive stores Q4_K packed instead of a dequantized fp16 weight → **less** VRAM, not more.

## Ledger

| field | value |
|---|---|
| candidate_id | `route_attn_k_q4k_primitive` |
| profile_id | qwen3-14b/32b Q4_K decode gfx1100 |
| role | attn_k 5120→1024 (Q4_K) |
| resolved_source | plain-Linear generic dequant (`r_8_32_4_20_4_2_32`), 38% of 14B decode |
| fix | add attn_k to `_q4k_policy` (DECODE_ROUTE_ATTN_K, default-off) → g3 route |
| wd_delta | 14B **+60%** (27.8→44.5 tok/s), byte-identical; 8B −4% (coop_partial path) |
| status | validated candidate, default-off; promote profile-scoped for the g3-anyshape class |
| reopen/next | wire attn_k→g3 under the large-model route policy; 32B transfer (expected similar, run timed out — rerun); resolve remaining reduce rows (attention combine ~5% @ctx512) |
| replay | `DEV=AMD JIT=1 DECODE_Q4K_G3_ANYSHAPE=1 DECODE_ROUTE_ATTN_K=1 QK_MODEL=.../Qwen3-14B-Q4_K_M.gguf QK_CKPTS=128,512 python3 extra/qk_decode_runtime_overhead.py` |
