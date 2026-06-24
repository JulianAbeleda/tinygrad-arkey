# Branch B Result: explicit attention on concrete KV — bogus refutation OVERTURNED, +1.16× fusion win (no WMMA)

Date: 2026-06-20
Repo: `/home/ubuntu/tinygrad-arkey`, branch `qk-prefill-flag-leak-resolution`. GPU gfx1100. Model Qwen3-8B-Q4_K_M.
Scope: `docs/prefill-branch-b-tc-attention-scope-20260620.md`. Step-1 audit: `docs/prefill-graph-route-attribution-result-20260620.md`.
Harnesses: `extra/qk_prefill_tc_attn_concrete_gate.py` (perf+correctness+kernel-identity),
`extra/qk_prefill_tc_attn_quality_gate.py` (dNLL+greedy).

## Headline

1. **The prior "TC attention REFUTED 0.79×" is INVALID and is overturned.** `qk_prefill_tc_attention_measure.py`
   set env `PREFILL_TC_ATTENTION` but the model reads `PREFILL_TC_ATTN` (typo → both arms ran SDPA), AND it
   bound a *symbolic* start_pos which fails the `isinstance(start_pos,int)` guard so the TC branch could never
   fire. The "symbolic KV blocks TC" conclusion was never tested.
2. **Measured correctly (concrete int start_pos=0, graph route ON both arms, same-process interleaved synced
   arbiter), the explicit attention path is a reproducible ~1.16× whole-forward win, byte-identical output.**
3. **But it is NOT tensor cores.** WMMA never fires on the attention matmuls (`tc_fired=False`, zero wmma
   kernels in either arm). The win is *better fusion of the attention reduce*, cutting attention from ~18% →
   ~5% of the concrete forward. The big llama-style prize (attention → ~5% via real flash/TC) is a separate,
   larger build (below).

## Measured (3 sessions, synced interleaved best-of-6, clock pinned high)

| arm | p50 ms/512 | tok/s | % of llama (170ms/3020) |
|---|---:|---:|---:|
| OFF (SDPA) | ~174.3 | ~2940 | ~97% |
| **ON (explicit attn)** | **~151.0** | **~3390** | **~113%** |

speedup (median of per-rep ratios): **1.161 / 1.152 / 1.157** across 3 sessions. OFF stable ~174 ms, ON stable
~151 ms; min-vs-min 1.15× → not a clock artifact (same-process interleaved).

### Gates (iron law) — ALL PASS
- correctness rel_RMSE(off,on) = **0.0** (byte-identical logits)
- quality: max_abs_dNLL = **0.0**, **greedy-exact** (argmax matches every window), finite — 4 windows
- kernel-identity: graphs differ ✓; **tc_fired = False** (no wmma) — recorded honestly, the win is fusion not TC
- synced arbiter, clock pinned, interleaved ✓

## Per-kernel (concrete start_pos=0, KV=512, graph route ON)

| | OFF (SDPA) | ON (explicit) |
|---|---|---|
| attention | `r_16_32_2_8_16_4_4_128_4` 10.7% + `r_32_16_8_8_16_4_4_32_4` 5.9% + small ≈ **~18%** | `r_8_32_32_…` 2.8% + small ≈ **~5%** |
| FFN graph GEMMs | ~73% | ~75% (bigger share; total shrank) |

The two big SDPA attention reduces vanish under the explicit formulation, replaced by one small reduce.

## Important reframe: concrete ≪ symbolic for attention

The Step-1 attribution measured attention at **47%** with a single 30% kernel — but that used a **symbolic**
start_pos (`vsp.bind(0)`). At a **concrete** int start_pos=0 the same attention is only **~18%** (the 30% kernel
`r_2_512_*start_pos*` becomes the 10.7% `r_16_32_*`). So symbolic-start_pos codegen makes attention ~3× more
expensive. Consequences:
- The promoted "graph route = 256 ms = 66% llama" ladder was measured **symbolic**; the real **concrete first
  chunk** (what `generate()` actually runs at start_pos=0, model.py:1285) is **~174 ms (97% llama) SDPA / ~151 ms
  (113% llama) with explicit attn** — much faster than the headline implies.
- The big attention cost lives in the **symbolic** (subsequent-chunk, long-context) regime, where the explicit
  path does **not** fire (needs concrete int). Capturing it there needs `PREFILL_CONCRETE_KV=1` (concrete every
  chunk, K-jit compile cost) and/or making the symbolic-start_pos attention codegen as cheap as concrete.

## Scope of applicability

By default only the **first 512-token chunk** (start_pos=0) is concrete → only it benefits. For a long prompt the
remaining chunks are symbolic and unaffected. So default-on Branch B helps short prompts (≤512) fully and long
prompts only on the first chunk. `PREFILL_CONCRETE_KV=1` extends it to all chunks at K-jit cost (separate gate).

## Decision

Branch B (explicit attention on concrete KV) is a **real, reproducible, byte-identical ~1.16× win on the
concrete-chunk regime** — and it **overturns** the prior refutation. It passes every iron-law gate. It is NOT the
tensor-core win; it is a fusion win on the attention reduce.

**PROMOTED default-on (owner-approved, 2026-06-20, commit `945c695d3`).** `PREFILL_TC_ATTN` now defaults on via
`_prefill_tc_attn_default()` — gfx1100-guarded, decided once at import like `_prefill_graph_gemm_default`;
`PREFILL_TC_ATTN=0/1` overrides. The route's `isinstance(start_pos,int)` guard confines it to concrete chunks
(start_pos=0 by default); symbolic chunks stay SDPA. Promotion gates all passed: rel_RMSE 0.0; dNLL 0.0 +
greedy-exact (4 windows); generation coverage 8/8 tokens identical default-on vs `=0` end-to-end; fallback
(non-gfx/non-AMD → off, symbolic → SDPA) + env override verified.

## The bigger prize (deferred): real flash/TC attention

To get attention from ~18–47% down to llama's ~4.4%, the attention matmuls must actually use WMMA (and ideally an
online-softmax flash kernel that avoids materializing the `Hq×T×KV` score tensor). Today WMMA does not fire
there: no warmstart TC-opt covers the attention shapes and we don't use BEAM. This is a real codegen build,
gated by the POWN WMMA-scheduling wall, and most valuable in the **symbolic/long-context** regime. Sequenced
after promoting the in-hand fusion win.

## Iron law
SYNCED only; same-process interleaved (no cross-process clock compare); rel_RMSE<1e-2 + dNLL≤0.01 + greedy-exact;
no BEAM; gfx1100; default-off unless owner-approved.
