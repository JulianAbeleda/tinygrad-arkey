# Decode Oracle Explanation & Schedule-Diff — Result (2026-06-23)

## Verdict: `DECODE_ORACLE_EXPLAINED` + `DECODE_8B_SEARCH_SURFACE_EXHAUSTED` + `DECODE_LOW_PRIORITY_SLOPE_RESIDUAL_ONLY` + `DECODE_NATIVE_CODEGEN_LEARNING_ONLY`
Consolidated *why* the current decode oracle is best into machine-readable primitive rows + a family matrix. The
oracle wins because it satisfies the **whole lifecycle simultaneously**; nearby generated variants only change local
instruction schedule within the same primitive boundary, so they don't transfer. No defaults changed, no new kernels,
no broad reruns, no prefill.

### Baseline refresh (2026-06-24)

Decode lifecycle periodic recheck bundle `20260624-141949` re-validates the same oracle stack and adds
single-pass versioned closure for:

- pre/post oracle gate checks (`qk_decode_search_gate.py`)
- pre/post unknown-lockstep closure (`DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN`)
- current/long/legacy-capture W==D A/B sweep capture in one handoff bundle

Latest bundle pointer:
- `bench/qk-decode-lifecycle-recheck-bundle/latest.json`

## 1. What the oracle is
Owned AMDGCN **whole-cache buffer-identity** attention tile (`owned_flash_tile_gqa_whole`, `DECODE_ATTN_KV_IDENTITY=1`
default-on, ctx≥512): passes the whole `cache_kv.after(store)` (no slice/reshape across the precompiled boundary), the
kernel offsets K/V halves; fp16 cache/Q/K/V, fp32 online softmax+PV; split-KV `S=48`, `TK=16`, `VEC=1`, `UNROLL=1`,
`combine=base`, 4-warp wave32, GQA `G=4`. ISA: `v_dot2` + 8 KB LDS + `__shfl_xor` cross-lane, **60 VGPR, 0 spill**.
W==D (canonical warp stack) **102.4 / 100.7 / 98.2 / 93.6 tok/s @ctx512/1024/2048/4096 = 101–105 % of llama.cpp.**

## 2. Why it beats gqa / slice / materialization
- **vs gqa_coop_vec:** the owned split-KV tile (`v_dot2`+LDS+cross-lane+online softmax) is +12–22 % — structural.
- **vs slice route (`KV_IDENTITY=0`):** the slice feeds *sliced* cache views across the precompiled boundary, which
  callify materializes into the full-MAXC `E_49152` copy on the critical path. The oracle removes it → **+13–19 %**.
  That single ABI choice (buffer identity, principle #12) is the largest decode win of the campaign.

## 3. Why S/combine policy variants don't beat it (Mode A)
`S ∈ {32,64,96}` and `combine=hd64` are all within the oracle's spread; **S96 is −1.1 %** (more KV-splits add combine
overhead). `min_ctx=1024` is correctly rejected at route-fire (route doesn't fire at the ctx512 test point). The
default `S48/base` is the **policy optimum** — searched and closed.

## 4. Why generated tile constants don't beat it (Mode B)
14 generated variants (`TK ∈ {8,16,32}` × `VEC ∈ {1,2,4}` × `UNROLL ∈ {1,2,4}` + S/combine) all PASS every gate and
all land within spread (best TK8 +0.4 %; S96 −1.5 %). They change only **local instruction schedule/loop constants
within the same primitive boundary and the same ABI** — no material ABI or primitive change → W==D within noise. The
win was the *boundary* (buffer identity) + the *primitive* (owned tile), both already in the oracle.

## 5. ISA/resource proof the intended primitive is present
`AMD_ISA_PRIMITIVE_CONFIRMED`: `v_dot2` (fused fp16 dot), 8 KB LDS staging, `__shfl_xor` cross-lane reduce, fp16
vector loads, **60 VGPR, 0 scratch, 0 spill**. Mode B variants keep the same flags (only schedule differs); the slice
route keeps the same compute but adds `E_49152`.

## 6. Why the gain shrinks at ctx4096 (ctx-slope) — `CTX_SLOPE_NO_ACTION`
Two confirmed mechanisms: (a) the removed materialization is **ctx-flat** (~1.52 ms fixed) — a fixed saving is a
shrinking % as per-token time grows; (b) the whole-cache tile has a **steeper ctx-slope** (0.245 vs llama 0.172
ms/1k = **1.43×**, the strided whole-cache read), eroding the saved-ms by −0.090 ms/1k. tinygrad stays **above llama
through MAXC** (103 % @4096); projected crossover ~**ctx 8335 > MAXC 4608**. The one bounded lever (strided-read
coalescing) is **< 2 %, long-ctx only** — below the action bar.

## 7. Remaining headroom (all low-priority / learning)
- **Whole-cache strided-read coalescing** — < 2 %, long-ctx only, crossover beyond MAXC → not action-worthy for 8B.
- **Native v_dot2 / cross-lane codegen** — the renderer can't yet emit them (native-codegen microsearch); codegen-
  *learning* target, **not an 8B-speed requirement**.

## 8. Closed decode search surfaces
Mode A policy (S/combine/min_ctx) — searched/oracle-best. Mode B generated tile constants — searched/oracle-best.
Buffer-identity ABI — solved/default-on. → **`DECODE_8B_SEARCH_SURFACE_EXHAUSTED`.**

## 9. Surfaces that remain only for learning / generalization
Strided-read coalescing (low-priority residual); native-codegen v_dot2/cross-lane (codegen learning); cross-shape /
model generalization (deferred until owner selects a target). Free-form attention-kernel generation is **disallowed**
without a new primitive audit.

## 10. What machine search should do next for decode
Nothing for **8B speed** — it's exhausted and at/above llama. The decode search machinery (oracle, gates, ledger)
now points naturally at **cross-shape generalization** (a new oracle per target) and **native-codegen learning**
(v_dot2/cross-lane lowering), both already scoped. The decode oracle is fully explained and recorded as ledger
`learned_rule`s for a future primitive-space proposer.

## Files changed
New: this doc + 8 artifacts under `bench/qk-decode-oracle-explanation/` (authority, oracle_fact_sheet,
alternative_family_matrix, primitive_decomposition, static_isa_explanation, ctx_slope_explanation,
search_surface_decision, learned_rules) + 1 ledger entry. **No `tinygrad/` source, no default change, no new kernel,
no prefill.** Historical docs preserved (superseding notes only).

## Git status
Clean before; adds 1 doc + 8 artifacts + 1 ledger line. Defaults unchanged.
