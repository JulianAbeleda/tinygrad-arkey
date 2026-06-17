# decode_attention_v3 prerequisite — WMMA custom-kernel revival: VERDICT (2026-06-17)

Prerequisite arc for `decode_attention_v3` (high-occupancy WMMA flash + cooperative GQA V-reuse). The gating
question: **can this repo express and run a SHAPED_WMMA custom_kernel tile again, correctly, through TinyJit,
with no CPU fallback?** Answer: **YES.** Phases 0–4 pass with proof and are committed. Phase 5 (GQA reuse) is
directionally positive but confounded; see below.

Hardware: RX 7900 XTX (gfx1100), 24 GB. No model.py integration, no default changes (per arc rules).

## 1. WMMA custom-kernel status — REVIVED ✅

The WR4 framing ("SHAPED_WMMA convention changed = a codegen rewrite, not kernel-authoring") was **too
pessimistic**. Two minimal fixes revive it — a **1-line spec rule + an authoring reorder**, no linearizer
surgery:

1. **Authoring (AFTER ordering):** `AFTER` must wrap the placeholder/movement, not an `INDEX`:
   `acc.after(k)[idx]`, NOT `acc[idx].after(k)`. The spec rejects `AFTER(INDEX, RANGE)` and `pm_mops` expects
   movement/INDEX *after* AFTER. (Fixed in `extra/gemm/amd_copy_matmul.py`.)
2. **Codegen (spec rule):** `Ops.SHAPED_WMMA` had **no `spec_tensor` rule**, so `type_verify` rejected it at
   `lower_sink_to_linear` — *before* `lower_shaped_wmma` (in rangeify's `pm_mops`) converts it to `Ops.WMMA`.
   Added one rule to `spec_tensor` (`tinygrad/uop/spec.py`). Purely additive; tensor-core suite 6 pass / 3 skip.

## 2. Exact SHAPED_WMMA convention (the modern idiom)

```
UOp(Ops.SHAPED_WMMA, dtype_out, (a_frag, b_frag, acc_frag), arg=((M, N, K), device, threads))
```
- `arg` = 3-tuple `((16,16,16), 'AMD', 32)` (dims, device, wave threads). Lowered by
  `lower_shaped_wmma` (`tinygrad/schedule/rangeify.py:25`) → `Ops.WMMA` (8-tuple arg) → rendered as
  `llvm.amdgcn.wmma.f32.16x16x16.f16`.
- **Fragments are shaped views** (reshaped/indexed LDS or REG), whose **last dim = per-lane element count**:
  a/b = `WMMA_K` (16) halfs, acc = `WMMA_ACC` = `WMMA_M // LANES_PER_WAVE_M` (8) floats. The lowering builds
  per-src UPCAST ranges + `contract`. NOT raw `INDEX(ptr)` operands passed directly; shaped views that the
  lowering then indexes+contracts.
- `acc_frag` for the reduce must be `acc.after(k)`-then-indexed (see fix #1). Output dtype `float.vec(8)`.
- Distinct from the **optimizer-TC `Ops.WMMA`** (register-vector operands, 8-tuple arg, emitted by
  `postrange.py:300`) which always worked — that path is whole-matmul, not in-kernel fragment control.

Smallest valid shape: 16×16×16, threads=32 (one wave). Exact pre-fix errors:
`amd_copy_matmul` → `UOp verification failed ... Ops.AFTER ...ptr(128,3)... [(INDEX,ptr),(RANGE,REDUCE)]`;
then (after AFTER fix) → `Ops.SHAPED_WMMA ... 3 [(INDEX,...),(INDEX,...),(INDEX,...)] ((16,16,16),'AMD',32)`
(no spec rule). `amd_flash_attention` fails earlier at the *warp-reduce* (`shapeless CUSTOM ds_bpermute → MAX`
shape assert) — unrelated to WMMA (use WR1–3 `CUSTOMI`, or LDS reductions).

## 3–5. Rung results

| phase | what | result | artifact |
|---|---|---|---|
| 1 | minimal 128×128×16 WMMA matmul, TinyJit | **PASS** — rel_err 0.0, AMD-only, capture/replay + input-subst | `bench/qk-wmma-custom-smoke/` |
| 3 | Q@Kᵀ via WMMA, Hd=16 & Hd=128 | **PASS** — rel_err ~1e-7..1e-6, AMD-only, replay | `bench/qk-wmma-qk-tile/` |
| 4 | decode-attention microtile (scores→softmax→P@V) | **PASS** — rel_err 1.6e-4 vs SDPA, AMD-only, 3 WMMA kernels, only `[M,L]` tile materialized | `bench/qk-decode-attention-v3-tile/` |
| 5 | GQA cooperative vs naive (V-reuse) | **3.79× — but confounded** (see below) | inline (amd_copy_matmul) |

**Phase 5 honesty:** the probe compared one batched `[512,4096]@[4096,128]` (cooperative group, V streamed once,
4 blocks) vs 4× `[128,4096]@[4096,128]` (naive per-head, V streamed 4×, 1 block each), FLOP-matched. Result
naive_total 1734µs vs coop 458µs = **3.79×**. BUT the per-head `[128,…]` matmul (458µs) is nearly as slow as
the 4× larger coop matmul — because **1 block underfills the 96-CU GPU**. So the win is **occupancy-dominated**,
not pure V-reuse. This is *consistent with the v3 thesis* (cooperative high-occupancy group processing ≫ naive
per-head) and clears the ≥1.2× gate, but: (a) it's a GEMM shape (M=128/512), not true T=1 decode (M=1–4 per
group); (b) it conflates occupancy with V-reuse. The decode-block map already measured the real decode kernel
(`flash_partial`) as **occupancy-bound at ~33 GB/s, Infinity-Cache-served (not HBM-bound)** — so the value
driver of v3 is **high occupancy**, and pure V-bandwidth-reuse is a *secondary* effect (V is cache-served).

## 6. Go / No-Go for full decode_attention_v3

**GO on expressibility — the prerequisite is cleared.** WMMA custom kernels compile, run on AMD, are correct,
and capture/replay through TinyJit; a correct WMMA decode-attention tile is expressible. The WR4 wall is removed.

**Qualified on value.** The win is **high-occupancy WMMA group processing**, NOT GQA-V-reuse per se (V is
IC-served). The full-v3 build must be a *high-occupancy* fused kernel and must beat the **current `hoisted`
flash at true decode shapes (T=1, symbolic KV)** on DEBUG=2 GPU time — that is the real gate, and the prior LDS
Phase-5 finding (IC-served baseline is hard to beat at single-query shapes) is the live risk.

Recommended next scope (a separate `[codegen]` build, gated):
1. Fused single custom_kernel: Q@Kᵀ (WMMA) → 2-pass row softmax (LDS reductions, shape-clean — avoid the
   ds_bpermute warp-reduce wall) → P@V (WMMA), **scores LDS-resident** (never global), GQA group as the
   high-occupancy M dimension (16/32 query-head rows per kv-group).
2. Symbolic start_pos (bound-twin-in-ranges, proven) + causal/KV-tiling at D=128, L≤128 (64 KB LDS cap).
3. Gates: isolated exact + ≥1.3× vs current flash at KV 1024/4096 (DEBUG2 tm) → in-model ≥5% decode @ctx1024,
   byte-identical greedy → only then a gated default. Kill if it can't beat the IC-served baseline.

## 7. Kill gates (carried into the full-v3 build)
- If the fused kernel can't beat current `hoisted` flash by ≥1.3× isolated at KV≥1024 → refute (IC-served wall).
- If 2-pass LDS softmax can't stay shape-clean / online-softmax hits the coupled-reduce wall → use 2-pass, don't
  rewrite the linearizer.
- No default flip without in-model ≥5% @ctx1024 + byte-identical greedy.

## 8. What NOT to do
- Don't use the `ds_bpermute` warp-reduce idiom (the `amd_flash_attention` wall) — use LDS reductions or WR1–3.
- Don't claim V-reuse as the value driver — it's occupancy; V is Infinity-Cache-served at decode KV sizes.
- Don't integrate into model.py or flip defaults in the prereq arc (done — this arc stayed isolated).
- Don't trust the Phase-5 3.79× as the expected decode gain — it's occupancy+GEMM-shape confounded; the honest
  projection is the decode-block map's Amdahl (+4–10% @ctx≤1024, +12–36% @ctx4096) *if* the fused kernel reaches
  llama-class occupancy at decode shapes.

## Status / commits
WMMA custom-kernel idiom **revived**; Phases 0–4 proven + committed; Phase 5 directional. Full v3 is **earned to
build** (expressibility unblocked) but **must pass the isolated ≥1.3× and in-model ≥5% gates** before shipping.
Commits: `[codegen] revive SHAPED_WMMA`, `[test] smoke`, `[test] QK tile`, `[test] microtile`, `[docs] this`.
