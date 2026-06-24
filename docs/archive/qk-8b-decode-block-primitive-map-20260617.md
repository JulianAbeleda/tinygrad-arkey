# 8B decode-block primitive map (post-hoisted-flash) — 2026-06-17

**Design/audit arc — no kernels built, no defaults changed, no codegen surgery.** Produces the per-layer
decode DAG, classifies the program boundaries, scores candidate decode-block primitives, and selects the single
best next hard target.

- Repo HEAD: master (flash-decode `hoisted`/L128 default shipped). Model: Qwen3-8B-Q4_K_M.
- Hardware: **RX 7900 XTX** (gfx1100) — `rocminfo` Marketing Name `AMD Radeon RX 7900 XTX` + 23.98 GiB VRAM
  (GRE is 16 GB; `rocm-smi`'s "GRE" Card-model string is a misread). XTX llama baseline applies.
- Census tool: `extra/qk_decode_block_map.py`; artifact `bench/qk-decode-block-map/result.json`.
- **Measured vs inferred vs hypothetical** are tagged throughout. GPU-time %s are the **eager DEBUG=2 `tm`
  relative proxy** (eager unbatches → absolute inflated; shares/ratios are the signal). tok/s is **W==D**
  (real decode wall vs dispatch ceiling, the robust path). Decode is GPU-bound here (W≈D).

---

## 1. Current post-hoisted decode census [measured]

| ctx | tok/s (W) | D ceiling | programs/token |
|---|---|---|---|
| 512 | 43.5 | 42.9 | 1001 |
| 1024 | 39.0 | 38.3 | 1001 |
| 4096 | 24.7 | 24.5 | 1001 |

**programs/token = 1001** = ~27.5/layer × 36 + 11 tail. This is **UP** from the stale ~780 (SDPA, short ctx)
and ~821 (SDPA, ctx1024): flash-decode adds ~6 kernels/layer + the score matmul. **Yet decode is faster at
long ctx** → *program count is not the decode bottleneck*; **decode is GPU-bound (W≈D)**, so GPU-time, not
kernel count, is what matters. This kills any "fuse to cut program count" thesis up front (and small-op fusion
was already refuted independently).

### Per-region split (GPU-time proxy %, and kernels/layer) [measured]

| region | ctx512 % | ctx1024 % | ctx4096 % | kernels/layer | bucket |
|---|---|---|---|---|---|
| ffn_down [GEMV] | 18.4 | 16.6 | 10.2 | 1 | GEMV |
| ffn_gate/up [GEMV] | 14.3 | 12.7 | 7.9 | 2 | GEMV |
| **attn_flash_partial** | 13.4 | 21.5 | **47.5** | 1 | attention |
| lm_head [GEMV] | 13.2 | 11.9 | 7.5 | tail (1) | GEMV |
| attn_q/o [GEMV] | 9.5 | 8.7 | 5.3 | 2 | GEMV |
| elementwise (rope/residual/cast) | 9.0 | 7.8 | 4.8 | **8.6** | small-op |
| reduce(other) (incl. q/k-norm, some scores) | 7.8 | 7.2 | 5.8 | **4.6** | small-op |
| attn_qk_scores (matmul) | 3.8 | 3.3 | 2.0 | ~0.06¹ | attention |
| attn_other | 2.7 | 2.4 | 1.5 | 1 | attention |
| rmsnorm | 2.1 | 1.9 | 1.2 | 2 | small-op |
| attn_k/v [GEMV] | 1.7 | 1.5 | 0.9 | 0.5 | GEMV |
| attn_flash_reduce (gmax/den/combine) | 1.6 | 1.7 | 2.2 | 2 | attention |
| attn_flash_max | 1.5 | 1.4 | 0.9 | 2 | attention |
| attn_flash_prob | 1.1 | 1.4 | 2.2 | 1 | attention |

¹ classification caveat: the per-layer score matmul mostly lands in `reduce(other)` (name didn't match the
`r_32_4/8` heuristic at all ctx); `reduce(other)` (4.6/layer) therefore bundles q/k-norm reduces **and** the
score matmuls. Treat `reduce(other)` + `attn_qk_scores` together as "attention-adjacent reduces".

### Aggregate by bucket [measured, approx]

| bucket | ctx512 | ctx1024 | ctx4096 | kernels/layer (count share) |
|---|---|---|---|---|
| **GEMV** (q/k/v/o, gate/up/down, lm_head) | ~57% | ~51% | ~32% | ~8 (29%) |
| **attention** (flash + scores) | ~24% | ~32% | ~55% | ~9 (33%) |
| **small-ops** (elementwise+reduce+rmsnorm) | ~19% | ~17% | ~12% | ~15 (**55%**) |

**Two-sentence summary:** GPU time lives in **GEMV (short ctx) and flash_partial (long ctx)**; program *count*
lives in **small-ops (~55% of kernels, only ~12–19% of GPU time)**. The high-time regions are refuted as
bounded levers; the high-count region is low-value to fuse (GPU-bound + small-op fusion refuted).

---

## 2. Per-layer decode DAG [measured shapes + inferred dependencies]

One transformer layer (dim=4096, 32 q-heads / 8 kv-heads, head_dim=128, FFN=12288), T=1 decode:

| # | node | shape (out) | dtype | kernels/layer | GPU% (ctx1024) | materialized? | consumed | fusion legality |
|---|---|---|---|---|---|---|---|---|
| 1 | residual in | [1,4096] | f16 | — | — | view | by 2,7 | — |
| 2 | RMSNorm (attn) | [1,4096] | f16 | ~1 (rmsnorm) | ~1% | yes (reduce) | by 3 | reduce → fuse-into-next blocked (single-acc) |
| 3 | q/k/v_proj [GEMV] | q[1,4096] k/v[1,1024] | f16 | 2.5 | ~10% | yes | by 4 | **horizontal fuse REFUTED (Q4K_FUSE −18%)** |
| 4 | q/k-norm + RoPE | [.,128]/head | f16 | part of elementwise+reduce | ~5% | yes | by 5 | elementwise, scheduler-fused; cheap |
| 5 | KV-cache write | [2,8,MAXC,128] | f16 | ~1 (copy) | <1% | store | by 6 | view/store; cheap |
| 6 | decode attention (flash) | [32,128] | f32→f16 | ~8 (prob/max/partial/gmax/den/combine + scores) | **~32%** | yes | by 7 | **flash_partial = the cost; refuted as bounded variant** |
| 7 | o_proj [GEMV] | [1,4096] | f16 | 1 | ~5% | yes | by 8 | fuse w/ attn → Q4K_FUSE-class risk |
| 8 | residual add | [1,4096] | f16 | part of elementwise | ~1% | yes | by 9,12 | elementwise; cheap |
| 9 | RMSNorm (ffn) | [1,4096] | f16 | ~1 | ~1% | yes (reduce) | by 10 | reduce; cheap |
| 10 | gate/up_proj [GEMV] | [1,12288]×2 | f16 | 2 | ~13% | yes | by 11 | horizontal fuse REFUTED |
| 11 | SwiGLU (silu·mul) | [1,12288] | f16 | part of elementwise | <2% | yes | by 12 | already scheduler-fused; removing .contiguous → 0% |
| 12 | down_proj [GEMV] + residual | [1,4096] | f16 | 1 + ew | ~17% | yes | next layer | largest GEMV; refuted |

Tail (×1, outside layer): final RMSNorm, **lm_head [GEMV] (~12% — q6k 151936)**, sampling/argmax, input upload.

**DAG reading:** the only node whose cost is *not* a refuted GEMV and *not* a cheap small-op is **node 6
(decode attention)** — and within it, `flash_partial`. Everything else is either an already-refuted GEMV
lever or a cheap elementwise/reduce.

---

## 3. Candidate boundary classification + scoring

Classification: **A** necessary · **B** avoidable-but-low-value · **C** searchable primitive · **D** deep
codegen/runtime.

| candidate boundary | cost share (ctx1024) | programs removable | class | risk | expected decode gain | verdict |
|---|---|---|---|---|---|---|
| **A. RoPE + KV-write + flash attention** | RoPE+KV ~5% (cheap); flash ~26% | ~3–5/layer (ew/copy) | C/D | high | small from fusion; real gain only from the flash WMMA inside | **fold into the attention-v3 target (the fusion bonus is minor; the value is the WMMA)** |
| B. RMSNorm + QKV prep | ~11% | ~2/layer | D | high | — | **REJECT — QKV horizontal fuse already refuted (Q4K_FUSE −18%)** |
| C. Attention block (qkv+rope+kv+flash+o) | ~37% | ~5/layer | D | very high | high but Q4K_FUSE-class occupancy regression risk on the projections | **DEFER — too broad; subsumes the refuted QKV/o fuse** |
| D. FFN block (gate/up+silu+down) | ~30% | ~3/layer | D | high | ~0 | **REJECT — Q4K_FUSE refuted; silu already scheduler-fused (contiguous removal = 0%)** |
| E. Norm/residual lifecycle | ~5% | ~6/layer | B | med | <3% (small-op fusion refuted) | **REJECT — program count, not GPU time; GPU-bound** |
| F. Whole layer | 100% | many | D | extreme | unknown | **REJECT — not a first target** |
| **G. Decode attention v3 (flash internals: WMMA + cooperative GQA V-reuse)** | **flash ~18% @512 / ~26% @1024 / ~53% @4096** | 0 (same/few kernels, faster) | **D** | high (WMMA-convention wall) | **+4–22% (see §4)** | **★ SELECTED — only high-GPU-time region with proven headroom** |

**Why G over the block fusions:** the block-fusion boundaries (A–F) chase the *small-op program count*
(~55% of kernels but ~12–19% of GPU time, GPU-bound, small-op fusion already refuted) or re-open the
*refuted GEMV horizontal fuse*. The **only** boundary that sits on a large GPU-time region **with measured,
unexploited headroom** is the decode-attention compute itself: `flash_partial` runs at ~**33 GB/s effective**
(ctx4096) — V (8 MB) is Infinity-Cache-served and prob is L1-served, so it is **occupancy/issue-bound, not
HBM-bound**. The textbook fix for that is a **high-occupancy WMMA kernel with cooperative GQA V-reuse (LDS)** —
exactly llama's structure — not register-blocking (refuted: 1.07–1.08×, regresses at BD≥4) and not bounded
knobs (the L-sweep is flat). This is `decode_attention v3`.

---

## 4. Selected target — `decode_attention_v3` primitive/search spec [hypothetical]

A high-occupancy fused decode-attention kernel that replaces the 6-kernel flash split with a WMMA/LDS structure
matching llama's occupancy, optionally absorbing RoPE + KV-write.

```
decode_attention_v3(q, k_cache, v_cache, start_pos, *, variant, L, tile_kv, warp_layout) -> [Hq, Hd]
  invariants: T==1 decode; symbolic start_pos (bound twin in kernel ranges; bound value via score slice)
  shapes:     Hq=32, Hkv=8 (G=4), Hd=128; KV=start_pos+1 symbolic; MAXC concrete buffers
  backend:    AMD gfx1100 (wave32); custom_kernel UOp path (DEFINE_LOCAL/BARRIER/SHAPED_WMMA)
  exactness:  byte-identical greedy vs current flash (flash is exact up to fp reassociation)
  reference:  numpy SDPA + current flash_decode_attention(variant=hoisted)
```

- **Core change:** cooperative **LDS K/V tiling** so the G=4 query heads of a kv-group reuse one V/K tile
  (kills the 4× GQA V re-read), + **register-resident online softmax** + **WMMA** for Q@Kᵀ / P@V at high
  occupancy (256-thread query blocks), instead of the current 1-output-per-thread, W=129-global-lane layout.
- **Search knobs:** `tile_kv` (LDS tile length, ≤128 at Hd=128 for the 64 KB LDS cap), `warp_layout`,
  whether RoPE is fused into the Q/K load, KV-write inside vs before, output layout, `L`/split, `variant`
  (keep `hoisted` as the searched fallback).
- **Reference in-repo:** `extra/gemm/amd_flash_attention.py` (full high-occupancy WMMA flash) — **the right
  structure but currently broken here** (shapeless `ds_bpermute` CUSTOM → shape-infer assert; no GQA/causal;
  square N×N). `extra/amd_warp_reduce.py` (WR1–3) revived shape-safe warp reductions; `amd_uop_matmul.py` has
  the LDS-tile + `c_regs` idiom.
- **Gates:** isolated exact + ≥1.3× vs current flash on the attention kernels (DEBUG=2 `tm`); in-model
  ctx512/1024/2048/4096 byte-identical greedy + ≥5% decode @ctx1024 (W==D) before any default flip.
- **Phasing:** (0) revive WMMA fragment-passing convention OR build flash from optimizer-TC matmuls +
  WR1–3 softmax; (1) GQA + causal + symbolic KV at D128; (2) LDS K/V tile reuse across G; (3) isolated gate;
  (4) in-model gate; (5) ship/refute.
- **Files likely touched:** `extra/qk_flash_decode.py` (new variant), `extra/gemm/amd_flash_attention.py`
  (revive), `tinygrad/llm/model.py` (gated `FLASH_VARIANT=wmma`, default unchanged until gated), tests.

---

## 5. Expected tok/s translation (Amdahl) [inferred from measured shares]

`total_speedup = 1 / ((1−f) + f/r)`, f = flash-attention GPU-time share (flash kernels), r = region speedup.
f ≈ **0.18 / 0.26 / 0.53** at ctx 512 / 1024 / 4096.

| region speedup r | ctx512 (43.5→) | ctx1024 (39.0→) | ctx4096 (24.7→) |
|---|---|---|---|
| **conservative 1.25×** | +3.7% → 45.1 | +5.5% → 41.1 | +11.9% → 27.6 |
| **base 1.5×** | +6.4% → 46.3 | +9.5% → 42.7 | +21.5% → 30.0 |
| **optimistic 2.0×** (llama-class) | +9.9% → 47.8 | +14.9% → 44.8 | +36% → 33.6 |

Honest read: meaningful only if the kernel reaches ~1.5×+ on the attention region (i.e. genuinely matches
llama's high-occupancy efficiency). At ctx≤1024 the upside is modest (+4–10%); the prize is **long context
(+12–36% @ctx4096)**. Helps 14B/32B too (attention share grows with model/context).

---

## 6. Kill gates (before any codegen surgery)

1. **WMMA-convention gate:** if reviving `SHAPED_WMMA` custom-kernel fragment passing is still stale (the WR4
   wall) AND the optimizer-TC-matmul route can't express symbolic-KV Q@Kᵀ/P@V at decode shapes → **STOP**,
   document, do not rewrite the linearizer.
2. **Isolated gate:** if the high-occupancy kernel doesn't beat current flash by ≥1.3× on DEBUG=2 `tm` at
   KV 1024/4096 → refute (matches the prior LDS Phase-5 "IC-served baseline wins" finding).
3. **In-model gate:** standalone win must survive to ≥5% decode @ctx1024 (W==D), byte-identical greedy.
4. **Occupancy-regression gate:** must not regress ctx512 (where attention is only ~18%).

---

## 7. Why this is the correct next primitive path

- It is the **only** high-GPU-time region (measured) that is **not refuted and has measured headroom**
  (occupancy-bound at 33 GB/s, not HBM-bound; not a bounded knob — register-blocking + L both flat/refuted).
- It is **the same primitive llama hand-builds** (high-occupancy WMMA flash), so the gap is concrete and the
  target is known-achievable in principle.
- It **scales to long context and to 14B/32B** (attention share grows), unlike short-ctx GEMV tweaks.
- The bounded version of this exact lever was already mined (hoisted-exp shipped; register-blocking refuted),
  so the remaining win is *specifically* the high-occupancy/WMMA shape — a clean, well-scoped (if hard) arc.

## 8. What NOT to do next

- **Do not** re-open QKV/FFN horizontal GEMV fusion (Q4K_FUSE −18%, refuted).
- **Do not** chase small-op (RoPE/residual/cast/norm) fusion for tok/s — ~55% of *programs* but ~12–19% of
  *GPU time*, decode is GPU-bound, and it's already refuted (each <3.5%, scheduler-fused, contiguous-removal 0%).
- **Do not** start a broad whole-layer fusion framework (candidate F) — extreme risk, no measured payoff.
- **Do not** trust program-count reduction as a proxy for tok/s here — flash *raised* program count and still
  won; decode is GPU-bound.
- **Do not** ship `decode_attention_v3` without the isolated ≥1.3× AND in-model ≥5% AND byte-identical gates.

---

## Status

Design map only — nothing built, no defaults changed. Selected next hard target: **`decode_attention_v3`
(high-occupancy WMMA flash + cooperative GQA V-reuse)**, a deep-`[codegen]` arc gated by the WMMA-convention
wall. All decode-block *fusion* boundaries are deferred/rejected with measured justification. Census:
`extra/qk_decode_block_map.py`, `bench/qk-decode-block-map/result.json`.
