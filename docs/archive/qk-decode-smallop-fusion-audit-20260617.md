# Decode small-op fusion audit — pick the first primitive to fuse (2026-06-17)

One output: which primitive to fuse first, by measured GPU time. `extra/qk_decode_smallop_audit.py`,
`bench/qk-decode-smallop-audit/result.json`. **Answer: the decode ATTENTION (SDPA) — and the fusion already
exists (flash-decode). The classic small-op fusions (RMSNorm/SwiGLU/residual) are each too small to matter.**

## Method

Kernel source metadata is empty in the decode path (checked both the DEBUG=2 `TRACEMETA=2` text and the
programmatic `call.arg.metadata` — both empty), so classification is structural: tinygrad auto-names kernels by
op+shape (`E_*`=elementwise, `r_*`=reduce); a per-layer op emits the SAME name in all 36 layers → grouping by
exact name gives clusters of ~36/72. Shape signature identifies the primitive — decisively, any reduce over the
**KV length (sp+1) or head/kv dims (128, 1024)** is attention (nothing else touches those). GPU time = eager
DEBUG=2 tm (relative proxy, per the census; sync-copy artifact excluded).

## Measured ranking of the 580 non-GEMV kernels (24.7% of decode GPU; GEMV is 75.3%)

| primitive | kernels | GPU (proxy) | % decode GPU |
|---|---:|---:|---:|
| **attention (SDPA: scores / softmax / @V over KV & head dims)** | 270 | **5.08 ms** | **15.8%** |
| elementwise (residual / norm-scale / cast, over hidden) | 126 | 1.07 ms | 3.3% |
| reduce (other) | 109 | 0.86 ms | 2.7% |
| RMSNorm (reduce over hidden 4096) | 73 | 0.86 ms | 2.7% |
| SwiGLU (silu·mul over ffn) | 2 | 0.04 ms | 0.1% |

## The decision

**First fusion target by GPU time = the decode attention (SDPA).** It is ~16% of decode GPU at short context
(KV=65) and *grows with context* (the long-context bench: baseline decode decays 3.4× to ctx 4096, attention-
dominated). The fusion that addresses it is **flash-decode — already implemented** (`extra/qk_flash_decode.py`,
gated `FLASH_DECODE`), measured **1.73× @ ctx 4096**. So the action is **make flash-decode the decode default
(or auto-enable above a context threshold)** — NOT build a new RMSNorm/SwiGLU fusion.

**The classic small-op fusions are refuted as first targets:**
- **SwiGLU silu·mul ≈ 0.1%** — tinygrad's scheduler already fuses the elementwise chain (only 2 stray kernels
  survive). Nothing to gain.
- **RMSNorm ≈ 2.7%, residual/norm-scale/cast ≈ 3.3%** — each tiny. Even fusing RMSNorm+scale+residual (llama's
  `norm.cu:147` fusion) addresses ≤ ~5% of decode GPU, below the +5–10% tok/s acceptance bar on its own, and
  would need to be stacked with others. Not worth a dedicated fusion arc at short context.

**So there is NO single classic small-op fusion that hits +5–10%.** The short-context gap vs llama (54–64 vs
80–100 tok/s) is the *sum* of many small inefficiencies + an equal-bytes GEMV, not one fat fusable primitive.
The one fat non-GEMV target is attention, and its fusion (flash-decode) exists.

## Recommended next action (the pick)

1. **Default/auto-enable flash-decode** — addresses the #1 non-GEMV consumer (attention) and the long-context
   decay (1.73× @4096), zero new kernel, quality already validated. Acceptance: same output; decode tok/s up at
   long context with no short-context regression; guard with a context threshold so short-ctx (where flash-
   decode is ~1.05×) isn't slowed. This is the gap-plan's #1 anyway — the audit confirms it's also the #1
   small-op-fusion target. **This is the first target.**
2. *(Only if net-new fusion is still wanted)* **RMSNorm + scale + residual** is the largest classic small-op
   group (~3–5% combined, adjacent, llama-proven local path) — but verify it's not already scheduler-fused, and
   expect a modest (~3–5%) gain. Lower priority than (1).

## Caveats / kill conditions

- Classification is **heuristic** (shape signatures; metadata unavailable). The KV-length/head-dim → attention
  mapping is strong, but the attention/RMSNorm split should be confirmed (metadata revival or an ablation that
  swaps SDPA→flash-decode in a *symbolic-sp* JIT path; the eager audit can't trigger flash-decode — concrete sp).
- GPU times are eager unbatched **relative proxies**, not batched-graph absolute time.
- **Do not** open a broad fusion arc to chase RMSNorm/SwiGLU — the data says the gain is below the gate.
- If flash-decode-default regresses short context, gate it by context length (already 1.05× @512, so a
  threshold like ctx≥1024 captures the win without short-ctx risk).

## Bottom line

The "small-op fusion" lever largely collapses into **"default flash-decode"** — the audit shows attention is
the only non-GEMV primitive big enough to matter, and its fusion is already built. RMSNorm/SwiGLU/residual are
each too small to justify a fusion arc. First target: **flash-decode default for long context.**
