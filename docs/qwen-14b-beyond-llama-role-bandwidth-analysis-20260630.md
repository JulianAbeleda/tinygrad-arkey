# 14B decode: role-bandwidth analysis and the path beyond llama

Date: 2026-06-30

Status: post-promotion analysis. The attn_k route-miss fix is shipped (8B now >llama). This maps exactly where the
remaining 14B/32B gap lives, by role-local kernel bandwidth, and names the concrete next lever. Hardware: gfx1100.

## Where we are

| model | tinygrad now (ctx512) | llama | % |
|---|---|---|---|
| **8B** | **107.4** | 98.4 | **109% — beyond llama** |
| 14B | 42.8 | ~65 | ~68% |
| 32B | 21.0 | ~31 | ~68% |

8B is already **beyond llama** — proof the kernels *can* win; they're just tuned for 8B shapes. Theoretical decode
ceiling = file_bytes / 960 GB/s: 8B 191, 14B 107, 32B 49 tok/s. Realistic parity (~62% of peak, where llama sits):
14B ~66, 32B ~30.

## Role-local kernel bandwidth (14B, synced TinyJit, the actual shipped route per role)

| role | shape | quant | GB/s | wall/tok | note |
|---|---|---:|---:|---:|---|
| ffn_gate | 5120→17408 | Q4_K | 406 | 4.9ms | efficient |
| ffn_up | 5120→17408 | Q4_K | 402 | 5.0ms | efficient |
| **ffn_down** | 17408→5120 | Q6_K | **253** | **11.5ms** | **biggest single wall** |
| attn_q | 5120→5120 | Q4_K | 121 | 4.9ms | ok |
| attn_output | 5120→5120 | Q4_K | 120 | 4.9ms | ok |
| **attn_k** | 5120→1024 | Q4_K | **24** | **4.9ms** | occupancy-starved |
| **attn_v** | 5120→1024 | Q6_K | **76** | **2.2ms** | occupancy-starved |
| lm_head | 5120→151936 | Q6_K | 706 | 0.9ms | efficient (huge output) |

Two structural bottlenecks emerge, and they explain the ~68% ceiling:

1. **Occupancy-starved KV projections** (attn_k, attn_v, `5120→1024`): only 1024 output rows → 1024 workgroups ×
   32 lanes ≈ 32k threads ≈ 26% GPU occupancy. attn_k does 1/5 of attn_q's work in the *same* time (24 vs 121
   GB/s). Combined ~7ms/tok (~30% of decode).
2. **ffn_down** (`17408→5120`, Q6_K half the layers): 253 GB/s, the single biggest role wall (11.5ms/tok). Enough
   occupancy (5120 rows), but long K (68 blocks → 17 serial/lane) and the Q6_K route.

## The lever, and why it's a build not a toggle

Split-K **helps the occupancy-starved roles** (the opposite of ffn_down, where SK4A refuted it): attn_k `5120→1024`
role-local goes **24 → 34.9 GB/s at split_k=4 (+45%), byte-correct** (split_k≥8 breaks correctness — a masked-tail
bug in the existing partial kernel). But:

- The existing partial kernel is a less-efficient range-reduce (34.9 GB/s), not the g3 wave kernel (which is
  efficient but occupancy-starved at 1024 output). Neither alone reaches attn_q's 121 GB/s.
- The partial kernel is wired for the prefill/batch path, **not the decode fast path** — excluding attn_k from g3
  to use it risks the slow fallback. So a clean win needs a **generated g3-wave-with-split-K decode kernel**
  (g3 efficiency × split-K occupancy) for small-output roles — the SK2 build that KT/SK deferred.

This is the honest pure-machine-search frontier: the machine must **author a split-K variant of the g3 lanemap**
for occupancy-starved shapes. It's a real generated-kernel build, correctness+W==D-gated, not a route toggle.

## Beyond-llama plan (concrete, measured)

| target | role | current | mechanism | est. |
|---|---|---:|---|---|
| KV-proj occupancy | attn_k+attn_v 5120→1024 | 24/76 GB/s | generated g3-split-K decode kernel (SK2) | ~7ms → ~3ms/tok |
| ffn_down efficiency | 17408→5120 Q6_K | 253 GB/s | Q6_K single-pass / better route for long-K | 11.5ms → ~8ms/tok |

If both land, 14B ~22ms → ~15ms/tok ≈ **~66 tok/s = llama parity**, with headroom to exceed (8B already does). The
KV-projection g3-split-K kernel is the highest-leverage next build and the cleanest pure-machine-search story: a
generated kernel authored for a measured shape class, promoted only on token-match + W==D.

## Shipped this session (recap)

- 8B flash-crossover fix (short-prompt ctx≥512 decode 54→104 tok/s).
- G3-anyshape + attn_k route promoted default-on: 8B +4% (>llama), 14B +60%, 32B +78%, byte-identical.
- KT/SK/LDR/RSR chain: proved the gap was a route-registration oversight (attn_k), not topology/split-K/GEMV.
