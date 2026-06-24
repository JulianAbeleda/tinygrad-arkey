# Q4K GEMV warp — Promotion Hardening + Same-Lever Expansion (Scope)

Date: 2026-06-22

Follow-on to `docs/decode-ffn-gemv-warp-result-20260622.md` (`Q4K_GEMV_WARP_WD_PASS`: the lossless FP
work-decomposition GEMV cleared W==D byte-identically, +9.78%@ctx1024 gate/up+down). **Harden for the owner
default-flip decision and extend ONLY the same proven lever** (Q6_K down, attn projections). No new primitive hunt,
no attention, no q8-default, no backend rewrite. Every expansion: local A/B → W==D (W==D decides).

## P0 — promotion readiness (the checklist)
| item | state |
|---|---|
| route flags | `Q4K_GEMV_WARP=1` (FFN gate/up) + `Q4K_GEMV_WARP_DOWN=1` (Q4_K down), read per-call via `getenv`, default **off** |
| arch/shape guard | gfx1100 (`DECODE_ATTN_AMDGCN_ARCH_OK`), in/out 4096↔12288, `parts==1` (gate/up), `k_blocks%4==0` |
| fallback | any guard miss or exception → default `q4k_gemv_partial` (the shipped path); `try/except` with DEBUG log |
| correctness | lossless FP (same Q4_K dequant/dot, reassociated); standalone rel ≤ 5e-6; in-model greedy 0 mismatches |
| artifact contract | `bench/qk-ffn-gemv-warp/{latest,wd}.json` (local A/B + W==D + repro band) |
| candidate registry | `q4k_gemv_warp_ffn` in `candidates.json` (flags, shapes, artifacts) |
| default_eligible | **true** (lossless + W==D pass + no ctx regression) |
| default_on | **false** — owner approval required to flip (the task boundary) |

## Phases
- **P1** reproduce 8B W==D (interleaved, repro band, byte-identical, no q8/B4/B5 enabled). Gate: reproduce ~+9–10%, no
  regression, correct.
- **P2** quality hardening: **real-generation greedy byte-identical** (prefill via GEMM + decode via warp) over a
  natural prompt — the decode-path quality gate (teacher-forced NLL uses the batched GEMM path, not the T==1 warp
  route, so it does not exercise this lever). Byte-identical preferred; else dNLL ≤ tol + explanation.
- **P3** registry: ensure `q4k_gemv_warp_ffn` carries flags/shapes/`default_eligible=true`/`default_on=false` + artifacts.
- **P4** Q6_K down expansion feasibility: the FFN down is Q4_K (×18, done) + **Q6_K (`q6k_coop_partial_4096_12288`,
  ×18, ~7% share)**. Q6_K block = 16 grp × 16 pos; a warp variant = 16 pos × 2 block_groups = 32 lanes,
  `warp_reduce_sum`, k_blocks=48 (÷2 ✓). Build ONLY if bounded; local A/B → W==D (must beat current Q4_K-warp result).
- **P5** projection GEMVs: attn q/o (Q4_K 4096×4096, ~8%, currently coop-routed) + k/v (Q6_K 1024×4096, ~2%). q/o is
  the same `q4k_gemv_warp` at 4096×4096 (k_blocks=16 ✓). Local A/B → W==D only if local passes. No attention tile work.
- **P6** cross-model: the kernel is shape-general (any out/in, `k_blocks%4==0`). Note 14B/32B shape support only if
  cheap; classify shape-general vs 8B-only.
- **P7** result + recommendation: one of `READY_FOR_OWNER_DEFAULT_DECISION` / `KEEP_OPT_IN_NEEDS_MORE_QUALITY` /
  `KEEP_OPT_IN_SHAPE_LIMITED` / `EXPANSION_Q6K_READY` / `EXPANSION_PROJ_READY` / `HARDENING_BLOCKED`.

## Boundaries
Same lever only. No attention, no q8-default, no backend rewrite, no lm_head-first, no default flip without owner
approval, no unrelated kernel work. Local A/B before W==D; W==D decides.
