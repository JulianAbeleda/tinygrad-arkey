# Q4K GEMV warp — Promotion Hardening + Same-Lever Expansion — Result

Date: 2026-06-22

Executes `docs/q4k-gemv-warp-promotion-hardening-scope-20260622.md`. Hardens the `Q4K_GEMV_WARP_WD_PASS` win for the
owner default-flip decision and tests **same-lever** expansions (Q6_K down, attn projections). No new primitive, no
attention work, no q8-default, no backend rewrite, no default change. Every expansion: local A/B → W==D (W==D decides).

## Recommendation: **`Q4K_GEMV_WARP_READY_FOR_OWNER_DEFAULT_DECISION`**
The promoted route (Q4_K FFN **gate/up + down**) is hardened: reproduced ~**+9.6%@ctx1024 / ~+8.5%@ctx4096**,
**byte-identical** on real text, lossless, `default_eligible=true`, fallback-safe — ready for the owner to flip
default-on. The two same-lever expansions (Q6_K down, attn q/o) are **feasible but do not transfer** to W==D and are
banked behind research flags.

## P1 — 8B W==D reproduced (promoted route Q4K_GEMV_WARP + Q4K_GEMV_WARP_DOWN)
| run | ctx512 | ctx1024 | ctx4096 |
|---|---|---|---|
| original (6a83f2b98) | +9.83% | +9.78% | +8.71% |
| repro 2 | +9.75% | +9.63% | +8.55% |
| repro 3 | +9.58% | +9.41% | +8.49% |
| **median** | **+9.75%** | **+9.63%** | **+8.55%** |

Reproducible (spread ~0.4%@ctx1024, clock variance), no ctx regression, tokens match, 90 warp kernels (72 gate/up +
18 Q4_K down). No q8/B4/B5 enabled.

## P2 — quality hardening (decode-path, the right gate)
Teacher-forced NLL uses the **batched GEMM** path (not the T==1 warp route), so it does not exercise this lever. The
decode-path gate is **real-generation greedy byte-identical**: prefill (GEMM) + 64 decode tokens (warp) on a natural
prompt → **0/64 mismatches, byte-identical** (a coherent "why the sky is blue / red at sunset" paragraph). The kernel
runs the **same** Q4_K dequant/dot as the default, only reassociated (warp tree-sum) — lossless up to fp reassoc, the
class already accepted for the coop/flash variants. **Quality clean → default-eligible.**

## P3 — registry
`q4k_gemv_warp_ffn` in `bench/qk-decode-eval/candidates.json`: flags (`Q4K_GEMV_WARP`, `Q4K_GEMV_WARP_DOWN`), supported
shapes (gfx1100, gate/up 4096→12288, Q4_K down 12288→4096), local-A/B + W==D artifacts, **`default_eligible=true`,
`default_on=false`** (owner approval required).

## P4 — Q6_K down expansion: feasible, NOT worth promoting
Built `q6k_gemv_warp_kernel` (same lever: 32 threads/row = 16 pos × 2 block_groups + in-kernel `warp_reduce_sum`),
correct (rel 2.5e-7, byte-identical), fires in-model (18 `q6k_gemv_warp_4096_12288`). **Local: only 1.09× over coop**
(38%→41%) — because the Q6_K down is **already coop-routed** (`Q6K_FFN_DOWN_COOP` default-on, ~51% in-model), unlike
the Q4_K down (which was on a weaker `parts=4` path → warp 1.37×). **W==D: no gain** (+9.58% with vs +9.63% without,
noise). → behind research flag **`Q6K_GEMV_WARP_DOWN`** (default-off); not promoted.

## P5 — attn q/o projection expansion: feasible locally, does NOT transfer
q/o is Q4_K 4096×4096, currently coop-routed. **Local: warp 1.32× over coop** (27%→36%). But **W==D did not improve**
(+9.33% with vs +9.63% without, within noise) despite firing (162 kernels). The q/o projection sits in the
**attention block and partly overlaps** in the JIT graph — the same transfer failure as B5 attention (local win ≠
in-graph). → behind research flag **`Q4K_GEMV_WARP_PROJ`** (default-off); not promoted.

**The transfer test keeps discriminating:** FFN gate/up+down weight GEMV is on the critical path and **transfers**
(+9.6%); attention-adjacent GEMVs (q/o) and already-served roles (Q6_K down) do not. This is exactly why W==D, not
% peak, is the gate.

## P6 — cross-model
The kernel is **shape-general** (any out/in with `k_blocks%4==0` for Q4_K / `%2==0` for Q6_K). The route guards are
**8B-specific** (gate/up 4096↔12288, down, q/o 4096²). 14B/32B Q4_K_M are present but their FFN shapes differ → the
guard **misses → safe fallback to the default** (no regression, by construction — the warp only runs for guarded
shapes). Generalizing the guards to per-model FFN shapes is a bounded follow-on; a 14B W==D was not run (9 GB model,
not "cheap"). Classification: **8B-shape-guarded route, shape-general kernel, safe fallback elsewhere.**

## Final state
- **Promoted (owner decision pending):** `Q4K_GEMV_WARP=1` + `Q4K_GEMV_WARP_DOWN=1` → lossless **~+9.6%@ctx1024 /
  ~+8.5%@ctx4096**, byte-identical, default-off, `default_eligible=true`. Decode 66.6→73.5 tok/s @ctx1024 (~67%→~73%
  of llama).
- **Research-only (banked, not promoted):** `Q6K_GEMV_WARP_DOWN` (1.09× local, already coop-served),
  `Q4K_GEMV_WARP_PROJ` (1.32× local but attention-overlapped, no transfer).
- **Follow-on (bounded):** generalize the route guards for 14B/32B (shape-general kernel already exists).

## Boundaries honored
Same lever only. No attention tile/combine work, no q8-default, no backend rewrite, no lm_head, no default flip
(default-off; owner call). Local A/B before W==D; W==D decided every call. Bench artifacts gitignored.
