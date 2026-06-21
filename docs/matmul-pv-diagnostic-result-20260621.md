# Matmul-PV Diagnostic Candidate — Result

Date: 2026-06-21

ISA-justified diagnostic from `docs/low-level-decode-attn-attribution-result-20260621.md`: replace coop's scalar
`flash_partial` PV path with `PV = prob @ V` as a tinygrad **matmul** so the tiled-GEMM codegen applies.

## Decision: **`MATMUL_PV_FAIL_LOCAL_AB`**

The candidate is **value-correct** but **0.872× / 0.634× / 0.345×** vs `gqa_coop_vec` @ctx512/1024/4096 (throughput)
— it **loses**, and regresses badly with ctx. The ISA-evidence hypothesis (matmul-PV would tile like the q·k matmul)
is **refuted by measurement**: the PV matmul runs at **68 GFLOPS** (worse than `flash_partial`'s 201) because the GQA
decode PV is a **skinny M=G=4 GEMM**. Per discipline, the local gate missed → **no W==D route**, banked.

## Phase 0 — design preface

- **Why reopened:** new ISA evidence named coop's `flash_partial` (24.7µs, **0 `v_dot2`, 0 LDS, scalar fp16 loads**)
  as the dominant inefficiency; the q·k matmul (LDS=64, tiled) is fast. Lever: route the PV through the same
  tiled-GEMM codegen. (NOT the closed "coop-qk-preserving" lane — that was timing-only; this is ISA-specific.)
- **PV expression:** `prob[Hkv,G,ctx] @ V[Hkv,ctx,Hd] → [Hkv,G,Hd]`. `prob` reshape `[Hq,ctx]→[Hkv,G,ctx]` is **free**
  (Hq=Hkv·G); V is `[Hkv,ctx,Hd]` contiguous → **no layout copy** erases the win.
- **Softmax semantics preserved:** standard max-subtract / exp / sum-normalize (same as coop, up to fp).
- **Kernel count / materialization:** candidate = q·k matmul + softmax (max/exp/sum/div) + PV matmul ≈ coop's count;
  the PV is a matmul, not the hand-rolled `flash_partial`.
- **Expected Amdahl:** best-case PV 24.7→~14µs ≈ 1.16× attention ≈ ~3–4% whole-decode (W==D-marginal even if local
  improved). **First gate:** total attention ≥1.05× @ctx1024. **Stop:** if PV improves but total doesn't, bank.

## Correctness (vs numpy)

| ctx | rel_rmse | max_abs | gate (≤1e-3) |
|---:|---:|---:|---|
| 512 | 4.7e-4 | 1.7e-4 | PASS |
| 1024 | 4.9e-4 | 1.2e-4 | PASS |
| 4096 | 5.2e-4 | 7.2e-5 | PASS |

(≤1e-5 unattainable with fp16; ~5e-4 matches coop's own fp-reassoc; no layout mismatch.)

## PV timing (DEBUG=2 single-run kernel breakdown @ctx1024)

| candidate kernel | GFLOPS | role |
|---|---:|---|
| q·k matmul (`r_16_8_16_4_4_32_4`) | 174 | scores |
| softmax reduces (`r_32_16_64`, `…n1`) | 10–15 | max / sum |
| **PV matmul (`r_2_8_16_4_4_256_4`)** | **68** | `prob @ V` — the dominant cost |

The PV matmul runs at **68 GFLOPS** — **worse than `flash_partial`'s 201 GFLOPS**. Root cause: GQA decode PV is
`[Hkv,G,ctx] @ [Hkv,ctx,Hd]` with **M=G=4 per kv-head** (a skinny GEMM, K=ctx scaling with context) → tinygrad's
GEMM tiling can't fill M=4. The q·k matmul is fast because its contraction is over Hd=128 (a fat reduction), not a
tiny-M batch. The only no-copy formulation is M=4 (slow); a fat-M version needs V-replication ×G (a layout copy).

## Total local A/B (throughput, clock-pinned, vs gqa_coop_vec)

| ctx | candidate µs | coop µs | **speedup** |
|---:|---:|---:|---:|
| 512 | 86.3 | 75.2 | 0.872× |
| 1024 | 134.0 | 84.9 | **0.634×** |
| 4096 | **414.0** | 142.7 | **0.345×** |

**Gate FAIL.** The candidate not only misses ≥1.05× — it regresses, **3× slower at ctx4096** (the M=4, K=ctx skinny
PV GEMM + full-ctx softmax reduces scale badly), while coop's flash decode (split structure) scales sub-linearly.

## Lifecycle verdict

decode_eval candidate `matmul_pv_diagnostic` (family `attention_split`, `ab_script`) → **`FAIL_LOCAL_AB`** →
`refute_candidate`, banked. Refutation added.

## W==D: NOT reached (local A/B failed — discipline = stop).

## Interpretation for the remaining llama gap

The bounded ISA lever is **refuted**: the matmul-PV is *worse* than coop's scalar `flash_partial` because the GQA
decode shape (M=G=4) defeats tinygrad's GEMM tiling. So "use the tiled-matmul codegen for the PV" — the one bounded
lever the attribution named — does **not** work. coop's hand-rolled `flash_partial`, despite its scalar ISA, is
better than a naive matmul-PV for this shape. **llama's advantage is specifically the fused `v_dot2_f32_f16` +
LDS-staged-K/V single tile** — a structure that is neither coop's split flash nor a standard matmul, and which
tinygrad's UOp codegen does not generate. There is **no bounded path** to it; the decode bounded space AND the
ISA-named bounded codegen lever are both now exhausted.

## Acceptance gates

| gate | result |
|---|---|
| G1 scalar PV issue restated from ISA | PASS |
| G2 candidate uses tiled-matmul path | PASS (PV is a GEMM; it just tiles poorly at M=4) |
| G3 correctness measured | PASS (rel_rmse ~5e-4) |
| G4 local A/B vs gqa_coop_vec | PASS (0.87/0.63/0.35×) |
| G5 through decode_eval/lifecycle | PASS (`FAIL_LOCAL_AB`) |
| G6 no W==D unless local passes | PASS (not added) |
| G7 no default/model change | PASS (`git diff tinygrad/` empty) |
| G8 no closed lane beyond this ISA diagnostic | PASS |
| G9 policy guard | PASS |
| G10 tree clean after commit | PASS (commit below) |

## Next action

**Rest the bounded decode frontier.** Both the bounded decode space (13+ refuted lanes) and the ISA-named bounded
codegen lever (matmul-PV) are exhausted; the only remaining path is the deep fused-`v_dot2`-LDS-tiled-flash codegen
capability tinygrad lacks (multi-week, unbounded). Honest recommendation: **`REST_DECODE`** — pivot to v2/search/
tooling-hardening; keep the llama oracle (non-promotable target) and these refutations as the standing evidence.

## Changed files
`extra/qk_matmul_pv_diagnostic_ab.py` (new), `bench/qk-decode-eval/candidates.json`, this doc,
`bench/qk-lifecycle-search/refutations.json`, handoff/READMEs.

## Boundary
No `tinygrad/` change, no model route/default, no W==D route, no closed lane reopened beyond the ISA-justified PV
diagnostic, no performance claim (it lost). Clock-pinned diagnostic; perf-state restored to `auto`.
