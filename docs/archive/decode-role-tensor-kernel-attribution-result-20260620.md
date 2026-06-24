# Decode Role/Tensor/Kernel Attribution — Final Ranked Report

Date: 2026-06-20

Executor: Claude

Scope: `docs/decode-role-tensor-kernel-attribution-solution-scope-20260620.md`
Deliverable-0 detail + method: `docs/decode-current-route-attribution-result-20260620.md`

Headline: **the current-route timed attribution overturns the scope's expected ranking.** In-model weight-GEMV
(the MMVQ-equivalent roles) is at/above llama parity; the entire decode-to-llama gap is **attention (flash-decode
overhead, ctx-slope)** and **elementwise fusion (the FFN SiLU·mul + rope + residual adds llama fuses away)**. No
kernel was built. No decode default changed.

## 1. Current route W/D by ctx (W = promotion authority; host-sync 0% → GPU-bound)

| mode | ctx | tok/s W | ms/tok W | D ceiling tok/s | host-sync % |
|---|---:|---:|---:|---:|---:|
| baseline | 512 / 1024 / 4096 | 68.5 / 66.9 / 61.2 | 14.59 / 14.95 / 16.35 | 65.5 / 64.1 / 58.6 | 0.0 |
| q8 | 512 / 1024 / 4096 | 72.8 / 71.0 / 64.5 | 13.73 / 14.08 / 15.50 | 65.6 / 67.5 / 61.4 | 0.0 |

llama.cpp reference: 98.6 / 97.6 / 92.2 tok/s (10.14 / 10.25 / 10.85 ms/tok) @512/1024/4096.

## 2. Ranked role/tensor/kernel attribution (baseline @ ctx1024, timed, rescaled to the 14.95 ms wall)

| role | tensor | calls/tok | ms/tok | %wall | eff BW (%HBM) | gap vs llama | confidence |
|---|---|---:|---:|---:|---:|---:|---|
| ffn_gate/up | Q4_K | 72 | 3.65 | 24.4 | 558 (58%) | +0.19 | timed |
| attention_flash | attention | 378 | 3.50 | 23.4 | — | **+2.73** | timed |
| elementwise | fp | 220 | 2.19 | 14.6 | — | **+1.83** | timed |
| ffn_down | Q6_K | 36 | 2.15 | 14.4 | 690 (72%) | +0.11 | timed |
| attn_q/o | Q4_K | 72 | 1.19 | 7.9 | 573 (60%) | +0.06 | timed |
| reduce/glue | fp | 204 | 0.95 | 6.3 | — | −0.13 | timed |
| lm_head | Q6_K | 1 | 0.59 | 4.0 | 863 (90%) | +0.03 | timed |
| rmsnorm | fp | 73 | 0.39 | 2.6 | — | −0.12 | timed |
| attn_k/v | Q6_K | 18 | 0.34 | 2.3 | 180 (19%) | +0.02 | timed |

## 3. Token math (gap fully decomposed; Σ family_gap == total gap)

| route@ctx | tinygrad ms | llama ms | gap ms | attention | elementwise | weight-GEMV | rmsnorm | glue/other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline@512 | 14.59 | 10.14 | 4.45 | +2.44 | +1.83 | +0.44 | −0.12 | −0.12 |
| baseline@1024 | 14.95 | 10.25 | 4.71 | +2.73 | +1.83 | +0.41 | −0.12 | −0.13 |
| baseline@4096 | 16.35 | 10.85 | 5.51 | +4.36 | +1.83 | −0.33 | −0.15 | −0.20 |
| q8@1024 | 14.08 | 10.25 | 3.83 | +2.69 | +1.94 | −0.71 | +0.07 | −0.14 |
| q8@4096 | 15.50 | 10.85 | 4.66 | +4.28 | +1.94 | −1.38 | +0.04 | −0.22 |

- **Attributed gap @ baseline ctx1024 = 4.84 ms** (all named families); **unattributed residual = −0.13 ms**
  (the glue/other bucket llama fuses away — tinygrad is already slightly faster there).
- **Attention + elementwise = 97% of the ctx1024 gap** and ~100%+ at ctx4096 (weight-GEMV goes negative).
- Weight-GEMV total = 7.93 ms (baseline@1024) vs llama mmvq 7.52 ms → **+0.41 ms only**; faster at ctx4096 and
  in q8 mode.

## 4. Build recommendations, ranked by expected full-model tok/s

The lever order is **inverted from the scope's expectation**. Rationale: weight-GEMV is already at llama parity
in-model, so Lanes 1–3 are near-dry; the recoverable mass is attention + elementwise fusion.

### Rank 1 — Attention flash-decode efficiency (scope Lane 4). Highest EV; grows with ctx.
- Evidence: +2.73 ms @1024, +4.36 ms @4096; share 21.9 → 23.4 → 31.6%. llama attention is only 7.5% (0.76 ms).
  The cost is flash-decode's fixed split/softmax/reduce overhead: `flash_partial_coop_vec` (0.90 ms) +
  `r_2_8_128…`/`r_1024_16…`/`r_2_…start_pos…` reduces (≈2.0 ms) + `flash_prob/max/den/gmax/combine` softmax-stat
  kernels (≈1.1 ms). At ctx512/1024 the KV is small (few L=128 chunks) so the fixed reduce/stat overhead dominates.
- Projected effect: closing attention to llama-class (e.g. 23% → 10% @1024) recovers ~2.0 ms/tok → ~77 tok/s
  @1024; at ctx4096 the payoff is larger (attention is 31.6% of wall).
- Build gate (from scope Lane 4): ctx4096 attention ≥15% (✓ 31.6%) or ctx1024 ≥10% (✓ 23.4%). Candidate must
  improve ctx4096 ≥5% with <1% regression at ctx512/1024.
- Next step: split the flash-decode kernel cost into partial-compute vs reduce/fixup vs softmax-stat (the
  `extra/qk_flash_decode.py` `gqa_coop_vec` path) and target the reduce/stat overhead; compare against llama
  `flash_attn_tile` + `stream_k_fixup` + `combine_results` geometry.

### Rank 2 — Elementwise / fusion (scope Lane 5, "small-op"). Flat ~1.8 ms at every ctx; q8 does NOT capture it.
- Evidence: +1.83 ms @ all ctx. Single biggest item is `E_49152_32_3` ≈ 1.4 ms (36×, 1/layer) = the FFN
  `silu(gate)*up` activation that llama fuses inline in its MMVQ; the rest is rope (`E_2_8_16…`) and residual
  adds (`E_32_32_4…`). llama's rope+elementwise total is only ~0.36 ms.
- Projected effect: fusing the FFN activation into the gate/up epilogue (or the q8 producer) and folding the
  residual/rope adds recovers up to ~1.4–1.8 ms/tok → ~74–76 tok/s @1024, **stacks on top of Rank 1**.
- Build gate (scope Lane 5): repeated op ≥0.25 ms (✓ `E_49152` at 1.4 ms); full W==D ≥1.02× and no ctx regression.
- Note: the q8 route fuses rmsnorm→gate/up but leaves the SiLU·mul `E_49152` as a separate kernel, so this lever
  is open in both modes.

### Rank 3 — q8 weight-GEMV route (scope Lane 6). Already shipped as opt-in; do not extend.
- q8 gives a stable ~1.06× (weight-GEMV gap goes negative: −0.71/−1.38 ms @1024/4096) and is already the latest
  route. It does not touch attention or elementwise, so it cannot close the residual gap. Keep default-off
  opt-in; no further q8 lifecycle work (consistent with the q8 clock-authority / model-route audits).

## 5. Lanes explicitly dropped (with evidence)

| lane | decision | reason |
|---|---|---|
| Lane 1 — Q6 big roles (`ffn_down`, `lm_head`) | **DROP** | timed in-model BW: ffn_down 690 GB/s (72% HBM), lm_head 863 (90%); gap +0.11 / +0.03 ms. Above llama (626). The stale proxy's 18%/13% "share" was DEBUG2-inflated; real gap is negligible. |
| Lane 2 — full MMVQ family quality | **DROP** | total weight-GEMV is +0.41 ms vs llama @1024 and **negative** at ctx4096 / in q8 mode. No recoverable mass; the older "44% HBM in-model" was a discarded PMC per-kernel estimate, refuted by ProfileGraphEvent timing. |
| Lane 3 — Q4 `ffn_gate/up` role join | **DROP as build** | gap +0.19 ms; q8 already over-closes it (−0.71 ms). Below the scope's own 0.5 ms / 3% build gate. |
| Lane 6 — q8 lifecycle | **CLOSE** (keep shipped opt-in) | does not own attention/elementwise; ~1.06× already captured. |
| Lane 7 — host/persistent runtime | **CLOSE** | host-sync 0% at every ctx; W/D within ~4% (no recoverable divergence). |

## 6. Expected whole-decode effect (from baseline ~66.9 tok/s @ctx1024)

| action | recovered ms/tok | approx tok/s @1024 |
|---|---:|---:|
| Rank 1 attention → llama-class | ~2.0 | ~77 |
| Rank 2 elementwise fusion | ~1.4 | ~74 |
| Rank 1 + Rank 2 stacked | ~3.4 | ~86–88 (approaching llama 97.6) |
| (q8 weight route, already shipped) | ~0.8 | ~71 (opt-in) |

These are first-order projections from the timed shares; each requires its own same-process A/B gate then full
W==D promotion per the scope. ctx4096 benefits more from Rank 1 (attention is 31.6% of wall there).

## 7. Whether default decode behavior changed

**No.** Instrumentation-only. q8 was env-gated (`Q8_FFN_HANDWRITTEN=1`) in its measurement child and restored
off. GPU perf-state `auto` verified before and after. No kernels implemented (the scope's stop condition: build
only after a trustworthy table — satisfied, and the next build is gated by per-lane A/B + W==D).

## 8. Exact commands

```bash
PYTHONPATH=. python3 extra/qk_decode_current_route_attribution.py \
  --modes baseline,q8 --ckpts 512 1024 4096 --nmeas 20 --warmups 8 \
  --out bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution.json
```

## 9. Artifacts

- `extra/qk_decode_current_route_attribution.py`
- `bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution.json`
- `bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution_{baseline,q8}.json`
- `docs/decode-current-route-attribution-result-20260620.md` (Deliverable-0 detail)

## 10. Caveats / confidence

- Per-role times are GPU-timestamp `timed`, rescaled onto the clean W wall by the timed busy-share (factor
  ~0.85–0.87); the rescale assumes roughly uniform PROFILE timestamp overhead, a small possible bias in absolute
  per-role ms (the *shares* and the family-gap signs are robust).
- Single host, best-of-`nmeas=20`, perf-state `auto` (clock-volatile per the q8 clock-authority note); tok/s
  reproduced within ~1 tok/s of the prior q8 model-route audit. The qualitative ranking (attention+elementwise ≫
  weight-GEMV) is large and stable across all six (mode×ctx) cells.
- ~0.25 ms of ffn_down glue (`E_128_32_3`) was initially misrouted to attention by a bare numeric rule; fixed
  (explicit `E_`/`copy` prefixes now win over numeric heuristics). Does not change any conclusion.
