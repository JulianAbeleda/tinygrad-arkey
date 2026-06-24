# Tinygrad-vs-Llama Decode Time-Tax Diff — Result (2026-06-22)

## Verdict: **LLAMA_DIFF_AUDIT_READY**

Apples-to-apples per-primitive decode time-tax diff built for Qwen3-8B-Q4_K_M on gfx1100,
full **per-role** at all four ctx (512/1024/2048/4096), validated and reconciled. Audit/tooling
/docs only — no kernel optimization, no default change, no new primitive. **Default behavior
changed: no.**

**Headline:** After `Q4K_GEMV_WARP`, tinygrad's **weight-GEMV is already at or below llama**
(combined gap **−1.1 ms/token** @ctx1024 — tinygrad slightly *faster*). The entire remaining
decode gap (~26 tok/s @ctx1024) is **non-GEMV**: elementwise/norm fusion + attention
flash-decode. **Next work should target attention and elementwise fusion, NOT more weight-GEMV.**
This quantifies, per bucket with measured data, the prior qualitative findings
`[decode-gap-is-attention-not-weight-gemv]` and `[llama-vs-tinygrad-decode-gap]`.

---

## 1. Data sources

| source | side | ctx | authority | confidence |
|---|---|---|---|---|
| `bench/qk-tinygrad-vs-llama-time-tax/tinygrad_default.json` | tinygrad default route | 512/1024/2048/4096 | ProfileGraphEvent GPU-busy (median-5) + wall token_ms (median-40, `.item()`); `qk_decode_time_tax_audit.py`, HEAD | HIGH |
| `bench/qk-tinygrad-vs-llama-time-tax/tinygrad_warp.json` | tinygrad `Q4K_GEMV_WARP=1 _DOWN=1` | same | same tool, warp flags (L5, ran 2026-06-22) | HIGH |
| `bench/qk-llama-decode-primitive-audit/llama_decode_kernel_trace_ctx1024.csv` | llama per-dispatch raw | 1024 | rocprofv3 `--kernel-trace`, llama.cpp `ac4cddeb` b9592 (existing oracle) | HIGH |
| `bench/qk-tinygrad-vs-llama-time-tax/llama_capture/llama_ctx{512,2048,4096}_kernel_trace.csv` | llama per-dispatch raw | 512/2048/4096 | **fresh rocprofv3 capture 2026-06-22**, same build/model/flags | HIGH |
| `bench/qk-llama-decode-primitive-audit/decode_kernel_trace.json` | llama per-family ledger | 512/1024/4096 | published oracle (cross-check authority) | HIGH |
| Qwen3-8B-Q4_K_M gguf header | tensor→quant→grid map | — | parsed 2026-06-22 | HIGH |

llama-bench clean wall tok/s (no rocprof): d512 97.71, d1024 97.39, d4096 92.37. The diff
compares **GPU time on both sides** (tinygrad ProfileGraphEvent vs llama rocprofv3); llama's
GPU-sum equals its `decode_ms_per_tok` (serial stream, ~no overlap), ~3% under its wall.

**Validation (CSV-derived vs published family ledger):** weight-GEMV total / decode total =
ctx512 −0.37%/−0.80%, **ctx1024 +0.00%/+0.00%** (same oracle CSV), ctx4096 −0.32%/−0.36%.
The fresh captures reproduce the published oracle to <1%. The per-role split is anchored to
the measured Q6_K-down dispatch count.

## 2. Mapping confidence

- **Tinygrad → 9 buckets**: reuses `classify()` dim-signatures (incl. `q4k_gemv_warp_<out>_<in>`,
  verified to carry the signature). HIGH.
- **Llama families → buckets** (`ffn_silu`, `flash_attn_*`, `rms_norm`/`rope`/`q8_1_quant`/
  `k_set_rows`/`k_bin_bcast`/copies): HIGH. Prefill kernels (`mul_mat_q`, Tensile `Cijk_*`,
  `flash_attn_ext_f16`, `dequantize_block_q6_K`, `quantize_mmq`) excluded.
- **Llama weight-GEMV per-role split** (the one combined `mmvq_weight_gemv` family): **HIGH**.
  gguf gives the exact tensor→quant→grid map (`Grid_Size_X = out_features × 32`): gate/up
  out=12288→grid 393216 (Q4_K); lm_head out=151936→grid 4861952 (Q6_K); k/v out=1024→grid
  32768; q/o/down out=4096→grid 131072. The out-4096 cell (q+o+down) is split by a **measured**
  bimodal per-dispatch duration: q/o (in=4096) ~17–22 µs vs ffn_down (in=12288, 3× K) ~50–55 µs
  with a clean empty gap at 35–50 µs; a 40 µs threshold separates them and the resulting down
  count matches the independently-measured Q6_K-down count. Not a model — a measurement.

## 3. Per-ctx bucket diff — **current state (`Q4K_GEMV_WARP` on), wall-normalized**

`gap_ms = tinygrad − llama` (positive = tinygrad slower). Ranked by gap_ms. tg%/ll% = share of
each side's per-token time. Σ gap_ms reconciles to `tinygrad_token_ms − llama_token_ms` at every ctx.

### ctx512 — tinygrad 76.1 tok/s vs llama 104.6 (gpu)
| bucket | tg_ms | llama_ms | gap_ms | ratio | tg% | ll% | notes |
|---|---|---|---|---|---|---|---|
| norm/rope/small ops | 3.426 | 1.359 | **+2.067** | 2.52 | 26.1 | 14.2 | unfused RMSNorm/RoPE/q8-quant/residual/KV-copy |
| attention qk/softmax/pv | 1.863 | 0.383 | **+1.479** | 4.86 | 14.2 | 4.0 | flash-decode vs llama `flash_attn_tile` |
| FFN activation | 1.322 | 0.065 | **+1.256** | 20.25 | 10.1 | 0.7 | unfused silu(gate)*up vs `unary_gated` |
| FFN down | 1.984 | 2.008 | −0.024 | 0.99 | 15.1 | 21.0 | parity (warp_down) |
| lm_head | 0.573 | 0.604 | −0.031 | 0.95 | 4.4 | 6.3 | parity |
| attention q/o/k/v proj | 1.511 | 2.000 | −0.489 | 0.76 | 11.5 | 20.9 | tinygrad faster |
| FFN gate/up | 2.468 | 3.138 | −0.671 | 0.79 | 18.8 | 32.8 | warp beats llama MMVQ |

### ctx1024 — tinygrad 74.0 tok/s vs llama 100.3 (gpu)
| bucket | tg_ms | llama_ms | gap_ms | ratio | tg% | ll% |
|---|---|---|---|---|---|---|
| norm/rope/small ops | 3.443 | 1.649 | **+1.794** | 2.09 | 25.5 | 16.5 |
| attention qk/softmax/pv | 2.151 | 0.507 | **+1.643** | 4.24 | 15.9 | 5.1 |
| FFN activation | 1.339 | 0.129 | **+1.210** | 10.38 | 9.9 | 1.3 |
| FFN down | 1.982 | 1.988 | −0.006 | 1.00 | 14.7 | 19.9 |
| lm_head | 0.580 | 0.601 | −0.021 | 0.97 | 4.3 | 6.0 |
| attention q/o/k/v proj | 1.508 | 1.984 | −0.476 | 0.76 | 11.2 | 19.9 |
| FFN gate/up | 2.506 | 3.113 | −0.607 | 0.80 | 18.6 | 31.2 |

### ctx2048 — tinygrad 71.0 tok/s vs llama 91.3 (gpu)
| bucket | tg_ms | llama_ms | gap_ms | ratio | tg% | ll% |
|---|---|---|---|---|---|---|
| attention qk/softmax/pv | 2.749 | 0.761 | **+1.988** | 3.61 | 19.5 | 6.9 |
| norm/rope/small ops | 3.469 | 2.250 | **+1.219** | 1.54 | 24.6 | 20.5 |
| FFN activation | 1.336 | 0.256 | **+1.080** | 5.22 | 9.5 | 2.3 |
| lm_head | 0.581 | 0.603 | −0.023 | 0.96 | 4.1 | 5.5 |
| FFN down | 1.943 | 1.985 | −0.042 | 0.98 | 13.8 | 18.1 |
| attention q/o/k/v proj | 1.485 | 2.001 | −0.516 | 0.74 | 10.5 | 18.3 |
| FFN gate/up | 2.530 | 3.100 | −0.571 | 0.82 | 18.0 | 28.3 |

### ctx4096 — tinygrad 67.0 tok/s vs llama 77.7 (gpu)
| bucket | tg_ms | llama_ms | gap_ms | ratio | tg% | ll% |
|---|---|---|---|---|---|---|
| attention qk/softmax/pv | 3.804 | 1.222 | **+2.582** | 3.11 | 25.5 | 9.5 |
| FFN activation | 1.353 | 0.509 | **+0.844** | 2.66 | 9.1 | 4.0 |
| norm/rope/small ops | 3.454 | 3.464 | −0.009 | 1.00 | 23.1 | 26.9 |
| lm_head | 0.589 | 0.620 | −0.032 | 0.95 | 3.9 | 4.8 |
| FFN down | 1.721 | 1.978 | −0.256 | 0.87 | 11.5 | 15.4 |
| attention q/o/k/v proj | 1.477 | 1.984 | −0.507 | 0.74 | 9.9 | 15.4 |
| FFN gate/up | 2.529 | 3.086 | −0.557 | 0.82 | 16.9 | 24.0 |

(`graph/runtime/host` and `unknown/unmapped` are ~0 at every ctx — decode is GPU-bound on both
sides and the family/role mapping leaves nothing material unmapped.)

## 4. What `Q4K_GEMV_WARP` closed (default → warp, L5)

| ctx | default tok/s | warp tok/s | Δtok/s | gpu_busy Δms | closed: gate/up | closed: down |
|---|---|---|---|---|---|---|
| 512 | 68.2 | 76.1 | +7.9 (+11.6%) | −1.42 | 1.13 ms | 0.20 ms |
| 1024 | 66.8 | 74.0 | +7.2 (+10.8%) | −1.52 | 1.12 ms | 0.19 ms |
| 2048 | 64.3 | 71.0 | +6.7 (+10.4%) | −1.55 | 1.14 ms | 0.19 ms |
| 4096 | 61.0 | 67.0 | +6.0 (+9.8%) | −1.49 | 1.09 ms | 0.20 ms |

`Q4K_GEMV_WARP` removes ~1.3 ms/token of GPU time, **entirely in FFN gate/up + down**, exactly
as the FFN-GEMV diagnostic predicted. It does so well that after warp tinygrad's gate/up
(2.51 ms @1024) **beats** llama's MMVQ (3.11 ms). It does **not** touch the non-GEMV buckets,
so it does not close the dominant remaining gap.

## 5. Ranked remaining gap_ms (`Q4K_GEMV_WARP` on)

| rank | bucket | gap @512 | @1024 | @2048 | @4096 | shape |
|---|---|---|---|---|---|---|
| — | **non-GEMV total** | **+4.80** | **+4.65** | **+4.29** | **+3.42** | the whole gap |
| 1 | attention qk/softmax/pv | +1.48 | +1.64 | +1.99 | +2.58 | **grows with ctx** |
| 2 | norm/rope/small ops | +2.07 | +1.79 | +1.22 | −0.01 | flat→fades (llama's grows w/ KV) |
| 3 | FFN activation | +1.26 | +1.21 | +1.08 | +0.84 | flat ~1.2 ms, ratio 10–20× |
| — | weight-GEMV total | **−0.71** | **−1.11** | **−1.15** | **−1.27** | tinygrad ahead (after warp) |

Robustness: the ranking is identical in the **raw gpu-busy view** (no wall-norm) — ctx1024 raw
gaps norm/small +2.39, attention +2.02, activation +1.44, all above gate/up +1.06. The
conclusion does not depend on the wall-normalization assumption.

## 6. What we know with confidence
- **Weight-GEMV is no longer the gap.** Gate/up + down + proj + lm_head combined: tinygrad is
  ~1.1 ms/token **faster** than llama @ctx1024 after `Q4K_GEMV_WARP`. FFN GEMV is tinygrad's
  largest *share* but a *negative* gap — share ≠ gap, exactly the inversion this audit set out to expose.
- **The remaining decode gap (~26 tok/s @ctx1024) is 100%+ non-GEMV**, in three buckets:
  attention flash-decode (ctx-growing), norm/rope/small ops (unfused tiny kernels), FFN
  activation (unfused silu*up). Ratios 2–20×.
- **Attention is the ctx-growing gap** (+1.48 → +2.58 ms from ctx512→4096) and the #1 bucket at
  ctx ≥ 2048. **Elementwise fusion (small-ops + activation)** is the flat gap (~3.0 ms @ctx512–1024),
  the largest combined at short/medium ctx.
- Cross-stack numbers validated: fresh captures reproduce the published oracle to <1%; per-bucket
  gap_ms reconciles to the measured wall token_ms gap at every ctx.

## 7. What remains unknown / caveats
- **Wall-norm assumes uniform overlap** across tinygrad buckets (tinygrad gpu_busy > wall by the
  HCQ graph overlap). Mitigated: the ranking holds in the raw view too. The true wall-vs-wall gap
  is ~0.3 ms smaller than shown (llama GPU-sum is ~3% under its wall).
- **`norm/rope/small ops` is a coarse bucket** (RMSNorm + RoPE + q8-quant + residual + KV copies).
  Acting on it needs a finer sub-breakdown — a cheap follow-up audit (the kernels are already
  named in the artifact's `top_kernels`).
- **rocprofv3 is blind to tinygrad's HCQ**; each side uses its own profiler (both HW-timestamp GPU
  time). Comparison is GPU-time vs GPU-time, not a single unified trace.
- llama attention numbers at ctx512/2048/4096 are now **freshly measured** (this capture), not the
  older derived constants.

## 8. Next-primitive recommendation — **ranked by gap_ms, not share**

1. **FFN activation fusion** — gap +1.2 ms flat, **ratio 10–20×**, highest transfer credibility.
   tinygrad runs silu(gate)*up as an unfused `E_49152` elementwise; llama fuses it into one
   `unary_gated_op_kernel`. Smallest absolute but cleanest, lossless, fusion is a known tinygrad
   lever. A natural first strike.
2. **norm/rope/small-ops fusion** — gap +1.8 ms @ctx1024 (largest single bucket at short/medium
   ctx). tinygrad emits many tiny unfused RMSNorm/RoPE/residual/q8-quant kernels (the 780-vs-260
   progs/token gap). Needs a finer sub-audit first to pick the fusable chain.
3. **attention flash-decode** — gap +1.6 → +2.6 ms, **the ctx-growing gap**, #1 at long ctx.
   tinygrad's flash-decode is 3–5× llama's `flash_attn_tile`. Highest ceiling but the attention
   arc is partly exhausted (`gqa_coop_vec` shipped); reopening needs the flash-decode kernel
   structure, a deeper lever.
4. **Do NOT** pursue more weight-GEMV (gate/up/down/proj/lm_head) — negative gap after warp.

Suggested order: (1) cheap lossless activation-fusion strike → (2) small-ops sub-audit to scope
the norm/rope fusion → (3) attention flash-decode for the long-ctx tail. (1)+(2) target the flat
~3 ms gap; (3) targets the ctx-growing tail.

## Artifacts
- `bench/qk-tinygrad-vs-llama-time-tax/latest.json` — full diff (both views, per-role, ranked, validation).
- `bench/qk-tinygrad-vs-llama-time-tax/{tinygrad_default,tinygrad_warp}.json` — tinygrad inputs.
- `bench/qk-tinygrad-vs-llama-time-tax/llama_capture/llama_ctx{512,2048,4096}_kernel_trace.csv` — fresh llama traces.
- `extra/qk_tinygrad_vs_llama_time_tax.py` — the diff tool. `docs/...-scope-20260622.md` — scope/method.
