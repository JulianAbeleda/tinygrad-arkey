# Norm/Rope/Small-Ops Sub-Audit — Result (2026-06-22)

## Verdict: **SMALL_OPS_BUCKET_MOSTLY_MISLABELED_GENUINE_NORM_NEAR_PARITY**

The coarse "norm/rope/small ops" bucket is **~55% mislabeled** (gpu-busy). The **genuine** RMSNorm/qk-norm
cost is ~0.9 ms gpu-busy (~0.77 ms wall-norm) — **at/below llama parity (gap −0.21 ms, ratio 0.79)**. The
bulk of the bucket is KV-projection and q8 activation-quant work that llama accounts in *other* families.
There is **no large bounded norm/rope primitive**. Audit only; default unchanged.

## Decomposition (gpu-busy µs/token, ctx1024)
| constituent | µs/tok | real norm/rope? | maps to llama family |
|---|---|---|---|
| MISLABEL: KV-projection (k/v-proj + rope + cache-write, `start_pos`+`uchar`) | 1310 | no | mmvq kv-proj + rope + k_set_rows |
| MISLABEL: q8 activation-quant / quant-reduce (`uchar`) | 902 | no | `quantize_q8_1` |
| **genuine: RMSNorm + qk-norm** (`sqrt`) | **885** | **yes** | `rms_norm` |
| other small (copy / elementwise) | 654 | partial | `residual_add` / copies |
| lm_head sampling (argmax over vocab) | 289 | no | (sampling, not in llama decode trace) |
| MISLABEL: attention reduce | 8 | no | flash |
| **bucket total** | **4048** | | |

Stable across ctx: genuine norm/rope ≈ 878/885/884 µs at ctx 512/1024/4096; mislabeled ≈ 55% at every ctx.

## Genuine norm/rope vs llama (wall-normalized, ctx1024)
| | tinygrad (genuine) | llama (rmsnorm+rope) | gap_ms | ratio |
|---|---|---|---|---|
| norm/rope | 0.768 ms | 0.979 ms | **−0.21** | 0.79 |

tinygrad's `r_16_256` RMSNorm (Σx²→rsqrt over 4096) and qk-norm reduces are **faster** than llama's
`rms_norm` + `rope` combined. The apparent +1.79 ms bucket gap in the diff was a **bucket-boundary
artifact**: tinygrad fuses k/v-projection + RoPE + cache-write into single `r_*`/`E_*` kernels that
`classify()` mislabels as "small ops", while llama splits the same work across `mmvq` (projection,
already counted in the diff's projection bucket), `rope`, and `k_set_rows`.

## Answers
| question | answer |
|---|---|
| Dominant constituent? | **KV-projection + q8-quant (mislabeled), ~2.2 ms** — not norms. |
| Genuine norm/rope gap? | **−0.21 ms (parity / tinygrad faster).** |
| Bounded norm/rope primitive? | **No.** |
| Where did the bucket gap go? | Bucket-boundary mismatch (tinygrad fuses proj+rope+cache; llama splits) + the KV-copy from Phase 1. |

## Handoff
The 1.31 ms "KV-projection" constituent is fused proj+rope+cache-write — adjacent to the KV-cache copy
(Phase 1). The norm/rope lane itself is exhausted. See the decision doc.

## Artifacts
`extra/qk_small_ops_time_tax_audit.py`, `bench/qk-small-ops-time-tax-audit/latest.json`.
