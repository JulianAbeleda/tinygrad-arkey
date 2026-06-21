# Decode Attention Fusion Analysis (Phase A)

Date: 2026-06-20

Scope: `docs/decode-fusion-build-scope-20260620.md` Phase A.

Verdict: `NOT_BUILT_EVIDENCE_BASED_NO_GO` — Phase A is **not pursued as a build**, on three converging grounds:
(1) the dominant + ctx-growing attention cost is **intrinsic O(KV) compute**, not fusible fixup; (2) the
genuinely fusible part is small *fixed* overhead, and Phase B (built, exact) decisively showed decode fusion
**conserves work** (no win); (3) the fully-fused flash kernel is **linearizer-walled** by design. Default decode
behavior NOT changed. This honors the scope's stop condition ("if fusing … cancels the win, stop and document").

## Finding 1 — `reduce_fixup` is the Q@K^T score reduction, not a fusible fixup

The cost-split (`docs/decode-attention-cost-split-result-20260620.md`) labels the dominant attention bucket
`reduce_fixup`, but its `r_*` kernels are the **score computation** (`(qg @ ks.transpose) * scale` →
`flash_decode_attention`, `qk_flash_decode.py:276-281`), which the UOp path materializes as a separate matmul
because fusing q·k into the flash kernel "trips the linearizer's range-ordering" (`qk_flash_decode.py:73-80`).

| ctx | reduce_fixup | `r_2_8_128…` (Hd-reduce, fixed) | `r_1024_16…` (fixed) | `r_2_…start_pos…` (O(KV), grows) |
|---:|---:|---:|---:|---:|
| 512 | 1.66 | 0.78 | 0.56 | 0.32 |
| 1024 | 1.79 | 0.78 | 0.56 | 0.45 |
| 4096 | 2.43 | 0.79 | 0.55 | **1.10** |

The **ctx-slope is `r_2_…start_pos…`** — the KV-length-dependent score reduce, i.e. inherent O(KV) attention
compute. This is not a "cross-chunk fixup" that fusion can remove; the scope's A2 premise is partly mismatched.

## Finding 2 — `softmax_stats`' slope is also intrinsic O(KV)

| ctx | softmax_stats | `flash_prob` (O(KV) exp/key) | `flash_combine` | `flash_max` (fixed) | `flash_den` (fixed) |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.79 | 0.19 | 0.18 | 0.19 | 0.13 |
| 1024 | 0.94 | 0.26 | 0.22 | 0.19 | 0.16 |
| 4096 | 1.85 | **0.71** | 0.50 | 0.20 | 0.30 |

`flash_prob` (the `exp(score-max)` over every key) and `flash_combine` grow with KV — intrinsic. The only
genuinely *fixed* fusible overhead is `flash_max` + `flash_den` + `flash_gmax` ≈ 0.5 ms/token.

## Finding 3 — Phase B proved decode fusion conserves work (so fusing the fixed stats recovers ~nothing)

Phase B1 (`docs/decode-ffn-activation-producer-fusion-result-20260620.md`) BUILT a byte-exact fusion that folds
the FFN `silu(gate)*up` into the up GEMV, eliminating the `E_49152` launch — and measured **0% speedup**. The
activation work moved into the kernel and ran serially for the same ~33 µs. Decode is GPU-execution-bound (D≈W,
host-sync 0%), so removing a launch saves ~nothing; only doing the work *more efficiently* (or hiding it under
memory latency, as llama does) helps. Fusing `flash_max`/`flash_den`/`flash_gmax` would likewise conserve their
work — the ~0.5 ms fixed overhead would reappear in the merged kernel. The local gate (≥0.4 ms@1024) would not be
met, by the same mechanism Phase B demonstrated.

## Finding 4 — the real attention lever is linearizer-walled

Matching llama's attention means its fully-fused flash kernel: `Q@Kᵀ + online-softmax + P@V` in **one** tiled
kernel, where the softmax/reduce ALU hides under the memory loads. tinygrad's flash-decode is split into 6 kernels
**precisely because** the coupled multi-accumulator reduce (q·k nested with the softmax reduce) "trips the
linearizer's range-ordering" (`qk_flash_decode.py:73-80`). That is the same deep codegen wall documented across
the prefill arc (POWN / software-pipelined-K-loop, BEAM-hang class). Building it is a multi-week linearizer
project, out of scope for a bounded fusion build.

## Decision

Do **not** build Phase A stat/reduce fusion. The dominant attention cost is intrinsic O(KV) compute (score
matmul + per-key exp); the fusible fixed overhead is ~0.5 ms and, per Phase B, work-conserved; and the real
fully-fused-flash lever is linearizer-walled. Reopen only if (a) the linearizer gains coupled-multi-reduce
support, or (b) the score matmul is re-optimized (a GEMV/codegen lane the scope says not to reopen).

## Boundary

No decode default changed. No kernel built for this phase (analysis only, from the existing pinned-clock split).
