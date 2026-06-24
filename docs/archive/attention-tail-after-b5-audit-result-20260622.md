# Attention Tail After B5 — Result (2026-06-22)

## Verdict: **ATTENTION_BOUNDED_LEVER_EXHAUSTED_NO_REOPEN**

The attention bucket is **correctly mapped** and **on the (ctx-growing) critical path**, but the **bounded**
attention lever is **exhausted**: the owned AMDGCN flash-decode tile saturates at **+5.7% @ctx4096 (< the
+7% promotion gate)** and **+0.23% @ctx1024**, and the deeper single-fused-LDS-`v_dot2`-tile lever is
codegen-blocked. The diff's attention gap_ms is real but **un-actionable by bounded means**. Do not reopen.
Audit only; default unchanged.

## B5 saturation — explained (the reopen gate)
The required explanation, from the closeout docs:

- **Mechanism: off-critical-path / overlap (measured), not Amdahl-projection.** A 3-point curve made the
  split-KV combine progressively cheaper (combine compute 9.7 → 5.8 → 4.0 µs) while whole-decode W==D
  **did not move** (a 2.4× cheaper combine added **+0.25%**; further cuts added ~0). Even a *free* combine
  extrapolates to **~+5.7% @ctx4096 < +7%** (`docs/b4-cheaper-combine-result-20260622.md:42-50`). The B4-era
  Amdahl/combine-tax projection (which assumed the combine's standalone time was fully serial) was
  **refuted** by this measurement.
- **Owned-tile W==D transfer** (the bounded lever's ceiling): +0.23% / +1.98% / **+5.66%** @ctx 1024/2048/4096
  (`docs/decode-time-tax-audit-result-20260622.md:37`). Ctx-dependent (small at 1024, grows with attention's
  share), but never clears the gate.
- **Transfer contrast** (ground truth): an FFN gate/up speedup (q8) **transfers** to W==D; the attention
  (B5) speedup **does not** — confirming attention is partly overlapped while weight-GEMV is serial.
- **Deeper lever is codegen-blocked**: the single fused LDS-tiled `v_dot2` flash tile is inexpressible —
  `fused-flash-concrete-gate FAIL_LOCAL_AB` (tiled-GEMM codegen and the `.set/.after` fusion idiom are
  mutually exclusive) and `matmul-pv BLOCKED_BY_LAYOUT` (symbolic split count can't reshape into a tiled
  batched matmul). Reopening needs an **unbounded renderer/codegen capability**, not a bounded primitive.

## Mapping (correct)
Attention = `flash_partial_coop_vec_32_128` + softmax chain (`flash_max/prob/den/gmax/combine`) + the
ctx-growing `start_pos` QK/PV reduce `r_2_28start_pos...`. All carry `start_pos` + `exp`; correctly the
`gqa_coop_vec` flash-decode path. No mislabel (unlike the activation/small-ops buckets).

## Ctx-scaling & gap (this audit)
| ctx | attention flash gpu-busy µs | diff gap_ms (wall-norm) | ratio |
|---|---|---|---|
| 512 | 1827 | +1.518 | 4.96 |
| 1024 | 2006 | +1.684 | 4.24 |
| 2048 | — | +2.024 | 3.66 |
| 4096 | 3154 | +2.641 | 3.16 |

ctx-slope +72.6% (512→4096) — the **only** large bucket that grows with ctx. This is why attention's wall
leverage grows with ctx (B5 +0.23%@1024 → +5.66%@4096) yet still under-clears the gate.

## Answers
| question | answer |
|---|---|
| Mapped correctly? | **Yes** — `flash_*` + ctx-growing `start_pos` reduce. |
| Critical-path or overlapped? | **Partly critical (ctx-growing, transfers +5.7%@4096), partly overlapped (combine off-path).** The diff gap_ms overstates the bounded wall opportunity. |
| Reopen justified? | **No** — bounded lever exhausted (< +7% gate); deeper lever codegen-blocked. |

## Artifacts
`extra/qk_attention_tail_after_b5_audit.py`, `bench/qk-attention-tail-after-b5-audit/latest.json`.
