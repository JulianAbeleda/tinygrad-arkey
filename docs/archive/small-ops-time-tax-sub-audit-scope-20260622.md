# Norm/Rope/Small-Ops Sub-Audit — Scope (2026-06-22)

**Phase 2 of 8B decode-gap exhaustion.** Audit only; no kernel/default change.

## Question
The diff's **"norm/rope/small ops"** bucket is coarse (~1.8 ms wall-norm / ~4.0 ms gpu-busy @ctx1024,
ratio ~2.1× to llama) and lumps many kernels. Break it into constituent taxes so the gap can be ranked
by gap_ms per real primitive, not by an aggregate that may hide mislabels.

## Method
- Use the rendered-source flags from `bench/qk-decode-kernel-probe/latest.json` (`start_pos`/`uchar`/
  `exp`/`sin`/`sqrt`) to assign each kernel in the bucket a **corrected role**:
  - `sqrt && !uchar && !start_pos` → genuine **RMSNorm/qk-norm**
  - `sin` → genuine **RoPE**
  - `start_pos && uchar` → **MISLABEL: KV-projection** (fused k/v-proj + rope + cache-write)
  - `uchar && !start_pos` → **MISLABEL: q8 activation-quant / quant-reduce**
  - vocab-1187 reduces → lm_head sampling (argmax)
- Compare each genuine constituent to its llama family (`decode_kernel_trace.json`: `rmsnorm`, `rope`,
  `q8_1_activation_quant`, `copy_cast_kv`, `residual_add`).
- Determine which constituent dominates and whether the **genuine** norm/rope gap is a real, bounded opportunity.

## Mapping caveat under test
`classify()` buckets by kernel-name only and cannot see that a `r_*` kernel takes `start_pos` + loads
`uchar` (a quantized projection). The audit corrects this from the rendered source.

## Deliverables
`extra/qk_small_ops_time_tax_audit.py`, `bench/qk-small-ops-time-tax-audit/latest.json`, this scope + result doc.

## Stop condition
If the genuine norm/rope cost is near llama parity (the bucket gap being mostly mislabeled proj/quant),
report it and do not propose a norm/rope primitive.
