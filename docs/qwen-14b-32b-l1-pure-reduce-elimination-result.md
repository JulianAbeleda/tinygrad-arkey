# L1 Pure Reduce-Elimination — Result (14B, real GPU)

Track: close the 14B/32B decode `reduce_partial` gap with a **generic** generated
capability — no handwritten kernel, no model-name check, no shape special-case.
BoltBeam diagnoses/evaluates; tinygrad only captures graph facts (RSR0) and (next)
gets a generic codegen capability.

Hardware: RX 7900 XTX (gfx1100). Model: Qwen3-14B-Q4_K_M. Defaults on
(`DECODE_Q4K_G3_ANYSHAPE=1`, `DECODE_ROUTE_ATTN_K=1`). Flash threshold ctx 512.

## L1A — Reduce-Source Resolution → `L1A_PASS_REDUCE_SOURCE_RESOLVED`

Method: `qk_decode_reduce_source_trace.py` captures the ordered decode graph and,
per hot `r_*` reduce, its nearest non-reduce producer/consumer + a ±4 window.
BoltBeam (`boltbeam/reduce_source.py`) classifies each by that producer/consumer
identity (not the reduce name): `*_partial` producer → coop-partial combine;
`flash_*`/`start_pos` neighbor → attention; entire loop-nest == hidden → RMSNorm
sum-of-squares; vocab-shaped neighbor with no weight producer → sampling.

**100% of the reduce bucket source-resolved at both contexts, 0% unknown:**

| context | reduce bucket (% decode) | attention_combine | coop_partial | rmsnorm | sampling |
|---|---|---|---|---|---|
| ctx128 (flash off) | 29.7% | 24.4% (high risk) | 2.4% | 2.8% | 0.1% (not removable) |
| ctx512 (flash on)  | 16.2% | 11.9% (high risk) | 2.2% | 2.1% | 0.05% (not removable) |

Sampling/gumbel is never marked removable (hard rule). The dominant reduce
`r_8_8_16_2_20_4_2_32` (~10% at both contexts) sits between the KV projection +
coop-partial combine and `flash_max_40` → it is the **attention score/combine**
reduce, confirmed by its ±4 window.

## L1B — Generic Capability Selection → `L1B_PASS_GENERIC_CAPABILITY_SELECTED`

Selected: **`generated_rmsnorm_reduce_scale_fusion`** — fuse the RMSNorm
`reduce(x²)` with its scale elementwise into one generated kernel.

- Coverage: 2.1% (ctx512) / 2.8% (ctx128) of whole decode → meets the "≥2% with
  low risk" bar. Low risk: a generic reduce→consumer fusion, no attention kernel.
- Transfers to 32B: identical RMSNorm structure, 64 layers × 2 norms (more, not
  fewer), so the same generic fusion applies unchanged.
- Rollback: `DECODE_RMSNORM_REDUCE_FUSION`. BoltBeam candidate:
  `decode_rmsnorm_reduce_fusion`.

**Deferred (recorded, not chased): the bigger prize is attention.** The largest
removable reduce bucket is `attention_combine` (12–24% of decode), but removing it
means fusing the attention score/combine into the flash kernel — a high-risk
attention/flash capability (`generated_attention_combine_reduce`), a different
track than pure reduce elimination. BoltBeam records it as the exact missing
capability (candidate `decode_attention_combine_reduce_fusion`) so the next real
work targets a generic attention fusion, not another model-specific kernel.

## Honest finding

Pure *generic* reduce elimination is a **low-Amdahl** lever for 14B/32B parity:
the clean, low-risk generic reduce (RMSNorm fusion) is ~2%, and the dominant
reduce bucket is attention — which flash already partially handles and whose
residual belongs to attention-kernel fusion. This redirects the parity effort
honestly instead of hand-writing a reduce kernel for a specific shape.

## Artifacts

- `bench/qwen-14b-32b-l1-reduce-source/latest.json` — L1A+L1B resolution (both ctx)
- `bench/qwen-14b-32b-l1-reduce-source/ordered_trace_ctx{128,512}.json` — RSR0 traces
- BoltBeam: `boltbeam/reduce_source.py` + `tests/test_reduce_source.py`; candidates
  `decode_rmsnorm_reduce_fusion` (selected) and `decode_attention_combine_reduce_fusion`
  (deferred).

## Next (L1C, gated)

Implement `generated_rmsnorm_reduce_scale_fusion` as a generic scheduler/UOp
fusion behind `DECODE_RMSNORM_REDUCE_FUSION` (flag-off byte-identical), microgate +
14B/32B token-match, then authority measurement (L1D) and BoltBeam promote/refute
(L1E). Note the modest expected ceiling (~2%); the attention-combine capability is
the higher-leverage follow-on.
