# Attention-Combine In-Kernel Fusion — Result (14B) — reachable search REFUTED, capability scoped

Target: remove the attention score/combine reduce (~12-24% of decode, the biggest
removable bucket) for 14B/32B, via generated codegen / search — NO handwritten kernel.

## What the attention combine is

14B/32B attention is the GENERATED `gqa_coop_vec` route (`extra/qk_flash_decode.py`;
the handwritten owned AMDGCN tile is 8B-only, default-off). It runs:
- `flash_partial`: Hq*S workgroups (one per head, split), each an online-softmax partial.
- `flash_gmax` + `flash_den` + `flash_combine`: **3 reduce kernels over the S splits** —
  the log-sum-exp (LSE) merge. These are the `attention_combine` bucket.

The combine is external because the S splits live across WORKGROUPS (occupancy).

## Reachable search — all REFUTED (measured)

| path | result |
|------|--------|
| **FLASH_L knob** (chunk count) | shrinks the combine (13.57%->12.01% @L512, reduce_eliminated=PASS) but regresses tok/s 50.2->47.6->43.5 @L128/256/512 — parallelism loss dominates |
| **generated wholecache/score-broadcast route -> 14B** (Hq=40,G=5) | token-identical but regresses ctx512 50.2->45.3 (-9.8%); the score-broadcast variant is 8B-tuned. BoltBeam refutes on the protected-context regression |
| **flash-on @ctx128** | slower (52.3->50.8) — the shipped threshold 512 is already optimal |

## Why the GEMV in-kernel combine does not transfer directly

`decode_q4k_inkernel_combine_kv` (built + proven, rel_rmse 3e-7) does a plain **SUM**
combine in LDS. The attention combine is an **LSE merge** (max + rescale-by-exp + sum) —
a different, coupled reduction. So attention needs its own in-kernel-LSE combine.

## Remaining capability (scoped, substantial)

A fused-flash kernel that places the S splits as **waves in ONE workgroup per head** and
does the LSE combine **in LDS** (each wave = one split's partial+max+denom; LDS+barrier
merge -> out[h,:]), removing the 3 combine reduce kernels WITHOUT the parallelism loss.
Generic codegen, NOT a handwritten kernel. Substrate exists:
`extra/qk_codegen_outer_b_lds_split.py` already expresses the flash LSE merge via LDS, and
the fused_xlane microgates combine score+PV in-kernel. This is a distinct, larger kernel
than the GEMV combine (online softmax is subtle; needs token-match on 14B/32B across ctx).

## Disposition

No default path changed (wholecache guard reverted to 8B-scoped; a comment records the
14B regression). BoltBeam candidate `decode_attention_combine_reduce_fusion` carries the
three refuted search results + the exact remaining capability. This is the honest end of
the *reachable* search: the attention combine is not profitably removable by any existing
knob/route for 14B, and the next step is the scoped fused-flash-LSE kernel.
