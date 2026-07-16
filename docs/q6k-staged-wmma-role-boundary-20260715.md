# Q6_K staged fp16-WMMA role boundary

The staged Q6_K fp16-WMMA overlay remains admitted only for the explicit
`ffn_down` contract `(M,N,K)=(512,4096,12288)` on `qwen3_14b_q6k_gfx1100`,
with an explicit memory budget and the `PREFILL_Q6K_WMMA=1` opt-in. Its
rollback is `direct_packed`.

No other Q6 role is promoted. The Q6 inventory includes `attn_kv`
`(512,1024,5120)` and a separate 14B `ffn_down` inventory shape
`(512,5120,17408)`; neither has an independently proven staged-emitter
provenance record plus whole-workload memory admission. A Q6 family label,
Q4 evidence, or a theoretical byte estimate is insufficient. Those shapes
therefore remain on the direct-packed fallback until both records exist.

The owned benchmark reports materialization, contraction, and combined times;
each sample uses `result.numpy()` as the synchronization point and labels its
wall accounting `synchronized_wall`, so combined timing includes dequantization
and GEMM.
