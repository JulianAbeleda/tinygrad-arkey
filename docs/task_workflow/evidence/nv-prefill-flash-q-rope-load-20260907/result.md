# NV pp512 Flash Q-RoPE load closure (2026-09-07)

The exact generated pre-Flash E2048 service is fp32 Q RoPE complex multiply plus fp16 cast over `[1,32,512,128]`, using fp32 frequencies `[512,128]` (cos first 64, sin second 64). It costs 9.984 us/layer, 0.359424 ms/token exposed over 36 layers.

A typed exact-NV-pp512 Flash descriptor accepted fp32 pre-RoPE Q and a read-only frequency PARAM. Its Q fragment load paired d/d+64, performed the exact fp32 complex multiply, then cast to fp16 before WMMA. A deterministic all-head/all-token fixture was bit-exact against explicit RoPE followed by current generated Flash: max_abs=0, mean_abs=0, finite and allclose. Full model smoke also passed token198, canonical252, exact logits and every deep stage across three replay cycles; E2048 was absent.

The initial fragment retained the KV RANGE dependency and measured 53.447752 ms smoke versus the ~48 ms current regime. A second form removed RANGE from the Q fragment so the transform was loop-invariant in the UOp graph. It remained bit-exact and removed E2048, but measured 53.458355 ms. Holding all transformed Q fragments live across the KV loop therefore preserves roughly the same ~5.4 ms resource/body regression as recomputing them; this overwhelms the 0.359 ms service ceiling.

Both forms are rejected and all production/compiler changes are reverted. The viable next seam is a bounded CTA-local Q stage loaded/rotated immediately before each head tile without extending all fragment live ranges, or producer-side fusion at the Q projection/RoPE boundary.
