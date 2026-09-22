# NVIDIA pp512 K residual service audit

## Measured exposure

The exact cross-runtime trace assigns K 2.210560 ms tinygrad active versus
1.251141 ms llama, an active debt of 0.959419 ms. This is 36 K projections on
each runtime; tinygrad's K population is Q4_K `(M,N,K)=(512,1024,4096)` and
the trace has no evidence that K is mixed-format. The K debt is only 2.3% of
positive active debt and is lower priority than V/Flash.

## Existing qualified route

The compiler-owned Q4 K route uses canonical GGUF weights, compact Q8 records,
signed IMMA, and the occupancy-safe 64x32x64 four-warp geometry with 256 CTAs.
Its ordered 36-call proxy is 2.580838 ms minimum / 2.592229 ms median (71.690
us/call), versus retained FP16 6.3845 ms. The fresh whole-model K integration
recovers 4.004704 ms wall (70.390585 ms vs 74.395289 ms control) with exact
canonical census and A/B/A replay.

## Decision

K is already qualified and default-off integrated; no new K implementation is
authorized. The remaining K residual is service/lifecycle rather than missing
arithmetic substrate. Existing four-warp decode K evidence is wall-neutral and
does not justify further geometry tuning.

## Ranked discriminator

If reopened, compare exact 36-call cold service against the existing K64
authority while recording producer-to-main intervals, queue placement, and L2
reuse. Only a repeated cold service win that survives the whole-model wall
bracket would justify a change. V's producer/lifecycle findings transfer only
as a measurement protocol (producer readiness, queue/cache intervals), not as
an assumed K mechanism.

Evidence: `nv-prefill-exact-cross-runtime-trace/cross-runtime-accounting.json`,
`nv-prefill-compiler-q4k-kv-role-result-20260828.md`, and
`nv-q4k-k-four-warp-route-result-20260822.md`.
