# NV 227 push: shared-Q8 Q4/Q4 K/V producer scope

Date: 2026-08-24
Base checkpoint: `b7ddda44b`
Target: RTX 5090 `NV sm_120`, Qwen3-8B-Q4_K_M decode at depth 512

## Starting ledger

The installed ordinary Q4/Q4 pair checkpoint is `4.518148 ms/token =
221.330 tok/s`. The 227 target is `4.405286 ms/token`, leaving `112.862
us/token` or `5.670 tok/s`.

The next closed population is the nine shared-Q8 blocks whose K and V
projections are both Q4_K: `4,5,7,8,10,11,14,16,17`. Their Q8_1 activation
provider is already shared and their two cooperative consumers already use
the promoted direct-output ABI. This experiment changes only K/V projection
ownership: two consumer launches become one dual-output launch.

## Admission and gates

- Exact shape: two `1024x4096` Q4_K projections in an already-admitted
  shared-Q8 block.
- Arithmetic: retain each consumer's four-warp block partition, staged
  shuffle association, and left-to-right four-partial merge independently.
- ABI: separate caller-owned fp32 K and V outputs from one opaque call. No
  `[2,1024]` slice boundary is allowed.
- Native gate: all 2048 fp32 words bit-identical; no spills; candidate faster
  than the two-consumer CUDA-event control.
- Production gate: identical token stream, exactly nine nodes removed, no
  adapters or completion kernels added.
- Wall gate: reps at least seven, then a stabilized reps=9 re-bracket if
  opening/closing control drift is comparable to the expected effect.
  Promotion requires the candidate to beat both controls.
- Target/rollback: only `NV sm_120`; explicit load-time rollback required.

The isolated saving is a ceiling, not booked tok/s. The route does not reduce
weight bytes and cannot by itself close the 227 gap.
