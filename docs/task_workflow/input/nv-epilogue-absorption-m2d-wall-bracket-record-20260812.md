# NV epilogue absorption M2d wall bracket record (all gates PASS, promotion NO-GO)

Status: **M2d LANDED with every correctness gate PASS, but the reverse wall
bracket does NOT promote. The fp16 flash-combine store (absorbing the
`E_32_32_4_0a5eb0ac` attention cast x36) is bit-exact: the exact full-logit
gate passes byte-identically to control and the census gate passes
(attention cast 36 -> 0, fp32 combine 36 -> 0 swapped 1:1 with the fp16
combine 0 -> 36, no opaque `3b0fcfbc` copy, net -36 kernels, no unrelated
program shift). The wall bracket measured candidate 5.1717 ms/token =
193.36 tok/s at +34.9 us (vs control A 5.2066 ms) / +36.6 us (vs control B
5.2083 ms), median +35.8 us/token — BELOW the +50 us promotion bar, so the
residual row's final item is LANDED but NOT BOOKED (same verdict shape as
M2b). The residual/cast/contiguous row is now exhausted: M2a -57.2 us
(booked), M2b+M2c -72.6 us (booked), M2d -35.8 us wall (landed).**

Scope: `docs/task_workflow/input/nv-epilogue-absorption-route-scope-20260810.md`.
Branch `nvidia-bringup-20260731`. Harness
`extra/llm_research/decode/nv_epilogue_absorption_ab.py`, run 2026-08-12 on
the RTX 5090 (sm_120, Qwen3-8B-Q4_K_M at fixed depth 512, count 32, 5 reps)
under the shared GPU bench lock with a fresh process per arm. Raw record:
`docs/task_workflow/output/nv-epilogue-absorption-m2d-wall-bracket-20260812.json`.

## Why M2d was wedged, and the fix

The candidate initially diverged from control at decode block 1 with
logits `d2c3e01f` vs control `6ef9191d`. Bisection (exec-order kernel
snapshots, both arms, fresh processes) traced the first divergence to the
kv cache at position 512 (378 bytes in fp16 low bits, near-equal values —
not a shifted write): the JIT=0 eager baseline forward at `depth` writes
position 512 via the legacy store, and in the candidate arm the fp16
combine lease rounded that pass's attention output. The eager graph is a
FRESH graph that cancels the fp32->fp16->fp32 attention roundtrip (full
fp32 contract), so the fp16 combine is a lossy rounding there; on the
captured decode graph the fp32->fp16 cast is materialized, which is where
the fp16 combine is bit-exact (verified: decode combine-f16 output ==
control cast output per layer once the cache is arm-invariant).

Fix (`6a22bc258`): `_without_flash_combine_fp16` clears the M2d lease for
the eager JIT=0 baseline in BOTH arms, making the cache at `depth`
arm-invariant; the exact-logits gate then measures exactly the decode swap.
This is a harness contract fix, not a policy change: the lease stays
closed by default and the model route is untouched.

## Gate evidence

| gate | control | candidate | result |
| --- | --- | --- | --- |
| NV render smoke | - | survive, fp16 combine present, no cast/copy/add | PASS |
| exact logits (32 rows, fp32 sha256) | `70838f5237ce2cf2` | `70838f5237ce2cf2` | PASS |
| census | 630 kernels, cast 36, combine f32 36 | 594 kernels, cast 0, combine f16 36 | PASS |
| wall bracket | A 5.2066 ms / B 5.2083 ms (median 5.2075) | 5.1717 ms | NO-GO (+35.8 us < +50 us) |

Census detail: control kernel_us 5534.02 (attention cast 60.98 us +
combine f32 123.36 us); candidate kernel_us 5489.50 (combine f16 120.54
us). Net program delta -36 == the attention-cast drop; population delta
only `residual_cast_contiguous` -36; token streams identical within and
between arms (`5ede6924...`).

## Ledger translation

193.36 tok/s candidate vs 192.03 tok/s control bracket: M2d books ~+1.3
tok/s of wall speed with bit-exact logits, short of the +50 us promotion
bar. Remaining path to ~194 tok/s (5.1546 ms/token): the norm-epilogue
row (M1) and then the quant GEMV row (~4050 us census; a 10% win lands
~203 tok/s).

## Decision

M2d stays LANDED (correctness gates are the booking authority for the
route; the +50 us bar is wall evidence only). The candidate conditions
(`_flash_combine_fp16_lease` harness-installed) are safe to carry forward:
fail-closed absent by default, bit-exact under the decode contract, and
the eager baseline is now arm-invariant by harness construction.
