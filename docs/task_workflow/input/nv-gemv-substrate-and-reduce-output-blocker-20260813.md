# NV GEMV substrate status + reduce-output body-free blocker (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `492313dab`)
Status: **diagnosis record. Read-only: no runtime change, no GPU arm, no lock.
Answers two questions grounded in the committed artifacts: (1) is the GEMV
core substrate missing, and (2) why does the reduce-output body-free admission
still reject on CPU.**

## 1. GEMV substrate is landed, not missing

llama's live decode GEMV is one route: MMVQ = Q8_1 activation + `__dp4a` +
one output row per CTA + four warps per CTA, zero dynamic smem
(`nv-decode-llama-live-gemv-route-audit-20260805.md`). That package is cloned
and promoted as the shared-Q8 DP4A four-warp lease
(`nv-gemv-substrate-landing-scope-20260808.md`), which is why the large shapes
are already at parity or better (gate/up 1.02x, Q/O <=1.0x, vocab 1.09x).

The remaining GEMV deficits are measured NO-GOs, not capability gaps:

| shape | ratio vs llama | status |
| --- | ---: | --- |
| Q4 FFN-down 4096x12288 | 2.27x | load-pattern sweep NO-GO 08-12 (same SASS as gate/up does not clear the down-geometry floor) |
| Q6 FFN-down 4096x12288 | 1.22x | MC2 coop NO-GO (installed kernel is a local optimum) |
| Q6 attention-V 1024x4096 | 3.65x | L2/MC2 partial NO-GO (local optimum) |
| Q4 attention-K remainder | 1.47x | small tail; shared-Q8 already on 26/54 |

What is still missing is a different substrate: the M3 generic topology-plan
executor plus generic primitive lowerers (`q4k_packed_block_dot`,
`q6k_packed_block_dot`, `external_reduce`), blocked in
`extra/llm_research/decode/m3_machine_search/m3_search_export.json`. That
unblocks *exploration*, not tok/s today: the closed rows above were already
swept by hand to local optima.

## 2. Why q/k markers stay `marker_not_eligible` on CPU

The generic cooperative reduce-output primitive (C1) is shipped. The CPU
census of the forced reduce-output graph
(`scratchpad/nv_reduce_output_rmsnorm_census.py`) reports 36 fused
`reduce_output_rmsnorm_1_4096` bodies and zero `*_weight_store*`, but also
`marker_not_eligible` for the q/k site.

Root cause, traced in the artifacts:

- The q/k marker input is `PERMUTE(CAST(...))`
  (`/tmp/nv_m1_promoted_20260813.json` `reduce_output.marker_input_histogram`:
  72 `[1,32,1,128]` + 72 `[1,8,1,128]`, `op=PERMUTE`, `base_op=CAST`).
- At marker creation, `Tensor._semantic_reduce_output_rmsnorm`
  (`tinygrad/tensor.py`) walks identity through only
  `RESHAPE / MEMORY_SEMANTIC / PERMUTE`. The walk stops at `CAST` (a compute
  op, no buffer/precompiled identity), so `input_identity_at_marker`,
  `owned_contiguous_candidate`, and `reduce_input_at_marker` are all False.
- At lowering, `lower_reduce_output_store` (`tinygrad/schedule/rangeify.py:604`)
  rejects with `marker_not_eligible`, and the marker falls back to the ordinary
  two-program reduce + epilogue pair.

This is the exact mechanism. The trace selector counts are 13 entries / 3
accepted / 10 `marker_not_eligible`, split as `16x32x8` (ffn-norm) 7/3/4 and
`32x32x4` (q) + `8x32x4` (k) 3/0/3 each. The 36-body program count comes from
the `1_4096` family; the trace's per-entry counters do not 1:1 map to emitted
bodies because the census captures multiple graphs and resets the trace
between arms.

## 3. The body-free "before" arm is uncapturable on CPU

The clean body-free delta is `promoted` vs `baseline-context` (callify
substrate held constant, reduce-output off). That arm raises
`flash_decode_attention_route: shape B=1 Hd=128 Hkv=8 Hq=32 is not served by
the generated live-split route` on `DEV=CPU`, even though the shape matches the
live-split structural class (`B==1, Hd==128, Hkv==8, Hq%Hkv==0`). The admission
fails because the current production flash graph transitively requires the
reduce-output route: `_decode_reduce_output_rmsnorm_promoted` is a single
all-sites flag, so "reduce-output off" also turns off the promoted fp32 q/k
site the live-split flash route now depends on.

Consequence: the `baseline` (no callify, no reduce-output, 261 programs) vs
`promoted` (734 programs) delta of +473 conflates the landed callify substrate
with reduce-output admission and is not a body-free removal verdict.

## 4. What this implies for the unblock

The reduce-output admission is construction-passing (36 `1_4096` bodies, zero
weight materializations) but not yet proven body-free on CPU. To close that
gap the substrate needs a **per-site admission knob**: hold the promoted fp32
q/k site open while the M1 ffn-norm site is independently gated, so the census
can measure the M1 chain's own `r_16_256 + E_32_32_4_f14a5cc0 -> 1_4096`
removal (target net -36) without collapsing the flash route. That is the next
CPU-only implementation step; no NV arm until it passes.

## References

- `nv-decode-gap-attribution-same-session-20260812.md` (ladder: ~193 needs -22 us;
  reduce-output epilogue 392 us -> ~201; full parity 245 needs flash + launch hiding)
- `nv-quant-gemv-llama-audit-20260812.md`,
  `nv-q4kd-load-pattern-measurement-record-20260812.md` (GEMV NO-GOs)
- `nv-m1-norm-epilogue-generic-primitive-scope-20260812.md` (body-free contract)
- `nv-reduce-output-site-absorption-scope-20260812.md` (per-site admission)
- `nv-boundary-free-ordinary-uop-gate-v4-reopen-record-20260813.md` (REDUCE_OUTPUT arm)
