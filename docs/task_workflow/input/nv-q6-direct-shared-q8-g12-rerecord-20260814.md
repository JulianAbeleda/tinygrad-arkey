# NV Q6 direct shared-Q8 g12 re-bracket (llama-MMVQ fix)

Status: **WALL_PASS (negative); default-off consumer stays closed until the
full promotion gate is run.**

Date: 2026-08-14
HEAD: `f31828e93` (branch `nvidia-bringup-20260731`)

The prior g12 re-bracket on 2026-08-09 was `WALL_NO_GO` at
`+27.281703125 us/token`.  That record used the flat four-warp Q6 consumer,
which decoded one Q6 byte at a time.  The missing substrate was not missing:
the vectorized llama MMVQ mapping was already representable in the UOp
layer, and the first implementation of it had a scale-index bug.  For the
two packed int8x4 terms llama reads `scales[scale_offset]` and
`scales[scale_offset + 4]`; the microgate reused `scales[scale_offset]` for
both.  Correcting that single offset produces bitwise-compatible attention-V
logits and a faster kernel.

## Primitive evidence

The corrected Q6 consumer is the llama `vec_dot_q6_K_q8_1_impl_mmvq`
geometry: one row per CTA, 128 threads (4 warps x 32 lanes), interleaved
Q6 blocks (`block = warp + 4*block_rel`), vectorized `ql`/`qh` uint32
loads, and two `__dp4a` terms.  `DEBUG=4` kernel census on the native
RTX 5090 showed the consumer at about `3.0 us` for 1024x4096, versus about
`57.7 us` for the established `q6k_gen_partial_1024_4096_4` control kernel
in the same process.

`extra/llm_research/decode/q6k_q8_llama_mmvq_microgate.py` is the
research-only microgate.  Its graph-replay timing field remains host-bound;
the `DEBUG=4` profiler census is the kernel-level authority.

## Settled g12 reverse bracket

All arms used `d512`, 32-token uninterrupted windows, five repetitions, two
feedback captures (composed ping-pong), the cooperative-Q4 g12 shared route,
and identical stream hashes.  Flags:

```text
--mode fused-timing --fused-groups 12 --cooperative-q4 --q6-direct-output
--composed --settled-continuous --depth 512 --count 32 --max-context 1024
--groups 0 --reps 5
```

| arm | ms/token |
| --- | ---: |
| control A | 5.0805794375 |
| direct-Q6 candidate B | 5.03689309375 |
| control A2 | 5.07130884375 |
| control midpoint | 5.075944140625 |
| B minus midpoint | **-0.039051046875 ms/token** |

The fresh wall is **-39.051046875 us/token** versus the control midpoint.
This is a real-token included-cost reverse bracket, so the Q6 direct-output
consumer is faster in this subset, not just smaller in a microgate.

## Semantic gate

A separate composed child compared g12 cooperative-Q4 control versus g12
cooperative-Q4 + Q6 direct-output on 8 decoded tokens:

```text
finite true
tokens_equal true
argmax_equal true
top_k_sets_equal true
top_k_order_equal true
relative_l2 0.0005006654530624455
max_abs 0.010866641998291016
perturbation_vs_min_margin 0.017828113485341815
```

All contract fields pass.  This does not yet mean the closed-default
`q6_direct_output` lease is promoted; that still requires the full promotion
protocol from `nv-gemv-substrate-landing-scope-20260808.md`, including the
all-depth wall, census, and record-policy equality checks.

## Raw artifacts

- `/tmp/nv-q6-direct-20260814-bracket.json` (aggregate reverse-bracket result)
- `/tmp/nv-q6-direct-20260814-census.json` and
  `/tmp/nv-q6-direct-20260814-census.npz` (candidate full logits)
- `/tmp/nv-q6-direct-20260814-control.json` and
  `/tmp/nv-q6-direct-20260814-control.npz` (control full logits)
