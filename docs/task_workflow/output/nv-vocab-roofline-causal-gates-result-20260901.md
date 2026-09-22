# NV vocabulary roofline causal gates

Date: 2026-09-01

## Goal

Recover the measured `17.338 us` gap between the installed Q6_K vocabulary
body (`317.402 us`) and the same-machine streaming-read roof (`300.064 us`).
The retained llama body is `300.930 us`.

## Results

| candidate | full-shape time | paired delta | disposition |
| --- | ---: | ---: | --- |
| installed one-warp/two-row FP16 control | 321.636 us in the causal session | - | control |
| Q8_1 four-warp lane-stage | 1373.903 us | +1047.953 us | stop |
| FP16 four-warp direct | **309.974 us** | **-11.546 us** | isolated pass |

The Q8 result rejects the current generated Q8 unpack emitter, not llama's
Q8 representation: the exact llama cubin remains at `300.930 us`.  The
generated Q8 body emits excessive scalar unpack/address work and cannot
service the 510 MB weight stream near the roof.

The FP16 result changes only row ownership and reduction geometry:

```text
control:   one 32-thread warp computes two rows
candidate: four 32-thread warps compute one row, direct FP32 logit output
```

It uses the canonical packed Q6_K weight, consumes FP16 directly, adds no
provider, performs no weight expansion, and recovers 66.6% of the practical
roofline gap in the isolated full-shape gate.  Its remaining distance is
`9.910 us` to the measured roof and `9.044 us` to llama.

## Numerical gate

The existing nonzero Q6 reference gate passed:

```text
candidate max abs vs dequantized Q6 reference: 1.9073486e-5
candidate vs installed control max abs:        2.9563904e-5
gate tolerance:                                2.0e-2
```

This route retains FP16 activation semantics and does not introduce Q8
approximation.

## Endpoint causal bracket

Fresh-process control/candidate/control, depth 512, five windows of 12 tokens
per arm, all llama pp512 oracle bindings disabled identically:

| arm | ms/token |
| --- | ---: |
| control A | 4.055613 |
| FP16 four-warp candidate | **4.051534** |
| control C | 4.062506 |
| control midpoint | 4.059059 |

```text
candidate recovery: 7.525 us/token
endpoint speedup:    0.1857%
token hashes equal:  yes
verdict:             WALL_PASS
```

The initial wall transfer was smaller than the isolated `11.546 us` recovery
but had the same sign and cleared correctness.

The larger frozen d512 R9x24 reverse bracket subsequently passed:

```text
control A:          4.067713 ms/token
candidate:          4.035563 ms/token
control C:          4.060289 ms/token
control midpoint:   4.064001 ms/token
candidate recovery: 28.438 us/token
endpoint speedup:    0.7047%
accepted windows:    27/27
token hashes equal:  yes
verdict:             PROMOTION_PASS
```

## Decision

The first roofline fix is promoted: the generated four-warp FP16 Q6 primitive
is bound to the exact `M=1, N=151936, K=4096` vocabulary role by the dedicated
Boltbeam authority `decode_q6k_vocab_four_warp_fp16`.  The explicit rollback is
`TINYGRAD_Q6K_VOCAB_FOUR_WARP_DISABLE=1`.  Do not promote the current Q8
full-shape emitter.  The next counter gate targets the remaining roughly
`9-10 us` isolated body distance.

Evidence directory:
`docs/task_workflow/evidence/nv-vocab-roofline-causal-gates-20260901/`
