# NVIDIA compiler-native packed projection gate result

## Decision

**GATE A PASS. DO NOT PROMOTE YET.**

The ordinary tinygrad matmul path can now compile a complete canonical
Q4_K x Q8_1 NVIDIA projection. The compiler owns packed Q4 codes, the compact
Q8 activation record, cooperative shared-memory staging, signed-int8 IMMA,
the K32 affine correction, and the final FP32 reduction. It does not allocate
an expanded global weight or a global `[K/32,M,N]` partial tensor.

At the required real shape `(M,N,K)=(512,12288,4096)`, tile K64 passes the full
6,291,456-element static-v4 oracle and the performance threshold:

| check | result |
|---|---:|
| finite / written / nonzero outputs | 6,291,456 / 6,291,456 / 6,291,456 |
| maximum absolute difference | 0.000213623046875 |
| mean absolute difference | 0.00000806683692644583 |
| full-output tolerance | `rtol=2e-5, atol=2e-3`: PASS |
| packed weight and compact Q8 record read-only | PASS |
| compiler program / candidate identity | one program / exact |
| generated minimum / median | 465.165 / 470.825 us |
| qualified v4 main+fixup minimum / median | 464.128 / 464.928 us |
| generated / v4 minimum | 1.0022 |

That is inside the required 3% threshold and effectively primitive parity.
An independent reverse run also landed at parity; the retained table above is
the canonical machine-readable run in `production_gate_staged_k64.json`.

## Compiler contract

The path is deliberately typed rather than an arbitrary ALU-DAG substitution:

1. `Q4KInt8FragmentProvider` maps logical `(N,K)` coordinates to signed-int8
   carriers containing Q4 codes `0..15`. Logical K32 ownership selects the
   low/high nibble before tensor-core range permutation. Both Q4 and Q8
   providers fail closed when `max(k_base % 32) + width > 32`, including for
   symbolic bases, so one fragment can never borrow metadata from two groups.
2. `Q8ActivationRecordTransform` owns one compact uint32 buffer containing
   row-major int8 values, FP32 K32 scales, and FP32 raw sums.
3. `Q4KQ8GroupAccumulatorContract` consumes one K32 int32 IMMA result and
   applies the proven half-roundpoint correction:

   ```text
   half(D*scale) * half(q8_scale) * int32_dot
     + half(-Dmin*minimum) * half(raw_sum)
   ```

4. Corrected K32 contributions enter the existing outer FP32 `Ops.REDUCE`.
   No global group-partial materialization is needed.

The bounded `(32,16,256)` gate is exact (`max_abs=0`) for tile K32, K64, and
K128. Hermetic tests pin Q8 signed-byte recovery, metadata offsets/group
ownership, the exact half2 formula, the K64 group0/group1 packet selection,
and aligned-versus-crossing K64 behavior through the actual staged builder.

## Why performance closed

Correct arithmetic alone initially took about 750 us. SASS localized the tax:
the K32 loop issued 67 global loads because Q4 and Q8 metadata were reloaded per
output accumulator. The typed producer now stages one half2 metadata packet per
cooperative vector owner and reuses it across the output tile.

| arm | min time | registers | shared | local traffic | static loop facts |
|---|---:|---:|---:|---:|---|
| K32, direct global metadata | 749.410 us | 234 | 9,216 B | none | 16 IMMA, 67 LDG, 24 LDS, 2 BAR |
| K32, staged metadata | 585.011 us | 170 | 13,312 B | none | 16 IMMA, 13 LDG, 40 LDS, 2 BAR |
| K64, staged metadata | 465.165 us | 222 | 21,504 B | none | 32 IMMA, 26 LDG, 80 LDS, 2 BAR |
| K128, staged metadata | 745.281 us | spill cliff | larger | 110 LDL / 174 STL | 64 IMMA, 2 BAR |

K64 is the clean point: it halves barrier frequency relative to K32 without
crossing the K128 register/spill cliff. K256 is rejected by NVRTC because the
value tile alone exceeds the target's admissible shared-memory budget.

The current generated SASS uses scalar `LDS`, not `LDSM`, but that is no longer
a Gate-A wall: K64 already matches the qualified v4 lifecycle.

## 72-real-weight proxy

The captured proxy contains exactly 72 compiler calls over the real 36-layer
gate/up packed-weight population. Replay is exact, representative outputs from
three real weights are distinct and finite, and no per-projection partial/fixup
workspace exists.

| arm | minimum captured wall | per call |
|---|---:|---:|
| compiler-generated K64 | 19.438525 ms | 269.980 us |
| matched v4 native main+fixup | 37.003343 ms | 513.935 us |

This proxy proves non-regression, not an end-to-end prefill claim. It removes
v4's partial/fixup lifecycle and amortizes synchronization over one captured
72-call graph; whole-model routing and the Q8 producer still belong to Gate B.

## Status

| requirement | status |
|---|---|
| typed logical Q4_K fragment / nibble parity | PASS |
| compact typed Q8 values, scales, raw sums | PASS |
| K32 correction before outer FP32 reduction | PASS |
| no expanded global weight or global group partial | PASS |
| full real-shape oracle / finite / sentinel / read-only | PASS |
| signed-int8 IMMA and exact capture identity | PASS |
| isolated within 3% of qualified v4 | PASS (1.0022x) |
| 72-real-weight proxy non-regression | PASS |
| production/default route | unchanged; Gate B still required |

No production route or default is enabled by this work. The next step is Gate
B: bind a compiler-owned Q8 producer from a computed norm output, capture the
72 gate/up chains with correct buffer ownership, and qualify the synchronized
whole-model wall and logits before any promotion.
