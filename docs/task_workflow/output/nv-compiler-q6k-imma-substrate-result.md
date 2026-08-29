# NVIDIA compiler-native Q6_K IMMA substrate result

## Decision

**PRIMITIVE PASS. REAL V PASS. REAL FFN-DOWN PASS. NOT MODEL-INTEGRATED OR PROMOTED.**

Subsequent default-off lifecycle testing has now been completed. Ownership and
correctness pass, but the combined V/down route is slower than the restored
70 ms-class gate/up+K control and is rejected for promotion. See
`docs/task_workflow/output/nv-compiler-q6k-model-lifecycle-result.md`. The
primitive result below remains valid; it must not be read as token recovery.

The missing compiler-native Q6_K/compact-Q8 substrate is now expressible and
qualified without an expanded global weight, a global group-partial tensor, or
a fixup kernel.  It is a normal compiler-owned packed matmul candidate.

Q6_K cannot reuse the Q4 K32 accumulator.  Q6 has one independently signed
scale per K16, while NVIDIA signed-int8 IMMA reduces K32.  Summing the K32
integer dot and applying either scale is invalid.  The admitted contract emits
two IMMA operations for each logical K32 step:

```text
low IMMA:  Q8[K0:32] dot Q6[K0:16, zero K16:32]
high IMMA: Q8[K0:32] dot Q6[zero K0:16, K16:32]

result = q8_scale * (round(D*q6_scale0)*low_dot
                   + round(D*q6_scale1)*high_dot)
```

The masks are applied while logical K is still explicit, before `CONTRACT`.
Masking the final fragment carrier was tested and rejected because later
output-axis unrolling permutes that carrier.  Typed Q6 metadata stages the two
rounded K16 coefficients as one half2 packet.

## Qualification

| fixture | outputs | CTAs | max abs | R9 minimum | R9 median | registers | shared | spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial K256 | 1,024 | 1 | 0.046875 | 3.936 us | 3.936 us | 96 | 6,144 B | 0 |
| real attention V `(512,1024,4096)` | 524,288 | 256 | 0.000305176 | 69.824 us | 70.464 us | 96 | 8,704 B | 0 |
| real FFN down `(512,4096,12288)` | 2,097,152 | 1,024 | 0.00122070 | 638.752 us | 642.592 us | 96 | 8,704 B | 0 |

Every fixture passes `rtol=2e-5, atol=2e-3`, finite/nonzero/full-write
sentinels, exact candidate identity, packed-weight and compact-record read-only
checks, signed-IMMA source/SASS checks, and zero local loads/stores.  The real
fixtures use block-0 canonical Q6_K weights from Qwen3-8B-Q4_K_M.

The adversarial fixture independently varies all Q6 low/high bit planes, all
16 signed K16 scales (including `-128` and `127`), block D, activation scales,
and low/high integer dots.  It therefore catches bit-unpack, sign, scale alias,
and K16-half swap errors before either real fixture runs.

## Compiler assets added

- `Q6KInt8FragmentProvider`: canonical uint16 Q6 storage to signed-char K16
  fragments, with format and boundary guards.
- `Q6KQ8SubgroupAccumulatorContract`: requires two independent integer
  subtotals; it has no single-K32-dot entry point.
- Q6-aware LDS metadata staging and logical-K-masked low/high B fragments.
- Q6-aware postrange paired-IMMA emission, guarded by the exact NVIDIA
  `m16n8k32 s8 -> s32` descriptor.
- Multi-packed-dtype warmstart discrimination.  A Q6 candidate sees uint16
  weight storage and the uint32 compact-Q8 record, so both dtypes must survive
  in the compiler key.

Q4 remains isolated by provider and accumulator type.  Its real attention-K
R9 regression gate still passes after these additions.

## Scope boundary

This result proves the isolated compiler substrate.  It does not select a
model route, install V/down bindings, claim a whole-prefill recovery, or claim
that the current geometry is performance-optimal.  Model integration belongs
behind separate default-off lifecycle, repeated-replay, full-logit, census,
and wall gates.

The concurrent scalar/vector LDS expectation has since been reconciled by the
K-lane work. The final focused packed/precontract/Q6/int8 suite passes 59/59.
The historical int8-WMMA wall test now asserts the intended split: generic
descriptor admission remains closed, while an exact typed candidate may use
the qualified renderer lowering.

Primary evidence is indexed in
`docs/task_workflow/evidence/nv-compiler-q6k-imma-20260828/README.md`.
