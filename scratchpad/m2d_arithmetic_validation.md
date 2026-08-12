# M2d census arithmetic validation

Date: 2026-08-11. Source: `/tmp/m2c-ab.json` (booked M2c full AB: smoke,
logits, census, wall bracket; control 715 kernels / 5619.52 us vs candidate
630 kernels / 5487.82 us; candidate 5.2030693125 ms/token vs control A
5.269300875 / control B 5.2820094375, bracket median 5.27565515625),
`docs/task_workflow/input/m5-flash-combine-normalization-measurement-record-20260802.md`
(non-landing M5: net zero because the fp16 copy replaced the absorbed cast
1:1), `docs/task_workflow/input/nv-epilogue-absorption-route-scope-20260810.md`
(ledger; section 1: "remaining in-row: attention cast `0a5eb0ac` x36 (~60 us
census) as the next item"), and the M2c validation doc
(`scratchpad/m2c_arithmetic_validation.md`), whose style this mirrors.

M2d is the booked M2c candidate plus one fold: absorb the standalone
fp32->fp16 attention cast `E_32_32_4_0a5eb0ac` (x36) into the flash combine
store (`flash_fused_gmax_combine_f16_*`, in-kernel
`value.cast(dtypes.float16)`), with the M5 typed-boundary validator
(`_cancel_lossless_fp16_roundtrip` / `_validated_typed_view`) canceling the
fp16->fp32->fp16 consumer roundtrip so no `E_32_32_4_3b0fcfbc` fp16 copy is
materialized. That last point is the entire difference from M5: M5 was net
zero (+copy replaces -cast); M2d is net -36 kernels.

## 1. Headline census fields (M2d on top of the booked M2c candidate)

| metric | control (booked) | candidate (booked M2c) | M2d swap | M2d expected candidate |
| --- | ---: | ---: | ---: | ---: |
| kernels | 715 | 630 | -36 | **594** |
| kernel_us | 5619.52 | 5487.82 | -61.88 | **5425.94 (~5425.9)** |
| attention_cast_count | 36 | 36 | -36 | **0** |
| attention_cast_us | 62.22 | 61.88 | -61.88 | **0.0** |
| legacy combine `flash_fused_gmax_combine_32_128` | 36 | 36 | -36 | 0 |
| fp16 combine `flash_fused_gmax_combine_f16_32_128` | 0 | 0 | +36 | 36 |
| fp16 copy `E_32_32_4_3b0fcfbc` (M5 artifact) | 0 | 0 | 0 (typed view) | 0 |

The M2d swap keeps the combine family count at 36 (legacy name -> fp16 name,
1:1) and removes only the 36 standalone cast kernels, so the census delta is
exactly the `0a5eb0ac` family: -36 kernels, -61.88 us. Every other census
line is byte-identical between arms.

## 2. Family counts (candidate-arm census, the arm being modified)

| family | count | candidate median | census us |
| --- | ---: | ---: | ---: |
| `E_32_32_4_0a5eb0ac` (fp32->fp16 attention cast, M2d target) | 36 | 1.63 us | 61.88 (sum) |
| `flash_fused_gmax_combine_32_128` (legacy fp32 combine, M2d swap) | 36 | 3.39 us | 122.04 (36 x 3.39) |
| `flash_fused_gmax_combine_f16_32_128` (new) | 0 | - | 0 |
| `E_32_32_4_3b0fcfbc` (fp16 copy, seen only in M5's variant) | 0 | - | 0 |

Note: 36 x 1.63 = 58.68 us, but the census `attention_cast_us` is the summed
runtime, 61.88 us; the projection uses the census sum (61.88), not the
median-times-count product. The control arm measured the same family at
62.22 us.

## 3. The replacement expression is bitwise-identical

**Standalone cast.** `E_32_32_4_0a5eb0ac` is a pure elementwise fp32->fp16
RNE cast (m5 record lines 8-10: captured CUDA source
`*((half4*)(data0_4096+alu0)) = make_half4((half)val0.x, ...)`, digest
`0a5eb0ac56c097a089f39541962d5d73b9bc613251a6320685824338d26b38c4`). It sits
between the fp32 combine AFTER (`flash_fused_gmax_combine_32_128`, output
`(Hq*Hd,)` fp32) and the o-proj GEMV. Its operand per element is the combine
store expression `final_acc / final_den` evaluated in fp32
(`flash_decode_attention.py:244`).

**In-kernel store.** The fp16 combine variant
(`flash_decode_attention.py:206-249`) computes the identical expression and
then casts on the store operand:
`value = final_acc[output_axis] / final_den`; `if output_fp16: value =
value.cast(dtypes.float16)` (lines 244-245); `out[head * Hd +
output_dim].store(value)` (line 247). `output_fp16` toggles only the cast;
the score/PV accumulation, the fp32 division, the lane/index math, and the
store address are unchanged from the legacy combine.

**Same operand, same instruction, same rounding -> same bytes.** Both
spellings compile to `cvt.rn.f16.f32` (CUDA `(half)` is RNE) applied to the
same fp32 bit pattern `final_acc/final_den`; IEEE round-to-nearest-even is
deterministic (no rounding mode, FTZ, or denormal state differs between the
two kernel instances, and both handle fp16 subnormal results identically).
The fp16 bytes at the attention boundary are therefore bitwise identical.

**Empirical evidence.** The 2026-08-02 record measured exactly this
replacement: d512 first token, token sha256
`9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` identical
3/3 between baseline and the fp16-combine variant (m5 record lines 46-48),
first-token shape 151936 identical 3/3, tok/s noise-only (+0.04). The record
states it directly (lines 50-52): "the fp16 store cast and the absorbed
generic cast are the same RNE fp32->fp16 conversion of the same fp32 value."
M2d reproduces that M5 replacement and adds the typed-boundary fold that M5
lacked.

## 4. fp16 -> fp32 -> fp16 roundtrip cancellation is exact

**Format arithmetic.** fp16 has 1 sign + 5 exponent + 10 explicit mantissa
bits, i.e. an 11-bit significand (10 explicit + 1 implicit), normal exponent
range 2^-14..2^15 and subnormal floor 2^-24. fp32 has 1 sign + 8 exponent +
23 explicit mantissa bits, a 24-bit significand, and exponent range
2^-126..2^127.

- fp16 -> fp32: every fp16 value's 11-bit significand is exactly representable
  in fp32's 24-bit significand, and the fp16 exponent range [2^-24, 2^15 x
  (2 - 2^-10)] is strictly inside fp32's. The upcast is injective and exact
  for every fp16 bit pattern (normals and subnormals alike): zero rounding.
- fp32 -> fp16 (RNE) of that value: the value is already an fp16 value, so it
  is its own nearest fp16; RNE returns it unchanged. Zero rounding.

Therefore fp16 -> fp32 -> fp16 is the identity on the fp16 set, and the
composed chain is exactly the original fp16 view. `_cancel_lossless_fp16_roundtrip`
(`kernel_program.py:222-238`) structurally enforces this: outer `CAST(fp16)`,
base `CAST(fp32)`, a pure `GroupOp.Movement` leg chain between them, bottoming
at an fp16 `AFTER`; it cancels the pair and recomposes the movement legs
(lines 229-238). Lossy shapes (bf16 round trips), arithmetic between the
casts, or non-identity movement reject (docstring lines 223-228).

**The real consumer chain.** Producer: the fp16 combine declares typed output
layout fp16 `(Hq*Hd,)` row-major, viewable `(Hq, Hd)`
(`flash_decode_attention.py:644-651`). Model: `attn =
out.reshape(B, Hq, T, Hd).cast(q.dtype)` (`model.py:887`); decode q is fp32,
so this is the fp16->fp32 upcast (with B=1, T=1 the reshape is a pure index
view). Consumer prelude: `x[:, 0, :].reshape(K).cast(fp16).contiguous()`
(`decode_routes.py:132`), the fp32->fp16 back-cast. The cancel produces the
fp16 view of the AFTER; `_validated_typed_view` (`kernel_program.py:241-289`)
then verifies the remaining chain is a contiguous offset-0 row-major reshape
(lines 256-261), that the declared layout exactly matches the request fp16
`(K,)` (lines 263-267), that `combine_fusion_admitted` and
`requires_combine_fusion` gates are open (lines 282-284), and that the
consumer is the single intended `attn_qo` Q4K GEMV (lines 285-288). The
attn_qo opt-in request is declared at `decode_routes.py:143-148`, and the
route threads `combine_fp16=binding.combine_fusion` at
`decode_routes.py:525-528`.

**Why `E_32_32_4_3b0fcfbc` becomes impossible.** In M5 there was no typed
view: the fixed-ABI combine output could not be re-laid-out to the consumer's
view, so the generic scheduler materialized the prelude's `contiguous()`
request as a per-input fp16->fp16 copy, 36 x ~1.58 us (m5 record lines 36-40,
33) - net zero kernels/us, which is why M5 did not land. With the typed view
accepted, `_fold_typed_input_views` (`kernel_program.py:292-310`) substitutes
the producer AFTER for the materialized contiguous input, so no `contiguous()`
request reaches the scheduler and no copy program can be emitted. The
validator is fail-closed: any rejection keeps the generic flat-buffer ABI
with its copy (byte-identical to control), which is exactly the M5
"opaque-kernel-boundary copy problem" condition the record named as the
promotion requirement (m5 record lines 58-60). The census requirement
"`3b0fcfbc` count 0" is the assertion that the fold succeeded.

## 5. M2d census swap arithmetic (projected, on the booked M2c candidate)

| family | delta count | delta census us |
| --- | ---: | ---: |
| `E_32_32_4_0a5eb0ac` fp32->fp16 cast | -36 | -61.88 |
| `flash_fused_gmax_combine_32_128` legacy fp32 combine | -36 | replaced 1:1 |
| `flash_fused_gmax_combine_f16_32_128` fp16 combine | +36 | ~0 incremental (in-kernel cast absorbs at ~zero cost) |
| `E_32_32_4_3b0fcfbc` fp16 copy | 0 | 0 (typed view prevents it) |
| **net** | **-36** | **-61.88** |

- kernels: 630 - 36 = **594**
- kernel_us: 5487.82 - 61.88 = **5425.94 (~5425.9)**
- attention_cast_count: 36 - 36 = **0**; attention_cast_us: 61.88 - 61.88 =
  **0.0**
- everything else byte-identical between arms: the combine family stays at 36
  (name/digest changes only), and no other census line moves (the M2c gate's
  "no unrelated program deltas" requirement carries forward).

The "fp16 combine ~= legacy combine" assumption is supported by M5's census:
with the copy present, the whole replacement (cast + legacy combine ->
fp16 combine + copy) netted only -5 us over 1021 kernels (m5 record lines
29-30), i.e. the in-kernel cast is absorbed at ~zero incremental cost; with
the copy gone (typed view), the swap is worth the full -61.88 us.

## 6. Projected wall ledger

Booked M2c bracket: candidate 5.2030693125 ms/token = **192.19 tok/s**
(1000/5.2030693125), control A 5.269300875, control B 5.2820094375, bracket
median 5.27565515625, booked at +72.6 us median vs the bracket (above the +50
us bar). The `0a5eb0ac` family is 61.88 us census, all of it pure kernel
removal (36 cast kernels disappear; no compute body changes - the fp16
combine body is the legacy combine body plus one store-side cast). The ledger
mapping rule (nv-epilogue-absorption-route-scope-20260810.md lines 44-47)
applies at ~1.0 for pure kernel REMOVAL ("for pure kernel REMOVAL expect
~1.0+ (kernel time + launch gap)"); the 0.61 figure is for body-adding
changes and does not apply. Precedent: M2a removed 36 cast kernels at a 0.99
census-to-wall ratio (-57.6 us census, -57.2 us wall).

| scenario | census delta | wall delta | ms/token | tok/s | vs target 5.1546 | vs +50 us bar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M2c booked baseline | - | - | 5.20307 | 192.19 | - | +72.6 us (PROMOTED) |
| M2d (x1.0 removal) | -61.88 us | -61.88 us | 5.14119 | **194.51** | crosses by 13.4 us | +134.5 us (MEETS) |
| M2d pessimistic (x0.61) | -61.88 us | -37.75 us | 5.16532 | 193.60 | below by 10.7 us | +110.3 us (MEETS) |

Projected wall (rounded as in the task): 5.2031 - 0.062 = **~5.141 ms/token
= ~194.5 tok/s**, crossing the ~194 tok/s target (1000/194 = 5.154639 ms) by
~13.4 us and clearing the +50 us booking bar by ~134.5 us vs the control
bracket median. Even at the pessimistic 0.61 mapping the booking bar is met
(+110.3 us), though the 194 tok/s target would be missed by ~10.7 us; the
x1.0 removal mapping is the ledger-justified projection.

## 7. Verification gates

- **AB exact-logits SHA gate (empirical bitwise check).** Full-logits fp32
  SHA-256 must be identical between arms. The booked M2c gate value is
  `70838f5237ce2cf215e937caed807c4827daa1336e69c3d6c396b1aad4434819`, shape
  [32, 1, 151936], `logits_sha256_equal` / `tokens_equal` / `gate_pass` all
  true (`/tmp/m2c-ab.json` `logits_gate`). M2d must reproduce this exact SHA:
  any byte change anywhere - a rounding-mode deviation, an fp32 promotion at
  the boundary, a dropped/reordered element, or a failed fold that changes
  what the o-proj GEMV reads - fails closed.
- **Census gate (exact swap table).** Asserts: `0a5eb0ac` 36 -> 0,
  `flash_fused_gmax_combine_32_128` 36 -> 0, `flash_fused_gmax_combine_f16_32_128`
  0 -> 36, `3b0fcfbc` 0 -> 0 (the typed view must hold - a re-materialized
  copy shows up as +36 and fails the gate), net -36 kernels (630 -> 594),
  kernel_us 5487.82 -> 5425.94, attention_cast_count/us -> 0 / 0.0, and no
  unrelated program deltas.

## Summary

- M2d absorbs `E_32_32_4_0a5eb0ac` (36 fp32->fp16 RNE casts, 61.88 us) into
  the flash combine store: the in-kernel `value.cast(fp16)` applies
  `cvt.rn.f16.f32` to the same fp32 `final_acc/final_den` operand, so the
  fp16 bytes are bitwise identical (2026-08-02 record: token sha256
  `9d6b3787...` identical 3/3).
- fp16 -> fp32 -> fp16 is the identity on the fp16 set (11-bit vs 24-bit
  significand, both upcast and back-cast exact), so the validator's cancel of
  the model.py:887 upcast + prelude back-cast yields an identical fp16 view of
  the AFTER; the accepted typed view makes the M5 `3b0fcfbc` fp16 copy
  impossible, which is what turns M5's net-zero into M2d's net -36.
- Census swap: -36 casts (-61.88 us), -36 legacy combines, +36 fp16 combines,
  net -36 kernels -> 630 - 36 = 594; kernel_us 5487.82 - 61.88 = 5425.94
  (~5425.9); attention_cast_count/us 0 / 0.0; everything else byte-identical.
- Projected wall at the ledger's ~1.0 pure-removal mapping: 5.2031 - 0.062 =
  ~5.141 ms/token = ~194.5 tok/s, crossing the ~194 tok/s target (5.1546 ms)
  by ~13.4 us and clearing the +50 us booking bar by ~134.5 us.
- Gates: the AB exact-logits SHA (`70838f52...4434819`, [32,1,151936]) is the
  empirical bitwise check; the census gate asserts the exact swap table
  (including `3b0fcfbc` staying at 0).
