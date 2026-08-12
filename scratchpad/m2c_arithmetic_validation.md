# M2c census arithmetic validation

Date: 2026-08-11. Source: `/tmp/m2b-ab-final.json` (authoritative M2b full AB,
smoke/logits/census/wall; candidate 679 kernels / 5631.68 us vs control 715
kernels / 5666.99 us), `/tmp/m2b-ab.children/*` (earlier census run, still
carrying the `86a23e1a` transport copies), ledger
`nv-epilogue-absorption-route-scope-20260810.md`, and
`nv_residual_epilogue_bitwise_probe.py` (re-run for this doc: `cast_bitwise
true`, `add_bitwise true`, `max_abs_cast 0.0`, `max_abs_add 0.0`).

## 1. Headline census fields (final AB)

| metric | control | candidate | delta |
| --- | ---: | ---: | ---: |
| kernels | 715 | 679 | -36 |
| kernel_us | 5666.99 | 5631.68 | -35.31 |
| residual_cast_contiguous population | 122 | 86 | -36 |
| ffn_residual_add_count | 36 | 0 | -36 |
| ffn_residual_add_us | 64.67 | 0 | -64.67 |
| ffn_down_resadd_count | 0 | 36 | +36 |
| ffn_down_resadd_us | 0 | 1129.79 | +1129.79 |

Note on the two census artifact sets: `/tmp/m2b-ab.children/candidate-census.json`
reports 751 kernels / 5732.57 us because it still contains the
`E_32_32_4_86a23e1a` fp32 transport copies (72 x 1.5 us). The final AB
(`/tmp/m2b-ab-final.json`) was re-run after the attn_qo typed-output
declaration closed that M2c-row sub-item: zero `86a23e1a`, 679 kernels /
5631.68 us. The final AB is the M2c baseline.

## 2. Family counts and histogram medians

| program family | control | candidate |
| --- | ---: | ---: |
| `E_32_32_4_fab82d40...` (fp32 block-output copy, M2c target) | 49 @ med 1.66 us | 49 @ med 1.76 us |
| `E_32_32_4_0a5eb0ac...` (attention residual cast fp32->fp16, M2c target) | 36 @ med 1.66 us | 36 @ med 1.66 us |
| `E_32_32_4_f14a5cc0...` (ffn norm epilogue, M1 row) | 37 @ med 2.3 us | 37 @ med 2.3 us |
| `E_32_32_4_02a9738c...` (fp32 h+ffn_out add, control-only) | 36 @ med 1.825 us | 0 |
| `q4k_g3_lanemap_gemv_4096_12288` -> `q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` | 18 @ med 26.32 us | 18 @ med 26.8 us |
| `q6k_gen_coop_4096_12288_inkernel` -> `q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd` | 18 @ med 34.94 us | 18 @ med 35.23 us |
| `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` | 36 @ med 38.96 us | 36 @ med 38.85 us |

The swapped families appear 1:1 with their plain twins (18 q4k + 18 q6k) and
`w1w3fused16` is unchanged at 36, so the -36 kernel delta reconciles exactly
with the add -> in-kernel epilogue swap and the copy/cast rows being unchanged
between arms. M2c does not touch any of these families.

## 3. M2c target arithmetic (pure kernel removals)

Measured medians from the candidate census (the arm being modified):

| family | count | candidate median | census us |
| --- | ---: | ---: | ---: |
| `fab82d40` fp32 copy | 49 | 1.76 | 86.24 |
| `0a5eb0ac` fp32->fp16 cast | 36 | 1.66 | 59.76 |
| total | 85 | - | **146.00** |

- Net census delta: **-146.00 us** (kernel removal, candidate-arm medians);
  kernels 679 -> **594** (-85).
- Using control-arm medians instead (fab82d40 1.66): -141.10 us. Using the
  ledger's scope figure for fab82d40 (49 x 1.79 = 87.7 us, per the ledger
  anatomy): -147.46 us. All variants land within ~1.5 us of each other; the
  ~87.7 us figure in the ledger is the 1.79 us estimate, the artifact medians
  are 1.66 (control) / 1.76 (candidate).
- These are pure kernel removals (no GEMV body changes), so the ledger mapping
  applies at ~1.0 census us -> wall us (kernel time + launch gap). The booked
  M2a precedent measured 0.99 (36 casts, -57.6 us census, -57.2 us wall).
- Projected wall: 5.2516235625 ms - 0.146 ms = **5.10562 ms/token**, candidate
  census 5631.68 - 146.00 = **5485.68 us**.

## 4. Bitwise safety analysis

The probe (`nv_residual_epilogue_bitwise_probe.py`) is a CPU hermetic check of
the residual-family in-core spellings. Re-run output: `cast_bitwise: true`,
`add_bitwise: true`, `max_abs_cast: 0.0`, `max_abs_add: 0.0`; the fp32->fp16
cast and the fp16 residual add reproduce the ordinary kernels' values bitwise.
Its docstring labels are stale (it names `fab82d40` as "the fp16 residual add"
and groups `0a5eb0ac` with the M2a-absorbed `E_128_32_3`); the ledger corrected
the anatomy: `E_128_32_3` is the ffn_activation_cast (absorbed by M2a),
`02a9738c` is the fp32 `h + ffn_out` add, and `fab82d40` is a pure fp32 copy.
The probe's math is family-independent, which is what matters here.

**Why `fab82d40` (fp32->fp32 copy) is bitwise-transparent to eliminate.** The
family is an identity copy: `out[i] = in[i]` in fp32 with no arithmetic, no
dtype change, and no rounding. The ledger root cause is explicit: "the cast
folds, the COPY does not" - the consumer prelude
`x[:, 0, :].reshape(K).cast(fp16).contiguous()` cannot fold `contiguous()`
over the opaque program CALL/AFTER boundary, so the M5 flat-buffer ABI leaves a
materializing copy kernel behind. Eliminating it is a boundary fold, not a
dtype change: the consumer reads the same fp32 bytes from the same producer
buffer, so every downstream value is byte-identical. Correctness contract:
the fp32 block output feeding the residual slot must remain byte-identical to
control (the attention resadd epilogue reads the fp32 residual as fp32). The
exact-logits SHA gate is the tripwire: any fold that changes values - an
accidental fp16 store, dropped/duplicated elements, a wrong reshape, or a
reordering - fails the stacked-rows fp32 SHA-256 and fails closed.

**Why `0a5eb0ac` (fp32->fp16 attention cast) is NOT freely eliminable.** The
cast is a lossy `cvt.rn.f16.f32` (round-to-nearest-even, 10-bit mantissa): it
is not an identity, so it cannot simply be dropped. The M2b premise correction
is the constraint: the residual slot stays fp32 for the attention resadd
epilogue, and the next block's attention consumes the fp32 block output, so
storing the residual fp16 would round it and break exact logits. The cast can
only vanish by fusing the identical cast into the consumer epilogue, so the
fp16 bytes produced in-kernel are bitwise-identical to the standalone kernel's
output (the probe's `cast_bitwise: true` spelling). Correctness contract: the
fp16 bytes at the attention boundary must be bitwise-identical to control AND
the fp32 residual slot must be preserved. The exact-logits SHA gate would catch
any rounding-mode deviation (truncation instead of RNE), any fp32 promotion at
the boundary (which shifts where fp16 rounding happens downstream), or any
reordering - any single differing byte fails closed.

## 5. tok/s projections and the +50 us booking bar

Baseline: M2b candidate 5.2516235625 ms/token = **190.42 tok/s**, wall win
+29.91 us vs the control bracket median (5.281533453125 ms; +27.47 vs control A
5.2790935625, +32.35 vs control B 5.28397334375). NOT booked: +29.91 us is
below the +50 us promotion bar (NO-GO). Removal mapping x1.0; measured medians.

| scenario | census delta | wall delta | ms/token | tok/s | vs +50 us bar |
| --- | ---: | ---: | ---: | ---: | --- |
| M2b baseline | - | - | 5.25162 | 190.42 | NO-GO (+29.91 us) |
| (a) `fab82d40` x49 only | -86.24 us | -86.24 us | 5.16538 | 193.60 | MEETS (+116.15 us) |
| (b) `0a5eb0ac` x36 only | -59.76 us | -59.76 us | 5.19186 | 192.61 | MEETS (+89.67 us) |
| (c) both x85 | -146.00 us | -146.00 us | 5.10562 | 195.86 | MEETS (+175.91 us) |

All three scenarios clear the +50 us bar with at least +89.7 us margin; (c)
projects ~195.9 tok/s. Sensitivity: with control-arm medians (a) = 193.41 tok/s,
(c) = 195.68 tok/s; with the ledger's 87.7 us fab82d40 figure (a) = 193.65,
(c) = 195.92. Even under a pessimistic x0.61 mapping, the projected wins are
(a) +82.5 us, (b) +66.4 us, (c) +119.0 us vs the bracket - all still above the
bar. M2b's underperformance came from the body-adding side (the absorbed-add
GEMVs), not from removals; `fab82d40`/`0a5eb0ac` are the same small epilogue
kernels M2a removed at a 0.99 census-to-wall ratio.

## Summary

- M2b final AB: candidate 679 kernels / 5631.68 us vs control 715 / 5666.99 us
  (net -36 kernels, -35.31 us census), wall 5.25162 ms = 190.42 tok/s, +29.91
  us vs bracket, below the +50 us bar (NO-GO).
- M2c removes `fab82d40` x49 (49 x 1.76 = 86.24 us) and `0a5eb0ac` x36 (36 x
  1.66 = 59.76 us): net census delta -146.00 us, kernels 679 -> 594.
- `fab82d40` is an fp32 identity copy (value-transparent; fold the contiguous
  boundary, keep the fp32 store); `0a5eb0ac` is a lossy fp32->fp16 cast that is
  NOT freely eliminable because the residual slot must stay fp32 - it can only
  fold into the consumer with bitwise-identical fp16 bytes.
- Projected tok/s (x1.0 removal): (a) 193.60, (b) 192.61, (c) 195.86; every
  scenario meets the +50 us booking bar, with (c) the strongest at +175.91 us.
- The exact-logits SHA-256 gate (`70838f52...4434819`, shape [32,1,151936])
  fails closed on any byte change, and is the correctness authority for both
  M2c folds.
