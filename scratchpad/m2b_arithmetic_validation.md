# M2b A/B census arithmetic validation

Date: 2026-08-11. Source: `/tmp/m2b-ab.children/control-census.json`,
`candidate-census.json`, `/tmp/m2b-ab.json` (gate FAILED, verdict NO-GO),
`/tmp/ro_resid_rows.txt` (control-side structural probe), smoke candidate
trace, ledger `nv-epilogue-absorption-route-scope-20260810.md`.

## 1. Headline census fields

| metric | control | candidate | delta |
| --- | ---: | ---: | ---: |
| kernels | 715 | 751 | +36 |
| kernel_us | 5675.17 | 5732.57 | +57.40 |
| ffn_residual_add_count | 36 | 0 | -36 |
| ffn_residual_add_us | 64.86 | 0 | -64.86 |
| ffn_down_resadd_count | 0 | 36 | +36 |
| ffn_down_resadd_us | 0 | 1135.56 | +1135.56 |

## 2. Family counts and histogram medians

| program family | control | candidate |
| --- | ---: | ---: |
| `E_32_32_4_02a9738c...` (fp32 residual add) | 36 @ med 1.82 us | 0 |
| `E_32_32_4_86a23e1a...` (fp32 block-output copy) | 0 | 72 @ med 1.5 us |
| `E_32_32_4_fab82d40...` (fp32 copy, control-only family) | 49 @ med 1.66 us | 49 @ med 1.66 us |
| `*_epi_ffnresadd` (absorbed-add GEMVs, q4k+q6k) | 0 | 36 (18 q4k 4096_12288 + 18 q6k inkernel) |
| `w1w3fused16` | 36 | 36 |

The swap families: `q4k_g3_lanemap_gemv_4096_12288` (18) and
`q6k_gen_coop_4096_12288_inkernel` (18) become the two `*_epi_ffnresadd`
variants 1:1. Kernel count reconciles exactly: -36 adds + 72 copies = +36.

## 3. Swap arithmetic vs observed census delta

- Control residual-add family: 36 x 1.82 = **65.52 us**
- Candidate copy family: 72 x 1.5 = **108.00 us**
- Net census delta from the swap: **+42.48 us** (copies cost 42.5 us more
  than the adds they replace)
- Observed kernel_us delta: 5732.57 - 5675.17 = **+57.40 us**
- Residual: +14.92 us, explained by the absorbed GEMV bodies running
  slightly slower (medians: q4k ffn_down 26.21 -> 26.895, q6k inkernel
  35.25 -> 35.49; ~+16.65 us across the 36 bodies), offset by ~-2 us of
  other shared-family drift. Median-based reconciliation lands at
  +59.1 us predicted vs +57.4 observed; the ~1.7 us remainder is
  histogram median-vs-actual noise (median-sums are ~56-77 us below
  `kernel_us` per arm).

## 4. Why 72 copies (2 per block output)

The smoke candidate trace shows the new `86a23e1a` body exactly twice per
block, in both boundary positions:

1. right after the ffn_down GEMV output (`epi_ffnresadd`), i.e. the
   block's own `.contiguous()` on the absorbed-add result;
2. at the next sub-block residual input boundary (the
   `r_16_256`/`f14a5cc0` norm chain entry), where the fp32 block output
   feeds the following block's attention residual input.

36 blocks x 2 = 72. Ground truth: `/tmp/ro_resid_rows.txt` is a
control-side probe (36 x `02a9738c`, 49 x `fab82d40`, 0 x `86a23e1a`, 0
epi_ffnresadd); the candidate-side row trace
(`/tmp/m2b-ab.children/smoke-candidate.json`) contains 72 x `86a23e1a`
and 36 x `*_epi_ffnresadd`, confirming the count.

## 5. tok/s projections (ledger mapping: 0.61 body-adding, ~1.0 kernel removal)

M2a booked: 189.67 tok/s at 5.2723 ms/token.

| scenario | census delta | wall delta | ms/token | tok/s |
| --- | ---: | ---: | ---: | ---: |
| M2a booked | - | - | 5.2723 | 189.67 |
| M2b ideal (copies eliminated) | -65.52 us (x1.0 removal) | -65.5 us | 5.2068 | **192.06** |
| M2b as implemented (net swap +42.48 us, x0.61 body-adding) | +42.48 us | +25.9 us | 5.2982 | **188.74** |
| decomposed as-implemented (-65.5 x1.0 removal + 108.0 x0.61 copies) | - | +0.36 us | 5.2727 | 189.66 |
| observed census delta (57.40 us, x0.61) | +57.40 us | +35.0 us | 5.3073 | 188.42 |

Conclusion: with the copy regression removed, M2b books to ~192.1 tok/s
(-65.5 us wall), matching the ledger's pre-bracket projection (~191.9).
As implemented, the 72 copies erase the absorption win: the census swap
nets +42.5 us and the decomposed mapping lands essentially neutral
(~189.7 tok/s, no better than booked M2a). The census gate failing is
correct: net kernel delta is +36 (not -36), kernel_us is +57.4 us, and the
campaign cannot meet the +50 us/token bracket bar as-is.
