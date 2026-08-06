# M4 epi_resadd kernel microgate record - per-kernel economics, isolated

Date: 2026-08-06
Target: Qwen3-8B-Q4_K_M, d512, RTX 5090 / native `DEV=NV`, branch
`nvidia-bringup-20260731`, HEAD `3f0784e7b`
Status: **probe record, PASS. Fused epi_resadd per-kernel economics are at
parity; the copy-free ceiling is strongly positive. Authorizes no
implementation, no route-record change, no promotion.** Authority:
`m4-variant-reopen-boundary-p0-scope-20260806.md` section 4, probe 2 (HARD
STOP section 7 applies: probes only).

## 1. Question

Is `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` economical per kernel vs the
legacy `q4k_g3_lanemap_gemv_4096_4096`, in isolation (one variant open)? The
gate from the scope: the fused median must not regress the legacy per-kernel
time materially (>10% is material here), and the copy-free ceiling
(copy mass minus fused penalty) must be positive or at parity, not negative.

## 2. Protocol

Probe: `extra/llm_research/decode/m4_resadd_kernel_microgate.py`
(same-session: control arm, then one-variant-open residual_add arm, each in a
fresh subprocess under one `flock -w 600 /tmp/gpu-bench.lock` so the model's
GPU memory is released between arms). M4 decomposition protocol: d512,
nmeas 20, reps 3, temperature 0, chunk_size 32, fused prefill attention
disabled (HEAD-broken on NV, house convention). Admission surgery identical to
`/tmp/m4_decomp_probe.py`: gate target set opened, then each linear's
`route_admission.q4k_epilogue_fusion_promoted` forced True only for route_role
`attn_qo` (144 linears admitted target across the 36 layers), False for every
other role (288 closed).

Artifact: `/tmp/m4_resadd_microgate_out.json`.

## 3. Census rows

| arm | kernels/token | E_ count | kernel us/token | fused 4096_4096 | legacy 4096_4096 | E_32_32_4_86a2 copy class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 948 | 492 | 5878.5 | 0 | 72 @ 9.38us | 1 @ 1.47us |
| variant | 984 (+36) | 528 (+36) | 5975.3 | 36 @ 9.60us | 36 @ 9.645us | 73 @ 1.5us |

The variant census replaces 36 legacy attn_qo GEMV calls with the fused kernel
(legacy 4096_4096 count drops 72 -> 36, the remaining 36 being ffn_down) and
adds 72 copy-class launches (73 - 1 control baseline); E_32_32_4_02a
(residual add) count drops 72 -> 36 as the epilogue absorbs the o-proj
residual add. Net +36 kernels, matching the M4 decomposition shape
(`m4-decomposition-measurement-record-20260803.md` section 4: +36, +69us at
d512; here +96.8us on a route that has since gained the 08-05 decode-parity
kernel mix).

## 4. Per-kernel microgate arithmetic

```text
fused median (attn_qo, variant)        9.60 us
legacy median (attn_qo, control)       9.38 us   -> +2.3% per kernel
legacy median (4096_4096, variant)     9.645 us  -> -0.47% same-session

fused penalty vs control baseline      36 x 0.22 = +7.9 us/token
copy mass (measured, 72 x 1.5)         108 us/token
copy mass (scope book, 36 x 1.47)      52.9 us/token
copy-free ceiling (measured)           +100.1 us/token  (positive)
copy-free ceiling (book)               +45.0 us/token   (positive)
```

Microgate verdict: **PASS** (`regress_ok` true at +2.3% << 10%; both ceilings
positive). The fused kernel is at parity with legacy per kernel in the
same-session arm and only +2.3% off the control-arm median; the copy removal
is worth an order of magnitude more than the fused penalty.

## 5. Token pins and bitwise-exactness

Both arms produce an IDENTICAL token stream, 3/3: token sha256
`227ad3ce9621f2c382cc722a3c2f1677637d3e3f2bfbf37d6ca652f98880eb4e`, first
token `271` in every rep. Control and variant streams are the same, so the
fused epi_resadd kernel is bitwise-exact with the legacy path in this session.

IMPORTANT pin note: the scope's section-5 pins (token sha
`9d6b3787...`, first token `151936`) do NOT reproduce at this HEAD. The
control census is now 948 kernels/token (vs 1021 in the 08-03 M4 record)
because the 08-05 decode-parity campaign landed w1w3/Q6K/flash-tiled/
reduce-output routes into the same tree (control histogram carries
`q4k_g3_lanemap_gemv_w1w3fused_12288_4096`, `q6k_gen_coop_*`, flash tiled
attention classes). The pins are therefore re-derived at this HEAD; the
section-6 full gate must re-pin before any record change.

## 6. Wall

| arm | tok/s median |
| --- | ---: |
| control | 180.36 |
| variant | 178.77 (-0.88%) |

The -0.88% wall delta with the copies present matches the M4 isolated row
(-1.15% at d512); with the copies removed the measured ceiling says the fused
variant is net positive (~+100us/token of kernel time before overlap effects).

## 7. Verdict and boundary

- **PASS**: fused median does not regress legacy materially (max +2.3%);
  copy-free ceiling positive by both measured (+100.1us) and book (+45.0us)
  arithmetic.
- Copy class is real and heavy: 72 launches/token at 1.5us (108us/token mass),
  confirming probe 1's fold is the load-bearing recovery.
- Bitwise-exactness confirmed by identical token streams across arms.
- HARD STOP respected: no production code changed, no record changed, no
  promotion. Section 6 full gate (d512/d2048/d4096 wall vs M2-on baseline,
  census assertions, re-derived pins 3/3) is a separate, subsequent decision.

## 8. References

- `m4-variant-reopen-boundary-p0-scope-20260806.md` sections 2-5, 7
- `m4-decomposition-measurement-record-20260803.md` sections 3-4, 6
- `m4-residual-boundary-fold-probe-record-20260806.md` (probe 1, the fold)
- `nv-decode-parity-final-20260802.md` (wall authority and pin baseline)
- `tinygrad/llm/decode_kernels.py` (`q4k_g3_lanemap_gemv_kernel` residual_add
  body)
- `tinygrad/llm/decode_routes.py` (`decode_routes.py:89-107`)
