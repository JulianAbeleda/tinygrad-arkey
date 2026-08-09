# NV Q4_K FFN-down layer-8 singleton re-bracket (post-census-fix HEAD)

Status: **WALL WIN; the predicted two-layer subset is not advanced.**

Date: 2026-08-09
HEAD: `d9c9cd20991bb897d8b312f3efd40a1405acf543` (branch
`nvidia-bringup-20260731`), the post-census-fix HEAD whose only GPU-family
delta since the 08-05 record is the census-fix stack plus the Q6 direct g12
re-bracket record.

Section 5.2 of `nv-gemv-substrate-landing-scope-20260808.md` authorizes ONE
re-bracket of the layer-8 singleton at this HEAD with the same settled
protocol as the 08-05 record
(`nv-q4k-ffn-down-mmvq-included-cost-and-one-layer-record-20260805.md`).
The 08-05 settled wall was `+6.204734 us/token` (`NO_GO_WALL`).  This arm
re-runs exactly that layer-8 singleton at the post-census-fix HEAD.

## Settled reverse bracket

All arms used d512, 32-token uninterrupted windows, five repetitions, no
rejected samples, composed ping-pong (P5), the accepted-attention max17
composition (P1/P2, `CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1`), Qwen3-8B-
Q4_K_M, `DEV=NV`.  Control = installed `q4k_g3_lanemap_gemv_4096_12288`;
candidate = `q8_1_llama_provider_12288` + `q4k_q8_mmvq_direct_4096_12288`
leased on layer 8 only.  Every GPU child ran under
`flock -w 600 /tmp/gpu-bench.lock`.

Flags: `--mode timing --indices 8 --composed --accepted-attention-max17
--depth 512 --count 32 --reps 5 --max-context 1024`.

| arm | ms/token |
| --- | ---: |
| control A | 5.2197369375 |
| layer-8 candidate | 5.21682915625 |
| control C | 5.22030165625 |
| control midpoint | 5.220019296875 |
| candidate minus midpoint | **-0.003190140625 ms/token** |

The fresh wall is **-3.190140625 us/token** (0.06115% speedup vs control
midpoint), a material negative settled delta at the post-census-fix HEAD.
Verdict: **WIN** (harness `WALL_PASS`).

## Verification

Exact token stream hash identical across all three arms
(`all_token_hashes_equal: true`):
`f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`.
No sample was rejected in any arm (0/5 high-side rejections each).  The
census child under the same lease flags reported `finite: true` over 8 x
1 x 151936 full logits, topology pass (2 provider, 2 consumer, 34 installed,
0 adapter), and accepted-attention max17 topology pass (34 fused providers,
86 cooperative-Q4 consumers, 0 legacy consumers).

## Boundary

This is a WIN for the layer-8 singleton only.  The predicted two-layer
subset (blocks 8,16) stays closed: no second layer is advanced or
recommended from this arm, per section 5.2's hard stop.  This record does
not alter the promotion record, the 08-05 FFN-down record, or any default-
closed admission.  No code changed.

## Raw artifacts

- `/tmp/nv-ffn-down-layer8-20260809-bracket.json` SHA-256
  `3821d6fe97dc54fc168a94dc7759bdf1ccb46f907da9b3dc32628c4ab305675e`.
- `/tmp/nv-ffn-down-layer8-20260809-bracket/control-0.json` SHA-256
  `6672bcc9f149ef9cbb1cdf44ac4b61a6faa73545fa3a06aa28715f0194528681`.
- `/tmp/nv-ffn-down-layer8-20260809-bracket/candidate-1.json` SHA-256
  `8dda2e6739bb1924e3d7d8758b1dee350edfe167ff2c675fded04a7ed02eac90`.
- `/tmp/nv-ffn-down-layer8-20260809-bracket/control-2.json` SHA-256
  `08cef505e393f22ab22e39eef60fc9fd49e3e137bc72981b366ffef74e501762`.
- `/tmp/nv-ffn-down-layer8-20260809-census.json` SHA-256
  `c21578665f715dd292caa17d544a97bf6dea62d07d790754c8c3f59b7977df57`.
- `/tmp/nv-ffn-down-layer8-20260809-census.npz` SHA-256
  `34574c67fb20d186156b9300dd4b5f52f3f0e651045393bffcc3833b3cb37797`.
