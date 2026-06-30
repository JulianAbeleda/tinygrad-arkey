# PMS-R5 G3 LaneMap Template Audit

Verdict: **PMS_R5_PASS_G3_TEMPLATE_PROVEN**  (lossless reconstruction = True)

Speed-equivalence (cited, not re-measured): `AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT` from `bench/amd-isa-backend-g3-weight-promotion/latest.json`.

| role | rows(N) | k(K) | kernel | UOp key == default | name match | packed-idx match |
|---|---:|---:|---|:--:|:--:|:--:|
| ffn_gate_up | 12288 | 4096 | `q4k_g3_lanemap_gemv_12288_4096` | True | True | True |
| ffn_down | 4096 | 12288 | `q4k_g3_lanemap_gemv_4096_12288` | True | True | True |
| attn_qo | 4096 | 4096 | `q4k_g3_lanemap_gemv_4096_4096` | True | True | True |

Template is lossless: emitting each eligible role FROM the LaneMapTemplate params yields a UOp program byte-identical (UOp .key) to the current default G3 emission, and the declared packed-word-index formula matches the LaneMap reference. Provenance + parameterization in `lanemap_template_schema`.
