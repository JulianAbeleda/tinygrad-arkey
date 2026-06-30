# TG1 LaneMapTemplate IR -- lossless re-emit of the promoted G3 route

Verdict: **TG1_PASS_IR_LOSSLESS_REEMITS_G3**  (topology separable = True, lossless re-emit = True)

## IR field taxonomy (topology FREE fields vs DATA inputs)

- **TOPOLOGY (free)**: block_groups, words_per_group, axis_roles, reduction_pattern, lane_ownership_index
- **QUANT (data)**: qk_k, q4k_words_per_block, q4k_quant_word_base, name, dequant_body
- **TARGET (data)**: lane_extent, name
- **SHAPE (data)**: rows, k, role

## 3-role lossless re-emit (IR instantiated with G3's actual topology)

| role | rows(N) | k(K) | kernel | UOp key == default | name match | lane-idx match |
|---|---:|---:|---|:--:|:--:|:--:|
| ffn_gate_up | 12288 | 4096 | `q4k_g3_lanemap_gemv_12288_4096` | True | True | True |
| ffn_down | 4096 | 12288 | `q4k_g3_lanemap_gemv_4096_12288` | True | True | True |
| attn_qo | 4096 | 4096 | `q4k_g3_lanemap_gemv_4096_4096` | True | True | True |

Each role instantiates `LaneMapTemplate` with G3's topology (block_groups=4, words_per_group=8, G3 axis roles row=GLOBAL/block_group+word_col=LOCAL/local_block+group_pair=REDUCE, cross-lane reduce, G3 lane index) and emits via the existing G2 lane map + G3 emitter (SAME path). The emitted UOp program is byte-identical (UOp .key) to the current promoted default emission.
