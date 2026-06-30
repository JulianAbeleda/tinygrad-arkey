# TG3 Quant Semantics Library -- verdict: **TG3_PASS_QUANT_SEMANTICS_READY**

Quant formats are now DATA (extra/qk_quant_semantics.py -> bench/qk-search-spaces/quant_semantics.json). TG1's QuantSpec reads the Q4_K row; the G2/G3 hardcoded 256/36/4 are now derived from the byte layout.

## Formats

| quant | elems | bytes | pack | metadata B | payload B | words/blk | quant_word_base | sym | quality |
|---|---:|---:|---|---:|---:|---:|---:|:--:|---|
| Q4_K | 256 | 144 | uint32 | 16 | 128 | 36 | 4 | False | lossy_4bit_kquant |
| Q5_K | 256 | 176 | uint32 | 16 | 160 | 44 | 4 | False | lossy_5bit_kquant |
| Q6_K | 256 | 210 | uint16 | 18 | 192 | 105 | None | True | lossy_6bit_kquant |
| Q8_0 | 32 | 34 | int8 | 2 | 32 | 34 | 2 | True | near_lossless_8bit |
| fp16 | 1 | 2 | fp16 | 0 | 2 | 1 | 0 | True | lossless_fp16 |

## TG3 proof gates

- **Q4_K row reproduces G3 quant facts** (qk_k=256/words_per_block=36/quant_word_base=4): True ({'block_elems': 256, 'words_per_block': 36, 'quant_word_base': 4})
- **TG1 re-emit lossless with data-driven QuantSpec.from_library('Q4_K')**: True (from_library drives G3: True)
- **Q6_K row -> shipped coop in known_good + half-warp direct marked refuted**: True
- **Unsupported quant (Q3_K) -> SEARCH_SPACE_INCOMPLETE, no Q4_K fallback**: True
- **quant_spec_fields refuses Q6_K (payload-first) instead of Q4_K-shaping it**: True
