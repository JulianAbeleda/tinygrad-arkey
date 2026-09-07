# Current252 native Flash role delta

Generated/native-Flash/generated R9 medians are 47.913109 / 47.264933 /
47.909971 ms. Generated Flash therefore costs 0.646607 ms against mean
controls. The native arm contains exactly 36 `nv_llama_fattn_mma_pp512` calls
and zero generated Flash calls; controls contain exactly 36
`nv_sm120_q16_grid_hd128_loop_attention` calls and zero native Flash calls.
All arms retain token 198, finite logits, and five exact replay cycles.
