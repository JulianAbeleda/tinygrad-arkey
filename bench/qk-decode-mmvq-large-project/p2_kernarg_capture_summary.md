# Decode MMVQ large project P2 kernarg capture

- verdict: `PASS`
- captures: `7`
- by type: `{'Q4_K': 4, 'Q6_K': 3}`

## Selected P3 Templates

- `q4_attn_q_or_o`: role `attn_q_or_o`, global `[131072, 1, 1]`, local `[32, 1, 1]`, num_workgroups `[4096, 1, 1]`, ncols_x `4096`, stride_row_x `16`, stride_col_dst `4096`, has_fusion `False`
- `q6_ffn_down`: role `ffn_down`, global `[131072, 2, 1]`, local `[32, 2, 1]`, num_workgroups `[4096, 1, 1]`, ncols_x `12288`, stride_row_x `48`, stride_col_dst `4096`, has_fusion `True`
- `q6_lm_head`: role `lm_head`, global `[4861952, 2, 1]`, local `[32, 2, 1]`, num_workgroups `[151936, 1, 1]`, ncols_x `4096`, stride_row_x `16`, stride_col_dst `151936`, has_fusion `False`
