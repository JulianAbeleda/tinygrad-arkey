# NV P9 feedback ping-pong nonflash R17

Fresh ordinary-model context-256/max-context-512 runs cover the symbolic
nonflash graph used by required context bands 128 and 256. Legacy
`rollout_logits_jit` and both `rollout_greedy_logits_pingpong_jits` slots
produce bit-identical 4x151936 FP32 logits (SHA-256 `b20052ee...`, max absolute
difference zero), identical tokens, and sampled-token/argmax agreement.

The separate production sampled-token census selects
`rollout_greedy_pingpong_jits`. Its two slots pass the distinct-fixed-return
and read-only-input contract, with zero shadows on both slots and the exercised
slot. The census token is 13, with 524 launches and a 5999.06 us DEBUG=2 sum.
This closes nonflash numerical and alias safety, not endpoint performance.
