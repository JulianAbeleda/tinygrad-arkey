# NV P9 feedback ping-pong ctx2048 R15

A fresh ordinary-model request at context 2048/max context 2560 uses public
request-scoped prewarming and selects
`rollout_greedy_pingpong_jits_flash_live[18]`. Both slots pass the
fixed-distinct-return/read-only-input contract with zero shadows; the exercised
slot also reports zero. The first token is 38835, the selected route has 416
launches, and its DEBUG=2 aggregate is 4043.43 us. Settled debug memory is
19.41 GB; an external snapshot during prefill reached about 30.6 GiB but the
request completed. This is capacity and alias-contract evidence, not a
replicated latency or full-logits gate.
