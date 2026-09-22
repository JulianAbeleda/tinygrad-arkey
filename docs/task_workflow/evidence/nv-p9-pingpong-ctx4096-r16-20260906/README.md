# NV P9 feedback ping-pong ctx4096 R16

A fresh ordinary-model request at context 4096/max context 4608 uses public
request-scoped prewarming and selects
`rollout_greedy_pingpong_jits_flash_live[34]`. Both slots pass the
distinct-fixed-return/read-only-input contract with zero shadows; the exercised
slot also reports zero. The token is 13, the route has 416 launches, and its
DEBUG=2 aggregate is 4273.89 us. Settled debug memory is 19.73 GB. External
snapshots observed a roughly 30.6 GiB prefill peak, which completed. This is
capacity and alias-contract evidence, not a replicated latency or full-logits
gate.
