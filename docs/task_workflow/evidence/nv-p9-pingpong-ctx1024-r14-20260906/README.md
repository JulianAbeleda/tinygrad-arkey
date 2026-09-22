# NV P9 feedback ping-pong ctx1024 R14

The fresh ordinary-model, request-scoped run at context 1024/max context 1536
selects `rollout_greedy_pingpong_jits_flash_live[10]`. Both captured slots pass
the distinct-fixed-return/read-only-input contract and have zero written-input
shadows. The exercised slot also reports zero shadows. The token is 13, with
416 launches and a 3935.05 us DEBUG=2 aggregate. Allocated memory in the debug
line is 19.25 GB.

Three preceding failures are retained because they exposed an invalid harness
setup. The old loader imported from `nv_predispatch_full_logits_qualification`
and explicitly disabled both normal prefill attention routes. Captured logits,
eager logits, and ping-pong census each failed during that diagnostic prefill
near 30.3 GB. The ordinary loader passes at 19.25 GB, so those failures do not
establish a production candidate capacity limit. Full-logit equality at this
band remains unavailable; qualification here covers normal-loader token output
and the exact captured-pair alias contract.
