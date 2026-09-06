# NV P9 continuous ctx1024 coverage

A fresh normal-route process with max context 1536 prefetched one 1024-token
prompt, excluded three decode warm/capture tokens, and measured three
sequential 10-token windows in that same request. The windows cover logical
contexts 1027--1037, 1037--1047, and 1047--1057. Their total latencies were
42.4249, 42.4755, and 42.5197 ms; median service was 4.247546 ms/token or
235.430 token/s.

The selected JIT was `rollout_jit_flash_live[10]`. Inside-window snapshots
record P0, 2527 MHz SM, 14001 MHz memory, 44--45 C, and no active throttle
reason. The census retains each actual source once by SHA-256. Total fresh
process time was 126.77 seconds.

This proves that one ctx1024 production request and its settled decode windows
fit. It does not replace independent-request coverage: retaining the census
request and starting a second full prompt request failed near 30.1 GiB at both
max-context 4608 and 1536. That retained-memory ownership remains open.
