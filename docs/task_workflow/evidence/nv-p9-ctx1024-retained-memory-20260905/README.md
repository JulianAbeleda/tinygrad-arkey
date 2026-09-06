# NV P9 ctx1024 retained-memory ownership

A fresh max-context-1536 process was sampled after model load, after the first
1024-token prompt yielded, and after three decode warm tokens. GlobalCounters
rose from 19,005,016,896 to 21,751,898,216 bytes after the first yield and by
only four more bytes after warmup. Four graph-cache entries existed.

Only `rollout_jit_flash_live[10]` was captured. Its concrete call buffers refer
to 5,265,857,620 unique base bytes, including shared model allocations. Its
written-input-shadow ownership is exactly one four-byte allocation after the
three warm tokens, ruling out the alias firewall as the material memory owner.

The concrete prefill-v2 JITs at offsets 0 and 512 are each `cnt=1` and remain
uncaptured after the first request. This originally made second-use capture a
hypothesis for the independent-request failure. Follow-up diagnostics that
cleared the uncaptured wrappers, and separately cleared both those wrappers and
the captured rollout graph, still failed during the next eager prefill near
30 GiB. The trigger claim is therefore withdrawn; see the 20260906 follow-up
checkpoint for the narrower retained-allocation findings.
