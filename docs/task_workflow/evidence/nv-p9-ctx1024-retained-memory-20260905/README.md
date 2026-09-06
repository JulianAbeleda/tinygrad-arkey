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
uncaptured after the first request. On the second independent request they
enter TinyJit's capture branch; both observed ctx1024 failures occur while
constructing/replaying that prefill graph near 30.1 GiB. This identifies the
second-use per-offset prefill-v2 capture as the trigger. A production fix still
needs target-neutral memory admission and must preserve the selected packed
kernels and alias safety.
