# NV P9 normal decode census and depth-512 endpoint

The normal production `Transformer.generate` route was captured at fixed KV
depth 512 with `max_context=1024` on the RTX 5090. No promotion was disabled.
The selected JIT was `rollout_jit_flash_s6`; it owns 418 concrete PROGRAM
launches with 29 unique program hashes. `tinygrad-program-census.json` records
each launch's JIT owner, capture order/path, source and binary SHA-256, launch
geometry, and available semantic identities. The graph-admission observer has
zero records because automatic horizon prewarming captured this JIT before the
observer was installed; the retained captured-linear census is the authority.

The census contains 155 launches with semantic role metadata. The remaining
263 generic launches have exact program identity and geometry but do not yet
carry explicit implementation provenance. Therefore this artifact proves the
normal route's concrete graph and ownership by the selected JIT, but it does
not prove that every unique program is generated or support a no-llama-cubin
claim.

The endpoint bracket used fresh processes and one GPU job at a time. tinygrad
ran three repetitions of 40 production `generate` tokens: 238.458, 238.785,
and 239.051 tok/s, with median 238.785 tok/s (4.187870 ms/token). All three
token streams were identical. llama.cpp commit `ac4cddeb0` ran three 40-token
generation repetitions at depth 512: 240.420, 245.826, and 246.482 tok/s, with
average 244.243 tok/s (4.094285 ms/token). tinygrad is 2.235% lower throughput
and 2.286% higher latency in this bracket.

This is an ordered cross-harness latency comparison. llama-bench and tinygrad
do not share the same prompt or sampled-token protocol, GPU clocks/power and
thermal state were not recorded, and the arms were not alternated. It is not a
correctness comparison or a strict promotion/parity gate. An alternating run
with system-state snapshots is required before interpreting this small gap.

Files:

- `tinygrad-program-census.json`: selected captured-JIT PROGRAM census.
- `tinygrad-census-run.json`: one-token normal-route timing and token evidence.
- `tinygrad-d512-r3.json`: three 40-token tinygrad repetitions.
- `llama-d512-r3.json`: fresh llama-bench prompt and generation rows.
- `*.log`: raw command output.
