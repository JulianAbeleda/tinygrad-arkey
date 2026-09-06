# NV P9 alternating endpoint pilot

This A/B/C pilot bracketed one fresh tinygrad process between two llama.cpp
processes at fixed depth 512. Every process held `/tmp/gpu-bench.lock`. Each
arm measured three repetitions of 40 decode tokens. Clocks were observed but
not locked.

Using the median total latency within each arm, llama A measured 4.060741 ms
per token, tinygrad B measured 4.220619 ms per token, and llama C measured
4.066252 ms per token. Relative to the mean of the two llama endpoint medians
(4.063497 ms per token), tinygrad had 3.8668% higher latency and 3.7229% lower
throughput. The tinygrad repetitions generated identical token streams with
SHA-256 `dbd3026bb808fd57a5c7466963b0309168ee3d5ca58ccaf829cacdea9e0d491b`.

No clock-throttle reason was active in the captured states. The llama CSVs are
process-boundary snapshots: they show an idle P8 state before each invocation
and an active P1 state after it. The tinygrad JSON was produced by the earlier
GPU-state harness revision, whose snapshots surround each request including
reset and prefill rather than the exact timed window. It recorded P0, 2482–2565
MHz SM clocks, 14001 MHz memory clocks, and 46–55 C across the three requests.

This is an ordered R3 cross-harness latency pilot. llama-bench and tinygrad use
different prompt and generated-token protocols, and the state scopes differ.
It is not token-correctness evidence or strict parity/promotion qualification.
A replicated alternating run using the current inside-window state callbacks is
still required for a strict endpoint conclusion.

Files:

- `summary.json`: derived medians, comparison, and explicit limitations.
- `llama-a.json`, `llama-c.json`: raw llama-bench output.
- `tinygrad-b.json`: raw tinygrad output, token evidence, and request-boundary states.
- `llama-*-before.csv`, `llama-*-after.csv`: process-boundary `nvidia-smi` snapshots.
