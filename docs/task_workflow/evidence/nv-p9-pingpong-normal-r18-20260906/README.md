# NV P9 ordinary-loader feedback service R18

Fresh ordinary-model processes, public request-scoped prewarm, context 512,
max context 1024, three 40-token repetitions per arm in legacy A / ping-pong B /
legacy C order. Medians are 4.183295, 4.064449, and 4.188350 ms/token.
Ping-pong is 2.8996% lower latency and 2.9862% higher throughput than the mean
of the two control medians. All nine token hashes are identical.

Every B repetition counter-selects
`rollout_greedy_pingpong_jits_flash_s6`; its contract is admitted with distinct
fixed returns, read-only inputs, and zero shadows. Inside-window GPU state is
P0, 2542--2572 MHz SM, 14001 MHz memory, and 43--50 C with no reported throttle
reason. A and C use the same prompt, generation API, horizon metadata, and
summary statistic. This is a strong same-runtime promotion gate; it does not by
itself establish external llama parity or quantify extra cold construction of
the second capture.
