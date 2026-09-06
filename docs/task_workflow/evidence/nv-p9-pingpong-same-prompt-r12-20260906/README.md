# NV P9 same-prompt feedback ping-pong R12

Current HEAD, context 512, max context 1024. The full-logits legacy and
ping-pong arms use the same prompt and retain eight complete 151936-wide FP32
rows. Their arrays are bit-identical (SHA-256 `1d6c6891...`, max and mean
absolute difference zero); sampled tokens equal argmax in both arms.

The steady-service bracket ran fresh processes in legacy A / ping-pong B /
legacy C order, three 40-token repetitions per process. Median milliseconds per
token are 4.193921, 4.075100, and 4.193351. Ping-pong B is 2.8266% lower latency
and 2.9088% higher throughput than the mean of the two control medians. Every
40-token output hash is identical. Inside-window state is P0, 2542--2572 MHz SM,
14001 MHz memory, and 47--54 C with no reported throttle reason.

The corrected counter-attributed census identifies the exercised pair as
`rollout_greedy_pingpong_jits_flash_s6`: both slots have distinct fixed int32
1x1 returns, read-only inputs, four captured graph programs, and zero written
input shadows. The production token has 416 PROGRAM launches versus 418 in the
normal full-logits route census. The earlier `pre-fix-census.json` is retained:
its null contract exposed that selecting the sole warm pair was invalid when
automatic prewarming had warmed more than one candidate.

A separate greedy-only R3 median is 4.185604 ms, only 0.192% below the mean legacy control. Ping-pong is 2.640% below greedy alone. Its exercised `rollout_greedy_jit_flash_s6` has one written-input shadow, while the ping-pong pair has zero. Both routes have 416 launches; the observed 4001.24 versus 3904.36 us debug sums and steady timings localize the material gain to removal of the recurrent alias shadow through alternating fixed returns.

This is a current same-prompt correctness and ordered R3 service gate. It does
not cover the 8/10/18/34-split context bands, quantify cold-start cost
precisely, separate direct-greedy from two-slot feedback, or establish a
replicated external llama parity result. Normal defaults remain closed.

A corrected R13 full-logits rerun explicitly enters the completed active-horizon
selector state that diagnostic mode normally skips. Counter deltas prove the
control used `rollout_logits_jit_flash_s6` and the candidate used both
`rollout_greedy_logits_pingpong_jits_flash_s6` slots. Their full arrays remain
bit-identical with the same hash and tokens, and the exact diagnostic s6 pair
passes the zero-shadow contract. These `s6-*` artifacts supersede the base-pair
logits files for s6 qualification; the originals remain as historical evidence.
