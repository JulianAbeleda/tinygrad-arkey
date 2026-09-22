# Current 216-role gate/up Stream-K promotion — 2026-09-06

The exact generated compiler pp512 arm now selects its retained unroll-8
Stream-K gate/up body and llama-compatible Q8 producer by default. Setting
`NV_COMPILER_Q4_GATE_STREAMK=0` restores the prior wide body. Admission remains
bounded by the existing explicit compiler mode and exact NV sm_120 dense
Qwen3-8B pp512 topology.

A wide/Stream-K/wide full-model bracket used 10 warmups and nine samples per
fresh process. All arms passed token 198, recurrent replay, 216 main/producer
calls, 216 canonical weight bases, and 36 remaining V/down overlays. Medians
were 61.893829, 59.524891, and 61.930238 ms. Stream-K improves 2.387143 ms or
3.856% against the mean controls. The B/C request-boundary snapshots were P0,
2572--2580 MHz, 45--46 C, and about 79.8 W; A began idle P8 but its timed
samples were tight after 10 warmups.

A separate 20-cycle run passes every recurrent cycle. Saved wide versus
Stream-K full logits are finite, select token 198, and pass rtol 0.02/atol 0.5
(max abs 0.12864208, mean abs 0.04389589). Arithmetic order is not bit-exact.

The ordinary-census diagnostic returns the correct replay token/logits but is
not a performance authority after the no-workload-reuse eager policy: it has no
captured PROGRAM census and repeatedly compiles, producing 2.7--3.1 s samples.
Its retained FAIL is a harness/policy limitation, not used for promotion.
