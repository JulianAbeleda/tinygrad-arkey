# NV P9 promoted endpoint R21

Commit `e4b5fc2a6` promotes the two-capture feedback ring only for the exact NV
sm_120 Qwen3-8B dense topology with native argmax and a caller-provided output
horizon. Unknown-horizon requests retain the legacy route; the environment
rollback is `TINYGRAD_DECODE_FEEDBACK_PINGPONG_DISABLE=1`.

Fresh llama A / tinygrad B / llama C processes measured three 40-token samples
at depth 512. Median service is 4.064018, 4.077346, and 4.081455 ms/token.
Promoted tinygrad is 0.1132% higher latency and 0.1131% lower throughput than
the mean of the llama control medians: effective endpoint parity in this
bracket. Tinygrad selects both s6 feedback slots with 416 programs/token and
all token hashes agree across its repetitions.

The promoted census contains 832 launches and 54 unique PROGRAM identities
across both slots. Every unique program has exact NV CUDARenderer construction
provenance and positive current SOURCE-to-binary transport; none is unknown or
native-precompiled.

This remains a cross-harness comparison: llama-bench and tinygrad use different
prompt/generated-token protocols, so it is latency parity rather than token
correctness evidence. Llama state snapshots are process boundaries (P8 before,
P1 after); tinygrad state is captured immediately around each timed window.
Clocks were observed rather than locked, and the bracket is ordered R3.
