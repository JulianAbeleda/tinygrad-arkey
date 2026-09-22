# Current 234-role Q4-down Stream-K promotion

The current216 pp512 route was bracketed against the existing graph-owned Q4
FFN-down U8 Stream-K lifecycle in fresh control/candidate/control processes.
Each arm used 10 warmups, R9 timing, and 20 recurrent replay cycles. Median
latencies were 59.534733, 54.935668, and 59.516347 ms. The candidate improves
4.589872 ms or 7.711% against the mean control median.

The candidate selects 18 Q4-down producers and 18 mains, retains 234 unique
canonical weight bases, and leaves only the 18 Q6-V FP16 overlays. All 20
recurrent cycles in every arm replay token 198 exactly. Candidate versus both
controls has finite full logits, the same token, max absolute difference
0.20824766, mean absolute difference 0.04733879, and passes rtol 0.02/atol 0.5.
The generated arithmetic is intentionally not bit-identical to FP16 fallback.

The exact compiler pp512 route now enables this Q4-down lease by default.
`NV_COMPILER_Q4_DOWN_STREAMK=0` restores the prior FP16 overlay. The first B
attempt is retained as a harness failure: the direct-forward authority had not
attached the new capture, its structural census correctly failed, and it is not
performance evidence.
