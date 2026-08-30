# NV pp512 Flash vector primitive F1.4 verdict

Packet F1.4: **STOP**.

The completed live F1.2 replay covered exactly 36 installed calls and exactly
36 candidate calls, with matched arm order `control_0`, `candidate_1`,
`control_2`. Full-output correctness, finite outputs, replay, exact logits,
and greedy token 198 passed.

Performance fails decisively: installed service was approximately 99-107
us/call, while the candidate was approximately 6703-6720 us/call. The
candidate therefore fails both the minimum and median requirements by far more
than measurement noise. F1.3 resource counters are unnecessary for this
decision. F2 composition is not authorized.
