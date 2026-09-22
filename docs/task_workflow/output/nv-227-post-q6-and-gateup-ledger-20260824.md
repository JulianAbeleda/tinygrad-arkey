# NV 227 ledger: Q6 packed lane map and gate/up four-warp recheck

Date: 2026-08-24  
Conservative endpoint: **4515.3957 us/token = 221.4645 tok/s**  
227 target: **4405.2863 us/token**  
Remaining: **110.1094 us/token = 5.5355 tok/s**

## Clean tests

### Q6 FFN-down packed lane map — no wall credit

The candidate is bit-exact and improves the isolated device pool, but the
fresh unprofiled d512/reps7 wall bracket is neutral: control midpoint
4472.3686 us/token, candidate 4472.2171 us/token, with the candidate not
beating the opening control.  The profile-only pool delta was 13.848 us/token
and is not booked because it did not survive the wall test.

### Gate/up four-warp plus current vector-load control — closed no-go

The clean CUDA microgate compares the existing installed
`q4k_g3_lanemap_gemv_w1w3vec16` route with the four-warp fp16 candidate:

| route | median us/launch | result |
|---|---:|---|
| current vector control | 22.0202 | reference |
| four-warp candidate | 23.5910 | **+7.13% slower** |

The outputs are bit-exact (zero mismatches, zero max absolute error).  At 36
gate/up launches per token, the measured regression is approximately **+56.5
us/token**, so this route is closed without a production wall bracket.

## Ledger decision

Neither tested mechanism changes the conservative endpoint.  The remaining
110.1 us/token to 227 therefore still requires a genuinely wall-positive
mechanism; microgate-only wins and profile-only pool deltas remain unbooked.
The next search should prioritize byte reduction or verified overlap/producer
effects, not another gate/up launch-shape variant.
