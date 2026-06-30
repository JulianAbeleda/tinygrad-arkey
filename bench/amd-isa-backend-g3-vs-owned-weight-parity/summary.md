# G3-vs-owned Q4_K weight-path parity gate

**Verdict:** AMD_ISA_G3_PARITY_PASS_MATCHES_OWNED

**Decision:** start_layout_reshuffle = False

Do NOT start the offline Q4_K weight-layout reshuffle. G3 is the pure speed-equivalent replacement for the owned warp custom-kernel. Next = generated-G3 promotion / search-binding hardening so BubbleBeam picks G3 without manual flags.

| ctx | owned tok/s | g3_bubblebeam tok/s (lag%) | g3_forced tok/s (lag%) | bubblebeam clean | forced clean |
|---|---|---|---|---|---|
| 512 | 103.95 | 103.64 (+0.3) | 104.13 (-0.17) | True | True |
| 1024 | 102.27 | 101.85 (+0.41) | 102.13 (+0.14) | True | True |
| 2048 | 99.63 | 99.36 (+0.27) | 99.88 (-0.25) | True | True |
| 4096 | 94.9 | 94.61 (+0.31) | 94.95 (-0.05) | True | True |

Parity threshold: 5.0%. Worst lag: 0.0%. NMEAS=12.

