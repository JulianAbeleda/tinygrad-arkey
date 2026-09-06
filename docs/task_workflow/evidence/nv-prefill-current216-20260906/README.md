# Current 216-role pp512 composition — 2026-09-06

This current-HEAD arm combines generated gate/up (72), K (36), Q/O (72), Q4-V
(18), and the promoted generated Q6 FFN-down population (18). It selects 216
producer/main pairs backed by 216 distinct canonical weights, leaves 36 V/down
FP16 overlays, and has no old Q4/Q6 fixup. Q6-down retains its qualified
Stream-K destination fixup.

`current216-r2.json` passes token 198, finite output, distinct-input output,
and exact recurrent replay. Its nine samples are 66.305659, 66.328885,
64.132186, 63.818517, 63.793128, 62.373687, 61.886702, 61.875039, and
61.890129 ms (minimum 61.875039, median 63.793128). The stepped within-run
warmup makes this a composition baseline and correctness/census authority;
a matched alternating performance gate is still required for any new kernel.
