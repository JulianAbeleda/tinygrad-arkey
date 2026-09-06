# Current252 tile-major HCQ ledger

The refreshed generated pp512 graph contains 252 canonical projection mains,
216 Q8 producers, 108 active fixups, 36 generated Flash calls, generated
vocabulary, and no V/down FP16 overlays. Two exact replay cycles pass.

The observer-bearing selected invocation spans 51.952448 ms. Projection mains
occupy 40.768928 ms of command intervals, producers 0.831968 ms, fixups 1.512352
ms, Flash 2.857952 ms, and unresolved support 3.446016 ms. Gate/up remains the
largest role population at 18.612416 ms. Q/O is 9.609984 ms, down 9.456736 ms,
V 2.464896 ms, and K 2.137248 ms.

The ledger adapter now recognizes the retained tile-major K identity and
`q8_ds4_fp16_pp512` producer. These are HCQ command intervals, not CUPTI kernel
active durations; the unprofiled model bracket remains endpoint authority.
