# Current252 native Q/K/V role delta

Generated/native-QKV/generated medians are 47.956002 / 44.805523 /
47.921975 ms on the exact current252 graph. The generated Q/K/V population
therefore costs 3.1334655 ms against the mean generated controls.

Every arm retains 252 projection mains, canonical packed weights, zero overlays,
token 198, finite logits, and five exact replay cycles. Cross-arm logits pass
rtol 0.02 / atol 0.5 with max absolute difference 0.13143349. The native QKV
programs are diagnostic only.
