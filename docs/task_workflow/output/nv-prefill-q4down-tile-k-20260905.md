# Q4-down tile-K experiment

The wide graph-owned Q4-down candidate was parameterized for tile K64, K128, and K256 while preserving K64 as default. K128 compiled and passed the 18-role live correctness/lifecycle gate, but measured `10.671209 ms` versus llama `3.686166 ms`, so it is rejected. K256 was rejected by PTXAS before runtime: static shared `0x14000` exceeds the `0xc000` limit. Stream-K remains explicitly K64 with unroll8 as the best generated route.
