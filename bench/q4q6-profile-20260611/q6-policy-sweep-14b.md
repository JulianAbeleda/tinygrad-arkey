target blk.0.ffn_down.weight shape=(5120, 17408)
target output.weight shape=(151936, 5120)
=== blk.0.ffn_down.weight fused_graph ===
pass quant_gbs=26.55
=== blk.0.ffn_down.weight local16_p1 ===
pass quant_gbs=163.0 opts=['LOCAL:0:16'] parts=1
=== blk.0.ffn_down.weight local32_p1 ===
pass quant_gbs=138.5 opts=['LOCAL:0:32'] parts=1
=== blk.0.ffn_down.weight local64_p1 ===
pass quant_gbs=154.56 opts=['LOCAL:0:64'] parts=1
=== blk.0.ffn_down.weight local128_p1 ===
pass quant_gbs=113.49 opts=['LOCAL:0:128'] parts=1
=== blk.0.ffn_down.weight local32_p2 ===
pass quant_gbs=218.13 opts=['LOCAL:0:32'] parts=2
=== blk.0.ffn_down.weight local64_p2 ===
pass quant_gbs=215.25 opts=['LOCAL:0:64'] parts=2
=== blk.0.ffn_down.weight local32_p4 ===
pass quant_gbs=132.57 opts=['LOCAL:0:32'] parts=4
=== blk.0.ffn_down.weight local64_p4 ===
pass quant_gbs=126.85 opts=['LOCAL:0:64'] parts=4
=== blk.0.ffn_down.weight local64_upcast2_p1 ===
wrong quant_gbs=None opts=['LOCAL:0:64', 'UPCAST:0:2'] parts=1
=== output.weight fused_graph ===
pass quant_gbs=120.14
=== output.weight local16_p1 ===
pass quant_gbs=130.63 opts=['LOCAL:0:16'] parts=1
=== output.weight local32_p1 ===
pass quant_gbs=94.58 opts=['LOCAL:0:32'] parts=1
=== output.weight local64_p1 ===
pass quant_gbs=94.31 opts=['LOCAL:0:64'] parts=1
=== output.weight local128_p1 ===
pass quant_gbs=92.59 opts=['LOCAL:0:128'] parts=1
=== output.weight local32_p2 ===
pass quant_gbs=109.52 opts=['LOCAL:0:32'] parts=2
=== output.weight local64_p2 ===
pass quant_gbs=111.02 opts=['LOCAL:0:64'] parts=2
=== output.weight local32_p4 ===
pass quant_gbs=110.11 opts=['LOCAL:0:32'] parts=4
=== output.weight local64_p4 ===
pass quant_gbs=110.6 opts=['LOCAL:0:64'] parts=4
=== output.weight local64_upcast2_p1 ===
wrong quant_gbs=None opts=['LOCAL:0:64', 'UPCAST:0:2'] parts=1
| tensor | candidate | status | quant GB/s | device ms | dot TFLOP/s | gemv | opts |
|---|---|---|---:|---:|---:|---:|---|
| blk.0.ffn_down.weight | fused_graph | pass | 26.55 | 2.754 | 0.06472691358024692 | 0.00121832 |  |
| blk.0.ffn_down.weight | local16_p1 | pass | 163.0 | 0.449 | 0.3970109576837416 | 0.00121832 | LOCAL:0:16 |
| blk.0.ffn_down.weight | local32_p1 | pass | 138.5 | 0.528 | 0.33760969696969695 | 0.00121832 | LOCAL:0:32 |
| blk.0.ffn_down.weight | local64_p1 | pass | 154.56 | 0.473 | 0.37686663847780133 | 0.00121832 | LOCAL:0:64 |
| blk.0.ffn_down.weight | local128_p1 | pass | 113.49 | 0.644 | 0.27679801242236024 | 0.00121832 | LOCAL:0:128 |
| blk.0.ffn_down.weight | local32_p2 | pass | 218.13 | 0.335 | 0.5321131940298507 | 0.00121784 | LOCAL:0:32 |
| blk.0.ffn_down.weight | local64_p2 | pass | 215.25 | 0.34 | 0.524288 | 0.00121784 | LOCAL:0:64 |
| blk.0.ffn_down.weight | local32_p4 | pass | 132.57 | 0.551 | 0.3235170961887477 | 0.00121856 | LOCAL:0:32 |
| blk.0.ffn_down.weight | local64_p4 | pass | 126.85 | 0.576 | 0.3094755555555556 | 0.00121856 | LOCAL:0:64 |
| blk.0.ffn_down.weight | local64_upcast2_p1 | wrong |  |  |  |  | LOCAL:0:64 UPCAST:0:2 |
| output.weight | fused_graph | pass | 120.14 | 5.312 | 0.29288867469879515 | 0.00166261 |  |
| output.weight | local16_p1 | pass | 130.63 | 4.885 | 0.3184902026612078 | 0.00166261 | LOCAL:0:16 |
| output.weight | local32_p1 | pass | 94.58 | 6.747 | 0.2305950259374537 | 0.00166261 | LOCAL:0:32 |
| output.weight | local64_p1 | pass | 94.31 | 6.767 | 0.22991349785724838 | 0.00166261 | LOCAL:0:64 |
| output.weight | local128_p1 | pass | 92.59 | 6.892 | 0.22574356355194428 | 0.00166261 | LOCAL:0:128 |
| output.weight | local32_p2 | pass | 109.52 | 5.826 | 0.2670485135599039 | 0.00166094 | LOCAL:0:32 |
| output.weight | local64_p2 | pass | 111.02 | 5.748 | 0.2706723451635351 | 0.00166094 | LOCAL:0:64 |
| output.weight | local32_p4 | pass | 110.11 | 5.795 | 0.26847707333908544 | 0.00166345 | LOCAL:0:32 |
| output.weight | local64_p4 | pass | 110.6 | 5.77 | 0.2696403188908146 | 0.00166345 | LOCAL:0:64 |
| output.weight | local64_upcast2_p1 | wrong |  |  |  |  | LOCAL:0:64 UPCAST:0:2 |

| tensor | shape | fused | best primitive | ratio | choice |
|---|---:|---:|---:|---:|---|
| blk.0.ffn_down.weight | 5120x17408 | 26.55 | 218.13 (local32_p2) | 8.215819209039548 | local32_p2 |
| output.weight | 151936x5120 | 120.14 | 130.63 (local16_p1) | 1.087314799400699 | local16_p1 |
