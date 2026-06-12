target blk.0.ffn_down.weight shape=(4096, 12288)
target output.weight shape=(151936, 4096)
=== blk.0.ffn_down.weight fused_graph ===
pass quant_gbs=21.24
=== blk.0.ffn_down.weight local16_p1 ===
pass quant_gbs=127.04 opts=['LOCAL:0:16'] parts=1
=== blk.0.ffn_down.weight local32_p1 ===
pass quant_gbs=115.65 opts=['LOCAL:0:32'] parts=1
=== blk.0.ffn_down.weight local64_p1 ===
pass quant_gbs=129.52 opts=['LOCAL:0:64'] parts=1
=== blk.0.ffn_down.weight local128_p1 ===
pass quant_gbs=105.79 opts=['LOCAL:0:128'] parts=1
=== blk.0.ffn_down.weight local32_p2 ===
pass quant_gbs=201.95 opts=['LOCAL:0:32'] parts=2
=== blk.0.ffn_down.weight local64_p2 ===
pass quant_gbs=197.1 opts=['LOCAL:0:64'] parts=2
=== blk.0.ffn_down.weight local32_p4 ===
pass quant_gbs=134.67 opts=['LOCAL:0:32'] parts=4
=== blk.0.ffn_down.weight local64_p4 ===
pass quant_gbs=138.37 opts=['LOCAL:0:64'] parts=4
=== blk.0.ffn_down.weight local64_upcast2_p1 ===
wrong quant_gbs=None opts=['LOCAL:0:64', 'UPCAST:0:2'] parts=1
=== output.weight fused_graph ===
pass quant_gbs=121.57
=== output.weight local16_p1 ===
pass quant_gbs=133.69 opts=['LOCAL:0:16'] parts=1
=== output.weight local32_p1 ===
pass quant_gbs=94.53 opts=['LOCAL:0:32'] parts=1
=== output.weight local64_p1 ===
pass quant_gbs=94.25 opts=['LOCAL:0:64'] parts=1
=== output.weight local128_p1 ===
pass quant_gbs=93.89 opts=['LOCAL:0:128'] parts=1
=== output.weight local32_p2 ===
pass quant_gbs=92.76 opts=['LOCAL:0:32'] parts=2
=== output.weight local64_p2 ===
pass quant_gbs=92.62 opts=['LOCAL:0:64'] parts=2
=== output.weight local32_p4 ===
pass quant_gbs=91.93 opts=['LOCAL:0:32'] parts=4
=== output.weight local64_p4 ===
pass quant_gbs=92.28 opts=['LOCAL:0:64'] parts=4
=== output.weight local64_upcast2_p1 ===
wrong quant_gbs=None opts=['LOCAL:0:64', 'UPCAST:0:2'] parts=1
| tensor | candidate | status | quant GB/s | device ms | dot TFLOP/s | gemv | opts |
|---|---|---|---:|---:|---:|---:|---|
| blk.0.ffn_down.weight | fused_graph | pass | 21.24 | 1.943 | 0.051808181163149766 | 0.00151622 |  |
| blk.0.ffn_down.weight | local16_p1 | pass | 127.04 | 0.325 | 0.30973321846153845 | 0.00151622 | LOCAL:0:16 |
| blk.0.ffn_down.weight | local32_p1 | pass | 115.65 | 0.357 | 0.2819700168067227 | 0.00151622 | LOCAL:0:32 |
| blk.0.ffn_down.weight | local64_p1 | pass | 129.52 | 0.319 | 0.3155589216300941 | 0.00151622 | LOCAL:0:64 |
| blk.0.ffn_down.weight | local128_p1 | pass | 105.79 | 0.39 | 0.2581110153846154 | 0.00151622 | LOCAL:0:128 |
| blk.0.ffn_down.weight | local32_p2 | pass | 201.95 | 0.204 | 0.49344752941176473 | 0.00151372 | LOCAL:0:32 |
| blk.0.ffn_down.weight | local64_p2 | pass | 197.1 | 0.209 | 0.48164256459330146 | 0.00151372 | LOCAL:0:64 |
| blk.0.ffn_down.weight | local32_p4 | pass | 134.67 | 0.307 | 0.3278934723127036 | 0.00151622 | LOCAL:0:32 |
| blk.0.ffn_down.weight | local64_p4 | pass | 138.37 | 0.298 | 0.33779629530201344 | 0.00151622 | LOCAL:0:64 |
| blk.0.ffn_down.weight | local64_upcast2_p1 | wrong |  |  |  |  | LOCAL:0:64 UPCAST:0:2 |
| output.weight | fused_graph | pass | 121.57 | 4.199 | 0.29641812622052877 | 0.00178289 |  |
| output.weight | local16_p1 | pass | 133.69 | 3.818 | 0.32599782922996334 | 0.00178289 | LOCAL:0:16 |
| output.weight | local32_p1 | pass | 94.53 | 5.401 | 0.2304498633586373 | 0.00178289 | LOCAL:0:32 |
| output.weight | local64_p1 | pass | 94.25 | 5.416 | 0.2298116159527326 | 0.00178289 | LOCAL:0:64 |
| output.weight | local128_p1 | pass | 93.89 | 5.437 | 0.22892398602170316 | 0.00178289 | LOCAL:0:128 |
| output.weight | local32_p2 | pass | 92.76 | 5.503 | 0.22617839578411775 | 0.0017817 | LOCAL:0:32 |
| output.weight | local64_p2 | pass | 92.62 | 5.512 | 0.22580909143686506 | 0.0017817 | LOCAL:0:64 |
| output.weight | local32_p4 | pass | 91.93 | 5.553 | 0.22414185341256979 | 0.00177741 | LOCAL:0:32 |
| output.weight | local64_p4 | pass | 92.28 | 5.532 | 0.22499271728127257 | 0.00177741 | LOCAL:0:64 |
| output.weight | local64_upcast2_p1 | wrong |  |  |  |  | LOCAL:0:64 UPCAST:0:2 |

| tensor | shape | fused | best primitive | ratio | choice |
|---|---:|---:|---:|---:|---|
| blk.0.ffn_down.weight | 4096x12288 | 21.24 | 201.95 (local32_p2) | 9.508003766478343 | local32_p2 |
| output.weight | 151936x4096 | 121.57 | 133.69 (local16_p1) | 1.0996956485975158 | local16_p1 |
