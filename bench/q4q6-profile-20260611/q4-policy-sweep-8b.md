=== blk.0.ffn_gate.weight (12288, 4096) fused_graph ===
pass device_q4_eff_gbs=87.72
=== blk.0.ffn_gate.weight (12288, 4096) local8_p1 ===
pass device_q4_eff_gbs=187.65 opts=['LOCAL:0:8'] parts=1
=== blk.0.ffn_gate.weight (12288, 4096) local16_p1 ===
pass device_q4_eff_gbs=313.54 opts=['LOCAL:0:16'] parts=1
=== blk.0.ffn_gate.weight (12288, 4096) local32_p1 ===
pass device_q4_eff_gbs=383.0 opts=['LOCAL:0:32'] parts=1
=== blk.0.ffn_gate.weight (12288, 4096) local64_p1 ===
pass device_q4_eff_gbs=420.05 opts=['LOCAL:0:64'] parts=1
=== blk.0.ffn_gate.weight (12288, 4096) local16_p2 ===
pass device_q4_eff_gbs=177.95 opts=['LOCAL:0:16'] parts=2
=== blk.0.ffn_gate.weight (12288, 4096) local32_p2 ===
pass device_q4_eff_gbs=182.72 opts=['LOCAL:0:32'] parts=2
=== blk.0.ffn_gate.weight (12288, 4096) local32_p4 ===
pass device_q4_eff_gbs=136.14 opts=['LOCAL:0:32'] parts=4
=== blk.0.ffn_gate.weight (12288, 4096) local32_upcast2_p1 ===
pass device_q4_eff_gbs=209.91 opts=['LOCAL:0:32', 'UPCAST:0:2'] parts=1
=== blk.0.ffn_gate.weight (12288, 4096) local32_upcast3_p1 ===
pass device_q4_eff_gbs=157.97 opts=['LOCAL:0:32', 'UPCAST:0:3'] parts=1
=== blk.0.attn_q.weight (4096, 4096) fused_graph ===
pass device_q4_eff_gbs=15.54
=== blk.0.attn_q.weight (4096, 4096) local8_p1 ===
pass device_q4_eff_gbs=111.49 opts=['LOCAL:0:8'] parts=1
=== blk.0.attn_q.weight (4096, 4096) local16_p1 ===
pass device_q4_eff_gbs=167.85 opts=['LOCAL:0:16'] parts=1
=== blk.0.attn_q.weight (4096, 4096) local32_p1 ===
pass device_q4_eff_gbs=175.1 opts=['LOCAL:0:32'] parts=1
=== blk.0.attn_q.weight (4096, 4096) local64_p1 ===
pass device_q4_eff_gbs=188.77 opts=['LOCAL:0:64'] parts=1
=== blk.0.attn_q.weight (4096, 4096) local16_p2 ===
pass device_q4_eff_gbs=136.88 opts=['LOCAL:0:16'] parts=2
=== blk.0.attn_q.weight (4096, 4096) local32_p2 ===
pass device_q4_eff_gbs=167.06 opts=['LOCAL:0:32'] parts=2
=== blk.0.attn_q.weight (4096, 4096) local32_p4 ===
pass device_q4_eff_gbs=162.54 opts=['LOCAL:0:32'] parts=4
=== blk.0.attn_q.weight (4096, 4096) local32_upcast2_p1 ===
pass device_q4_eff_gbs=96.51 opts=['LOCAL:0:32', 'UPCAST:0:2'] parts=1
=== blk.0.attn_q.weight (4096, 4096) local32_upcast3_p1 ===
illegal-opt device_q4_eff_gbs=None opts=['LOCAL:0:32', 'UPCAST:0:3'] parts=1
=== blk.4.ffn_down.weight (4096, 12288) fused_graph ===
pass device_q4_eff_gbs=16.17
=== blk.4.ffn_down.weight (4096, 12288) local8_p1 ===
pass device_q4_eff_gbs=132.22 opts=['LOCAL:0:8'] parts=1
=== blk.4.ffn_down.weight (4096, 12288) local16_p1 ===
pass device_q4_eff_gbs=182.29 opts=['LOCAL:0:16'] parts=1
=== blk.4.ffn_down.weight (4096, 12288) local32_p1 ===
pass device_q4_eff_gbs=196.71 opts=['LOCAL:0:32'] parts=1
=== blk.4.ffn_down.weight (4096, 12288) local64_p1 ===
pass device_q4_eff_gbs=212.24 opts=['LOCAL:0:64'] parts=1
=== blk.4.ffn_down.weight (4096, 12288) local16_p2 ===
pass device_q4_eff_gbs=188.93 opts=['LOCAL:0:16'] parts=2
=== blk.4.ffn_down.weight (4096, 12288) local32_p2 ===
pass device_q4_eff_gbs=252.62 opts=['LOCAL:0:32'] parts=2
=== blk.4.ffn_down.weight (4096, 12288) local32_p4 ===
pass device_q4_eff_gbs=268.78 opts=['LOCAL:0:32'] parts=4
=== blk.4.ffn_down.weight (4096, 12288) local32_upcast2_p1 ===
pass device_q4_eff_gbs=101.38 opts=['LOCAL:0:32', 'UPCAST:0:2'] parts=1
=== blk.4.ffn_down.weight (4096, 12288) local32_upcast3_p1 ===
illegal-opt device_q4_eff_gbs=None opts=['LOCAL:0:32', 'UPCAST:0:3'] parts=1
| tensor | candidate | status | q4 GB/s | device Q4 GB/s | ms | gemv | opts |
|---|---|---|---:|---:|---:|---:|---|
| blk.0.ffn_gate.weight | fused_graph | pass | 8.22 | 87.72 | 3.445 |  |  |
| blk.0.ffn_gate.weight | local8_p1 | pass | 76.51 | 187.65 | 0.37 | 0.00179243 | LOCAL:0:8 |
| blk.0.ffn_gate.weight | local16_p1 | pass | 92.08 | 313.54 | 0.307 | 0.00179243 | LOCAL:0:16 |
| blk.0.ffn_gate.weight | local32_p1 | pass | 97.78 | 383.0 | 0.29 | 0.00179243 | LOCAL:0:32 |
| blk.0.ffn_gate.weight | local64_p1 | pass | 100.05 | 420.05 | 0.283 | 0.00179243 | LOCAL:0:64 |
| blk.0.ffn_gate.weight | local16_p2 | pass | 124.43 | 177.95 | 0.228 | 0.00179029 | LOCAL:0:16 |
| blk.0.ffn_gate.weight | local32_p2 | pass | 126.59 | 182.72 | 0.224 | 0.00179029 | LOCAL:0:32 |
| blk.0.ffn_gate.weight | local32_p4 | pass | 101.8 | 136.14 | 0.278 | 0.00179052 | LOCAL:0:32 |
| blk.0.ffn_gate.weight | local32_upcast2_p1 | pass | 78.81 | 209.91 | 0.359 | 0.00179243 | LOCAL:0:32 UPCAST:0:2 |
| blk.0.ffn_gate.weight | local32_upcast3_p1 | pass | 70.71 | 157.97 | 0.4 | 0.00179243 | LOCAL:0:32 UPCAST:0:3 |
| blk.0.attn_q.weight | fused_graph | pass | 3.64 | 15.54 | 2.593 |  |  |
| blk.0.attn_q.weight | local8_p1 | pass | 31.14 | 111.49 | 0.303 | 0.00238562 | LOCAL:0:8 |
| blk.0.attn_q.weight | local16_p1 | pass | 33.96 | 167.85 | 0.278 | 0.00238562 | LOCAL:0:16 |
| blk.0.attn_q.weight | local32_p1 | pass | 35.15 | 175.1 | 0.269 | 0.00238562 | LOCAL:0:32 |
| blk.0.attn_q.weight | local64_p1 | pass | 36.0 | 188.77 | 0.262 | 0.00238562 | LOCAL:0:64 |
| blk.0.attn_q.weight | local16_p2 | pass | 68.82 | 136.88 | 0.137 | 0.00238013 | LOCAL:0:16 |
| blk.0.attn_q.weight | local32_p2 | pass | 74.54 | 167.06 | 0.127 | 0.00238013 | LOCAL:0:32 |
| blk.0.attn_q.weight | local32_p4 | pass | 73.3 | 162.54 | 0.129 | 0.00238061 | LOCAL:0:32 |
| blk.0.attn_q.weight | local32_upcast2_p1 | pass | 29.8 | 96.51 | 0.317 | 0.00238562 | LOCAL:0:32 UPCAST:0:2 |
| blk.0.attn_q.weight | local32_upcast3_p1 | illegal-opt |  |  |  |  | LOCAL:0:32 UPCAST:0:3 |
| blk.4.ffn_down.weight | fused_graph | pass | 7.49 | 16.17 | 3.781 |  |  |
| blk.4.ffn_down.weight | local8_p1 | pass | 65.18 | 132.22 | 0.434 | 0.00208139 | LOCAL:0:8 |
| blk.4.ffn_down.weight | local16_p1 | pass | 77.19 | 182.29 | 0.367 | 0.00208139 | LOCAL:0:16 |
| blk.4.ffn_down.weight | local32_p1 | pass | 77.78 | 196.71 | 0.364 | 0.00208139 | LOCAL:0:32 |
| blk.4.ffn_down.weight | local64_p1 | pass | 80.51 | 212.24 | 0.352 | 0.00208139 | LOCAL:0:64 |
| blk.4.ffn_down.weight | local16_p2 | pass | 129.3 | 188.93 | 0.219 | 0.00208473 | LOCAL:0:16 |
| blk.4.ffn_down.weight | local32_p2 | pass | 156.6 | 252.62 | 0.181 | 0.00208473 | LOCAL:0:32 |
| blk.4.ffn_down.weight | local32_p4 | pass | 160.83 | 268.78 | 0.176 | 0.00207901 | LOCAL:0:32 |
| blk.4.ffn_down.weight | local32_upcast2_p1 | pass | 57.41 | 101.38 | 0.493 | 0.00208139 | LOCAL:0:32 UPCAST:0:2 |
| blk.4.ffn_down.weight | local32_upcast3_p1 | illegal-opt |  |  |  |  | LOCAL:0:32 UPCAST:0:3 |

| tensor | shape | fused | best primitive | ratio | choice |
|---|---:|---:|---:|---:|---|
| blk.0.ffn_gate.weight | 12288x4096 | 87.72 | 420.05 (local64_p1) | 4.788531691746466 | local64_p1 |
| blk.0.attn_q.weight | 4096x4096 | 15.54 | 188.77 (local64_p1) | 12.147361647361649 | local64_p1 |
| blk.4.ffn_down.weight | 4096x12288 | 16.17 | 268.78 (local32_p4) | 16.622139764996906 | local32_p4 |
