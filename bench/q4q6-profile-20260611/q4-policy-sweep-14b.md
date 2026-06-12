=== blk.0.ffn_gate.weight (17408, 5120) fused_graph ===
pass device_q4_eff_gbs=58.25
=== blk.0.ffn_gate.weight (17408, 5120) local8_p1 ===
pass device_q4_eff_gbs=189.37 opts=['LOCAL:0:8'] parts=1
=== blk.0.ffn_gate.weight (17408, 5120) local16_p1 ===
pass device_q4_eff_gbs=325.65 opts=['LOCAL:0:16'] parts=1
=== blk.0.ffn_gate.weight (17408, 5120) local32_p1 ===
pass device_q4_eff_gbs=369.97 opts=['LOCAL:0:32'] parts=1
=== blk.0.ffn_gate.weight (17408, 5120) local64_p1 ===
pass device_q4_eff_gbs=362.37 opts=['LOCAL:0:64'] parts=1
=== blk.0.ffn_gate.weight (17408, 5120) local16_p2 ===
pass device_q4_eff_gbs=198.61 opts=['LOCAL:0:16'] parts=2
=== blk.0.ffn_gate.weight (17408, 5120) local32_p2 ===
pass device_q4_eff_gbs=188.29 opts=['LOCAL:0:32'] parts=2
=== blk.0.ffn_gate.weight (17408, 5120) local32_p4 ===
pass device_q4_eff_gbs=144.55 opts=['LOCAL:0:32'] parts=4
=== blk.0.ffn_gate.weight (17408, 5120) local32_upcast2_p1 ===
pass device_q4_eff_gbs=238.97 opts=['LOCAL:0:32', 'UPCAST:0:2'] parts=1
=== blk.0.ffn_gate.weight (17408, 5120) local32_upcast3_p1 ===
illegal-opt device_q4_eff_gbs=None opts=['LOCAL:0:32', 'UPCAST:0:3'] parts=1
=== blk.0.attn_q.weight (5120, 5120) fused_graph ===
pass device_q4_eff_gbs=19.37
=== blk.0.attn_q.weight (5120, 5120) local8_p1 ===
pass device_q4_eff_gbs=136.41 opts=['LOCAL:0:8'] parts=1
=== blk.0.attn_q.weight (5120, 5120) local16_p1 ===
pass device_q4_eff_gbs=197.7 opts=['LOCAL:0:16'] parts=1
=== blk.0.attn_q.weight (5120, 5120) local32_p1 ===
pass device_q4_eff_gbs=210.63 opts=['LOCAL:0:32'] parts=1
=== blk.0.attn_q.weight (5120, 5120) local64_p1 ===
pass device_q4_eff_gbs=247.11 opts=['LOCAL:0:64'] parts=1
=== blk.0.attn_q.weight (5120, 5120) local16_p2 ===
pass device_q4_eff_gbs=181.42 opts=['LOCAL:0:16'] parts=2
=== blk.0.attn_q.weight (5120, 5120) local32_p2 ===
pass device_q4_eff_gbs=222.89 opts=['LOCAL:0:32'] parts=2
=== blk.0.attn_q.weight (5120, 5120) local32_p4 ===
pass device_q4_eff_gbs=188.19 opts=['LOCAL:0:32'] parts=4
=== blk.0.attn_q.weight (5120, 5120) local32_upcast2_p1 ===
pass device_q4_eff_gbs=122.33 opts=['LOCAL:0:32', 'UPCAST:0:2'] parts=1
=== blk.0.attn_q.weight (5120, 5120) local32_upcast3_p1 ===
illegal-opt device_q4_eff_gbs=None opts=['LOCAL:0:32', 'UPCAST:0:3'] parts=1
=== blk.5.ffn_down.weight (5120, 17408) fused_graph ===
pass device_q4_eff_gbs=21.6
=== blk.5.ffn_down.weight (5120, 17408) local8_p1 ===
pass device_q4_eff_gbs=141.94 opts=['LOCAL:0:8'] parts=1
=== blk.5.ffn_down.weight (5120, 17408) local16_p1 ===
pass device_q4_eff_gbs=211.88 opts=['LOCAL:0:16'] parts=1
=== blk.5.ffn_down.weight (5120, 17408) local32_p1 ===
pass device_q4_eff_gbs=225.55 opts=['LOCAL:0:32'] parts=1
=== blk.5.ffn_down.weight (5120, 17408) local64_p1 ===
pass device_q4_eff_gbs=269.93 opts=['LOCAL:0:64'] parts=1
=== blk.5.ffn_down.weight (5120, 17408) local16_p2 ===
pass device_q4_eff_gbs=243.65 opts=['LOCAL:0:16'] parts=2
=== blk.5.ffn_down.weight (5120, 17408) local32_p2 ===
pass device_q4_eff_gbs=318.78 opts=['LOCAL:0:32'] parts=2
=== blk.5.ffn_down.weight (5120, 17408) local32_p4 ===
pass device_q4_eff_gbs=248.75 opts=['LOCAL:0:32'] parts=4
=== blk.5.ffn_down.weight (5120, 17408) local32_upcast2_p1 ===
pass device_q4_eff_gbs=127.39 opts=['LOCAL:0:32', 'UPCAST:0:2'] parts=1
=== blk.5.ffn_down.weight (5120, 17408) local32_upcast3_p1 ===
illegal-opt device_q4_eff_gbs=None opts=['LOCAL:0:32', 'UPCAST:0:3'] parts=1
| tensor | candidate | status | q4 GB/s | device Q4 GB/s | ms | gemv | opts |
|---|---|---|---:|---:|---:|---:|---|
| blk.0.ffn_gate.weight | fused_graph | pass | 17.11 | 58.25 | 2.929 |  |  |
| blk.0.ffn_gate.weight | local8_p1 | pass | 103.91 | 189.37 | 0.482 | 0.00317574 | LOCAL:0:8 |
| blk.0.ffn_gate.weight | local16_p1 | pass | 133.46 | 325.65 | 0.376 | 0.00317574 | LOCAL:0:16 |
| blk.0.ffn_gate.weight | local32_p1 | pass | 141.39 | 369.97 | 0.355 | 0.00317574 | LOCAL:0:32 |
| blk.0.ffn_gate.weight | local64_p1 | pass | 138.7 | 362.37 | 0.361 | 0.00317574 | LOCAL:0:64 |
| blk.0.ffn_gate.weight | local16_p2 | pass | 155.96 | 198.61 | 0.321 | 0.00317621 | LOCAL:0:16 |
| blk.0.ffn_gate.weight | local32_p2 | pass | 148.46 | 188.29 | 0.338 | 0.00317621 | LOCAL:0:32 |
| blk.0.ffn_gate.weight | local32_p4 | pass | 120.4 | 144.55 | 0.416 | 0.00317287 | LOCAL:0:32 |
| blk.0.ffn_gate.weight | local32_upcast2_p1 | pass | 115.41 | 238.97 | 0.434 | 0.00317574 | LOCAL:0:32 UPCAST:0:2 |
| blk.0.ffn_gate.weight | local32_upcast3_p1 | illegal-opt |  |  |  |  | LOCAL:0:32 UPCAST:0:3 |
| blk.0.attn_q.weight | fused_graph | pass | 5.27 | 19.37 | 2.797 |  |  |
| blk.0.attn_q.weight | local8_p1 | pass | 44.55 | 136.41 | 0.331 | 0.00561523 | LOCAL:0:8 |
| blk.0.attn_q.weight | local16_p1 | pass | 51.0 | 197.7 | 0.289 | 0.00561523 | LOCAL:0:16 |
| blk.0.attn_q.weight | local32_p1 | pass | 51.44 | 210.63 | 0.287 | 0.00561523 | LOCAL:0:32 |
| blk.0.attn_q.weight | local64_p1 | pass | 51.88 | 247.11 | 0.284 | 0.00561523 | LOCAL:0:64 |
| blk.0.attn_q.weight | local16_p2 | pass | 97.47 | 181.42 | 0.151 | 0.00559807 | LOCAL:0:16 |
| blk.0.attn_q.weight | local32_p2 | pass | 108.54 | 222.89 | 0.136 | 0.00559807 | LOCAL:0:32 |
| blk.0.attn_q.weight | local32_p4 | pass | 98.98 | 188.19 | 0.149 | 0.0056076 | LOCAL:0:32 |
| blk.0.attn_q.weight | local32_upcast2_p1 | pass | 43.79 | 122.33 | 0.337 | 0.00561523 | LOCAL:0:32 UPCAST:0:2 |
| blk.0.attn_q.weight | local32_upcast3_p1 | illegal-opt |  |  |  |  | LOCAL:0:32 UPCAST:0:3 |
| blk.5.ffn_down.weight | fused_graph | pass | 11.49 | 21.6 | 4.363 |  |  |
| blk.5.ffn_down.weight | local8_p1 | pass | 86.11 | 141.94 | 0.582 | 0.00282097 | LOCAL:0:8 |
| blk.5.ffn_down.weight | local16_p1 | pass | 108.67 | 211.88 | 0.461 | 0.00282097 | LOCAL:0:16 |
| blk.5.ffn_down.weight | local32_p1 | pass | 113.31 | 225.55 | 0.442 | 0.00282097 | LOCAL:0:32 |
| blk.5.ffn_down.weight | local64_p1 | pass | 122.67 | 269.93 | 0.409 | 0.00282097 | LOCAL:0:64 |
| blk.5.ffn_down.weight | local16_p2 | pass | 181.42 | 243.65 | 0.276 | 0.00282812 | LOCAL:0:16 |
| blk.5.ffn_down.weight | local32_p2 | pass | 221.87 | 318.78 | 0.226 | 0.00282812 | LOCAL:0:32 |
| blk.5.ffn_down.weight | local32_p4 | pass | 185.24 | 248.75 | 0.271 | 0.00282669 | LOCAL:0:32 |
| blk.5.ffn_down.weight | local32_upcast2_p1 | pass | 80.56 | 127.39 | 0.622 | 0.00282097 | LOCAL:0:32 UPCAST:0:2 |
| blk.5.ffn_down.weight | local32_upcast3_p1 | illegal-opt |  |  |  |  | LOCAL:0:32 UPCAST:0:3 |

| tensor | shape | fused | best primitive | ratio | choice |
|---|---:|---:|---:|---:|---|
| blk.0.ffn_gate.weight | 17408x5120 | 58.25 | 369.97 (local32_p1) | 6.351416309012876 | local32_p1 |
| blk.0.attn_q.weight | 5120x5120 | 19.37 | 247.11 (local64_p1) | 12.757356737222509 | local64_p1 |
| blk.5.ffn_down.weight | 5120x17408 | 21.6 | 318.78 (local32_p2) | 14.758333333333331 | local32_p2 |
