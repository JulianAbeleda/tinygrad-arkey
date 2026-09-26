| role | M tok | roofline | vLLM | ours prod (hi/lo) | ours/vLLM | ours plain bf16 | production route (model limiter) | best derived measured @2M rows | ours NCU top stalls (TP%) | vLLM NCU (TP%) | vLLM pick in derived space |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ssm_in | 8 | 65.0 | 72.0 | 98.0 | 1.36 | 118.0 | fp32 matvec (rows<=16) | - | long_sb 94, wait 2 (0) | long_sb 82, wait 9 (12) | ragged_k |
| ssm_in | 16 | 65.2 | 72.7 | 166.9 | 2.3 | 117.7 | fp32 matvec (rows<=16) | - | long_sb 93, wait 2 (0) | long_sb 81, wait 9 (12) | ragged_k |
| ssm_in | 32 | 65.6 | 73.5 | 83.2 | 1.13 | 126.9 | 64x128x64 w2x2 s1 p3r (grid<sms) | - | math_pipe 44, wait 27 (48) | math_pipe 33, wait 28 (47) | admitted |
| ssm_in | 64 | 66.4 | 74.0 | 132.0 | 1.78 | 82.1 | 128x128x32 w2x4 s1 p4r (grid<sms) | 124.64 (128x64x64 w2x2 s1 p2r) | math_pipe 55, barrier 16 (58) | math_pipe 32, wait 28 (47) | admitted |
| ssm_in | 128 | 68.0 | 91.4 | 220.4 | 2.41 | 132.3 | 128x128x32 w2x4 s1 p4r (waves) | 209.63 (256x128x32 w4x2 s1 p4r) | math_pipe 60, wait 15 (67) | math_pipe 50, wait 42 (74) | admitted |
| ssm_in | 512 | 237.1 | 296.6 | 1016.8 | 3.43 | 523.3 | 2x 512-row sync2 chunks | 677.54 (128x64x64 w1x4 s1 p2r) | math_pipe 44, mio 32 (57) | math_pipe 51, wait 43 (87) | admitted |
| ssm_in | 1024 | 474.1 | 546.4 | 2214.8 | 4.05 | 1049.1 | 4x 512-row sync2 chunks | 1205.15 (128x128x64 w2x2 s1 p3r) | math_pipe 44, mio 32 (57) | math_pipe 52, wait 44 (93) | admitted |
| ssm_out | 8 | 28.6 | 44.0 | 46.0 | 1.05 | 37.0 | fp32 matvec (rows<=16) | - | long_sb 85, lg 10 (0) | long_sb 60, wait 22 (8) | admitted |
| ssm_out | 16 | 28.6 | 44.0 | 83.8 | 1.9 | 35.5 | fp32 matvec (rows<=16) | - | long_sb 65, lg 29 (0) | long_sb 59, wait 22 (8) | admitted |
| ssm_out | 32 | 28.9 | 34.5 | 52.3 | 1.52 | 64.0 | 64x64x32 w2x2 s8 p4r (waves) | - | long_sb 40, barrier 25 (37) | long_sb 30, math_pipe 29 (45) | admitted |
| ssm_out | 64 | 29.3 | 36.0 | 73.8 | 2.05 | 52.4 | 128x64x32 w4x2 s3 p2 (grid<sms) | 61.98 (128x64x64 w2x2 s3 p2r) | math_pipe 35, mio 31 (43) | math_pipe 30, long_sb 29 (46) | admitted |
| ssm_out | 128 | 30.1 | 41.3 | 105.1 | 2.55 | 72.9 | 64x64x32 w2x2 s4 p4r (l2) | 110.75 (64x128x64 w1x4 s5 p2r) | math_pipe 57, barrier 14 (80) | math_pipe 49, wait 41 (76) | admitted |
| ssm_out | 512 | 104.0 | 141.8 | 385.5 | 2.72 | 189.9 | 2x 512-row sync2 chunks | 347.01 (64x128x64 w1x4 s2 p2r) | math_pipe 45, mio 29 (60) | math_pipe 48, wait 41 (81) | admitted |
| ssm_out | 1024 | 208.0 | 310.5 | 796.7 | 2.57 | 369.6 | 4x 512-row sync2 chunks | 628.19 (64x128x64 w1x4 s1 p2r) | math_pipe 45, mio 29 (61) | math_pipe 53, wait 44 (77) | admitted |
| attn_q | 8 | 19.0 | 23.5 | 42.4 | 1.81 | 37.9 | fp32 matvec (rows<=16) | - | long_sb 96, wait 2 (0) | long_sb 64, wait 19 (10) | ragged_k |
| attn_q | 16 | 19.1 | 23.5 | 60.0 | 2.56 | 36.7 | fp32 matvec (rows<=16) | - | long_sb 94, wait 2 (0) | long_sb 64, wait 19 (10) | ragged_k |
| attn_q | 32 | 19.3 | 23.9 | 36.8 | 1.54 | 39.3 | 64x128x32 w1x4 s7 p4r (waves) | - | long_sb 32, math_pipe 21 (36) | long_sb 40, math_pipe 25 (41) | ragged_k |
| attn_q | 64 | 19.6 | 24.7 | 39.3 | 1.59 | 36.2 | 64x64x32 w1x4 s1 p4r (l2) | - | math_pipe 40, wait 28 (58) | long_sb 38, math_pipe 26 (42) | ragged_k |
| attn_q | 128 | 20.2 | 24.4 | 58.4 | 2.39 | 40.3 | 64x128x64 w1x4 s1 p4r (compute) | - | math_pipe 56, wait 31 (73) | math_pipe 44, wait 39 (76) | admitted |
| attn_q | 512 | 69.3 | 84.0 | 231.9 | 2.76 | 118.0 | 2x 512-row sync2 chunks | - | math_pipe 46, mio 27 (66) | math_pipe 52, wait 44 (86) | admitted |
| attn_q | 1024 | 138.7 | 165.9 | 513.4 | 3.09 | 231.2 | 4x 512-row sync2 chunks | 418.53 (128x128x64 w2x8 s1 p2r) | math_pipe 46, mio 27 (65) | math_pipe 52, wait 44 (88) | admitted |
| attn_kv | 8 | 3.8 | 7.5 | 38.1 | 5.1 | 26.7 | fp32 matvec (rows<=16) | - | long_sb 97, wait 2 (0) | long_sb 73, wait 10 (7) | ragged_k |
| attn_kv | 16 | 3.9 | 19.1 | 38.2 | 2.0 | 24.8 | fp32 matvec (rows<=16) | - | long_sb 97, wait 2 (0) | long_sb 59, wait 21 (2) | ragged_k |
| attn_kv | 32 | 4.0 | 8.0 | 12.9 | 1.63 | 24.8 | 64x32x32 w2x2 s7 p4r (waves) | - | long_sb 38, math_pipe 19 (28) | long_sb 58, wait 16 (14) | ragged_k |
| attn_kv | 64 | 4.1 | 8.1 | 18.0 | 2.22 | 11.9 | 64x64x32 w2x2 s7 p4r (waves) | - | math_pipe 39, wait 19 (39) | long_sb 25, math_pipe 25 (26) | admitted |
| attn_kv | 128 | 4.4 | 9.5 | 23.1 | 2.45 | 16.8 | 64x64x32 w2x2 s7 p4r (waves) | - | math_pipe 43, wait 13 (57) | math_pipe 30, wait 30 (43) | ragged_k |
| attn_kv | 512 | 13.9 | 23.5 | 71.6 | 3.04 | 35.0 | 2x 512-row sync2 chunks | - | math_pipe 35, long_sb 27 (45) | math_pipe 46, wait 40 (62) | admitted |
| attn_kv | 1024 | 27.7 | 43.0 | 140.0 | 3.26 | 76.0 | 4x 512-row sync2 chunks | 117.47 (64x256x32 w1x8 s1 p2r) | math_pipe 35, long_sb 26 (45) | math_pipe 51, wait 43 (67) | admitted |
| attn_o | 8 | 19.0 | 30.8 | 32.6 | 1.06 | 26.3 | fp32 matvec (rows<=16) | - | long_sb 82, lg 11 (0) | long_sb 59, wait 22 (7) | admitted |
| attn_o | 16 | 19.1 | 31.0 | 56.4 | 1.82 | 24.7 | fp32 matvec (rows<=16) | - | long_sb 76, lg 17 (0) | long_sb 60, wait 21 (7) | admitted |
| attn_o | 32 | 19.3 | 24.9 | 38.1 | 1.53 | 35.3 | 64x64x32 w2x2 s8 p4r (waves) | - | long_sb 39, barrier 23 (35) | long_sb 32, math_pipe 29 (43) | ragged_k |
| attn_o | 64 | 19.6 | 26.1 | 50.6 | 1.94 | 38.1 | 64x64x32 w2x2 s4 p4r (waves) | - | math_pipe 47, wait 14 (60) | long_sb 34, math_pipe 28 (42) | ragged_k |
| attn_o | 128 | 20.2 | 29.9 | 74.8 | 2.5 | 51.9 | 64x64x32 w2x2 s4 p4r (l2) | - | math_pipe 54, barrier 14 (77) | math_pipe 47, wait 40 (70) | ragged_k |
| attn_o | 512 | 69.3 | 98.2 | 250.8 | 2.55 | 141.1 | 2x 512-row sync2 chunks | - | math_pipe 42, mio 25 (66) | math_pipe 50, wait 43 (81) | ragged_k+n_pad_to_tile |
| attn_o | 1024 | 138.7 | 182.0 | 531.4 | 2.92 | 267.0 | 4x 512-row sync2 chunks | 431.04 (64x128x64 w1x4 s1 p2r) | math_pipe 41, mio 26 (67) | math_pipe 50, wait 42 (84) | admitted |
| ffn_up | 8 | 46.6 | 49.8 | 73.3 | 1.47 | 87.1 | fp32 matvec (rows<=16) | - | long_sb 95, lg 1 (0) | long_sb 83, wait 9 (13) | ragged_k |
| ffn_up | 16 | 46.8 | 49.9 | 115.0 | 2.3 | 86.3 | fp32 matvec (rows<=16) | - | long_sb 94, wait 2 (0) | long_sb 82, wait 9 (13) | ragged_k |
| ffn_up | 32 | 47.1 | 53.2 | 65.2 | 1.23 | 94.6 | 64x128x64 w1x4 s1 p2r (grid<sms) | - | math_pipe 54, wait 30 (44) | long_sb 34, barrier 24 (46) | ragged_k |
| ffn_up | 64 | 47.6 | 53.5 | 107.0 | 2.0 | 59.9 | 64x64x64 w2x2 s1 p3r (waves) | 110.4 (64x64x64 w2x2 s1 p3r) | math_pipe 58, wait 20 (53) | math_pipe 45, wait 38 (45) | admitted |
| ffn_up | 128 | 48.8 | 64.0 | 165.2 | 2.58 | 104.1 | 64x64x64 w2x2 s1 p4r (occupancy:lds) | 176.99 (64x64x64 w2x2 s1 p3r) | math_pipe 47, wait 29 (63) | math_pipe 45, wait 40 (68) | ragged_k |
| ffn_up | 512 | 169.9 | 187.1 | 617.6 | 3.3 | 307.0 | 2x 512-row sync2 chunks | 490.18 (64x256x32 w1x8 s1 p2r) | math_pipe 42, mio 25 (70) | math_pipe 51, wait 43 (86) | admitted |
| ffn_up | 1024 | 339.8 | 412.8 | 1341.8 | 3.25 | 639.2 | 4x 512-row sync2 chunks | 914.78 (128x128x32 w2x4 s1 p3r) | math_pipe 42, mio 25 (71) | math_pipe 52, wait 44 (88) | admitted |
| ffn_down | 8 | 46.6 | 70.5 | 73.2 | 1.04 | 58.4 | fp32 matvec (rows<=16) | - | long_sb 89, lg 6 (0) | long_sb 60, wait 22 (8) | admitted |
| ffn_down | 16 | 46.8 | 70.3 | 134.7 | 1.92 | 56.6 | fp32 matvec (rows<=16) | - | long_sb 61, lg 35 (0) | long_sb 61, wait 21 (8) | admitted |
| ffn_down | 32 | 47.1 | 52.7 | 77.6 | 1.47 | 109.0 | 64x64x32 w2x2 s14 p6r (waves) | - | mio 30, long_sb 19 (34) | math_pipe 31, long_sb 27 (49) | ragged_k |
| ffn_down | 64 | 47.6 | 53.7 | 96.1 | 1.79 | 80.3 | 64x64x64 w2x2 s4 p3r (waves) | 108.29 (64x64x64 w4x2 s4 p2r) | math_pipe 60, wait 21 (67) | math_pipe 31, long_sb 27 (49) | ragged_k |
| ffn_down | 128 | 48.8 | 58.7 | 160.6 | 2.74 | 94.2 | 64x64x32 w2x2 s4 p4r (l2) | 178.94 (64x64x64 w2x2 s4 p4r) | math_pipe 59, barrier 14 (83) | math_pipe 51, wait 43 (79) | ragged_k |
| ffn_down | 512 | 169.9 | 214.7 | 646.3 | 3.01 | 316.8 | 2x 512-row sync2 chunks | 529.79 (64x128x64 w1x4 s2 p2r) | math_pipe 44, mio 24 (70) | math_pipe 49, wait 42 (87) | ragged_k |
| ffn_down | 1024 | 339.8 | 424.3 | 1315.7 | 3.1 | 621.1 | 4x 512-row sync2 chunks | 979.87 (64x128x64 w2x8 s1 p2r) | math_pipe 44, mio 24 (70) | math_pipe 49, wait 42 (88) | ragged_k |
| output | 8 | 486.8 | 490.0 | 543.6 | 1.11 | 833.7 | fp32 matvec (rows<=16) | - | long_sb 92, barrier 3 (0) | long_sb 85, wait 8 (13) | ragged_k |
| output | 16 | 488.0 | 493.8 | 1072.3 | 2.17 | 840.7 | fp32 matvec (rows<=16) | - | long_sb 93, barrier 3 (0) | long_sb 68, lg 13 (13) | single_stage |
| output | 32 | 490.6 | 523.7 | 551.3 | 1.05 | 898.5 | 64x64x64 w2x2 s1 p6r (occupancy:lds) | - | math_pipe 33, wait 23 (50) | long_sb 35, math_pipe 25 (46) | admitted |
| output | 64 | 495.6 | 529.5 | 681.3 | 1.29 | 516.7 | 128x64x64 w2x2 s1 p4r (occupancy:lds) | 656.64 (128x64x64 w2x2 s1 p4r) | math_pipe 56, wait 32 (79) | long_sb 35, math_pipe 25 (45) | admitted |
| output | 128 | 505.8 | 553.3 | 1317.3 | 2.38 | 639.0 | 128x64x64 w2x2 s1 p4r (compute) | 1239.46 (128x64x64 w2x2 s1 p4r) | math_pipe 56, wait 32 (80) | math_pipe 49, wait 41 (88) | admitted |
