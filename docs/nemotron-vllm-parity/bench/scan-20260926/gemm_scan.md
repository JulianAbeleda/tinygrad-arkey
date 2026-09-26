| role | M | GEMM rows | roofline us | vLLM us | ours prod us | ours/vLLM | promoted route (limiter) | promoted us | derived measured best us | derived model best (limiter) | vLLM pick in derived space | deferred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ssm_in | 8 | 16 | 65.03 | 72.02 | 97.96 | 1.36 | 16x64x32 w1x2 s2 p2 (waves) | 107.07 | - | 16x32x64 w1x4 s1 p2r (waves) | ragged_k | ragged_k,stream_k |
| ssm_in | 16 | 32 | 65.23 | 72.69 | 166.91 | 2.3 | 32x128x32 w1x4 s2 p2 (waves) | 118.53 | - | 32x32x64 w2x4 s1 p2r (waves) | ragged_k | ragged_k,stream_k |
| ssm_in | 32 | 64 | 65.62 | 73.47 | 83.18 | 1.13 | 64x128x64 w2x2 s1 p3r (grid<sms) | 76.9 | - | 64x64x64 w2x8 s1 p2r (waves) | admitted | ragged_k,stream_k |
| ssm_in | 64 | 128 | 66.4 | 74.05 | 131.98 | 1.78 | 128x128x32 w2x4 s1 p4r (grid<sms) | 127.62 | 124.64 | 64x128x64 w1x16 s1 p2r (waves) | admitted | ragged_k,stream_k |
| ssm_in | 128 | 256 | 67.96 | 91.36 | 220.45 | 2.41 | 128x128x32 w2x4 s1 p4r (waves) | 220.32 | 209.63 | 64x128x32 w1x8 s2 p2r (compute) | admitted | ragged_k,stream_k |
| ssm_in | 512 | 1024 | 237.07 | 296.57 | 1016.83 | 3.43 | - | - | 677.54 | 64x128x64 w1x16 s1 p2r (compute) | admitted | ragged_k,stream_k |
| ssm_in | 1024 | 2048 | 474.15 | 546.44 | 2214.78 | 4.05 | - | - | 1205.15 | 64x128x64 w1x16 s1 p2r (compute) | admitted | ragged_k,stream_k |
| ssm_in | 2048 | 4096 | 948.29 | 1061.43 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k |
| ssm_in | 4096 | 8192 | 1896.58 | 2087.43 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k |
| ssm_in | 8192 | 16384 | 3793.17 | 3886.46 | - | - | - | - | - | 64x128x32 w1x8 s2 p2r (compute) | admitted | ragged_k |
| ssm_out | 8 | 16 | 28.55 | 43.97 | 46.0 | 1.05 | 16x128x64 w1x4 s4 p2 (grid<sms) | 34.56 | - | 16x16x256 w1x2 s1 p2r (waves) | admitted | ragged_k,stream_k |
| ssm_out | 16 | 32 | 28.65 | 44.04 | 83.8 | 1.9 | 32x64x32 w1x2 s3 p2 (grid<sms) | 59.36 | - | 16x32x256 w1x4 s1 p2r (waves) | admitted | ragged_k,stream_k |
| ssm_out | 32 | 64 | 28.86 | 34.48 | 52.3 | 1.52 | 64x64x32 w2x2 s8 p4r (waves) | 45.73 | - | 32x64x256 w2x8 s2 p2r (waves) | admitted | ragged_k,stream_k |
| ssm_out | 64 | 128 | 29.26 | 36.03 | 73.85 | 2.05 | 128x64x32 w4x2 s3 p2 (grid<sms) | 112.32 | 61.98 | 64x64x128 w4x8 s3 p2r (waves) | admitted | ragged_k,stream_k |
| ssm_out | 128 | 256 | 30.08 | 41.29 | 105.1 | 2.55 | 64x64x32 w2x2 s4 p4r (l2) | 120.51 | 110.75 | 64x128x128 w2x16 s5 p2r (compute) | admitted | ragged_k,stream_k |
| ssm_out | 512 | 1024 | 104.02 | 141.78 | 385.46 | 2.72 | - | - | 347.01 | 64x128x64 w1x16 s8 p2r (compute) | admitted | ragged_k,stream_k |
| ssm_out | 1024 | 2048 | 208.03 | 310.47 | 796.67 | 2.57 | - | - | 628.19 | 64x128x128 w2x16 s4 p2r (compute) | admitted | ragged_k,stream_k |
| ssm_out | 2048 | 4096 | 416.07 | 500.33 | - | - | - | - | - | 64x128x128 w2x16 s12 p2r (compute) | admitted | ragged_k,stream_k |
| ssm_out | 4096 | 8192 | 832.14 | 975.34 | - | - | - | - | - | 64x128x128 w2x16 s6 p2r (compute) | admitted | ragged_k,stream_k |
| ssm_out | 8192 | 16384 | 1664.28 | 1803.89 | - | - | - | - | - | 64x128x128 w2x16 s3 p2r (compute) | admitted | ragged_k,stream_k |
| qkv | 8 | 16 | 26.65 | 29.96 | - | - | - | - | - | 16x32x64 w1x4 s1 p2r (waves) | ragged_k | ragged_k,stream_k |
| qkv | 16 | 32 | 26.75 | 30.02 | - | - | - | - | - | 32x32x64 w2x4 s1 p2r (waves) | ragged_k | ragged_k,stream_k |
| qkv | 32 | 64 | 26.94 | 32.86 | - | - | - | - | - | 32x64x64 w1x8 s1 p2r (waves) | ragged_k | ragged_k,stream_k |
| qkv | 64 | 128 | 27.33 | 33.77 | - | - | - | - | - | 64x64x64 w2x8 s1 p2r (waves) | ragged_k | ragged_k,stream_k |
| qkv | 128 | 256 | 28.11 | 38.55 | - | - | - | - | - | 64x128x32 w1x8 s2 p2r (waves) | admitted | ragged_k,stream_k |
| qkv | 512 | 1024 | 97.08 | 113.43 | - | - | - | - | - | 64x128x32 w1x8 s2 p2r (compute) | admitted | ragged_k,stream_k |
| qkv | 1024 | 2048 | 194.17 | 227.75 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k,stream_k |
| qkv | 2048 | 4096 | 388.33 | 459.37 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k,stream_k |
| qkv | 4096 | 8192 | 776.66 | 884.9 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k |
| qkv | 8192 | 16384 | 1553.33 | 1654.89 | - | - | - | - | - | 64x128x32 w1x8 s2 p2r (compute) | admitted | ragged_k |
| attn_q | 8 | 16 | 19.04 | 23.46 | 42.36 | 1.81 | 16x64x32 w1x2 s2 p2 (memory) | 28.03 | - | 16x16x64 w1x2 s1 p2r (memory) | ragged_k | ragged_k,stream_k |
| attn_q | 16 | 32 | 19.12 | 23.48 | 60.01 | 2.56 | 32x64x32 w1x2 s2 p2 (memory) | 34.85 | - | 16x32x64 w1x4 s1 p2r (memory) | ragged_k | ragged_k,stream_k |
| attn_q | 32 | 64 | 19.28 | 23.86 | 36.81 | 1.54 | 64x128x32 w1x4 s7 p4r (waves) | 33.86 | - | 32x64x64 w1x8 s1 p2r (memory) | ragged_k | ragged_k |
| attn_q | 64 | 128 | 19.59 | 24.7 | 39.26 | 1.59 | 64x64x32 w1x4 s1 p4r (l2) | 42.18 | - | 64x64x64 w2x8 s1 p2r (l2) | ragged_k | ragged_k,stream_k |
| attn_q | 128 | 256 | 20.21 | 24.45 | 58.43 | 2.39 | 64x128x64 w1x4 s1 p4r (compute) | 71.9 | - | 64x128x64 w1x16 s1 p2r (compute) | admitted | ragged_k,stream_k |
| attn_q | 512 | 1024 | 69.34 | 84.04 | 231.88 | 2.76 | - | - | - | 64x128x64 w1x16 s1 p2r (compute) | admitted | ragged_k,stream_k |
| attn_q | 1024 | 2048 | 138.69 | 165.89 | 513.41 | 3.09 | - | - | 418.53 | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k,stream_k |
| attn_q | 2048 | 4096 | 277.38 | 329.19 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k,stream_k |
| attn_q | 4096 | 8192 | 554.76 | 654.2 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k,stream_k |
| attn_q | 8192 | 16384 | 1109.52 | 1206.09 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k,stream_k |
| attn_kv | 8 | 16 | 3.83 | 7.47 | 38.08 | 5.1 | 16x64x32 w1x2 s2 p2 (grid<sms) | 24.99 | - | 16x8x32 w1x1 s2 p2r (waves) | ragged_k | ragged_k,stream_k |
| attn_kv | 16 | 32 | 3.87 | 19.1 | 38.23 | 2.0 | 16x64x32 w1x2 s2 p2 (grid<sms) | 25.54 | - | 16x16x32 w1x2 s2 p2r (waves) | ragged_k | ragged_k,stream_k |
| attn_kv | 32 | 64 | 3.95 | 7.95 | 12.95 | 1.63 | 64x32x32 w2x2 s7 p4r (waves) | 11.42 | - | 32x32x16 w1x2 s4 p2r (waves) | ragged_k | ragged_k,stream_k |
| attn_kv | 64 | 128 | 4.11 | 8.1 | 18.01 | 2.22 | 64x64x32 w2x2 s7 p4r (waves) | 15.74 | - | 32x64x16 w1x2 s4 p2r (waves) | admitted | ragged_k,stream_k |
| attn_kv | 128 | 256 | 4.42 | 9.46 | 23.14 | 2.45 | 64x64x32 w2x2 s7 p4r (waves) | 24.61 | - | 64x64x16 w1x4 s4 p2r (waves) | ragged_k | ragged_k,stream_k |
| attn_kv | 512 | 1024 | 13.87 | 23.52 | 71.58 | 3.04 | - | - | - | 64x64x16 w1x4 s4 p2r (waves) | admitted | ragged_k,stream_k |
| attn_kv | 1024 | 2048 | 27.74 | 42.96 | 140.04 | 3.26 | - | - | 117.47 | 64x128x16 w1x4 s4 p2r (waves) | admitted | ragged_k,stream_k |
| attn_kv | 2048 | 4096 | 55.48 | 82.75 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k,stream_k |
| attn_kv | 4096 | 8192 | 110.95 | 166.98 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k,stream_k |
| attn_kv | 8192 | 16384 | 221.9 | 276.9 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | ragged_k | ragged_k,stream_k |
| attn_o | 8 | 16 | 19.04 | 30.82 | 32.59 | 1.06 | 16x128x64 w1x4 s4 p2 (grid<sms) | 23.42 | - | 16x16x256 w1x2 s1 p2r (waves) | admitted | ragged_k,stream_k |
| attn_o | 16 | 32 | 19.12 | 30.98 | 56.4 | 1.82 | 16x128x64 w1x4 s4 p2 (waves) | 43.2 | - | 16x32x256 w1x4 s1 p2r (waves) | admitted | ragged_k,stream_k |
| attn_o | 32 | 64 | 19.28 | 24.87 | 38.06 | 1.53 | 64x64x32 w2x2 s8 p4r (waves) | 34.3 | - | 32x64x256 w2x8 s2 p2r (waves) | ragged_k | ragged_k,stream_k |
| attn_o | 64 | 128 | 19.59 | 26.08 | 50.6 | 1.94 | 64x64x32 w2x2 s4 p4r (waves) | 53.86 | - | 64x64x128 w4x8 s4 p2r (waves) | ragged_k | ragged_k,stream_k |
| attn_o | 128 | 256 | 20.21 | 29.86 | 74.76 | 2.5 | 64x64x32 w2x2 s4 p4r (l2) | 89.5 | - | 64x128x128 w2x16 s5 p2r (occupancy:lds) | ragged_k | ragged_k,stream_k |
| attn_o | 512 | 1024 | 69.34 | 98.22 | 250.83 | 2.55 | - | - | - | 64x128x128 w2x16 s5 p2r (compute) | ragged_k+n_pad_to_tile | ragged_k,stream_k |
| attn_o | 1024 | 2048 | 138.69 | 181.97 | 531.36 | 2.92 | - | - | 431.04 | 64x128x128 w2x16 s4 p2r (compute) | admitted | ragged_k,stream_k |
| attn_o | 2048 | 4096 | 277.38 | 338.64 | - | - | - | - | - | 64x128x128 w2x16 s2 p2r (compute) | admitted | ragged_k,stream_k |
| attn_o | 4096 | 8192 | 554.76 | 661.66 | - | - | - | - | - | 64x128x128 w2x16 s8 p2r (compute) | admitted | ragged_k,stream_k |
| attn_o | 8192 | 16384 | 1109.52 | 1214.29 | - | - | - | - | - | 64x128x128 w2x16 s4 p2r (compute) | admitted | ragged_k,stream_k |
| ffn_up | 8 | 16 | 46.61 | 49.8 | 73.26 | 1.47 | 16x64x32 w1x2 s2 p2 (waves) | 62.11 | - | 16x32x64 w1x4 s1 p2r (waves) | ragged_k | ragged_k,stream_k |
| ffn_up | 16 | 32 | 46.76 | 49.91 | 114.99 | 2.3 | 32x128x32 w1x4 s2 p2 (waves) | 82.27 | - | 32x32x64 w2x4 s1 p2r (waves) | ragged_k | ragged_k,stream_k |
| ffn_up | 32 | 64 | 47.06 | 53.17 | 65.16 | 1.23 | 64x128x64 w1x4 s1 p2r (grid<sms) | 65.98 | - | 64x64x64 w2x8 s1 p2r (waves) | ragged_k | ragged_k,stream_k |
| ffn_up | 64 | 128 | 47.65 | 53.46 | 106.98 | 2.0 | 64x64x64 w2x2 s1 p3r (waves) | 110.98 | 110.4 | 64x64x64 w2x8 s1 p2r (waves) | admitted | ragged_k,stream_k |
| ffn_up | 128 | 256 | 48.83 | 64.05 | 165.17 | 2.58 | 64x64x64 w2x2 s1 p4r (occupancy:lds) | 177.86 | 176.99 | 64x128x32 w1x8 s2 p2r (compute) | ragged_k | ragged_k,stream_k |
| ffn_up | 512 | 1024 | 169.89 | 187.07 | 617.61 | 3.3 | - | - | 490.18 | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k,stream_k |
| ffn_up | 1024 | 2048 | 339.79 | 412.8 | 1341.76 | 3.25 | - | - | 914.78 | 64x128x32 w1x8 s2 p2r (compute) | admitted | ragged_k,stream_k |
| ffn_up | 2048 | 4096 | 679.58 | 791.23 | - | - | - | - | - | 64x128x64 w1x16 s1 p2r (compute) | admitted | ragged_k,stream_k |
| ffn_up | 4096 | 8192 | 1359.16 | 1515.85 | - | - | - | - | - | 64x128x64 w1x16 s1 p2r (compute) | admitted | ragged_k |
| ffn_up | 8192 | 16384 | 2718.32 | 2781.74 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k |
| ffn_down | 8 | 16 | 46.61 | 70.48 | 73.25 | 1.04 | 16x128x64 w1x4 s4 p2 (grid<sms) | 51.81 | - | 16x16x256 w1x2 s1 p2r (waves) | admitted | ragged_k,stream_k |
| ffn_down | 16 | 32 | 46.76 | 70.32 | 134.71 | 1.92 | 32x64x32 w1x2 s8 p2 (waves) | 89.57 | - | 16x32x256 w1x4 s1 p2r (waves) | admitted | ragged_k,stream_k |
| ffn_down | 32 | 64 | 47.06 | 52.7 | 77.59 | 1.47 | 64x64x32 w2x2 s14 p6r (waves) | 68.26 | - | 32x64x128 w2x8 s2 p2r (waves) | ragged_k | ragged_k,stream_k |
| ffn_down | 64 | 128 | 47.65 | 53.66 | 96.1 | 1.79 | 64x64x64 w2x2 s4 p3r (waves) | 110.24 | 108.29 | 64x64x64 w2x8 s4 p2r (waves) | ragged_k | ragged_k,stream_k |
| ffn_down | 128 | 256 | 48.83 | 58.68 | 160.62 | 2.74 | 64x64x32 w2x2 s4 p4r (l2) | 177.73 | 178.94 | 64x128x32 w1x8 s8 p2r (compute) | ragged_k | ragged_k,stream_k |
| ffn_down | 512 | 1024 | 169.89 | 214.69 | 646.29 | 3.01 | - | - | 529.79 | 64x128x128 w2x16 s14 p2r (compute) | ragged_k | ragged_k,stream_k |
| ffn_down | 1024 | 2048 | 339.79 | 424.31 | 1315.69 | 3.1 | - | - | 979.87 | 64x128x128 w2x16 s7 p2r (compute) | ragged_k | ragged_k,stream_k |
| ffn_down | 2048 | 4096 | 679.58 | 802.91 | - | - | - | - | - | 64x128x128 w2x16 s7 p2r (compute) | admitted | ragged_k,stream_k |
| ffn_down | 4096 | 8192 | 1359.16 | 1555.53 | - | - | - | - | - | 64x128x128 w2x16 s7 p2r (compute) | admitted | ragged_k,stream_k |
| ffn_down | 8192 | 16384 | 2718.32 | 3018.38 | - | - | - | - | - | 64x128x16 w1x4 s16 p2r (compute) | admitted | ragged_k,stream_k |
| output | 8 | 16 | 486.76 | 490.02 | 543.65 | 1.11 | 16x64x32 w1x2 s2 p2 (memory) | 768.93 | - | 16x32x64 w1x4 s1 p2r (memory) | ragged_k | ragged_k,stream_k |
| output | 16 | 32 | 488.03 | 493.8 | 1072.28 | 2.17 | 32x128x32 w1x4 s2 p2 (memory) | 835.04 | - | 32x32x64 w2x4 s1 p2r (memory) | single_stage | ragged_k,stream_k |
| output | 32 | 64 | 490.56 | 523.66 | 551.32 | 1.05 | 64x64x64 w2x2 s1 p6r (occupancy:lds) | 514.98 | - | 64x64x64 w2x8 s1 p2r (memory) | admitted | ragged_k,stream_k |
| output | 64 | 128 | 495.64 | 529.48 | 681.3 | 1.29 | 128x64x64 w2x2 s1 p4r (occupancy:lds) | 655.39 | 656.64 | 64x128x64 w1x16 s1 p2r (occupancy:registers) | admitted | ragged_k,stream_k |
| output | 128 | 256 | 505.78 | 553.3 | 1317.34 | 2.38 | 128x64x64 w2x2 s1 p4r (compute) | 1235.97 | 1239.46 | 64x128x32 w1x8 s2 p2r (compute) | admitted | ragged_k,stream_k |
| output | 512 | 1024 | 1775.23 | 1889.61 | - | - | - | - | - | 64x128x32 w1x8 s2 p2r (compute) | admitted | ragged_k |
| output | 1024 | 2048 | 3550.46 | 3740.64 | - | - | - | - | - | 64x128x64 w1x16 s1 p2r (compute) | admitted | ragged_k |
| output | 2048 | 4096 | 7100.92 | 7351.44 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k |
| output | 4096 | 8192 | 14201.83 | 14591.94 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k |
| output | 8192 | 16384 | 28403.66 | 29334.32 | - | - | - | - | - | 64x128x16 w1x4 s4 p2r (compute) | admitted | ragged_k |
