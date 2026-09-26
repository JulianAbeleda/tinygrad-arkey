| role | M | ours us (GEMM+aux) | vLLM us | roofline us | ours grid/regs/occ/TP/DRAM | ref grid/regs/occ/TP/DRAM | ours top stalls | gap us | known buckets | UNEXPLAINED us |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_kv | 8 | 39.04 (39.04+0.0) | 10.208 | 3.832 | (128, 1, 1) / 56 / 8% / 0% / 0.167 | (8, 8, 13) / 124 / 9% / 7% / 0.917 | long_scoreboard 97%, wait 2%, no_instruction 0% | 28.832 | aux_kernels=-3.136, wave_tail=9.495, stall_excess:long_scoreboard=7.024, stall_excess:no_instruction=0.112 | 15.337 |
| attn_kv | 16 | 39.424 (39.424+0.0) | 20.288 | 3.872 | (128, 2, 1) / 54 / 13% / 0% / 0.168 | (8, 8, 1) / 124 / 2% / 2% / 0.322 | long_scoreboard 97%, wait 2%, no_instruction 1% | 19.136 | rows=3.336, wave_tail=-2.91, stall_excess:long_scoreboard=9.839, stall_excess:no_instruction=0.144 | 8.727 |
| attn_kv | 32 | 13.056 (7.68+5.376) | 9.888 | 3.95 | (32, 7, 1) / 79 / 11% / 28% / 0.889 | (8, 4, 5) / 84 / 8% / 14% / 0.943 | long_scoreboard 38%, math_pipe_throttle 19%, barrier 15% | 3.168 | aux_kernels=2.528, rows=1.42, wave_tail=2.206, stall_excess:barrier=0.561, stall_excess:math_pipe_throttle=0.676 | -4.224 |
| attn_kv | 64 | 17.696 (10.88+6.816) | 10.688 | 4.107 | (16, 2, 7) / 100 / 11% / 39% / 0.665 | (8, 2, 7) / 88 / 8% / 26% / 0.911 | math_pipe_throttle 39%, wait 19%, long_scoreboard 16% | 7.008 | aux_kernels=3.648, rows=2.516, wave_tail=1.146, stall_excess:math_pipe_throttle=0.661 | -0.963 |
| attn_kv | 128 | 26.24 (16.352+9.888) | 12.544 | 4.422 | (16, 4, 7) / 100 / 20% / 57% / 0.491 | (16, 2, 5) / 96 / 8% / 43% / 0.771 | math_pipe_throttle 43%, wait 13%, long_scoreboard 12% | 13.696 | aux_kernels=6.752, rows=5.168, wave_tail=1.434, stall_excess:math_pipe_throttle=1.168 | -0.827 |
| attn_kv | 512 | 95.04 (40.128+54.912) | 29.376 | 13.869 | (16, 8, 1) / 122 / 8% / 45% / 0.24 | (64, 2, 1) / 96 / 8% / 62% / 0.329 | math_pipe_throttle 35%, long_scoreboard 27%, wait 26% | 65.664 | aux_kernels=54.912, rows=19.226, wave_tail=2.656, stall_excess:long_scoreboard=2.511 | -13.641 |
| attn_kv | 1024 | 193.888 (38.912+154.976) | 53.76 | 27.738 | (16, 8, 1) / 122 / 8% / 45% / 0.248 | (64, 2, 1) / 158 / 8% / 67% / 0.24 | math_pipe_throttle 35%, long_scoreboard 26%, wait 26% | 140.128 | aux_kernels=154.976, rows=17.568, wave_tail=-3.668, stall_excess:long_scoreboard=2.728 | -31.475 |
| attn_o | 8 | 37.696 (37.696+0.0) | 32.96 | 19.043 | (784, 1, 1) / 54 / 37% / 0% / 0.856 | (8, 25, 1) / 124 / 2% / 7% / 0.977 | long_scoreboard 82%, lg_throttle 11%, wait 2% | 4.736 | wave_tail=-10.645, stall_excess:long_scoreboard=7.793, stall_excess:lg_throttle=3.819 | 3.769 |
| attn_o | 16 | 103.68 (103.68+0.0) | 33.568 | 19.121 | (784, 2, 1) / 48 / 69% / 0% / 0.313 | (8, 25, 1) / 124 / 2% / 7% / 0.962 | long_scoreboard 76%, lg_throttle 17%, mio_throttle 2% | 70.112 | rows=1.142, wave_tail=-5.772, stall_excess:mio_throttle=2.077, stall_excess:long_scoreboard=15.657, stall_excess:lg_throttle=16.348 | 40.659 |
| attn_o | 32 | 40.512 (31.584+8.928) | 27.84 | 19.277 | (50, 8, 1) / 99 / 19% / 35% / 1.058 | (8, 4, 6) / 158 / 8% / 43% / 1.325 | long_scoreboard 39%, barrier 23%, math_pipe_throttle 16% | 12.672 | aux_kernels=5.6, rows=3.38, wave_tail=-3.858, stall_excess:barrier=4.86, stall_excess:long_scoreboard=1.343 | 1.347 |
| attn_o | 64 | 50.176 (41.44+8.736) | 30.048 | 19.589 | (50, 2, 4) / 100 / 18% / 60% / 0.823 | (8, 4, 6) / 158 / 8% / 42% / 1.313 | math_pipe_throttle 47%, wait 14%, barrier 13% | 20.128 | aux_kernels=3.68, rows=9.595, wave_tail=-1.941, stall_excess:barrier=3.035, stall_excess:math_pipe_throttle=4.296 | 1.463 |
| attn_o | 128 | 80.352 (66.624+13.728) | 38.72 | 20.213 | (50, 4, 4) / 100 / 21% / 77% / 0.531 | (16, 4, 3) / 158 / 8% / 70% / 1.006 | math_pipe_throttle 54%, barrier 14%, wait 13% | 41.632 | aux_kernels=8.288, rows=21.087, wave_tail=-10.568, stall_excess:barrier=5.742, stall_excess:math_pipe_throttle=3.002 | 14.081 |
| attn_o | 512 | 309.12 (138.912+170.208) | 114.272 | 69.345 | (50, 8, 2) / 122 / 15% / 66% / 0.274 | (64, 2, 3) / 230 / 8% / 81% / 0.328 | math_pipe_throttle 42%, mio_throttle 25%, wait 17% | 194.848 | aux_kernels=170.208, rows=69.456, wave_tail=-20.061, stall_excess:mio_throttle=15.414 | -40.169 |
| attn_o | 1024 | 629.824 (136.864+492.96) | 201.792 | 138.69 | (50, 8, 2) / 122 / 15% / 67% / 0.278 | (128, 4, 2) / 158 / 8% / 84% / 0.211 | math_pipe_throttle 41%, mio_throttle 26%, wait 16% | 428.032 | aux_kernels=492.96, rows=67.711, wave_tail=-20.098, stall_excess:mio_throttle=15.917 | -128.458 |
| attn_q | 8 | 44.544 (44.544+0.0) | 24.576 | 19.043 | (640, 1, 1) / 56 / 31% / 0% / 0.723 | (8, 40, 1) / 124 / 4% / 10% / 1.309 | long_scoreboard 96%, wait 2%, lg_throttle 1% | 19.968 | wave_tail=1.175, stall_excess:long_scoreboard=13.505, stall_excess:lg_throttle=0.282 | 5.006 |
| attn_q | 16 | 46.08 (46.08+0.0) | 25.472 | 19.121 | (640, 2, 1) / 54 / 62% / 0% / 0.701 | (8, 40, 1) / 124 / 4% / 10% / 1.265 | long_scoreboard 94%, wait 2%, mio_throttle 1% | 20.608 | rows=0.599, wave_tail=1.212, stall_excess:mio_throttle=0.524, stall_excess:long_scoreboard=12.984 | 5.289 |
| attn_q | 32 | 39.84 (29.728+10.112) | 29.216 | 19.277 | (40, 7, 1) / 130 / 13% / 36% / 1.094 | (8, 5, 4) / 158 / 8% / 41% / 1.239 | long_scoreboard 32%, math_pipe_throttle 21%, barrier 15% | 10.624 | aux_kernels=7.008, rows=2.457, wave_tail=3.71, stall_excess:barrier=3.324 | -5.875 |
| attn_q | 64 | 48.0 (41.696+6.304) | 29.408 | 19.589 | (80, 2, 1) / 97 / 8% / 58% / 0.79 | (8, 5, 4) / 158 / 8% / 42% / 1.296 | math_pipe_throttle 40%, wait 28%, long_scoreboard 12% | 18.592 | aux_kernels=2.016, rows=4.011, wave_tail=0.975, stall_excess:math_pipe_throttle=4.905, stall_excess:wait=1.35 | 5.335 |
| attn_q | 128 | 72.128 (63.68+8.448) | 31.808 | 20.213 | (40, 4, 1) / 128 / 8% / 73% / 0.53 | (16, 10, 1) / 96 / 8% / 76% / 1.036 | math_pipe_throttle 56%, wait 31%, mio_throttle 3% | 40.32 | aux_kernels=8.448, rows=23.087, wave_tail=1.875, stall_excess:mio_throttle=1.07, stall_excess:math_pipe_throttle=4.236 | 1.604 |
| attn_q | 512 | 298.56 (129.44+169.12) | 103.296 | 69.345 | (40, 4, 2) / 124 / 28% / 66% / 0.273 | (16, 10, 1) / 230 / 8% / 86% / 0.343 | math_pipe_throttle 46%, mio_throttle 27%, barrier 15% | 195.264 | aux_kernels=169.12, rows=64.72, wave_tail=1.538, stall_excess:mio_throttle=15.257, stall_excess:barrier=8.689 | -64.06 |
| attn_q | 1024 | 630.592 (130.752+499.84) | 188.128 | 138.69 | (40, 4, 2) / 124 / 28% / 65% / 0.271 | (32, 10, 1) / 230 / 8% / 88% / 0.205 | math_pipe_throttle 46%, mio_throttle 27%, barrier 15% | 442.464 | aux_kernels=499.84, rows=63.267, wave_tail=-3.375, stall_excess:mio_throttle=16.006, stall_excess:barrier=9.022 | -142.295 |
| ffn_down | 8 | 105.824 (105.824+0.0) | 76.992 | 46.611 | (784, 1, 1) / 54 / 36% / 0% / 0.749 | (8, 25, 1) / 124 / 2% / 8% / 1.025 | long_scoreboard 89%, lg_throttle 6%, wait 2% | 28.832 | wave_tail=-23.486, stall_excess:long_scoreboard=28.363, stall_excess:lg_throttle=6.206 | 17.749 |
| ffn_down | 16 | 347.136 (347.136+0.0) | 78.336 | 46.759 | (784, 2, 1) / 48 / 66% / 0% / 0.231 | (8, 25, 1) / 124 / 2% / 8% / 1.01 | long_scoreboard 61%, lg_throttle 35%, mio_throttle 2% | 268.8 | rows=2.582, wave_tail=-5.302, stall_excess:mio_throttle=5.406, stall_excess:long_scoreboard=2.077, stall_excess:lg_throttle=109.635 | 154.402 |
| ffn_down | 32 | 93.6 (81.184+12.416) | 56.704 | 47.056 | (50, 14, 1) / 105 / 16% / 34% / 1.065 | (8, 4, 6) / 158 / 8% / 49% / 1.491 | mio_throttle 30%, long_scoreboard 19%, barrier 16% | 36.896 | aux_kernels=9.088, rows=4.238, wave_tail=-8.908, stall_excess:mio_throttle=18.907, stall_excess:barrier=9.805 | 3.765 |
| ffn_down | 64 | 102.24 (92.064+10.176) | 59.2 | 47.648 | (50, 2, 4) / 98 / 14% / 67% / 0.941 | (8, 4, 6) / 158 / 8% / 49% / 1.496 | math_pipe_throttle 60%, wait 21%, long_scoreboard 7% | 43.04 | aux_kernels=4.768, rows=11.486, wave_tail=-3.558, stall_excess:math_pipe_throttle=17.568 | 12.776 |
| ffn_down | 128 | 167.136 (150.336+16.8) | 78.56 | 48.834 | (50, 4, 4) / 100 / 21% / 83% / 0.608 | (16, 4, 3) / 158 / 8% / 79% / 1.127 | math_pipe_throttle 59%, barrier 14%, wait 14% | 88.576 | aux_kernels=11.296, rows=55.635, wave_tail=-22.958, stall_excess:barrier=12.135, stall_excess:math_pipe_throttle=6.882 | 25.585 |
| ffn_down | 512 | 646.848 (299.68+347.168) | 243.328 | 169.895 | (50, 8, 2) / 122 / 16% / 70% / 0.336 | (64, 4, 5) / 158 / 8% / 87% / 0.389 | math_pipe_throttle 44%, mio_throttle 24%, wait 17% | 403.52 | aux_kernels=347.168, rows=145.299, wave_tail=3.315, stall_excess:mio_throttle=33.044 | -125.306 |
| ffn_down | 1024 | 1304.064 (297.952+1006.112) | 455.424 | 339.79 | (50, 8, 2) / 122 / 16% / 70% / 0.338 | (128, 4, 5) / 158 / 8% / 88% / 0.239 | math_pipe_throttle 44%, mio_throttle 24%, wait 17% | 848.64 | aux_kernels=1006.112, rows=148.976, wave_tail=-9.263, stall_excess:mio_throttle=31.593 | -328.777 |
| ffn_up | 8 | 85.056 (85.056+0.0) | 51.264 | 46.611 | (1568, 1, 1) / 56 / 70% / 0% / 0.928 | (8, 98, 1) / 124 / 9% / 13% / 1.537 | long_scoreboard 95%, lg_throttle 1%, wait 1% | 33.792 | wave_tail=2.624, stall_excess:long_scoreboard=9.741, stall_excess:lg_throttle=1.123 | 20.305 |
| ffn_up | 16 | 119.104 (119.104+0.0) | 53.696 | 46.759 | (1568, 2, 1) / 54 / 71% / 0% / 0.67 | (8, 98, 1) / 124 / 9% / 13% / 1.468 | long_scoreboard 94%, wait 2%, mio_throttle 2% | 65.408 | rows=1.338, wave_tail=-0.703, stall_excess:mio_throttle=1.725, stall_excess:long_scoreboard=13.795 | 49.253 |
| ffn_up | 32 | 66.72 (60.608+6.112) | 56.992 | 47.056 | (98, 1, 1) / 133 / 8% / 44% / 1.326 | (8, 25, 5) / 88 / 16% / 46% / 1.39 | math_pipe_throttle 54%, wait 30%, long_scoreboard 6% | 9.728 | aux_kernels=6.112, rows=1.331, wave_tail=24.552, stall_excess:math_pipe_throttle=11.621, stall_excess:wait=10.249 | -44.136 |
| ffn_up | 64 | 117.856 (108.896+8.96) | 56.192 | 47.648 | (196, 2, 1) / 98 / 15% / 53% / 0.757 | (8, 13, 1) / 158 / 8% / 45% / 1.409 | math_pipe_throttle 58%, wait 20%, long_scoreboard 8% | 61.664 | aux_kernels=8.96, rows=13.128, wave_tail=3.38, stall_excess:math_pipe_throttle=9.355 | 26.841 |
| ffn_up | 128 | 183.552 (170.528+13.024) | 79.392 | 48.834 | (196, 4, 1) / 80 / 8% / 63% / 0.497 | (16, 7, 3) / 230 / 8% / 68% / 1.008 | math_pipe_throttle 47%, wait 29%, long_scoreboard 14% | 104.16 | aux_kernels=13.024, rows=68.212, wave_tail=12.307, stall_excess:long_scoreboard=4.877, stall_excess:math_pipe_throttle=1.655 | 4.085 |
| ffn_up | 512 | 628.896 (292.448+336.448) | 235.104 | 169.895 | (196, 8, 1) / 122 / 16% / 70% / 0.337 | (64, 13, 1) / 158 / 8% / 86% / 0.359 | math_pipe_throttle 42%, mio_throttle 25%, wait 16% | 393.792 | aux_kernels=336.448, rows=134.368, wave_tail=17.729, stall_excess:mio_throttle=34.169 | -128.923 |
| ffn_up | 1024 | 1334.592 (290.144+1044.448) | 439.872 | 339.79 | (196, 8, 1) / 122 / 16% / 71% / 0.337 | (32, 25, 1) / 230 / 8% / 88% / 0.209 | math_pipe_throttle 42%, mio_throttle 25%, wait 16% | 894.72 | aux_kernels=1044.448, rows=145.072, wave_tail=-3.346, stall_excess:mio_throttle=30.848 | -322.302 |
| output | 8 | 552.096 (552.096+0.0) | 494.944 | 486.76 | (16384, 1, 1) / 56 / 73% / 0% / 1.504 | (8, 1024, 1) / 124 / 10% / 13% / 1.675 | long_scoreboard 92%, barrier 3%, wait 1% | 57.152 | wave_tail=-4.651, stall_excess:barrier=19.1, stall_excess:long_scoreboard=39.068 | 3.635 |
| output | 16 | 1088.864 (1088.864+0.0) | 498.464 | 488.028 | (16384, 2, 1) / 54 / 74% / 0% / 1.522 | (8, 1024, 1) / 96 / 20% / 13% / 1.666 | long_scoreboard 93%, barrier 3%, wait 2% | 590.4 | rows=11.018, wave_tail=-6.864, stall_excess:barrier=36.782, stall_excess:long_scoreboard=270.34, stall_excess:wait=16.954 | 262.169 |
| output | 32 | 538.848 (514.464+24.384) | 531.456 | 490.565 | (2048, 1, 1) / 80 / 8% / 50% / 1.657 | (8, 128, 1) / 158 / 8% / 46% / 1.571 | math_pipe_throttle 33%, wait 23%, long_scoreboard 21% | 7.392 | aux_kernels=24.384, rows=10.203, wave_tail=-36.424, stall_excess:math_pipe_throttle=37.944, stall_excess:wait=9.186 | -37.902 |
| output | 64 | 697.056 (651.68+45.376) | 538.848 | 495.637 | (2048, 1, 1) / 128 / 8% / 79% / 1.357 | (8, 128, 1) / 158 / 8% / 45% / 1.563 | math_pipe_throttle 56%, wait 32%, long_scoreboard 3% | 158.208 | aux_kernels=45.376, rows=24.863, wave_tail=-27.397, stall_excess:math_pipe_throttle=181.085, stall_excess:wait=62.149 | -127.869 |
| output | 128 | 1328.256 (1234.624+93.632) | 595.36 | 505.782 | (2048, 2, 1) / 128 / 8% / 80% / 1.438 | (16, 128, 1) / 158 / 8% / 88% / 1.437 | math_pipe_throttle 56%, wait 32%, long_scoreboard 4% | 732.896 | aux_kernels=93.632, rows=515.657, wave_tail=1.095, stall_excess:math_pipe_throttle=47.129 | 75.383 |
| output | 512 | 7628.48 (3572.128+4056.352) | 2036.128 | 1775.229 | (1024, 4, 1) / 124 / 33% / 53% / 0.995 | (16, 256, 1) / 230 / 8% / 94% / 0.466 | mio_throttle 47%, math_pipe_throttle 31%, barrier 13% | 5592.352 | aux_kernels=4056.352, rows=1776.81, wave_tail=55.657, stall_excess:mio_throttle=783.714, stall_excess:barrier=217.714 | -1297.895 |
| output | 1024 | 16659.296 (3576.512+13082.784) | 3961.088 | 3550.458 | (1024, 4, 1) / 124 / 33% / 53% / 0.994 | (32, 256, 1) / 230 / 8% / 96% / 0.272 | mio_throttle 47%, math_pipe_throttle 31%, barrier 13% | 12698.208 | aux_kernels=13082.784, rows=1788.256, wave_tail=63.974, stall_excess:mio_throttle=781.527, stall_excess:barrier=217.716 | -3236.049 |
| ssm_in | 8 | 94.336 (94.336+0.0) | 76.768 | 65.03 | (2188, 1, 1) / 56 / 62% / 0% / 1.213 | (8, 137, 1) / 124 / 9% / 12% / 1.49 | long_scoreboard 94%, wait 2%, lg_throttle 1% | 17.568 | wave_tail=-5.125, stall_excess:long_scoreboard=11.66, stall_excess:lg_throttle=1.4 | 9.633 |
| ssm_in | 16 | 144.224 (144.224+0.0) | 76.992 | 65.225 | (2188, 2, 1) / 54 / 72% / 0% / 1.037 | (8, 137, 1) / 124 / 9% / 12% / 1.487 | long_scoreboard 93%, wait 2%, mio_throttle 2% | 67.232 | rows=1.568, wave_tail=-4.646, stall_excess:mio_throttle=2.587, stall_excess:long_scoreboard=17.333 | 50.389 |
| ssm_in | 32 | 86.272 (79.136+7.136) | 76.448 | 65.615 | (137, 1, 1) / 124 / 8% / 48% / 1.505 | (8, 18, 1) / 158 / 8% / 47% / 1.493 | math_pipe_throttle 44%, wait 27%, barrier 12% | 9.824 | aux_kernels=7.136, rows=1.684, wave_tail=3.67, stall_excess:barrier=7.481, stall_excess:math_pipe_throttle=6.466 | -16.613 |
| ssm_in | 64 | 135.136 (125.12+10.016) | 76.896 | 66.395 | (137, 1, 1) / 128 / 17% / 58% / 0.948 | (8, 18, 1) / 158 / 8% / 47% / 1.485 | math_pipe_throttle 55%, barrier 16%, wait 15% | 58.24 | aux_kernels=10.016, rows=9.912, wave_tail=12.527, stall_excess:barrier=14.456, stall_excess:math_pipe_throttle=20.27 | -8.942 |
| ssm_in | 128 | 231.712 (214.592+17.12) | 103.104 | 67.955 | (137, 2, 1) / 128 / 17% / 67% / 0.847 | (16, 18, 1) / 158 / 8% / 74% / 1.121 | math_pipe_throttle 60%, wait 15%, barrier 14% | 128.608 | aux_kernels=17.12, rows=91.968, wave_tail=25.887, stall_excess:barrier=11.048, stall_excess:math_pipe_throttle=7.751 | -25.166 |
| ssm_in | 512 | 991.264 (467.712+523.552) | 323.776 | 237.073 | (137, 4, 1) / 124 / 29% / 57% / 0.531 | (64, 18, 1) / 158 / 8% / 87% / 0.375 | math_pipe_throttle 44%, mio_throttle 32%, barrier 14% | 667.488 | aux_kernels=523.552, rows=233.856, wave_tail=80.452, stall_excess:mio_throttle=45.552, stall_excess:barrier=20.561 | -236.485 |
| ssm_in | 1024 | 2150.56 (469.312+1681.248) | 583.36 | 474.146 | (137, 4, 1) / 124 / 29% / 57% / 0.53 | (128, 18, 1) / 158 / 8% / 93% / 0.23 | math_pipe_throttle 44%, mio_throttle 32%, barrier 14% | 1567.2 | aux_kernels=1681.248, rows=234.656, wave_tail=72.473, stall_excess:mio_throttle=45.824, stall_excess:barrier=20.545 | -487.547 |
| ssm_out | 8 | 59.456 (59.456+0.0) | 49.248 | 28.549 | (784, 1, 1) / 54 / 37% / 0% / 0.814 | (8, 25, 1) / 124 / 2% / 8% / 0.981 | long_scoreboard 85%, lg_throttle 10%, wait 2% | 10.208 | wave_tail=-15.662, stall_excess:long_scoreboard=13.811, stall_excess:lg_throttle=5.68 | 6.379 |
| ssm_out | 16 | 192.608 (192.608+0.0) | 49.088 | 28.651 | (784, 2, 1) / 48 / 66% / 0% / 0.253 | (8, 25, 1) / 124 / 2% / 8% / 0.987 | long_scoreboard 65%, lg_throttle 29%, mio_throttle 3% | 143.52 | rows=1.74, wave_tail=-5.257, stall_excess:mio_throttle=4.454, stall_excess:long_scoreboard=10.166, stall_excess:lg_throttle=51.516 | 80.901 |
| ssm_out | 32 | 53.728 (44.992+8.736) | 38.272 | 28.856 | (50, 8, 1) / 99 / 19% / 37% / 1.114 | (8, 4, 6) / 158 / 8% / 45% / 1.398 | long_scoreboard 40%, barrier 25%, math_pipe_throttle 16% | 15.456 | aux_kernels=5.312, rows=3.466, wave_tail=-5.465, stall_excess:barrier=7.874, stall_excess:long_scoreboard=3.143 | 1.126 |
| ssm_out | 64 | 86.88 (78.016+8.864) | 39.904 | 29.264 | (50, 3, 1) / 76 / 17% / 43% / 0.655 | (8, 4, 6) / 158 / 8% / 46% / 1.412 | math_pipe_throttle 35%, mio_throttle 31%, wait 12% | 46.976 | aux_kernels=3.808, rows=10.415, wave_tail=-5.991, stall_excess:mio_throttle=18.257, stall_excess:math_pipe_throttle=2.635 | 17.852 |
| ssm_out | 128 | 111.072 (96.032+15.04) | 52.192 | 30.082 | (50, 4, 4) / 100 / 21% / 80% / 0.554 | (16, 4, 3) / 158 / 8% / 76% / 1.074 | math_pipe_throttle 57%, barrier 14%, wait 14% | 58.88 | aux_kernels=9.568, rows=30.461, wave_tail=-14.688, stall_excess:barrier=8.462, stall_excess:math_pipe_throttle=4.48 | 20.597 |
| ssm_out | 512 | 460.096 (207.616+252.48) | 160.512 | 104.017 | (25, 4, 3) / 124 / 27% / 60% / 0.279 | (64, 4, 4) / 158 / 8% / 81% / 0.349 | math_pipe_throttle 45%, mio_throttle 29%, barrier 14% | 299.584 | aux_kernels=252.48, rows=102.715, wave_tail=2.035, stall_excess:mio_throttle=23.226, stall_excess:barrier=11.605 | -92.476 |
| ssm_out | 1024 | 925.312 (203.776+721.536) | 325.408 | 208.035 | (25, 4, 3) / 124 / 27% / 61% / 0.283 | (128, 4, 1) / 158 / 8% / 77% / 0.197 | math_pipe_throttle 45%, mio_throttle 29%, barrier 14% | 599.904 | aux_kernels=721.536, rows=101.888, wave_tail=-56.421, stall_excess:mio_throttle=22.684, stall_excess:barrier=11.256 | -201.039 |

## Research queue

Unexplained residual, largest share of our time first:

- ffn_up M=32: -44.136 us unexplained of a 9.728 us gap (-66.1% of ours)
- ffn_down M=16: 154.402 us unexplained of a 268.8 us gap (44.5% of ours)
- ssm_out M=16: 80.901 us unexplained of a 143.52 us gap (42.0% of ours)
- ffn_up M=16: 49.253 us unexplained of a 65.408 us gap (41.3% of ours)
- attn_kv M=8: 15.337 us unexplained of a 28.832 us gap (39.3% of ours)
- attn_o M=16: 40.659 us unexplained of a 70.112 us gap (39.2% of ours)
- ssm_in M=16: 50.389 us unexplained of a 67.232 us gap (34.9% of ours)
- attn_kv M=32: -4.224 us unexplained of a 3.168 us gap (-32.4% of ours)
- ffn_down M=1024: -328.777 us unexplained of a 848.64 us gap (-25.2% of ours)
- ffn_up M=1024: -322.302 us unexplained of a 894.72 us gap (-24.1% of ours)
- output M=16: 262.169 us unexplained of a 590.4 us gap (24.1% of ours)
- ffn_up M=8: 20.305 us unexplained of a 33.792 us gap (23.9% of ours)
- ssm_in M=512: -236.485 us unexplained of a 667.488 us gap (-23.9% of ours)
- ffn_up M=64: 26.841 us unexplained of a 61.664 us gap (22.8% of ours)
- ssm_in M=1024: -487.547 us unexplained of a 1567.2 us gap (-22.7% of ours)

Unknown techniques (reference opcodes the taxonomy does not map):

- UVIMNMX x640 in void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params) (attn_kv M=32)
- B2R x36148 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8>(T1::Params) (attn_o M=512)
- CGAERRBAR x312 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8>(T1::Params) (attn_o M=512)
- B2R x3200 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params) (attn_o M=1024)
- CGAERRBAR x800 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params) (attn_o M=1024)
- B2R x6000 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params) (ffn_down M=512)
- CGAERRBAR x1000 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params) (ffn_down M=512)
- B2R x8000 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params) (ffn_down M=1024)
- CGAERRBAR x2000 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params) (ffn_down M=1024)
- B2R x15588 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params) (ffn_up M=32)
- CGAERRBAR x980 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params) (ffn_up M=32)
- B2R x25460 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8>(T1::Params) (ffn_up M=128)
- CGAERRBAR x294 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8>(T1::Params) (ffn_up M=128)
- UVIMNMX x8192 in void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8>(T1::Params) (output M=16)
- B2R x4456 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params) (ssm_out M=512)
- CGAERRBAR x800 in void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params) (ssm_out M=512)

Technique gaps (reference uses, ours does not):

- async_global_to_shared (attn_kv M=512): reference 221184; lowering fact async_copy = True
- matrix_fragment_load (attn_kv M=512): reference 403456; lowering fact matrix_fragments = True
- async_global_to_shared (attn_kv M=1024): reference 313344; lowering fact async_copy = True
- matrix_fragment_load (attn_kv M=1024): reference 605184; lowering fact matrix_fragments = True
- async_global_to_shared (attn_o M=512): reference 703040; lowering fact async_copy = True
- matrix_fragment_load (attn_o M=512): reference 1074944; lowering fact matrix_fragments = True
- async_global_to_shared (attn_o M=1024): reference 1612800; lowering fact async_copy = True
- matrix_fragment_load (attn_o M=1024): reference 3091200; lowering fact matrix_fragments = True
- async_global_to_shared (attn_q M=512): reference 646400; lowering fact async_copy = True
- matrix_fragment_load (attn_q M=512): reference 1008640; lowering fact matrix_fragments = True
- async_global_to_shared (attn_q M=1024): reference 1292800; lowering fact async_copy = True
- matrix_fragment_load (attn_q M=1024): reference 2017280; lowering fact matrix_fragments = True
- async_global_to_shared (ffn_down M=512): reference 1977600; lowering fact async_copy = True
- matrix_fragment_load (ffn_down M=512): reference 3787200; lowering fact matrix_fragments = True
- async_global_to_shared (ffn_down M=1024): reference 3955200; lowering fact async_copy = True
- matrix_fragment_load (ffn_down M=1024): reference 7574400; lowering fact matrix_fragments = True
- async_global_to_shared (ffn_up M=512): reference 1919232; lowering fact async_copy = True
- matrix_fragment_load (ffn_up M=512): reference 3706752; lowering fact matrix_fragments = True
- async_global_to_shared (ffn_up M=1024): reference 3167360; lowering fact async_copy = True
- matrix_fragment_load (ffn_up M=1024): reference 4942336; lowering fact matrix_fragments = True
- async_global_to_shared (output M=512): reference 16547840; lowering fact async_copy = True
- matrix_fragment_load (output M=512): reference 25821184; lowering fact matrix_fragments = True
- async_global_to_shared (output M=1024): reference 33095680; lowering fact async_copy = True
- matrix_fragment_load (output M=1024): reference 51642368; lowering fact matrix_fragments = True
- async_global_to_shared (ssm_in M=512): reference 2683008; lowering fact async_copy = True
- matrix_fragment_load (ssm_in M=512): reference 5181888; lowering fact matrix_fragments = True
- async_global_to_shared (ssm_in M=1024): reference 5366016; lowering fact async_copy = True
- matrix_fragment_load (ssm_in M=1024): reference 10363776; lowering fact matrix_fragments = True
- async_global_to_shared (ssm_out M=64): reference 158400; lowering fact async_copy = True
- matrix_fragment_load (ssm_out M=64): reference 291600; lowering fact matrix_fragments = True
- async_global_to_shared (ssm_out M=512): reference 1228800; lowering fact async_copy = True
- matrix_fragment_load (ssm_out M=512): reference 2323200; lowering fact matrix_fragments = True
- async_global_to_shared (ssm_out M=1024): reference 2342400; lowering fact async_copy = True
- matrix_fragment_load (ssm_out M=1024): reference 4617600; lowering fact matrix_fragments = True
