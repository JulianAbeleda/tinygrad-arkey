# NV Q4_K warp-ownership static audit — 2026-08-05

## Question

Does the installed Q4_K G3 decode GEMV lose useful CUDA parallelism relative to llama.cpp MMVQ by assigning too little workgroup width to one output row? This is a CPU/static audit only. It neither changes a default route nor makes a performance claim.

## Grounded comparison

The installed tinygrad body is `tinygrad/llm/decode_kernels.py:q4k_g3_lanemap_gemv_kernel`. At `K=4096` it launches `(global=rows, local=32)`: one warp/output row. `lane//8` owns one of four disjoint four-Q4_K-block stripes, `lane%8` one packed word in every one of the stripe's eight 32-value groups. Thus every lane does `4 blocks * 8 groups * 4 values = 128` fp16 products and warp-reduces one row.

The local checked llama.cpp source is `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`. On the RTX 5090's Blackwell path `get_device_table_id()` falls through to `MMVQ_PARAMETERS_GENERIC`. Its `calc_nwarps(Q4_K, ncols_dst=1, GENERIC)` is 4 and `calc_rows_per_block(1, GENERIC)` is 1, so MMVQ launches `dim3(warp_size, 4, 1)`: four warps per output row. The Q4_K MMVQ fragment uses `VDR_Q4_K_Q8_1_MMVQ=2` and Q8_1 activation; distributing 4096 scalar values over 128 threads is 32 values/thread. It is not merely a spelling difference: its Q8 representation and DP4A fragment are also different.

| body | threads/output | K blocks/warp | scalar values/lane | activation |
| --- | ---: | ---: | ---: | --- |
| tinygrad installed G3 | 32 | 16 collectively | 128 | fp16 |
| llama generic MMVQ Q4_K | 128 | 4 | 32 | Q8_1 + DP4A |

This establishes a real ownership asymmetry for attention Q/O and FFN-down, the production Q4_K roles with `K=4096`. It does not prove a speedup: four warps can cost more registers, synchronization, and memory issue traffic. This is reinforced by the Q6 four-warp included-cost gate, which was flat/slightly slower (`+0.185 us`): symmetry alone is not a Q4 admission argument.

## Research construction and static gate

`extra/llm_research/decode/q4k_warp_ownership_static.py` adds a closed, research-only fp16-input witness. It uses one flat 128-thread LOCAL axis, four four-block warp stripes, and writes `[row, warp]` partials. Its pure coordinate witness covers every K element exactly once and changes per-lane work from 128 to 32 values. No runtime route imports this file.

The sm_120 render test is `test/unit/test_q4k_warp_ownership_static.py`; it passed together with the Q6 ownership mapping tests (7 passed). NVRTC PTX counts are:

| variant | local size | static `ld.global` | `shfl.sync` | fp32 FMA |
| --- | ---: | ---: | ---: | ---: |
| installed Q4 G3 | 32 | 40 | 5 | 32 |
| flat-LOCAL witness | 128 | 40 | 5 | 28 |

The 40 global-load instructions remain **per thread** in the 128-thread witness. The control-masked static group selection expands each possible Q4 group instead of expressing the packed dynamic MMVQ fragment. Consequently the witness would issue roughly four times the body-wide static load footprint per output row (`128*40` versus `32*40`) before considering its partial consumer. It is therefore explicitly **not GPU-authorized**.

## Verdict and next precise move

The ownership hypothesis is TRUE; the available naïve UOp construction is not a valid implementation of it. The next admissible Q4 experiment must first render a dynamic packed-address fragment where each of the 128 lanes loads only its two owned words per Q4 block, with the Q8_1/DP4A ABI represented explicitly or separately justified. Its pre-GPU static gate is strict:

1. exact 4096-element ownership and four partials per row;
2. `local_size=(128,1,1)` without factorization;
3. body-wide static load issue not greater than installed G3 after multiplying the per-thread PTX count by local width; and
4. an explicit numerical contract for the changed Q8_1/reduction order.

Until that exists, do not time, promote, or book this route. This leaves the previous vector-carrier spelling result closed: lane ownership and packed fragment representation are the distinct remaining substrate question.
