# vLLM baseline: Nemotron 3 Nano 4B BF16 rollouts on the RTX 5090

Date: 2026-09-25. vLLM is used here only as an oracle and benchmark reference. It is not on the product path.
Everything was measured on this box under `gpu-run time` (exclusive GPU). Raw data, scripts, per-kernel tables and
log digests are in `vllm-nemotron-prof/` next to this file. The venv is `~/vllm-bench/.venv` and the runs are in
`~/vllm-bench/runs/`.

## 0. Headline

**The fp32-state default at the RL shape (n=8, T=1, logprobs=1, 4096 fixed tokens)** gives:

| B | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| decode tok/s, P=200 | 193 | 1205 | 3428 | **4680** | 4286* | 4306* |
| decode tok/s, P=2k | 191 | 1197 | 3119 | 4332 | 4238* | 4283* |
| decode tok/s, P=10k | 189 | 1151 | 2968 | 3630 | 3401* | 2900* |

The starred cells are *not* resident. vLLM's hybrid cache cannot hold 128 sequences of about 4.3-14k tokens with fp32
state in 32 GB, so it queues or preempts, and throughput plateaus at the B≈64-100 level (§2.3).

**The resident steady state (P=2k, N=1024, so every sequence fits)** gives the real per-step cost:

| config | B=64 step / tok/s | B=128 step / tok/s |
|---|---|---|
| default (fp32 SSM state, prefix caching, align) | 13.36 ms / 4791 | **22.30 ms / 5739** |
| bf16 SSM state (prefix caching off) | 10.30 ms / 6214 | 16.83 ms / 7607 |
| fp32 state + **ReplaySSM** (opt-in, runner V1) | 10.44 ms / 6128 | 16.93 ms / 7559 |

- **10k-token prefill: 0.37 s** (0.368-0.377 over 3 reps, single prompt, max_tokens=1), which is about 27k prompt tok/s.
  A 2k prefill takes 0.080 s. A group of 8 with n=8 takes 0.48 s at 10k (0.099 s at 2k), so the prompt is computed
  **once** per group (§4.4).
- **At B=128, 62-65% of the step is `selective_state_update`** (Triton, 13.4 ms, running at about 93% of DRAM BW on the
  fp32 state), 27% is cuBLASLt GEMM, and 2-7% is attention. At B=8, 86% of the step is GEMM.
- Against the audit's CALC floors (§0 table of the gap audit), vLLM reaches 80% of the floor at B=8 (1.2k vs 1.5k),
  89% at B=64 (4.79k vs 5.4k), and **86% at B=128 (5.74k vs 6.65k)** with fp32 state. The parity target for W1-W5 is
  therefore **22.3 ms/step at B=128 and 13.4 ms at B=64**, not "10k tok/s". The 10k figure assumed bf16 state.

## 1. Versions and config

| item | value |
|---|---|
| vLLM | **0.30.0** (pip wheel), V1 engine, **V2 model runner** (`gpu_worker.py:441 "Using V2 Model Runner"`) |
| torch / CUDA / triton | 2.13.0+cu130 / 13.0 (wheel); system driver 595.84 (CUDA 13.2) / triton 3.7.1 |
| flashinfer | 0.6.18.post1 (installed; its sampler JIT is disabled, see flags) |
| GPU | RTX 5090 (sm_120), 32 GB, max SM clock 3135 MHz |
| weights | local HF `~/storage/models/NVIDIA-Nemotron-3-Nano-4B-BF16/model.safetensors` (no download needed) |
| LLM() args | `dtype=bfloat16, trust_remote_code, mamba_ssm_cache_dtype=float32, max_num_seqs=256, max_model_len=14400, gpu_memory_utilization=0.85, seed=0` |
| SamplingParams | `n=8` (n=1 at B=1), `temperature=1.0, top_k=0, top_p=1.0, logprobs=1, max_tokens=4096, ignore_eos=True` (natural variant: `ignore_eos=False`) |
| env | `VLLM_USE_FLASHINFER_SAMPLER=0`. FlashInfer's sampler JIT needs `curand.h`, which the system CUDA lacks. With top_k=0 and top_p=1 the V2 runner uses its own Triton Gumbel sampler anyway, so no measured path changes. |
| util | 0.92 OOMs during sampler warmup; 0.85 is used |

**What vLLM chose** (from the engine log, `runs/*.log.digest`):

- **CUDA graphs**: `cudagraph_mode=FULL_AND_PIECEWISE` (forced for mamba models, `model_executor/models/config.py:613`). Capture sizes are
  `[1,2,4,8,16,24,…,256 (step 8),272,…,512]`, with `max_cudagraph_capture_size=512`. At `max_num_seqs=256`, **35 FULL
  decode graphs** are captured (sizes 1-256) plus 51 PIECEWISE graphs. **The profile confirms that every decode step is one
  `cudaGraphLaunch`** (299 launches for 299 steps). The graph holds 227 kernels at B=8 and 269 at B=128.
- **torch.compile**: `mode=VLLM_COMPILE`, Inductor, `custom_ops=['none']` with `ir_op_priority rms_norm/fused_add_rms_norm=['native']`.
  All RMSNorms are therefore **Inductor-generated Triton** kernels, not vLLM's CUDA `fused_add_rms_norm`.
  `splitting_ops` includes `vllm::mamba_mixer2` and `vllm::unified_attention_with_output`, so the mixer body and attention are
  graph breaks for PIECEWISE but are inside the FULL graph.
- **Attention backend**: `FLASH_ATTN`, **FlashAttention 2** (`cuda.py:538`, `flash_attn.py:1116`), out of
  `[FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION]`. The KV layout is LBNHC.
- **Mamba SSU backend**: Triton (`ssu_dispatch.py:505`). The log warns "Using default Mamba SSU config" because there is no
  tuned json for sm_120. Because `is_blackwell` matches only sm_100 (`mamba_mixer2.py:565`), dstate=128 gets
  `BLOCK_SIZE_M=4, num_warps=4` (`ops/mamba_ssm.py:116-133`).
- **mamba_ssm_cache_dtype**: float32. NemotronH forces fp32 when "auto" (`model_executor/models/config.py:660-687`,
  "Only float32 is known to have no accuracy issues"). The conv state stays bf16.
- **Chunked prefill**: on, `max_num_batched_tokens=8192` (`scheduler.py:288`).
- **Prefix caching**: on by default, which makes **`mamba_cache_mode='align'`** (`config.py:625`).
- **Hybrid block sizing**: the attention block size is raised to **976 tokens** so that an attention page ≥ a mamba page
  (`platforms/interface.py:934`). The mamba page is padded 0.18%. **3 padding layers** are added (21 mamba layers padded
  to 24, i.e. 6 groups of 4 plus 1 attention group of 4; `v1/core/kv_cache_utils.py:1554`). KV cache is 628,266 "tokens" at
  0.85 util.
- max_num_seqs = 256. Resident concurrency is capped by memory, not by this setting (§2.3).

## 2. Throughput

Method (`bench.py`): for each (P, B), submit B/8 prompts with n=8 (distinct random-token prompts per run). T1 is the
wall for `max_tokens=1` (the prefill phase). TN is the wall for `max_tokens=N`. decode tok/s = B·(N−1)/(TN−T1),
ITL = (TN−T1)/(N−1), and e2e = B·N/TN. All sequences have exactly N tokens (asserted), and logprobs are present for every token.

### 2.1 Fixed length, N=4096, default config (fp32 state)

| P | B | T1 prefill phase (s) | ITL (ms) | decode tok/s | e2e tok/s |
|---|---|---|---|---|---|
| 200 | 1 | 0.03 | 5.19 | 193 | 192 |
| 200 | 8 | 0.08 | 6.64 | 1205 | 1202 |
| 200 | 32 | 0.27 | 9.34 | 3428 | 3405 |
| 200 | 64 | 0.50 | 13.68 | 4680 | 4640 |
| 200 | 128 | 0.94 | 29.86* | 4286* | 4255 |
| 200 | 256 | 1.92 | 59.45* | 4306* | 4273 |
| 2000 | 1 | 0.09 | 5.23 | 191 | 191 |
| 2000 | 8 | 0.10 | 6.68 | 1197 | 1193 |
| 2000 | 32 | 0.38 | 10.26 | 3119 | 3092 |
| 2000 | 64 | 0.81 | 14.77 | 4332 | 4276 |
| 2000 | 128 | 1.55 | 30.20* | 4238* | 4186 |
| 2000 | 256 | 3.27 | 59.77* | 4283* | 4228 |
| 10000 | 1 | 0.39 | 5.30 | 189 | 185 |
| 10000 | 8 | 0.46 | 6.95 | 1151 | 1133 |
| 10000 | 32 | 1.77 | 10.78 | 2968 | 2854 |
| 10000 | 64 | 3.53 | 17.63 | 3630 | 3462 |
| 10000 | 128 | 7.09 | 37.64* | 3401* | 3252 |
| 10000 | 256 | 16.54 | 88.28* | 2900* | 2774 |

\* The sequences are not all resident, so "ITL" here is wall/steps rather than a per-step latency. See §2.3.

### 2.2 Resident steady state, P=2000, N=1024 (with scheduler stats logging on)

| config | B | Running (log) | ITL ms | decode tok/s | T1 s |
|---|---|---|---|---|---|
| default (fp32, PC on, align), V2 runner | 64 | 64 | 13.36 | 4791 | 0.73 |
| | 128 | **128** (KV 79-90%) | **22.30** | **5739** | 1.46 |
| | 256 | 161 running / 95 waiting | 47.84* | 5351* | 2.92 |
| PC off (`none`), fp32 | 64 | 64 | 13.77 | 4648 | **4.42** |
| | 128 | ≤125 (KV 99.7%) | 30.23* | 4234* | 8.86 |
| | 256 | ≤125 | 55.57* | 4607* | 17.84 |
| PC off, **bf16 state** | 64 | 64 | 10.30 | 6214 | 4.37 |
| | 128 | 128 | **16.83** | **7607** | 8.75 |
| | 256 | ≤201 | 39.55* | 6472* | 17.58 |
| V1 runner, default | 8 / 64 / 128 | full | 6.59 / 13.38 / 22.80 | 1214 / 4783 / 5615 | |
| V1 runner + **use_replayssm** (fp32) | 8 / 64 / 128 | full (128 at KV 99.3%) | 6.33 / 10.44 / **16.93** | 1264 / 6128 / **7559** | 0.31 / 2.40 / 4.82 |

Takeaways:

- The V2 and V1 runners are within 2% of each other, so the runner is not the lever.
- **bf16 state gives −25% step at B=128** (22.3 → 16.8 ms).
- **ReplaySSM gives the same gain with fp32 state.** That is, the fp32/bf16 policy question (W6) can be sidestepped by cutting
  state *writes* instead of state *width*. ReplaySSM's T1 is 3.3× worse because admission waits on memory: the ring
  buffers cost about 4% of KV capacity, and siblings queue ("Running 65 / Waiting 63" during the prefill phase).
- **Without prefix caching, prefill for a group is about 6× slower** (4.42 vs 0.73 s at B=64, P=2k), because each of the n=8
  children re-prefills the prompt.

### 2.3 Why B≥128 at N=4096 is not resident (vLLM memory layout, measured)

Per sequence, vLLM allocates from one shared block pool with page = 976 attention tokens ≈ one mamba layer state (3.93 MB
fp32 SSM + conv). In `align` mode, each of the 6 mamba groups keeps ≥2 state blocks per request
(`v1/kv_cache_interface.py:1040-1043`: `2 + spec + prefill_checkpoint`). The attention group needs ⌈len/976⌉ blocks.
At P=2k+4096 this is roughly 7 + 12 blocks of 16 MB, about 300 MB/seq (an estimate from the spec; the logs below are the measurement). The actual need is 82.6 MB of fp32 state plus
16 KB/token of KV (≈100 MB at 6k tokens). The logs show that the ceiling is reached: 128 resident at 3k tokens (KV 90%),
161 at B=256, and ≤125 with PC off. The 976-token block granularity, the 3 padding layers (14%) and the double state
blocks under align are why the RL shape plateaus at about 4.3k tok/s. **The audit's plan (27.3 GB at B=128 with
fp32 state and a 10k shared prompt) packs tighter than vLLM does.**

### 2.4 Natural EOS (real-text chat prompt ending in a math question, thinking on, max 4096)

| P | B | mean len | max len | wall s | e2e tok/s |
|---|---|---|---|---|---|
| 200 | 8 | 672 | 798 | 5.1 | 1048 |
| 200 | 64 | 925 | 4096 | 29.5 | 2006 |
| 200 | 128 | 748 | 1449 | 20.3 | 4716 |
| 200 | 256 | 789 | 4096 | 50.1 | 4033 |
| 2000 | 8 | 992 | 2005 | 11.8 | 674 |
| 2000 | 64 | 1024 | 4096 | 31.3 | 2094 |
| 2000 | 128 | 1003 | 2905 | 34.3 | 3739 |
| 2000 | 256 | 971 | 4096 | 63.1 | 3936 |

A single sequence that runs to 4096 halves group throughput (B=64: 2.0k vs 4.7k fixed). vLLM keeps the other slots
busy only if more requests are queued. With one batch submitted and nothing queued, it is tail-bound just like us, so W8's slot refill is a real
lever and not a vLLM-parity item.

### 2.5 Prefill

| prompt | wall (max_tokens=1) | note |
|---|---|---|
| 2,000 | 0.080 s (0.079-0.082) | single prompt |
| 10,000 | **0.372 s** (0.368-0.377) | single prompt; 2 chunks of ≤8192 (chunked prefill) |
| 10,000 × group of 8 (n=8) | 0.480 s | siblings hit the prefix cache (KV blocks + mamba align checkpoint) |
| 2,000 × group of 8 | 0.099 s | |
| 8 groups × 10k (B=64) | 3.53 s | about 0.44 s/prompt |

The audit's W9 gate (≤1.0 s warm at 10k) is looser than vLLM (0.37 s). The compute floor at R is 0.28 s.

## 3. Per-step kernel breakdown (nsys, `--cuda-graph-trace=node`, steady decode window = middle 60%)

Tables are `vllm-nemotron-prof/prof_*.txt`. Profiled wall matched unprofiled wall within 1%.

### 3.1 B=8 (P=200, default): step 6.51 ms, 98.6% GPU-busy, **306 kernels/step (227 in the FULL graph)**

| kernel | class | /step | µs/step | share |
|---|---|---|---|---|
| cuBLAS `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8`, grid (8,25), N=3136 (ssm_out+down+o) | GEMM, WMMA TC, 1-warp CTAs | 42 | 2487 | 38.8% |
| same kernel, grid (8,137), in_proj N=17504 | GEMM | 21 | 1516 | 23.6% |
| same kernel, grid (8,98), up_proj N=12544 | GEMM | 17 | 900 | 14.0% |
| same kernel, grid (8,1024), lm_head N=131072 (**outside the graph**) | GEMM | 1 | 489 | 7.6% |
| `_selective_scan_update_kernel` (Triton), grid (20,B,96) | selective_state_update | 21 | 455 | 7.1% |
| same GEMM kernel, grid (8,56), qkv N=7168 | GEMM | 4 | 130 | 2.0% |
| `_topk_log_softmax_kernel` (Triton, V2 runner logprobs) | sampling | 1 | 80 | 1.2% |
| `flash_fwd_splitkv_kernel` FA2 + `splitkv_combine` | attention | 4+4 | 67 | 1.1% |
| `_causal_conv1d_update_kernel` (Triton) | conv update | 21 | 39 | 0.6% |
| Inductor `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_silu_view` | gated RMSNorm with silu(z) | 21 | 29 | 0.5% |
| Inductor `triton_red_fused_fused_add_rms_norm_{0..3}` | residual add + RMSNorm | 42 | 67 | 1.0% |
| `_gumbel_sample_kernel` (Triton) | sampling | 1 | 18 | 0.3% |

By class: **GEMM 5.52 ms (86.1%)**, SSU 0.45 (7.1%), sampling 0.15, attention 0.07, norms 0.10, conv 0.04, relu² 0.01.
The GEMMs move the 7.12 GB of weights at **1.29 TB/s, which is 76% of the 1.7 TB/s measured BW, using tensor cores at M=8**.

### 3.2 B=128 (P=2000, default): step 21.91 ms, 99.3% busy, **348 kernels/step (269 in-graph)**

| # | kernel | class | /step | µs/step | share |
|---|---|---|---|---|---|
| 1 | `_selective_scan_update_kernel` (Triton, grid (20,128,96), 4 warps, BLOCK_M=4) | **selective_state_update** | 21 | **13,439** | **61.8%** |
| 2 | cuBLASLt `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8`, **split-K=3** (grid z=3), N=3136 (ssm_out 21 + down 17 + o 4) | GEMM (mma.sync m16n8k16) | 42 | 2,036 | 9.4% |
| 3 | same 128x64_64x3 kernel, in_proj N=17504, no split | GEMM | 21 | 1,854 | 8.5% |
| 4 | `flash_fwd_splitkv_kernel` FA2, grid (1,B,Hkv=8) (GQA heads packed in M) | attention | 4 | 1,606 | 7.4% |
| 5 | cuBLASLt `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4`, split-K=3, up_proj N=12544 | GEMM | 17 | 1,141 | 5.2% |
| 6 | 128x64 kernel, lm_head (outside graph) | GEMM | 1 | 565 | 2.6% |
| 7 | 128x64 kernel, qkv N=7168 | GEMM | 4 | 199 | 0.9% |
| 8 | `_causal_conv1d_update_kernel` | conv update | 21 | 165 | 0.8% |
| 9 | `cublasLt::splitKreduce_kernel` | GEMM | 42 | 105 | 0.5% |
| 10 | `_gumbel_sample_kernel` + `_topk_log_softmax_kernel` + torch radix topk + `_ranks_kernel` | sampling | ~7 | ~290 | 1.3% |
| 11 | `precopy_mamba_align_fused_kernel` + `postprocess_mamba_fused_kernel` | align-mode state copies | 2 | 83 | 0.4% |
| 12 | gated RMSNorm / add+RMSNorm / relu² (Inductor Triton) | norm/fused | 81 | 183 | 0.8% |

By class: **SSU 13.44 ms (61.8%)**, GEMM 5.90 (27.1%), attention 1.61 (7.4%), sampling 0.31, conv 0.17, norms 0.18,
other 0.15.

- SSU bytes: 128 × 21 × 3.93 MB is 10.57 GB read plus the same written, 21.1 GB/13.4 ms = **1.58 TB/s, about 93% of BW**. It is
  at its floor for an r+w fp32 update.
- GEMM: 0.91 TFLOP/step in 5.9 ms is **154 TFLOP/s**.
- Other B=128 runs: at P=200 (`prof_default_B128_P200.txt`) the step is 20.72 ms, with SSU 65.2% and attention only 0.47 ms.
  With PC off at P=2000 (`prof_nopc_B128_P2000.txt`) the step is 23.0 ms and **attention is 2.80 ms, against 1.61 ms with PC on**:
  sibling sequences reading the *same* physical prompt blocks get L2 reuse, so prompt sharing is about 1.2 ms/step at P=2k.
  PC off also drops the 2 align kernels, giving 303 kernels/step.
- ReplaySSM at B=128 (P=2000, V1 runner; about 113-128 resident): the step is 16.27 ms. `_replayssm_output_only_kernel` takes
  6.91 ms (43%) in place of SSU's 13.4. The state is only *read* each step, with a small ring of the last 16 (x, dt, B) inputs, and
  the full state is written back only on flush. GEMM is 5.64 and attention 2.03. There are 376 kernels/step (290 in-graph).

**Kernels outside the CUDA graph, per step (B=128):** the lm_head GEMM, the whole sampler (Gumbel argmax, logprob LSE, torch topk
radix select for top-1, ranks), input prep (`_prepare_pos_seq_lens`, `_gather_block_tables`, `_compute_slot_mappings`), the
align bookkeeping, and about 50 small torch elementwise/index kernels. That is 79 kernels and ≈1.1 ms on the GPU, but the GPU stays
99% busy, so host overhead is hidden by async scheduling.

## 4. How vLLM gets there (code, `vllm/` = `~/vllm-bench/.venv/lib/python3.12/site-packages/vllm`)

Full notes with more citations: `vllm-code-notes.md` (same scratchpad).

### 4.1 Model / fused ops (`model_executor/models/nemotron_h.py`)
- **Merged QKV**: `QKVParallelLinear` (`nemotron_h.py:454-462`), split at `:491`. HF q/k/v are stacked at load (`:730-734`). There is no RoPE.
- **Mamba in_proj is one 5-way merged GEMM** `[z | x | B | C | dt]` (`layers/mamba/mamba_mixer2.py:348-360`), N=17504. The conv
  weight is a separate merged `[x|B|C]` (`:336-346`).
- **Residual + RMSNorm**: every layer calls `self.norm(hidden, residual)` (`nemotron_h.py:404-417` etc.). With `custom_ops=none`,
  Inductor lowers each to one Triton reduction (`triton_red_fused_fused_add_rms_norm_*`). The residual is **bf16** (model dtype).
  vLLM does *not* keep an fp32 residual.
- **Gated RMSNorm**: `Mixer2RMSNormGated` (`mamba_mixer2.py:79-188`). With n_groups=8 (group 960), `forward_cuda` falls back to
  `forward_native` (`:171-172`; with `custom_ops=none` the native path is used anyway). Inductor fuses silu(z)·x, the per-group mean-square, rsqrt and the weight into **one** Triton kernel
  (`triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_silu_view_0`). This runs *outside* the mixer custom op, in the compiled graph
  (`mamba_mixer2.py:596-602`).
- **relu²**: `ReLUSquaredActivation` via `maybe_fused_act_quant` (`nemotron_h.py:123-128`). In practice it is a **separate** Inductor
  pointwise kernel (`triton_poi_fused_pow_relu`) and is *not* in the GEMM epilogue.
- **GEMMs**: `default_unquantized_gemm` → `torch.nn.functional.linear` (`model_executor/layers/utils.py:84-90`), so cuBLASLt
  heuristics pick the kernel. There is no autotune and no Triton GEMM. On sm_120 cuBLASLt picks the sm80 CUTLASS kernels: WMMA 16×16×16 with 1-warp
  CTAs at M=8, and mma.sync m16n8k16 128×64 (K-stage 64, 3 stages) or 256×64 (32×4) at M=128, with **split-K=3 + a
  `splitKreduce` kernel** for the N=3136 and N=12544 outputs.
- **torch.compile scope**: `@support_torch_compile` on `NemotronHModel` only (`nemotron_h.py:550-558`). `compute_logits` and lm_head
  (`:901-906`) run eagerly outside the graph.

### 4.2 Decode path (Mamba2 mixer, `mamba_mixer2.py`)
- `forward` (`:567-606`): in_proj GEMM → **custom op `torch.ops.vllm.mamba_mixer2`** (`:590-594`, registered `:1281-1285`) →
  gated norm → out_proj. The custom op is a torch.compile splitting op, but it *is* captured in the FULL decode graph.
- `conv_ssm_forward` (`:706-1167`) splits decode tokens (first) from prefill tokens (`:781-791`). Decode does:
  1. `causal_conv1d_update` (`:1048-1060`), which is **Triton** `_causal_conv1d_update_kernel` (`ops/causal_conv1d.py:762`, driver
     `:1096`). It updates the rolling conv state in place, grid (B, 9728/256=38).
  2. `selective_state_update` (`:1151-1167`) → `ssu_dispatch.py:530` → `TritonSSUBackend` (`:154-209`) → **Triton**
     `_selective_scan_update_kernel` (`ops/mamba_ssm.py:241`). One kernel per layer does softplus(dt+bias), dA=exp(dt·A), the
     state load (fp32, `:385`), the update, y=Σ state·C + D·x (`:450`) and the state store (`:494`). **One read and one write of the
     state per step.** z-gating is *not* in the kernel; it is done by the gated norm.
     In align mode the read index and write index differ (`:1036-1041`, copy-on-write for prefix caching). With PC off they are
     the same block (`:1043-1045`).
  3. Optional **ReplaySSM** (`:1124-1149`, `ops/selective_state_update_replayssm_output_only.py`, `config/cache.py:193-203`) is
     opt-in. It keeps the last 16 (x, dt, B) inputs per sequence, computes y from the checkpoint state plus the ring (read-only
     state), and writes the state only on flush. It requires runner V1 for Triton. The measured result is SSU 13.4 → 6.9 ms and step
     22.8 → 16.9 ms at B=128.
- **All Mamba kernels are Triton**: conv fwd/update, SSU, the 5 SSD kernels and the gated norm. CUDA-C++ code is used only by FA2,
  cuBLASLt, `reshape_and_cache_flash`, and the torch topk/elementwise kernels.

### 4.3 Prefill path
- `causal_conv1d_fn` (`mamba_mixer2.py:856-871`, Triton `_causal_conv1d_fwd_kernel`, `ops/causal_conv1d.py:16`, varlen over
  packed sequences, carries conv state across chunked-prefill boundaries).
- `mamba_chunk_scan_combined_varlen` (`mamba_mixer2.py:894-916`; `ops/ssd_combined.py:27-227`) runs **5 Triton kernels** with chunk
  256: `_chunk_cumsum_fwd` (`ssd_chunk_state.py:303`), `_chunk_state_fwd` (`:353`), `_state_passing_fwd`
  (`ssd_state_passing.py:102`, the only sequential-over-chunks part), `_bmm_chunk_fwd` (`ssd_bmm.py:148`, C·Bᵀ per chunk), and
  `_chunk_scan_fwd` (`ssd_chunk_scan.py:418`). The final state per sequence is written to the cache (`mamba_mixer2.py:1001-1013`).
  The SSD kernels are warmed at init (`:615`).
- Multiple prompts are **packed varlen** into one ≤8192-token prefill step (metadata in `v1/attention/backends/mamba2_attn.py:22-172`),
  and mixed prefill+decode batches run under PIECEWISE graphs.

### 4.4 Hybrid cache and prefix caching
- `MambaSpec` (`v1/kv_cache_interface.py:993-1057`) and `MambaManager` (`v1/core/single_type_kv_cache_manager.py:1444+`) share one
  block pool with attention. Block alignment is in `platforms/interface.py:778` (`_align_hybrid_block_size`). Groups are formed
  and padding layers added in `v1/core/kv_cache_utils.py:1535-1560`.
- **n=8 is fan-out, not fork**: `v1/engine/parallel_sampling.py` (`ParentRequest`) creates 8 independent child requests. Sharing
  comes *only* from the prefix cache. The first child computes the prompt and the other 7 hit the cached KV blocks and the
  mamba "align" checkpoint. Measured: a group of 8 at 10k takes 0.48 s against 0.37 s for one prompt, and PC off is 6× slower.
  With PC on, the siblings' decode attention reads the same physical prompt blocks, and the L2 reuse measured above is 1.61 vs 2.80 ms.
- Align-mode cost: 2 state blocks per request per group, plus 2 copy kernels per step (`v1/worker/mamba_utils.py:369,551`, ≈83 µs
  at B=128).

### 4.5 Attention
FA2 `flash_fwd_splitkv_kernel` over the paged KV. With grid (1, B, 8), the 5 query heads of each KV head are packed into the M
tile, so one CTA reads each (seq, kv-head) KV stream once. It splits the KV only at small B: B=8 uses split + combine and B=128 uses no split.

### 4.6 Sampling and logprobs (V2 runner, `v1/worker/gpu/sample/`)
- `_gumbel_sample_kernel` (`gumbel.py:193`): per (token, vocab-block) Gumbel-max with a counter-based RNG from (seed, pos), then a
  second-stage argmax. There is no softmax and no probs tensor.
- Logprobs: `_topk_log_softmax_kernel` (`logprob.py:17`) makes three passes over the bf16 logits row (max, Σexp, gather) per
  request. `torch.topk` gives the top-1 id (`logprob.py:126/141`) and `_ranks_kernel` (`:61`) gives the rank. `logprobs_mode`
  defaults to raw logprobs. The whole sampler costs about 0.3 ms at B=128 and runs eagerly.

## 5. Against our plan (gap audit W1-W10)

Covered by the plan (vLLM does the same): bf16 GEMM operands on tensor cores (W1), merged QKV and a merged ssm_in GEMM (W4 rows),
residual-add+RMSNorm and gated-norm+silu fusion (W5), a one-read/one-write fused SSU plus conv update (W3), a Gumbel sampler with
logprobs from one LSE pass (W5), per-B CUDA graphs (W10: vLLM captures 35 sizes), prompt prefilled once per group (have), and
chunk-256 SSD prefill (W9).

**vLLM techniques not in our plan:**

1. **ReplaySSM, i.e. deferred state write-back.** This is the single biggest measured lever: SSU 13.4 → 6.9 ms and B=128 step
   22.8 → 16.9 ms (+35% tok/s) **with fp32 state**, matching bf16-state throughput. W3 plans one-pass r+w and W6 plans bf16 state.
   Neither considers "read the checkpoint and keep a 16-step input ring, write every 16 steps". This could make W6's
   numerics gamble unnecessary. It needs a machine-searched descriptor (output-only kernel + flush), not a port.
2. **Split-K GEMMs for the N=3136 and N=12544 roles at M=128** (cuBLASLt split-K=3 + a reduce kernel), and **tensor-core GEMM at
   M=8** (WMMA with split along grid.x=8, 1-warp CTAs, 76% of BW). W4 lists tile shapes and n=64 tiles but not split-K. W1b/1b
   plan CUDA-core matvec up to M=16; vLLM uses TC even at M=8.
3. **Implicit prompt sharing through shared physical KV blocks, which gives L2 reuse without cascade attention.** This is
   measured at 1.2 ms/step at B=128 P=2k. W2 gets it explicitly, so it is not missing, but it is **evidence that W2's gain is
   real** and that a single-segment kernel over shared pages already gets most of it.
4. **GQA head packing in decode attention** (5 q-heads per KV head in the M dimension, grid (1,B,Hkv)) and **B-dependent
   split-KV** (split at small B, none at B=128). W7 extends G5 with a batch axis but does not say to drop the split at large B.
5. **Varlen multi-prompt packed prefill in one ≤8192-token step**, with conv and SSM state carried across chunk boundaries,
   interleaved with decode (mixed batches in PIECEWISE graphs). W9 is per-prompt and W8 mentions interleaving but not packing
   several prompts into one SSD varlen call.
6. **lm_head and sampler outside the graph with async scheduling** (GPU stays 99% busy). This is a detail: our W8 device-resident loop is the
   equivalent.

**Things the plan does that vLLM does not (the headroom over vLLM):**
- An fp32 residual (the plan keeps it; vLLM is bf16 end to end).
- relu² in the GEMM epilogue (vLLM uses a separate kernel).
- Tighter memory. vLLM's 976-token pages, 3 padding layers and align-mode double state blocks cap it below B=128 at the RL
  shape (≈4.3k tok/s at N=4096), whereas the plan fits B=128 × (10k shared + 4096) in 27 GB.
- A tuned SSU config for sm_120 (vLLM uses the untuned default).
- Explicit slot refill for the natural-EOS tail (§2.4).

**Parity targets derived from these measurements:** B=8 at 6.6 ms/step, B=64 at 13.4 ms, and B=128 at **22.3 ms** (fp32 state,
resident). With ReplaySSM-class or bf16 state, B=128 at 16.9 ms. A 10k prefill at 0.37 s. At the literal RL shape (N=4096
fixed, fp32): **≈4.3k tok/s** at B≥64 (P≤2k) and 3.6k at P=10k.
