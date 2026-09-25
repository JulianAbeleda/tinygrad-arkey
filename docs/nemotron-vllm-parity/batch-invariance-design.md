# Batch invariance for one-stack RLOO: how vLLM gets bitwise train/inference parity, and what it means for Nemotron-H on tinygrad

Status: design, 2026-09-25. Research and code reading only: no GPU runs, no product code changed.
Problem it answers: the one-stack smoke hit the parity stop. Real 4B model, 4k-token rollouts: mean 1.8e-3 nats,
p99 0.024, max 0.31, 5/4090 tokens > 0.1. Divergence jumps at block 12, the first attention block, from ~1e-5 to
~5e-4 relative, and forcing one matmul kernel cuts it ~5x.

Sources: vLLM 0.30.0 installed at `~/vllm-bench/.venv/lib/python3.12/site-packages/vllm/` (paths below are relative
to it); TorchTitan `main` @ 2f44117f (2026-09-25), `torchtitan/rl/docs/bitwise_parity.md`,
`torchtitan/rl/model/{attention,gdn}.py` and `torchtitan/models/qwen3_5/gdn.py`; Thinking Machines, "Defeating
Nondeterminism in LLM Inference"; the vLLM blog "bitwise-consistent train-inference" (2025-11-10); vLLM RFC #55524
(Mamba2 exact-replay decode); arXiv 2511.17826 (TBIK); arXiv 2605.14220 / verl-project/vexact.

---

## 1. The principle

**Property.** Take any output element y[r, j], such as the logprob of token r or a hidden unit of token r. Every
kernel on the path to it has to evaluate one fixed **expression DAG**: the same operands, the same association
order for every sum and product, and the same rounding points (casts, fma contractions, exp and rsqrt calls). That
DAG may depend only on r's own data and on constants fixed in advance. It must not depend on:

1. **Batch size**: how many other rows or sequences share the launch (M of a GEMM, B of a decode step).
2. **Chunk size**: how many tokens of this sequence run in one call (prefill piece, grad slice, parity slice, the
   64-token scan chunk, the 16-slot replay ring).
3. **Position or length bookkeeping**: the KV length, the suffix bucket, whether a key came from cache or from the
   same call, where a key sits relative to the prompt or suffix boundary, and the padding.

Kernels are free to vary only the mapping of independent outputs onto hardware (M/N tiles, grid, which SM runs
row r). Thinking Machines puts it this way: "the reduction order for a given token does not depend on how many
other tokens from its sequence are being simultaneously processed... when processing the 1000th query token in a
sequence, the reduction order must be identical regardless of whether 0 tokens are in the KV cache (prefill) or
999 tokens are in the KV cache (decoding)."

**Why fp32 alone does not fix it.** fp32 addition is not associative, so a different grouping gives a different last
bit, about 1e-7 relative. The stack then amplifies that difference instead of averaging it away:
- **Rounding discontinuities.** Every later rounding to bf16 (K/V storage, softmax weights `exp(...).cast(bf16)`,
  the routed activation cast) turns ulp noise into occasional full bf16 ulps. Our own measurement: fp32-order noise
  becomes a 5e-3 mean logprob gap once activations are rounded to bf16 (`_Linear` docstring, `nemotron_h.py:97-100`).
  `~/scratchpad/fp16/RESULTS.md` models it: a fraction of about eps/u of elements flip one ulp, so the noise grows
  from eps toward sqrt(eps*u).
- **Recurrence and depth.** Mamba state carries an error to every later token. 42 blocks compound it, and the
  attention softmax exponentiates score noise.
- **Different formulas, not only different orders.** Several of our sites compute different expressions: hi+lo bf16
  versus an fp32 matvec, a global-row softmax max versus a per-chunk max, chunked SSD versus the replay ring. More
  precision narrows those gaps. Only one shared DAG closes them.

Bitwise identity is also the only tolerance that stays stable under training. Any nonzero gap has a tail (our max is
0.31) that importance-sampling correction (TIS) only truncates. TorchTitan's measured run: `logprob_diff max` 1.56
with batch invariance off, and exactly 0 at every step with it on (table in §2.4).

**Useful exact identities.** Padding and masking do not break invariance, because these are exact in IEEE:
`x + 0 = x`, `x * 1 = x`, `exp(-inf) = 0`, and a bf16 x bf16 product is exact in fp32. **The condition:** padded or
masked terms must sit in the same positions of the same reduction tree. Zeros appended to a sequential accumulation
are harmless. Zeros that change the shape of a tree reduction (a longer span, so a different GROUP/UNROLL split) are
not, because the grouping of the nonzero terms moves.

---

## 2. vLLM's implementation, op by op (vLLM 0.30.0)

Switch: `VLLM_BATCH_INVARIANT=1` (`envs.py:92,619`). `init_batch_invariance()` runs at worker init
(`v1/worker/gpu_worker.py:1524-1526`, `model_executor/determinism/batch_invariant.py:1162-1172`).

### 2.1 Matmul (mm / addmm / matmul / linear / bmm, plus lm_head)
- **Ampere (SM80):** aten `mm/addmm/matmul/linear` are replaced by a Triton **persistent** GEMM
  (`batch_invariant.py:1068-1074`, kernel `:47-135`, launcher `matmul_persistent :280-353`). Each output tile runs
  `for ki in range(k_tiles): accumulator = tl.dot(a, b, accumulator)` (`:98-116`), a sequential fp32 accumulation
  over fixed BLOCK_K slices. There is no split-K and no atomics.
- **Tile choice may depend on M, but BLOCK_K may not.** The tuned table `batch_invariant_configs.py` picks
  BLOCK_M/BLOCK_N/warps/stages per M bucket, and BLOCK_K per (N, K) only: "shape-wide BLOCK_K keeps this
  batch-invariant" (`:316`, `_get_matmul_config :298-326`). The XPU docstring says the same: "Only BLOCK_SIZE_M/N and
  the launch parameters vary with M; dtype alone decides BLOCK_SIZE_K, so the K-reduction order never depends on
  batch size" (`:341-342`). vLLM relies on `tl.dot` giving the same bits for a given BLOCK_K whatever BLOCK_M is
  (different MMA instruction shapes). For our TC path that is an assumption to test, not a given.
- **Hopper and Blackwell:** vLLM keeps cuBLAS(Lt) and removes split-K with a workspace clamp:
  `CUBLAS_WORKSPACE_CONFIG=:16:8`, `CUBLASLT_WORKSPACE_SIZE=1` (`:1075-1080`), and forces the cublaslt backend
  (`:1137-1138`).
- **Model layers:** `UnquantizedLinearMethod.apply` calls `linear_batch_invariant` (`model_executor/layers/linear.py:234-237`),
  and so does the lm_head (`vocab_parallel_embedding.py:78-81`). `aten::bmm` gets its own kernel (`:357`, `:792-877`,
  registered `:1123-1126`).
- **Precision knobs:** reduced-precision bf16/fp16 reductions off (`:1128-1136`); TF32 off for matmul, conv and rnn
  (`:1170-1172`).

### 2.2 Row reductions: log_softmax, softmax, mean, RMSNorm
- `aten::_log_softmax` becomes `_log_softmax_kernel` (`:490-549`): **one program per row**. The row is walked in
  BLOCK_SIZE column blocks for max, then sum-exp, then output. The order depends only on n_cols.
- `aten::softmax/_softmax` becomes amax, exp and sum in torch (`:890-897`). `aten::mean.dim` becomes `mean_kernel`,
  iterative per dim, forced fp32 (`:597-730`, `:900-921`).
- **RMSNorm:** `RMSNorm.forward_cuda` calls `rms_norm_batch_invariant` (`layers/layernorm.py:101-110`), a kernel with
  one program per row, fixed BLOCK_SIZE=1024, fp32 sum of squares (`:925-974`, `:977-1036`). The fused-residual
  variant delegates to vLLM's own CUDA `fused_add_rms_norm` (`:1004`), which is also one block per row. The fused
  all-reduce+RMSNorm pass is disabled (`config/vllm.py:184-186`).

### 2.3 Attention: backend and split settings
- **Allowed backends:** only those whose `supports_batch_invariance()` returns True, namely FLASH_ATTN
  (`v1/attention/backends/flash_attn.py:343`), TRITON_ATTN (`triton_attn.py:335`), FLEX_ATTENTION
  (`flex_attention.py:126`), TRITON_MLA and FLASHATTN_MLA. The base default is False (`v1/attention/backend.py:204`),
  and selection rejects the rest (`:335`).
- **FlashAttention:**
  - `max_num_splits = 1` (`flash_attn.py:817-818`) and `num_splits=1` on every call (`:1715`, `:2024`, `:2049`).
    This is **no split-KV**: each query row walks its KV blocks in one CTA.
  - The AOT scheduler is disabled because its "schedule varies with max_seqlen_q/k" (`:784-787`).
  - FA4 is refused and falls back to FA2, because FA4 "uses batch-shape-dependent scheduling heuristics"
    (`fa_utils.py:169-176`).
  - Cascade (shared-prefix) attention is disabled (`config/vllm.py:1924-1932`).
- **Triton unified attention:** batch invariance forces the **2D kernel** (no 3D segmented split-KV,
  `triton_unified_attention.py:1040-1054`). The 2D kernel uses `TILE_SIZE_PREFILL` (`:1082-1084`), so decode rows
  and prefill rows walk the KV in the **same tile size** (32 by default, `_get_tile_size :789-803`).
- **FlexAttention:** BLOCK_M = BLOCK_N = 16 (`flex_attention.py:1236-1237`, `config/attention.py:103-111`),
  `IS_DIVISIBLE=False` (`:1400-1401`), and blocks must be at most the KV cache block size (`layers/attention/attention.py:395-410`).
- **FlashInfer:**
  - Fixed split *sizes* (`decode_fixed_split_size=2048`, `prefill_fixed_split_size=4096`) and
    `disable_split_kv=True` (`flashinfer.py:707-710`).
  - A 2 GiB workspace (`:94`, `:1017-1018`).
  - TRTLLM attention is disabled (`utils/flashinfer.py:599-601`).
  - Prefix caching is disabled for FLASHINFER and TRITON_MLA (`layers/attention/attention.py:371-386`).
- **Default backend on XPU:** TRITON_ATTN (`engine/arg_utils.py:2506-2516`).

### 2.4 Communication, compile, graphs
- **Tensor parallelism:** custom all-reduce, symmetric memory and FlashInfer all-reduce are off
  (`config/parallel.py:1058-1059`, `cuda_communicator.py:65-69,299,525`, `all_reduce_utils.py:136,165`, `symm_mem.py:114`).
  NCCL is pinned to one channel, tree all-reduce, Simple protocol, one thread (`batch_invariant.py:1141-1159`).
- **What is not disabled:**
  - CUDA graphs: TorchTitan's measured run keeps `FULL_DECODE_ONLY` graphs on.
  - torch.compile: only AOT compile is turned off (`:1159`).
  - Chunked prefill: I found no BI-conditional code that disables it. With per-row-invariant kernels it does not
    need disabling.
- **Mamba/SSM is not supported.** `_cached_get_mamba_attn_backend` **raises** "VLLM batch_invariant mode is not
  supported for ..." for any Mamba backend without `supports_batch_invariance()` (`v1/attention/selector.py:228-240`).
  No Mamba, GDN, linear or short-conv backend overrides it. **In 0.30.0, Nemotron-H cannot run with
  VLLM_BATCH_INVARIANT=1.** The open work is RFC #55524 and PRs #55905 and #56244 (§3.3).

### 2.5 Measured cost

| Source | Setting | Cost |
|---|---|---|
| Thinking Machines | Qwen3-8B, vLLM, 1000 completions | 26 s default; 55 s deterministic (naive attention); 42 s with the improved attention kernel. Their invariant matmul is "about 20%" behind cuBLAS |
| vLLM blog (TorchTitan) | Qwen3-1.7B GSM8K RL | whole RL run 2.4x slower; KL "always 0.0" |
| TorchTitan `bitwise_parity.md` | Qwen3-8B Search-R1, 8xH100, TP2/TP2, CUDA graphs on | generator per-token latency 9.6 → 22.8 ms (2.4x); decode 2.6x; trainer fwd/bwd 370 → 127 tok/s (2.9x); **wall-clock per step ~1.0x** (orchestration-bound); kl_div / logprob_diff max 0.0084 / 1.56 → **0 / 0** |
| vLLM RFC #55524 (Mamba2 replay) | B=64, 128-token partial chunks | one SSD-replay decode call 0.80–1.05 ms vs 0.37–0.40 ms for one recurrent (SSU) step; about +45 MiB per sequence (27 layers) |
| VeXact | Qwen3-30B-A3B RL | KL exactly 0 throughout; about 2x final AIME24 versus vLLM rollouts. Overhead not extracted |

---

## 3. The crux: incremental decode versus full recompute

### 3.1 Attention and matmul: one per-row kernel, not "decode runs prefill"
Neither vLLM nor TorchTitan runs decode through a prefill kernel, and neither makes the trainer imitate decode.
**They make every kernel per-row invariant.** A row's result is then the same whether it is computed alone at
decode, inside a prefill chunk, or inside the trainer's full-sequence forward:

- **Matmul:** the K-order is fixed per (N, K) (§2.1). A token row gets the same bits at M=1 (decode) and M=4096
  (trainer).
- **Attention:** one CTA per query row with no split-KV (`num_splits=1`, or the 2D Triton kernel with the prefill
  tile for decode too). It walks the KV in fixed-size tiles from key 0 with one online-softmax recurrence.
  Thinking Machines adds that the KV cache and page table are updated *before* the kernel runs, so a key produced
  in the same call and a key read from cache look identical to it. Where split-KV is kept (FlashInfer, the TML
  kernel), the split **size** is fixed and the split count varies ("instead of fixing the # of splits, we fix the
  size of each split"). The combine tree is then a function of the KV length only.
- **Trainer side (TorchTitan):** the trainer and the generator run **the same model definition**.
  - Attention: the generator runs TorchTitan's own `VarlenInnerAttention` through vLLM's `CUSTOM` backend
    (`torchtitan/rl/model/attention.py`, `TorchTitanVarlenInnerAttentionBackend`). "The same FA3 varlen kernel as
    the trainer. The shared `num_splits=1` fix above removes the only batch-dependent split-k, so the two match
    bitwise" (`bitwise_parity.md`).
  - Row ops: `mm/addmm/_log_softmax/mean.dim` come from `batch_invariant_ops` in both processes.
  - Logprobs: vLLM's fused Triton logprob kernel is **replaced by the trainer's `compute_logprobs`**
    (`force_logprobs_fn_for_batch_invariance`), so the final log-softmax is one code path.
  - Precision: the trainer forward is bf16 through the FSDP mixed-precision cast, like the generator.
  - Backward: custom backward passes are registered (`batch_invariant_backward.py`). Only the forward needs to be
    bitwise identical.

**So:** the trainer uses the prefill kernels (a full-sequence varlen forward). Incremental decode is **made equal to
prefill by per-row kernel invariance**: identical KV tile walk and identical GEMM K-order. It is not made equal by
replaying decode as prefill. The residual assumption is that the kernel's KV tile size is the same for q_len = 1 and
q_len = L. vLLM enforces this for Triton (`TILE_SIZE_PREFILL` in both) and relies on FA2/FA3 at `num_splits=1` for
FA; TorchTitan measured it at 0/0.

### 3.2 Why that cannot work for Mamba: scan versus recurrence
Attention's per-row sum over keys can be re-chunked freely. A linear recurrence's cannot, because the chunked form
and the sequential form are different algebraic expressions of y_t:

- **Sequential (SSU):** `S_t = exp(dt_t A) ⊙ S_{t-1} + (dt_t x_t) ⊗ B_t` rounded every step, then `y_t = S_t · C_t`.
  y_t reads a state that has been rounded t times.
- **Chunked (SSD):** `y_t = exp(L_t) (C_t · S_0) + Σ_{s≤t} exp(L_t − L_s) (C_t · B_s) (dt_s x_s)` with
  `L = cumsum(dt A)`. The decay is `exp` of a difference of sums, not a product of `exp`s, and C·B is contracted
  before scaling by x.

`exp(a+b) ≠ exp(a)·exp(b)` and `C·(Σ w_s x_s B_s) ≠ Σ w_s x_s (C·B_s)` in floating point. So no implementation
detail makes a chunk-Q SSD equal to the step recurrence for Q > 1. They differ in which quantities are rounded, not
only in order. The RFC's wording: "neither is 'wrong'; they are two roundings of the same math". The only
bitwise-safe options pick **one** form and use it on **both** sides:

- **(a) Recurrent form everywhere.** TorchTitan's answer for GDN, the closest shipped analogue of Mamba2. In batch-invariant mode:
  - Trainer forward, generator prefill and generator decode all run the sequential `recurrent_gdn` kernel with a
    **float32** state (`models/qwen3_5/gdn.py:85-139`, `rl/model/gdn.py` `_forward_gdn`: "Batch-invariant recurrence
    uses the same Attention Gym scan as the trainer"; `autotune=False`, since "Triton autotuning breaks batch
    invariance").
  - The **backward** comes from the parallel chunk kernel (`_chunk_gdn_gradients`, registered as the autograd rule
    of `_recurrent_gdn_fwd`). The forward is exact; the gradient is the mathematically equal chunked one.
  - fp32 state is required, because decode would otherwise round the state to bf16 every token "unlike a single
    prefill call".
- **(b) Chunked form everywhere, with decode replaying the chunk from the last boundary.** vLLM RFC #55524 and
  PRs #55905/#56244:
  - The cache holds the fp32 **state at the last chunk boundary** plus the **partial chunk's inputs** (x, raw dt,
    B, C since the boundary).
  - Each decode step runs the SSD chunk kernel over "its buffered partial chunk plus the new token", so every path
    "see[s] the chunk grid of a single-shot prefill and produce[s] identical bits". Cost is in §2.5.
- **(c) A fixed-order chunking shared by both sides.** A generalization of (b): any chunk size Q, with boundaries
  anchored at the same absolute positions and the same per-row DAG inside a chunk.

**Our sampler already implements (b) at Q = ring = 16.** `mamba_replay_step` (`nemotron_h_decode.py:162-201`)
computes `y_t = exp(L_t)(S_ckpt·C_t) + Σ_j exp(L_t−L_j)(x_j dt_j)(B_j·C_t)` from a read-only checkpoint and a ring of
the chunk's inputs. `mamba_replay_flush` (`:210-229`) is the chunk-boundary state update. What differs is the
trainer's scan: Q=64 (`scan_chunk`), a different anchor, and a different DAG (`_scan_chunk`, `nemotron_h.py:173-197`).

---

## 4. Our stack, op by op

Paths:
- **Sampler (S):** `NemotronHPrefill` primes; then the batched decode (`nemotron_h_sampler.py`) at B rows per step.
- **Trainer (T):** DayCare `rloo_tinygrad.py`. `prompt_caches` reuses the **same prefill buffers** after
  `prompt[:-1]`. `rollout_features` recomputes the generated segment through `block.cached` in 512-token buckets,
  4 rollouts at a time (rows = 2048), to get the hidden state entering the final block. `policy.logprobs` then runs
  final block + norm + lm_head + log_softmax at 16 rows (parity) or 512 rows (gradient).

| Site | S (decode) | T (recompute) | Invariant today? | Change that makes it invariant |
|---|---|---|---|---|
| Prompt KV and Mamba state | prefill buffers | the same buffers | **Yes, shared by construction** | keep |
| Last prompt token | inside the prefill piece; its hidden gives token 0 | row 0 of `cached` over `[last]+tokens` | **No** | both sides: prefill `prompt[:-1]`, then run `last` as decode step 0 (ring slot 0) |
| `_Linear` projections | B ≤ 16: fp32 x widened-bf16 **matvec** (heuristic schedule); 17–128: routed TC candidate, hi/lo, **split_k S from the route table per (role, rows, N, K)** | rows > 128: generic tinygrad hi/lo TC, heuristic opts | **No: three formulas and three K-orders by row count** (`nemotron_h.py:116-132`, `dense_candidate_gemm.py:126-211`) | one canonical projection for every row count: hi/lo bf16 TC, rows padded to 16, one candidate tile family with **K-loop and split_k fixed per (N, K)**, M tiles free (vLLM §2.1). Replace the matvec: B ≤ 8 hi/lo stacks into exactly one M=16 tile |
| LoRA `x@A`, `(xA)@B` | generic matmul, B rows | generic matmul, 16 or 512 rows | Unverified; heuristic opts depend on M | the same pinned-reduce matmul; rank-r K is tiny, so a pinned sequential reduce is cheap |
| RMSNorm (`attn_norm`, `output_norm`) | `nn.RMSNorm`, B rows | same op, 2048 / 16 / 512 rows | Unverified; tinygrad's heuristic uses GROUP/GROUPTOP for small row counts, which is the case TML says breaks | an invariant row reduce (below) |
| Gated group RMSNorm (Mamba) | `_block_output`, B×1 rows | `cached`, B×L rows | same risk | same fix |
| Causal conv (K=4) | `(window * w.T).sum(axis=1)`: a reduce, oldest tap first (`nemotron_h_decode.py:38`) | Python sum `x_t·w3 + x_{t-1}·w2 + ...`, newest first (`nemotron_h.py:155-171`) | **No: different association order** | one function for both, e.g. an explicit elementwise chain in one fixed order. Watch fma contraction |
| Attention | two segments (prompt, own suffix); **per-1024-key chunk max**, `exp(s − chunk_max)` rounded to bf16; LSE combine over a chunk count that varies with the suffix bucket; bucket < 1024 is one chunk of `bucket` length (`nemotron_h_attention.py:64-133`) | `_attention`: **global row max**, `exp(s − row_max)` rounded to bf16; query chunks of 128; one reduce over the whole key span (`nemotron_h.py:322-341`, `:382-406`) | **No: different rounding points, not just order. This is the block-12 jump** | a canonical tiled attention on both sides (below) |
| Mamba scan | replay ring Q=16, anchored at ring flushes; flush = unrolled rank-16 update | `_scan_chunk` Q=64, anchored at the segment start, C@Bᵀ⊙decay DAG | **No** (§3.2) | the trainer runs the sampler's replay DAG (below) |
| SSD prefill (prompt) | `ssd_cached` Q=256, `precision="float"` | the same buffers | shared | keep |
| lm_head + bias + /T + log_softmax | `_sample`: B rows (`nemotron_h_sampler.py:110-118`) | `policy.logprobs`: 16 or 512 rows | Unverified; lm_head is a `_Linear` (row-count routes); log_softmax over 131k columns is a row reduce | canonical projection plus an invariant row reduce, in **one shared function** (TorchTitan's `compute_logprobs` lesson) |
| Elementwise (silu, softplus, exp, residual add) | fused into various kernels | fused differently | usually yes, but **fma contraction** (NVRTC `--fmad=true` default) can turn `a*b+c` into an fma in one fusion and not the other | identical expression trees; if a site still differs, test with fmad off to rule contraction in or out |

### 4.1 Invariant row reductions in tinygrad
tinygrad chooses reduce opts (GROUP, GROUPTOP, UNROLL on the reduce axis, split reduces) from the **whole** kernel
shape, row count included. So a row reduce is not batch-invariant by default. Two ways to get it:

1. **Pin the reduce opts.** Apply explicit opts that are a function of (reduce length, dtype) only; allow only
   upcasts and locals on the row axis. This is `dense_candidate_gemm`'s pattern (`_CANDIDATE_OPTS`), extended to
   norms and log_softmax.
2. **Encode the order structurally**, the TBIK idea: pad the reduce axis with zeros to a fixed power of two and
   write the reduction as a fixed binary tree of **elementwise** halving adds (`x[..., :n/2] + x[..., n/2:]`). The
   zeros sit at fixed positions and are exact, and no reduce opt can reorder it. This costs a few more kernels or
   fused loads but survives any scheduler or BEAM change. Use it for RMSNorm (hidden dim), the group norm and the
   log-softmax sum over the vocab; the max needs no care, because max is associative and exact.

### 4.2 Canonical attention (both sides)
Define attention for query t over keys `[prompt 0..P) | generated 0..t]`:
- **Tiles** of T keys, aligned to each **segment's** start (prompt tiles at prompt-relative multiples of T,
  generated tiles at generated-relative multiples). The sampler's segment layout thereby becomes the definition,
  and the trainer knows P, so it can use the same tiles.
- **Per tile:** scores via `_exact_dot` (bf16 operands, exact products, fp32 accumulation with a **pinned per-tile
  reduce order** over hd=128); `peak = max`; `w = exp(s − peak).cast(bf16)`; `sum = Σw` and `out = w·V`, both with
  pinned reduce order over T keys.
- **Combine:** over a **fixed-length** tile array: `prefix_capacity/T + suffix_capacity/T` slots, with masked or
  unread tiles set to (peak −inf, sum 0, out 0). Use one pinned (or elementwise-tree) reduce. Empty tiles are exact
  identities because `exp(-inf − peak) = 0`, and the fixed length makes the tree independent of the bucket. The
  sampler's suffix bucket may still limit *reads*; only the partial-array length must be fixed.
- **Trainer:** computes the same per-tile partials for every generated row. Query rows are the parallel axis, and
  the KV tiling does not depend on how many query rows share a launch (the vLLM property).
- **T:** 1024 keeps today's sampler prefix path unchanged but forces the suffix to be read in whole 1024 tiles.
  256 is a better default, since the suffix bucket minimum becomes 256. Measure both.

### 4.3 Mamba on the trainer: run the sampler's replay DAG
Of the three options in §3.2, (a) as a literal per-token recurrence is not what our sampler does, so it would not
match. The cheapest bitwise-safe option is (b)/(c) at **Q = ring**, using the sampler's own functions:
- **Split the generated segment** into chunks at the sampler's flush points (token index i goes to ring slot `i %
  ring`, with i = 0 the last prompt token).
- **Within a chunk,** the output of position k is `mamba_replay_step`'s DAG with `slot = k` over the chunk's ring
  contents. Fold the chunk position into the batch axis: 16 "virtual decode steps" evaluated in parallel, each with
  its own `count` mask, reading one checkpoint. Use the same `_ring_weights` (masked ring cumsum over the fixed
  16-slot array), the same `_readout` reduce over N, the same `coef` construction, and the same
  `recent + current + ckpt` association.
- **Between chunks,** call `mamba_replay_flush` itself: an unrolled j = 0..15 update in fixed order. That is 256
  sequential elementwise flushes for a 4k rollout.
- **Batched decode rows must not change a row's reductions.** They reduce over N, the ring and the group dims; batch
  is parallel. Check that with the unit test below.

This is also the RFC's design (checkpoint plus buffered partial chunk) with Q=16. The alternative, making the
sampler replay a Q=64/256 SSD (RFC-literal), costs sampler speed (RFC: ~2.5x per mixer step at Q=128) for no gain.

### 4.4 The frozen stack does not need recomputing
The predeclared adapter is LoRA on the **final MLP block only** (`attach_lora`, `last_k=1`), and blocks 0..40 are
frozen. `rollout_features` recomputes them solely to get the hidden state entering block 41. The sampler already
computed those exact vectors while sampling. **Capturing** them in the sampler (one `assign` per step of `hidden`
before the final block into a `[B, capacity, dim]` fp32 buffer, or a host copy per window) makes the frozen stack
**bitwise identical by construction**:
- It is not an approximation. The stack is frozen, so the features are the same function at every update.
- It is faster: `rollout_features` took 443 s in the T0 profile.
- It removes every Mamba, attention and conv site from the parity problem while LoRA stays on the tail.
  §4.2–4.3 are needed only once adapters move below block 41. Even then, activations entering the lowest adapted
  block can still be captured; only the blocks from there upward need invariant recompute.

**What remains after capture** is the tail: final-block `attn_norm`, `ffn_up` (+LoRA), relu², `ffn_down` (+LoRA),
the residual add, `output_norm`, lm_head, bias, /T and log_softmax, at B rows (sampler) versus 16 or 512 rows
(trainer). Five matmuls and three row reductions: exactly vLLM's op list.

---

## 5. Prioritized plan

Every step's gate is **exact**: fp32 outputs compared as uint32 bit patterns, never `allclose`. The per-op harness is
TML's batch-invariance test: `f(X)[r] == f(X[r:r+1])[0]` bitwise, for row counts {1, 2, 8, 15, 16, 17, 64, 128, 129,
512, 2048} and several r. Run it under `gpu-run check`. Timings go under `gpu-run time`.

| # | Change | Effort | Speed cost (estimate, to be measured) | Confirm |
|---|---|---|---|---|
| 0 | **Invariance harness**: the per-op bitwise test above, plus a per-site sampler-versus-trainer diff that feeds the sampler's block input into the trainer's block, so each site is isolated before chaining. Reuse `~/scratchpad/fp16/localize.py`, `sites.py` | 0.5 day | none | reproduces today's per-site gaps; block 12 dominant |
| 1 | **Capture final-block inputs in the sampler**; delete the `rollout_features` recompute for the frozen stack. Also route the last prompt token through the decode step, so token 0 has one path | 0.5–1 day (sampler + DayCare) | **saves ~443 s/update** (T0); +B×dim×4 B per step of writes | the trainer's input hash equals the sampler's capture; parity error then comes **only from the tail**. Expect it to fall well below today's 1.8e-3 mean |
| 2 | **Canonical projection** (`_Linear` + LoRA + lm_head): one hi/lo TC candidate family, rows padded to 16, K-order and split_k fixed per (N, K), no fp32 matvec, no heuristic path for > 128 rows (chunk rows over the promoted tile, as `plan_rows` already does) | 1–2 days | decode B ≤ 16: matvec → one M=16 TC tile, both weight-bandwidth-bound, expect ±10%. Routes that used split_k > 1 at small M lose parallelism: TML reports ~20%; per-projection 10–30% at B=16–64. Trainer tail: negligible | per-op test passes for every row count; **tail parity max error = 0.0** once step 3 lands |
| 3 | **Invariant row reductions**: RMSNorm, output norm, log_softmax (vocab 131k), pinned opts or an elementwise tree; one shared `logprobs(hidden)` function used by sampler and trainer | 1 day | <5% of a decode step (a few reductions per token) | per-op bitwise test; after steps 1–3: **update-0 parity max = 0.0**, `rho ≡ 1`; TIS becomes a no-op at update 0 |
| 4 | *(only when adapters reach below block 41 or full-stack recompute is needed)* **Canonical tiled attention** (§4.2) on both sides | 2–3 days | sampler: +0–10% (fixed partial arrays, suffix read in whole T tiles); trainer attention about equal FLOPs | block-12 isolated site diff = 0; then chained |
| 5 | *(same condition)* **Trainer Mamba = the sampler's replay DAG** (§4.3) and the conv tap order | 2–3 days | trainer Mamba forward 2–4x slower than chunk-64 (256 sequential flushes and 16x-expanded readouts), small against the ~4000 s backward | per-Mamba-block isolated diff = 0; chained block outputs 0..41 bit-equal |
| 6 | Residual risks: fma contraction differences, the TC instruction shape varying with the M tile, BEAM or `TC_OPT` env drift between processes | ongoing | none | run the harness with and without fmad; pin env flags in the run manifest (see the three-way getenv split note) |

**Order rationale:**
- Steps 1–3 are enough for the **current** RLOO design (LoRA on the tail) and should take the gate from max 0.31 to
  exactly 0.
- Step 1 is the largest single win and is free in speed. Doing 4–5 first would spend most of the effort on sites
  that step 1 removes.
- Steps 4–5 are the general one-stack answer. They follow vLLM for attention and TorchTitan/RFC #55524 for the SSM.
- Expected end-to-end cost of 1–3: the sampler within about ±10%; the update faster, since the features are gone.
  That compares with vLLM/TorchTitan's 2.4–2.9x compute cost for full-stack invariance, which steps 4–5 would begin
  to approach.

**What not to do:**
- Don't keep tuning tolerances or relying on TIS for a gap that has a 0.31 tail.
- Don't make the sampler run a Q ≥ 64 SSD to match the trainer: that reverses the cheap direction.
- Don't trust `allclose`-style checks. The only gate that composes over 42 layers and 4k tokens is bit equality.
