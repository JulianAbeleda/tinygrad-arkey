# Nemotron 3 Nano 4B rollout sampling: tinygrad-arkey vs vLLM / SGLang / TRT-LLM / llama.cpp gap audit

Date: 2026-09-24. Read-only audit: no repo edits and no GPU use. Worktree `~/tinygrad-self-training` (branch
`tinygrad-self-training-mvp-20260907`, HEAD `31612e6a8`, with uncommitted in-flight edits to `heuristic.py` and
`nemotron_h.py`). `~/tinygrad-arkey` is at `caa752ac4` (nvidia-bringup). Line numbers refer to committed HEAD.

Evidence classes: **OBS** means measured in a repo doc. **CODE** means read from source. **CALC** means arithmetic from
measured R/BW. **EXT** means an external source. **INF** means an inference that has not been measured.

---

## 0. Findings that change the plan

1. **Tensor cores are not blocked by promotion. They are blocked by operand dtype.** In `nemotron_h.py`, the residual
   stream is fp32 (`prefix()`/`advance()` use `.float()`, lines 418 and 440), and every `nn.Linear` receives an fp32
   activation (`ssm_in(hidden)` at l.158/219, `attn_q/k/v` at l.290-292, `ffn_up` at l.352; `ssm_out` gets
   `.cast(hidden.dtype)`, which is fp32, at l.208/275). A bf16 weight multiplied by an fp32 activation becomes an
   fp32×fp32 MUL. The generic TC opt matches the MUL's source dtypes (`postrange.py:434-447`). fp32 input has only
   the tf32 descriptor, and that descriptor is disabled on NV unless `ALLOW_TF32=1` (`postrange.py:438,446`;
   `helpers.py:274`). As a result, **no Nemotron matmul can reach `mma.sync` today**, at any M, whether or not
   anything is promoted. The bf16 descriptor `cuda_81616` (bf16 in, fp32 out, m16n8k16) is already present on sm_120
   (`tc.py:113-119,139-149`; `cuda.py:299`). HF and vLLM feed bf16 into the GEMMs and keep only the residual in fp32.
   (CODE)
2. **The promoted sm_120 candidate set cannot be used on this model as is. Its schedule can be reused (see §1b).** It is now promoted in the worktree
   (`prefill_candidate_runtime.py:28-29`, `generated/prefill_sm120_lds_dbuf_candidate_set.json`, commit `98ed11fa5`).
   However, it covers exactly 4 exact shapes for **Qwen3-8B fp16 at M=512** (`attn_kv 512×1024×4096`, `attn_qo`,
   `ffn_down 512×4096×12288`, `ffn_gate_up`). It is bound to a GGUF Q4_K inventory (`automatic_promoted_prefill_graph_policy`,
   l.269+) and requires exact tile divisibility with tile 128×128×32 (l.~218). Nemotron's dim is 3136 = 64·49, so
   3136 is not a multiple of 128, and ssm_in N = 17504 = 32·547 is not a multiple of 64. **Nemotron needs a bf16 row and its own shape rows on that schedule,
   not a new search family.** The candidate set is present in both trees
   (`bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-sm120-v1/`). (CODE)
3. **The "12k tok/s" ceiling leaves out Mamba state traffic, which dominates at batch ≥64.** Per sequence,
   the SSM state is 21 layers × 96 heads × 80 × 128 = **82.6 MB in fp32** (41.3 MB in bf16). Each step reads it once
   and writes it once at minimum. NVIDIA's card tells vLLM to use `--mamba_ssm_cache_dtype float32` (EXT, HF card).
   Floors, with R = 255.4 TF measured and BW = 1700 GB/s measured (`extra/llm_research/microbench/README.md`), weights
   of 7.12 GB/step excluding the embedding, and an average of 2048 generated tokens per sequence with the shared prompt
   KV read once (CALC):

   | B | weights | SSM state r+w (fp32 / bf16) | KV (cascade / no-share) | compute@R | floor tok/s (fp32 / bf16 state) |
   |---|---|---|---|---|---|
   | 1 | 4.19 ms | 0.10 / 0.05 | 0.12 | 0.03 | 227 / 230 |
   | 8 | 4.19 | 0.78 / 0.39 | 0.25 / 0.93 | 0.22 | 1.5k / 1.7k |
   | 64 | 4.19 | 6.22 / 3.11 | 1.36 / 7.43 | 1.79 | 5.4k / 7.4k |
   | 128 | 4.19 | 12.4 / 6.2 | 2.6 / 14.9 | 3.6 | **6.65k / 9.8k** |
   | 256 | 4.19 | 24.9 / 12.4 | 5.2 / 29.7 | 7.2 | does not fit / 11.8k |

   With fp32 state, the realistic target is about 5k tok/s at B=128 (75% of the floor). A 6-10k target needs **bf16
   SSM state and prefix-shared KV**. Memory at B=128 with fp32 state, 4096-token generation, and a 10k shared prompt
   is 27.3 GB (fits). B=256 needs 46 GB and does not fit. **If each sequence gets its own copy of the prompt KV, as
   today's sampler does (`sampler.py:36,54`), B=128 needs +21 GB and does not fit at all.** Throughput above B≈128 is
   state-bound regardless of how good the GEMMs are.
4. **The current sampler does full-capacity attention and copies the prompt.** The KV buffers are
   `[B, 8, capacity, 128]` per attention layer (`sampler.py:36`). The prompt KV is broadcast into every row (l.54).
   Every step's SDPA reads the entire capacity under a mask (l.82-84), so the attention cost is O(capacity), not
   O(length). The Mamba step writes a fresh state and then `assign`s it into the buffer (l.87-90), which costs at
   least one extra full state read and write. Each step does a host round-trip (`.numpy()` at l.101-110). (CODE)
5. **Nemotron 3 Nano 4B ships no MTP heads.** The local GGUF has 263 tensors and none match `mtp` or `nextn` (CODE).
   The HF card mentions no MTP or speculative decoding (EXT). NeMo-RL's 1.3-1.8× comes from EAGLE-3/MTP on other
   models. At B≥64 we are state- and weight-bound, not latency-bound, so speculation has little value here.
6. **Per-kernel launch cost on NV/HCQ is already small.** HCQGraph replay is CUDA-graph class: 1648 launches with
   277 µs of total positive gaps, p50 = 0 and p95 = 1.75 µs (`nv-*` replay record; grep "exactly1648 launches"). DEBUG
   `tm` overhead is about 0.65 µs/kernel (`five-lever-test-record-20260803.md` §1). At 799 kernels that is about
   0.1-0.5 ms per step, which is not the bottleneck at batch. The kernel-count problem is that each kernel moves bytes
   (materialized intermediates) and runs small ops with poor occupancy, not the launch itself. (OBS)

## 1. Model facts (from the local BF16 GGUF, CODE)

- 42 blocks: **21 Mamba-2, 17 MLP, 4 attention** (layers 12, 17, 24, 32). dim = 3136, vocab = 131072, and
  `output.weight` is separate from `token_embd`.
- Mamba: `ssm_in` 3136→17504 (z 7680 | xBC 9728 | dt 96), conv k=4 over 9728 channels, 96 heads × headdim 80,
  8 groups, state size 128, gated RMSNorm with group size 960, `ssm_out` 7680→3136.
- Attention: 40 q heads, 8 kv heads (GQA 5), head dim 128, no RoPE. KV is 16 KB per token across all 4 layers.
- MLP: 3136→12544→3136 with squared ReLU and no gate.
- Weight bytes per step: SSM 3.32 GB, MLP 2.67, attention 0.31, lm_head 0.82, for a total of **7.12 GB**
  (7.1 GFLOP/token).
- The Hq=40/Hkv=8/Hd=128 attention geometry is **identical to Qwen3-14B**. The promoted flash-decode route
  `FLASH_DECODE_G5` (Hq 40, split 32, qg 2, stage width 4) already matches it, but only at B=1
  (`flash_decode_attention.py:1217-1231`).

## 1b. Qwen route/kernel reuse map (kernels transfer unless hardware- or format-specific)

Rule: a Qwen route counts as **have (reuse)** when its mechanism is width- and shape-parametric and already
promoted for NV sm_120. Redo is needed only for gfx1100-only artifacts. A route bound to packed Q4_K/Q6_K bytes
does not read BF16, so it counts as **partial (reuse spec, new dtype row)**.

| Qwen asset | promoted sm_120? | applies to Nemotron dense layers? | verdict |
|---|---|---|---|
| Dense WMMA/LDS-dbuf GEMM candidate schedule (`prefill_sm120_lds_dbuf_candidate_set.json`: 128×128×32, 4×2 warps, LDS double buffer, `cuda_mma_f32_8x16x16_f16_lds2_static`) | yes (fp16 overlay, `98ed11fa5`) | The schedule and lowering are shape-parametric; this is the dense path Nemotron needs. Two things block it. (a) The precontract path admits only `dtype_in ∈ {half, char}` (`codegen/opt/kernel_lds.py:54-57`), so it needs a bf16 row (same m16n8k16 instruction, bf16 variant). (b) Entries are exact-shape rows, so Nemotron role shapes need new rows, with N=3136 needing n-tiles of 64. | **have (reuse schedule)**: add a bf16 dtype and Nemotron shape rows through BB. Not a new search family. |
| Generic TC descriptor `cuda_81616` bf16→fp32 (`tc.py:113-119`) | NV generic | yes, once operands are bf16 (finding 1) | **have (reuse)** |
| Flash decode `FLASH_DECODE_G5` (Hq 40 / Hkv 8 / Hd 128, split 32, qg 2) + `FlashCombineSpec` (`flash_decode_attention.py:1228-1231`) | yes | The geometry is identical to Nemotron's attention. `shape_ok` requires B=1 (l.1217-1221). Nemotron has no RoPE, which the kernel already supports via `rope=False`. The KV dtype on the Qwen route is fp16, so use an fp16 KV cache or add a bf16 loader row. | **have (reuse) at B=1**. Batch axis, per-sequence lengths and the shared-prefix segment are the new work (W7). |
| RMSNorm semantic lowering `Ops.RMSNORM` → `DecodeRMSNormSpec` (`schedule/rangeify.py:320-396`, `decode_kernels.py:649`; policy `decode-rmsnorm-native-lowering`) | yes (site-filtered to Qwen's 4096-wide sites) | The rows/dim/dtype parameters cover B rows × 3136 in fp32 | **have (reuse)**: emit `Ops.RMSNORM` from Nemotron's norms and admit the 3136-wide sites (policy row). Gated RMSNorm (group 960 × z·silu) is an extension. |
| `reduce-output-rmsnorm`, `decode-norm-fusion` (model.py:975/1185, 2498-2626) | yes | The mechanism is generic. It is wired only in `Transformer` (`model.py`) | **have (reuse)**: needs binding in `nemotron_h.py`. |
| Q4_K/Q6_K decode GEMVs (`q4k_g3_lanemap_gemv_*`, `Q6KGEMVRouteSpec`, four-warp gate/up, vocab coop) | yes | Byte format mismatch: they decode Q4_K/Q6_K blocks, not bf16. Lane-map, cooperative-row and vocab-coop structure transfers. They would apply directly to the local `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`, but that is off-policy for RL. | **partial (reuse spec structure)**: add a bf16 row to the GEMV spec family if the in-flight heuristic plateaus |
| Epilogue/resadd fusions (`decode-q4k-epilogue-*`, `decode-ffn-down-resadd`) | yes | Fused into the Q4_K/Q6_K mmvq bodies (`q4k_ffn_down_mmvq.py`, `q6k_ffn_down_mmvq.py`) | **partial**: the epilogue contract carries over once a bf16 GEMV row exists |
| Native argmax / vocab top-1 (`packed_argmax.py`, `emit_q6k_vocab_top1_reduce_kernel`) | yes | Top-1 over the vocab works for Gumbel-max if noise is added in the operand, plus a logsumexp side output for the logprob | **partial (reuse)** |
| NV prefill fused attention | NV prefill currently runs a native (llama-derived) Flash binary. The generated replacement is open (M4 in `nv-generated-inference-completion-plan-20260906.md`). AMD `flash_prefill_attention` is generated and Hd-generic. | Only 4 layers, so the prefill share is small | **partial**: the generated kernel is the AMD artifact; sm_120 needs the M4 work. Not a Nemotron-specific item. |
| HCQGraph/TinyJit replay, PDL scaffolding (`cuda.py:11-45`, research) | yes | model-agnostic | **have (reuse)** |
| Mamba-2 anything | — | no Qwen analogue | **missing** (genuinely new) |

Net: GEMM schedule, flash decode, RMSNorm lowering, the graph runtime and argmax all carry over. Genuinely new work:
the bf16 operand policy (W1), bf16 and shape rows on existing schedules (W4), a batch axis on flash decode (W7), the
Mamba kernels (W3, W9), and the scheduling/memory work (W2, W8).

## 2. Gap matrix

Legend for tinygrad-arkey: **H** = have, **P** = partial, **M** = missing. The build routes are **SCHED** =
`tinygrad_scheduler_generated` (ordinary Tensor code plus the heuristic), **BB** = `machine_authored_generated`
(BoltBeam descriptor, search, and promotion), and **RT** = runtime/Python infrastructure (no kernel authorship). A
**✗** marks a route that is illegal on the default path under the purity contract (`docs/pure-machine-search.md`).
Porting Triton, CUDA, or FlashInfer kernels is ✗; they may be used only as oracles.

| # | capability | vLLM | SGLang | TRT-LLM | llama.cpp | tinygrad-arkey (evidence) | expected impact on us | legal route | effort | deps |
|---|---|---|---|---|---|---|---|---|---|---|
| 1a | TC GEMM, bf16, M=64-256 (decode at batch) | cuBLAS/cuBLASLt heuristics, torch.compile+Inductor; sm_120 uses mma.sync (no wgmma/tcgen05) (EXT) | cuBLAS + FlashInfer/Triton | own cuBLASLt/CUTLASS plugins, autotuned tactics | `mmf.cu` (mma tiles for small batch), cuBLAS above (EXT) | **P (reuse)**: TC descriptor present (`tc.py:113`), and the promoted sm_120 LDS-dbuf schedule can be reused (§1b). Blocked by fp32 operands (finding 1), by bf16 not being admitted in `kernel_lds.py:54`, and by missing Nemotron shape rows (finding 2). | Compute at B=128 is 3.6 ms at R. Scalar FMA runs ≲10% of R, so ≥36 ms. **This is the single largest lever at B≥32: 5-10× step time.** | SCHED (cast activations to bf16 → generic TC), then BB (Nemotron role shapes; split ssm_in into z/xBC/dt for divisibility; n=64 tiles for N=3136) | S (dtype) + M (bf16 + shape rows on the reused schedule) | model dtype decision; trainer shares the code |
| 1b | TC GEMM, M=8-32 | cuBLAS / "skinny" paths | same | same | `mmf` for ≤16 cols, `mmvf` below (EXT) | **P**: in-flight batched-matvec heuristic (MV_MAX_BATCH=16, working-tree `heuristic.py`) reuses weight loads across M ≤ 16 rows on CUDA cores | bandwidth-bound regime; a good matvec is enough to about M=16-32 | SCHED (in flight); BB for a small-M TC tile | S-M | in flight |
| 1c | TC GEMM for prefill (M = chunk 512-4096) | cuBLAS / Triton | same | same | mmf/cuBLAS | **P (reuse)**: same schedule and same blockers. Qwen NV prefill evidence: generated pp512 47.8 ms vs llama 38.8 ms (`nv-generated-inference-completion-plan-20260906.md`), i.e. the path exists for exact-shape Q4_K | a 10k prime is 71 TFLOP, which takes 0.28 s at R. The scalar path is minutes. | SCHED then BB (same search as 1a, M=512/1024) | M | 1a |
| 2 | GEMV M=1 bandwidth | CUDA-graph'd cuBLAS gemv / custom (EXT: no hard numbers) | same | same | `mmvf` bf16 | **P (reuse structure)**: Qwen Q4_K/Q6_K GEMV specs are promoted on sm_120 but read packed formats, not bf16 (§1b); 23% of BW (caller); GGUF lazy-decode bug fixed in flight (materialize weights, `nemotron_h.py` diff). Qwen Q4_K generated decode reached llama parity at d512 (+0.113%) (completion plan) | only B=1/8 latency; irrelevant to B=128 throughput | SCHED (in flight); BB GEMV spec (like `Q6KGEMVRouteSpec`) if the heuristic plateaus | S-M | in flight |
| 3 | Launch overhead / graphs | FULL_AND_PIECEWISE CUDA graphs, per-batch-size capture (`cudagraph_capture_sizes`) (EXT) | CUDA graph per bs | CUDA graph | CUDA graphs | **H**: TinyJit→HCQGraph direct GPFIFO; gaps p50 = 0, ~0.17 µs/launch in replay (OBS). One graph per fixed B; the position is a symbolic UOp (`sampler.py:43`) | small (≤0.5 ms/step) | RT: capture one JIT per B bucket | S | 9 |
| 4a | norm+residual, QKV/gate fusion, activation epilogue | fused_add_rms_norm CUDA op; merged QKV column-parallel; SiLU-and-mul / relu² epilogue (EXT, general) | same (sgl-kernel) | plugins | ggml graph fusion (rms_norm+mul, add) | **P (reuse)**: the scheduler fuses elementwise into reductions. RMSNorm semantic lowering and norm fusion are promoted on sm_120 and reusable (bind in `nemotron_h.py`). The resadd/epilogue fusions are inside Q4_K/Q6_K GEMV bodies and carry over once a bf16 GEMV row exists (§1b). 799 kernels/step (caller) | 10-25% at B≥64 via fewer materialized [B,17504] / [B,12544] intermediates | SCHED (merge q/k/v into one weight; relu² folds into the ffn_up/ffn_down boundary) | S-M | 1a |
| 4b | sampling kernel | FlashInfer sorting-free top-k/p (rejection) + Gumbel; logprobs_mode raw/processed (EXT) | FlashInfer sampling | fused sampling | CPU/GPU samplers | **P (reuse)**: Gumbel-max plus full fp32 log_softmax over 131072 × B, several kernels (`sampler.py:61-67`). The promoted sm_120 native argmax / vocab top-1 (`packed_argmax.py`) is reusable with a logsumexp side output | ~0.1-0.3 ms at B=128 (logits 67 MB); correctness matters more | SCHED: one online-softmax pass (max, sum-exp, Gumbel argmax, chosen logit) plus a gather | S | — |
| 5 | Mamba-2 decode: fused `selective_state_update` + `causal_conv1d_update` | Triton SSU (`ssu_dispatch`) + causal_conv1d_update in `mamba_mixer2.py` (EXT) | Triton (mamba_ssm-derived) | selectiveScan plugin | `ssm_conv` / `ssm_scan` CUDA ops | **M**: step runs `cached()` → `_scan_chunk` with L=1 (matmuls over 1×1 decay, cumsum, masks), then `contiguous().realize()` and `assign` (`sampler.py:87-90`). The state is read ≥2× and written ≥2× (INF) | **large at B≥64**: the floor is 12.4 ms of state traffic at B=128 in fp32. Today's ≥2× multiplier costs another ≥12 ms | SCHED first (a dedicated 1-token update written as `state.assign(state*dA + dt·x⊗B)`, `y = (state·C)` + D·x); BB descriptor if one-pass read/write needs a multi-output kernel | M | — |
| 5b | SSM state dtype | fp32 recommended for this model (`--mamba_ssm_cache_dtype float32`, EXT) | configurable | configurable | f32 | fp32 (`Tensor.zeros` default) | bf16 state = +48% at B=128 (6.65k → 9.8k floor) and allows B=256; logprob drift risk | policy decision; oracle-gated | S | 11 |
| 6 | Mamba-2 prefill: SSD chunked scan + chunked prefill | `mamba_chunk_scan_combined` (chunk 256, Triton, varlen), chunked prefill for hybrids in V1 (EXT) | same, plus varlen | plugin | sequential `ssm_scan` | **P**: exact chunked SSD form (chunk 64; `nemotron_h.py:129-153`), but a **Python loop per chunk with realize** (l.257-266): 157 chunks × 21 layers ≈ 3300 serial groups at 10k. Attention prefill does per-128-query-chunk SDPA with a growing key length (l.325-337). Every chunk is a new shape, so each one compiles a new kernel (INF, probable prime-stall cause) | 10k prime: stalls → target ≤1 s (compute floor 0.28 s) | SCHED: vectorize the intra-chunk diagonal across all chunks at once, then run the inter-chunk state pass over L/256 chunks; fixed-size attention chunks padded to bucketed key lengths | M | 1c |
| 7a | Attention decode (paged, split-KV, GQA) | FlashAttention/FlashInfer paged + split-KV; GQA packed heads (EXT) | FlashInfer / Triton | XQA / MMHA | FA-vec kernels, unified KV | **H (reuse) at B=1 / P at batch**: the promoted generated flash-decode G5 route (Hq40/Hkv8/Hd128) is Nemotron's exact geometry, but `shape_ok` admits only B=1 (`flash_decode_attention.py:1217-1221`). The sampler does not use it yet; it runs plain SDPA over the full capacity | at B=128 and a 2k average, KV is 2.6 ms of the floor; today's O(capacity) read is ≥5× that | BB: extend `FlashDecodeAttentionSpec` with a batch axis and per-sequence lengths; SCHED interim: slice to a length bucket | M | 7b |
| 7b | Shared-prompt KV (prefix sharing / cascade) | block sharing via prefix caching; cascade attention exists but is off by default (EXT: vLLM PR 55652) | RadixAttention shares blocks | KV reuse | `-np` slots with a copied or shared prefix | **M**: the prompt is copied into every row (`sampler.py:54`) | **enables B=128 at all** (−21 GB); saves up to 12 ms/step of KV reads at B=128 with a 10k prompt | SCHED: two-segment attention (shared prefix [1,8,P,128] plus per-sequence suffix), merged by LSE; BB later inside 7a | S-M | — |
| 8 | Continuous batching / variable lengths / preemption / memory planning / hybrid cache manager | V1 scheduler; HybridKVCacheCoordinator unifies attention pages and Mamba state pages; preemption by recompute (EXT) | overlap scheduler; MambaRadixCache | in-flight batching | slots (`-np`), unified KV | **M**: fixed B, run until all sequences finish (`sampler.py:103-110`); finished rows keep computing | RL long tail: at a max of 4096 with a median of about 1-1.5k, **≈2-3× effective throughput** (INF) | RT: slot table with a per-slot position vector (not a scalar `position` UOp), per-slot prefix id, refill finished slots with the next group's sequences; state/KV indexed by slot | M-L | 7a, 7b |
| 9 | Prefix caching / prompt prefill once per group | automatic prefix caching; Mamba "align" mode, experimental (EXT) | MambaRadixCache | KV reuse | prompt cache | **H** (per group): `prime()` runs the prompt once and broadcasts it (`sampler.py:46-59`). Cross-group caching is missing (not needed for RL) | done, apart from the broadcast copy (7b) | — | — | — |
| 10 | FP8 on sm_120 | FP8 W8A8 via CUTLASS/Marlin; some SM12x kernels gated out (EXT: vLLM #43906) | FP8 | FP8/NVFP4 | Q8_0 etc. | cuda_81632_f8 descriptors exist (`tc.py:120-124,145`) | ~1.8× on weight bytes, but the policy is BF16 and FP8 widens the train/rollout mismatch (EXT arXiv 2510.26788 finds BF16 already mismatches). **Not for the policy** | — | — | — |
| 11 | Speculative decoding / MTP | EAGLE/MTP; NeMo-RL reports 1.3-1.8× (EXT) | EAGLE | Medusa/EAGLE/MTP | draft models | **M**. The model has no MTP heads (finding 5) | ≈0 at B≥64 (not latency-bound); only B=1 eval | — | — | — |
| 12 | RL integration: weight sync, logprob parity, async rollouts | NeMo-RL refit (`prepare_refit_info`), verl/OpenRLHF weight broadcast; trainer recomputes logprobs; async GRPO (EXT) | same | — | — | **H (structurally)**: a single tinygrad stack, same weights in place, **no weight sync**. Logprobs come from the same logits used to sample (`sampler.py:62-66`). A parity gate against the trainer's full-sequence recompute is missing | correctness, not speed; prerequisite for any precision change | SCHED | S | — |
| 13 | Other: detokenize off-path, pinned buffers, host overlap | detokenizer process; async scheduling overlaps CPU with GPU (EXT) | overlap scheduler | C++ runtime | C++ | **M**: host sync plus Python list work every step (`sampler.py:101-110`) | ~0.2-1 ms/step; matters once the step is ≤10 ms | RT: device-resident token/logprob ring buffer, read back every N steps; stop mask on device | S | — |

## 3. Build plan (ordered by impact per effort, respecting dependencies)

A **step** is one `NemotronHBatchSampler.step` replay. "Logprob parity" means: for 8 sequences × 256 tokens, the
per-token logprobs from sampling match a fresh full-sequence recompute through the training path
(`model.advance`/forward with the same precision policy), **mean |Δ| ≤ 1e-3 and max ≤ 1e-2**, and the argmax tokens
match under a fixed Gumbel seed. "Oracle" means HF/torch BF16, used only as a checker.

- [ ] **W0: Ledger and harness (S).** Reuse `HCQ_GRAPH_PROFILE_JSON`, DEBUG=2 `tm`, and the W==D method
  (`extra/llm_research/decode/decode_runtime_overhead.py`). Do not build a new harness. Add a per-step class census
  (GEMM, SSM, attention, norm/elementwise, sampling) at B ∈ {1, 8, 64, 128}, plus the CALC floor table above as the
  denominator. **Gate:** the census sum is within 5% of the measured step wall, and every kernel is assigned to a class.
- [ ] **W1: bf16 GEMM operands → generic tensor cores (S, highest impact per effort).** Keep the fp32 residual and
  cast the normed activation to bf16 at every Linear input: attn_norm output, gated-norm output before `ssm_out`,
  attended before `attn_output`, and relu² before `ffn_down`. This matches HF/vLLM numerics. Then verify that the
  generic TC opt fires (`cuda_81616` bf16→fp32). This is a numerics change shared with the trainer, so Julian must
  sign off. **Gate:** DEBUG=4 shows `mma.sync ... bf16` in every projection kernel at B=64 and at prefill M=512. Logprob
  parity holds, and the oracle Δ is no worse than today's fp32 path. The B=64 step is ≥3× faster than before.
  *This supersedes "promote the sm_120 candidate set first": that set is shape-bound to Qwen, and dtype is the actual
  blocker.*
- [ ] **W2: Prefix-shared, length-bounded KV (S-M).** Store the prompt KV once as `[1, 8, P, 128]` and give each
  sequence its own suffix buffer. Use two-segment SDPA with an LSE merge in ordinary Tensor code, and bound the suffix
  read to a length bucket (for example, powers of two, one JIT per bucket) instead of the full capacity. **Gate:**
  B=128 at P=10k with 4096-token capacity fits with ≥3 GB headroom. Attention per step is ≤1.5× the CALC KV term at
  a 2k average. Parity holds.
- [ ] **W3: One-pass Mamba decode update (M).** Write a dedicated 1-token path (no `_scan_chunk`, cumsum, or causal
  mask): conv as a 4-tap dot over (conv_tail, new), `dA = exp(dt·A)`, and `state.assign(state*dA + (dt·x)⊗B)` in
  place, with `y = state·C + D·x`. No separate realize-then-assign copy. If the scheduler cannot produce a single
  read and write, file a BB descriptor (multi-output reduce; reuse the flash composite-reduce substrate). **Gate:**
  SSM-class bytes per step are ≤1.25× (2 × B × 82.6 MB) at B=64. SSM-class time is ≤1.3× its CALC floor. State
  matches the chunked path bit-for-bit, or within 1e-6 relative in fp32.
- [ ] **W4: Reuse the promoted sm_120 LDS-dbuf GEMM schedule with bf16 and Nemotron shape rows (M; was L before the reuse check).** Admit bf16 in `kernel_lds.py:54-57` (same m16n8k16 family), then run a BoltBeam re-bind and small search around the existing geometry. Roles: ssm_in, split into z
  7680 / xBC 9728 / dt 96 (or padded); ssm_out 7680→3136; ffn_up 3136→12544; ffn_down 12544→3136; qkv fused 3136→7168;
  o 5120→3136; lm_head 3136→131072. M buckets are {16, 32, 64, 128, 256} for decode and {512, 1024, 2048} for
  prefill. N=3136 requires n-tiles of 64 or 32·k. Promote through the TG8 target-parametric artifact path
  (`prefill_candidate_runtime.py:162-176`). The current path is bound to GGUF-inventory M=512, so it needs a
  bf16/Nemotron profile row. **Gate:** each role at M=128 reaches ≥50% of R (≥128 TF), and at M=64 reaches ≥65% of BW.
  Full step at B=128 with fp32 state is ≤25 ms (≥5.1k tok/s). Parity holds.
- [ ] **W5: Fusion and kernel-count reduction (S-M).** Fuse q/k/v into one weight. Fuse residual add with the next
  RMSNorm. Fold relu² into the ffn_up epilogue, and fold z-gate·silu with the group RMSNorm. Use a one-pass sampler
  (online max, sum-exp, Gumbel argmax, and chosen logit in one reduction). **Gate:** ≤300 kernels/step (llama.cpp
  class is about 6-8 per block). Non-GEMM, non-SSM time at B=128 is ≤10% of the step.
- [ ] **W6: SSM-state dtype decision (S, policy).** A/B fp32 vs bf16 state against the oracle over full 4096-token
  rollouts. **Gate:** bf16 is adopted only if logprob parity with a bf16-state trainer recompute holds and the KL
  against fp32-state rollouts is ≤1e-3/token. Otherwise stay on fp32 and cap at B=128. Recording the result is enough.
- [ ] **W7: Batched flash-decode (BB, M; reuses the promoted G5 route).** Extend `FlashDecodeAttentionSpec` G5 (already the right geometry) with a
  batch axis, per-sequence lengths, and the shared-prefix segment. **Gate:** attention time at B=128 with a 2k average
  is ≤1.2× the CALC KV floor, and parity holds.
- [ ] **W8: Device-resident loop and slot scheduler (M-L).** Keep tokens, logprobs, and the finished mask in device
  buffers, read back every 32 steps, and pass a per-slot position vector plus a per-slot prefix id. Refill finished
  slots with the next group's sequences, priming new prompts in chunked prefill between decode steps. **Gate:** on a
  recorded GameTerm length distribution, the active-slot fraction is ≥90%, host time is ≤0.3 ms/step, and aggregate
  throughput is ≥0.85 × (B × 1/step_time).
- [ ] **W9: Prefill / prime (M; depends on W1 and W4).** Vectorize the SSD intra-chunk term across chunks with chunk
  256, run the sequential inter-chunk state pass, and use fixed-shape attention chunks so there is no per-chunk
  recompile. **Gate:** a 10k-token prime takes ≤1.0 s warm and ≤ +30 s of cold compile on first use. Output states
  match the whole-sequence forward within 1e-4 relative.
- [ ] **W10: Per-B-bucket JIT capture (S).** Capture graphs at B ∈ {8, 16, 32, 64, 128}, the analogue of
  vLLM's `cudagraph_capture_sizes`, so shrinking batches at the end of the tail stay efficient. **Gate:** in the tail
  phase, step time follows the bucket and not the maximum B.

Projected (INF): W1+W2+W3 take B=64 from "unmeasured, likely ≥150 ms" to about 20-30 ms (2-3k tok/s). W4+W5 reach
about 5k tok/s at B=128 with fp32 state. W6 (if bf16 passes) and W8 are needed for ≥6-8k. Reaching vLLM-level (~10k)
requires bf16 state, the long-tail fix, and ≥75% of every floor.

## 4. Already handled by the in-flight sampler agent (do not duplicate)

- **B>1 matvec heuristic:** batched MV path with `MV_MAX_BATCH=16`, reading bf16 weights through a widening CAST
  (working-tree `tinygrad/codegen/opt/heuristic.py`, `_loaded` + batch upcast).
- **B=1 GEMV bandwidth:** the GGUF lazy-decode fix materializes weights (`nemotron_h.py` working-tree diff: "a decode
  matvec ran at ~3% of memory bandwidth") and continues from the 23% figure.
- **Kernel-count reduction in the sampler step:** 799 kernels, work in progress. W5 above is the follow-on target, so
  coordinate before starting it.
- **Long-prompt prime stall:** W9 overlaps with it. Hand the agent the per-chunk-recompile hypothesis: variable
  `key_stop`/`stop` shapes at `nemotron_h.py:326-337`, plus about 3300 serial realized scan chunks at l.257-266.

Note for that agent: its matvec work tops out at about M=16. Nothing it is doing unblocks tensor cores, because W1 is
the blocker at M≥32.

## 5. Dead ends to avoid

- **Concurrent multi-queue / compute overlap on NV:** one GPFIFO per HCQGraph, and measured replay overlap is 0.0%
  (`five-lever-test-record-20260803.md` §1, `decode-replay-overlap-measurement-record-20260803.md`).
  `HCQ_NV_MULTI_QUEUE_*` knobs exist, but the probe showed no win. Do not reopen for this model.
- **Binding the Qwen sm_120 candidate *rows* to Nemotron unchanged:** they are exact-shape, fp16, and GGUF-inventory bound (finding 2). Reuse the *schedule* with new bf16/shape rows (W4). Do not start a fresh search family.
- **Enabling TF32 (`ALLOW_TF32=1`) to get tensor cores on fp32 activations:** runs at half the bf16 rate
  (m16n8k8-class; `README` measured 125.8 TF for k8) and adds a numerics mode that neither the trainer nor the oracle
  uses. Use bf16 operands instead.
- **Porting vLLM/mamba_ssm Triton or FlashInfer CUDA kernels:** `external_handwritten_kernel`, never a default route.
  They are acceptable only as off-path oracles.
- **Speculative decoding / MTP:** no heads; the regime is not latency-bound at B≥64.
- **FP8 weights:** policy is BF16, and FP8 worsens the rollout/train mismatch.
- **Chasing launch overhead:** HCQ replay gaps are already sub-µs. Kernel count matters only through bytes and occupancy.
- **Runtime BEAM:** removed. Search happens offline in BoltBeam.
- **Tier-3 attention micro-levers before W2/W7:** split-KV combine tax, waitcnt tuning, and DS-offset folding were
  measured losses or flat results on the Qwen path (`kernel-golf-method-20260830.md` §2).
- **Using the "12k tok/s" ceiling:** it leaves out SSM state traffic. Use the §0 table.

## 6. Biggest unknowns

1. **How well the generic TC opt does on sm_120 for these shapes once W1 lands.** No LDS staging or double
   buffering. The Qwen record suggests generic kernels sit well below the searched candidates. This decides whether
   W4 (L effort) is needed immediately or later.
2. **Whether the tinygrad scheduler can express the Mamba decode update as one state read plus one write**,
   including the y readout. If it cannot, W3 needs a BB multi-output descriptor, which is M→L effort.
3. **Whether bf16 SSM state and bf16 GEMM operands keep RL logprob parity** over 4096-token rollouts, given that NVIDIA
   recommends an fp32 state. This sets the ceiling: about 6.6k (fp32, B≤128) or about 10-12k (bf16, B up to 256).

## Sources

- Repo: `docs/nv-prefill-decode-diagnosis-20260801.md`, `docs/kernel-golf-method-20260830.md`,
  `docs/pure-machine-search.md`, `docs/README.md`, `docs/task_workflow/input/five-lever-test-record-20260803.md`,
  `docs/task_workflow/input/nv-generated-inference-completion-plan-20260906.md`,
  `extra/llm_research/microbench/README.md` (R = 255.4 TF, BW = 1700 GB/s).
- HF card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (`--mamba_ssm_cache_dtype float32`, 4
  attention layers, no MTP mentioned).
- vLLM: https://docs.vllm.ai/en/stable/design/cuda_graphs.html ;
  https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager.html ; cascade off by default:
  https://github.com/vllm-project/vllm/pull/55652 ; hybrid prefix-cache tracking:
  https://github.com/vllm-project/vllm/issues/26201 ; SM12x FP8 gating: https://github.com/vllm-project/vllm/issues/43906
- SGLang MambaRadixCache: https://pytorch.org/blog/hybrid-models-meet-sglang
- FlashInfer sampling: https://flashinfer.ai/2025/03/10/sampling.html
- NeMo-RL speculative decoding: https://research.nvidia.com/labs/nemotron/rl-speculative-decoding
- Training-inference mismatch (FP16 vs BF16): https://arxiv.org/pdf/2510.26788
- TRT-LLM row: general knowledge, not re-verified in this pass. Treat its cells as INF.
