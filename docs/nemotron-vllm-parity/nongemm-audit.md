# Non-GEMM kernel audit: Mamba prefill (move 3) and decode non-GEMM kernels (move 4), 2026-09-26

Reverse-engineer + test half of moves 3 and 4 from `scan-20260926.md`. Measure-only: no product code changed.
Nemotron 3 Nano 4B BF16, RTX 5090, tinygrad `exp` @ 43aa3318d, vLLM 0.30.

Sources:
- Per-kernel tables come from the same 2026-09-26 traces the scan's `lifecycle-compare` used (ours: HCQ graph
  profile jsonl; vLLM: nsys sqlite). They are re-attributed with BoltBeam's own `lifecycle_compare` functions and
  grouped by kernel name (`bench/nongemm-audit-20260926/per_kernel.py` -> `per_kernel.json`). No new captures.
- The "tested" rows are isolated runs of the production code paths (a stub 4B block with random weights), under
  `gpu-run time`, with per-kernel times from `DEBUG=2` and graph walls from TinyJit replays. Scripts are in
  `bench/nongemm-audit-20260926/`; raw logs are in `/home/ubuntu/storage/audit-nongemm/`.
- No ncu this round. Every limiter below is identified from achieved FLOP/s or bytes/s against the roofline
  (bf16 tensor ~250 TF, fp32 FMA ~105 TF, DRAM 1.69 TB/s). NCU confirmation of the three replication targets is
  queued in section 4.

## 0. Headline

- **Mamba prefill is 10x slower because the SSD contractions run as fp32 CUDA-core reductions** (`precision="float"`,
  which production prefill uses). They reach 5-7 TF/s (5-7% of fp32 peak, 2-3% of the bf16 tensor roofline). vLLM runs
  the same five-step SSD (chunk 256, like ours) as Triton `tl.dot` on bf16 operands with fp32 accumulation.
  Tested: the same graph at `precision="bf16"` cuts the SSD core from 1917 to 393 us per 1024-token piece (4.9x).
  That is about **-250 ms per 10k prefill**, with y 1.5e-3 rms-relative and state 2.4e-3 from float. `"split"` (3 bf16
  products, 5.7e-5 / 4.5e-6) only reaches 1.4x (about -90 ms). The rest of the gap (about 64 vs 26 ms of SSD, plus
  50 vs 10 ms of conv/combine/norm) is fusion: vLLM computes intra-chunk, inter-chunk and the D skip in one
  `_chunk_scan_fwd_kernel`, while we materialize scores, weights, diagonal and previous in fp32.
- **Decode non-GEMM, ranked by recoverable time** (ms/step at B=32 / 128; all but the last are measured in isolation):
  1. sampler tail, full log-softmax materialized plus a recomputed split-reduce combine: **-0.32 / -1.36**
  2. decode attention, fp32-widened `_exact_dot` (CUDA cores) instead of the bf16 tensor-core dot: **-0.37 / -1.2 to -3.1**
  3. Mamba decode aux kernels (about 19 small kernels per layer, several at 10-30% of DRAM bandwidth): **-1.2 / -4.3**
     against our own floor, estimated, not tested. We already beat vLLM on the Mamba class as a whole.
  4. ring flush `r_*_6_427_8_16_3_16` at 21% of DRAM bandwidth: **-0.1 / -0.37**
  5. residual + RMSNorm as 3 kernels (reduce at 9% of bandwidth, latency-bound) vs vLLM's one fused kernel:
     **-0.54 / -0.59**; BEAM=4 does not change the reduce kernel
- **Attribution corrections for the scan's decode table.** At B=128, 0.52 ms of "sampling" is ring-flush kernels
  that sit in flush graphs after the first. And 1.03 ms of "mamba" is the ssm_out input prep (`E_2_640_32_4_3_4`,
  the hi/lo split of the gated-norm output), which is GEMM aux. Corrected B=128 deltas: sampling +1.38 (not +1.90),
  flush +1.02 (not +0.50), mamba -2.55, ssm_out aux +1.03.
- **BoltBeam importer fix (not committed; patch + test in `bench/nongemm-audit-20260926/boltbeam_importer_fix.patch`):**
  `boltbeam/profiler/ncu_counters.py:139-144` `mark_gemm` marks only the single longest launch as GEMM, so the other
  3 chunk launches of a prefill op count as aux. `boltbeam/profiler/ncu_audit.py:93-97` `_side` then takes one
  launch as `gemm_us`. The fix marks every launch of the GEMM kernel's name as GEMM and sums them. On the scan's
  `oncu.raw.csv`, ssm_in M=1024 goes from gemm 469 / aux 1681 us to 1860 / 291 us (the scan measured aux 317 us),
  and ffn_up M=1024 from 290 / 1044 to 1155 / 179 us. BoltBeam's 15 ncu tests pass. The new
  `test_ncu_chunked_gemm.py` fails on unpatched BoltBeam and passes with the patch.

## 1. Move 3: Mamba prefill

### 1a. vLLM's kernels (source: `vllm/model_executor/layers/mamba/ops/ssd_*.py`; trace `prof_prefill10k.sqlite`)

vLLM runs 10k as two chunked-prefill steps (8192 + 1808 tokens), so each layer launches each kernel twice. Chunk size
is 256 (`chunk_size` in config.json). Times are us per launch at 8192 / 1808 tokens:

| # | kernel | grid (8192 tok) | block / regs / smem | computes | dtypes | us 8192 / 1808 |
|---|---|---|---|---|---|---|
| 0 | `_causal_conv1d_fwd_kernel` | 1024x38 | 128 / 34 / 0 | depthwise conv (K=4) + bias + SiLU over xBC | bf16 in/out | 208 / 27 |
| 1 | `_chunk_cumsum_fwd_kernel` | 32x48 | 128 / 26 / 2 KB | dt = softplus(dt+bias), dA_cumsum = cumsum(dt*A) per chunk | fp32 out | 7.4 / 2.8 |
| 2 | `_chunk_state_fwd_kernel` | 6x32x96 | 128 / 80 / 16.9 KB | states[c,h] = x^T @ (B * exp(dA_last - dA_k) * dt_k): the decay is applied to the B tile in registers once per K tile | `tl.dot` bf16 x bf16 -> fp32; states fp32 | 200 / 36 |
| 3 | `_state_passing_fwd_kernel` | 40x1x96 | 128 / 26 / 0 | sequential over chunks: s = exp(dA_chunk) * s + new, in blocks of 256 of P*N=10240 | fp32 | 152 / 12.5 |
| 4 | `_bmm_chunk_fwd_kernel` | 32x256 | 64 / 84 / 24.6 KB | CB = C @ B^T per (chunk, group) | bf16 dot, fp32 out | 62 / 14 |
| 5 | `_chunk_scan_fwd_kernel` | 24x32x96 | 128 / 90 / 16.9 KB | out = exp(dA_m) * (C @ prev_state) + sum_k [CB * exp(dA_m - dA_k) * dt_k, causal] @ x + D*x, all in one kernel; the causal K loop stops at the diagonal | bf16 dots (CB and state cast to bf16 per tile), fp32 acc | 627 / 128 |
| 6 | `triton_red_fused_..._rsqrt_silu` | 32768 | 256 / 40 | gated RMSNorm over 8 groups (torch.compile) | bf16 | 222 / 31 |

Per 1024 tokens: SSD core (1-5) about 131 us, conv 26 us, gated norm 28 us, **total about 185 us**; per 10k, 37.4 ms.
Autotune picked BLOCK_M x BLOCK_N = 32x32 (grid x = 8x3 = 24) for chunk_scan at 8192 tokens and 64x64 (4x2 = 8) at 1808.

### 1b. Ours (`nemotron_h_ssd.ssd_mixer`, piece 1024, chunk 256, precision "float"): per piece, from the prefill trace

| kernel | us | what (matched through an isolated CPU/NV schedule) | useful work | achieved | limiter |
|---|---|---|---|---|---|
| `E_1027_152_16_4` | 77.5 | conv input = cat(prev conv tail, raw xBC) materialized fp32 | 40 MB | 0.5 TB/s | extra materialization (vLLM reads the tail in-kernel) |
| `E_512_6_5_2_16_4_4` | 28.5 | conv + SiLU + x*dt pack | | | |
| `r_48_16_2_4_16_64_4` | 22 | cumsum of log_a per chunk | | | |
| `r_2_2_8_4_8_4_8_16_2_2_32_4` | 92.8 | scores = C @ B^T per group (fp32 SIMT) | 0.54 GF | 5.8 TF/s | fp32 CUDA core; 2.3% of the bf16 roofline |
| `r_2_2_96_2_5_2_2_4_16_2_2_64_4` | 280.5 | local states = x^T @ (B * exp(last - cum)) per head | 2.0 GF | 7.2 TF/s | fp32 CUDA core; the decay is recomputed per output row |
| `r_2_2_96_2_4_5_2_16_4_2_2_64_4` | 809.2 | diagonal = (scores * decay) @ x per head; exp(cum_t - cum_s) is recomputed for every P column (80x) | 4.0 GF | 5.0 TF/s | fp32 ALU-bound (tinygrad's op estimate is 52 GF, about 13 ALU ops per MAC) |
| `r_96_5_10_2_8_16_4_5` | 19.8 | state passing as a (chunks+1)^2 matmul | 0.05 GF | | fine |
| `r_2_2_8_4_8_3_5_8_4_4_2_2_32_4` | 306.2 | previous = C @ entering-state per group (fp32 SIMT) | 2.0 GF | 6.6 TF/s | fp32 CUDA core |
| `E_32_8_12_5_32_4_4` | 118 | y = diag + previous*exp(cum) + D*x, times silu(gate) | 60 MB | 0.5 TB/s | materialized fp32 diag/previous (vLLM keeps them in the accumulator) |
| `r_64_8_8_4_4_120` + small | 16 | gated group-norm reduce | | | |
| **sum** | **about 1770** | | 8.6 GF | | |

The three reductions the scan names (`r_2_2_96_2_4_5...` 173 ms, `r_2_2_8_4_8_3_5...` 64, `r_2_2_96_2_5_2...` 59) are the
diagonal, previous and local-state contractions: 297 ms per 10k.

### 1c. Why 10x (tested, `bench/nongemm-audit-20260926/run_ssd.sh`, one layer, 1024-token piece)

TinyJit graph wall of the whole mixer (ssm_in/ssm_out stubbed out), and the SSD-core kernel sum (cumsum through
previous, non-JIT DEBUG=2 per-kernel times), against float:

| variant | graph wall ms | SSD core us | y rms rel / max rel vs float | state rms rel |
|---|---|---|---|---|
| float, chunk 256 (production) | 2.11 | 1917 | 0 | 0 |
| float + decay operands materialized (MAT) | 2.26 | ~1790 (local 372 -> 234, diag 951 -> 1007) | 0 (bit-identical) | 0 |
| split, chunk 256 | 1.63 | ~1350 | 5.7e-5 / 1.2e-3 | 4.5e-6 |
| **bf16, chunk 256** | **0.86** | **393** | 1.5e-3 / 5.0e-3 | 2.4e-3 |
| float + MAT, chunk 128 | 1.70 | | 1.7e-5 / 1.2e-3 | 2.2e-6 |
| bf16 + MAT, chunk 128 | 0.84 | 356 | 1.5e-3 | 2.4e-3 |
| split + MAT, chunk 128 | 1.33 | | 5.7e-5 | 5.0e-6 |

Readings:
1. **The dominant cause is the fp32 CUDA-core contraction, not the recompute.** Materializing the decay-scaled
   operands (so the exp is computed once) leaves float unchanged (2.11 -> 2.26 ms). Moving the same dots to bf16
   tensor cores cuts the core 4.9x: diag 951 -> 139 us, previous 379 -> 95, local 372 -> 49, scores 126 -> 16.
2. `"split"` triples K and adds hi/lo split/concat kernels, so it gains only about 1.4x. Chunk size is not a lever:
   ours is already vLLM's 256, and 128 changes bf16 by 2%.
3. Even in bf16 our tensor-core contractions run at 21-29 TF/s (diag 4 GF in 139 us). They are batched small matmuls
   with fp32 intermediates between them. vLLM's `_chunk_scan_fwd_kernel` does diag + previous + D-skip for about
   2 us per (chunk, head) with no intermediate traffic. The remaining gap after bf16 is fusion: about 64 vs 26 ms of
   SSD, and 50 vs 10 ms of conv-cat / y-combine / norm.
4. Numerics: vLLM's dots are bf16 (the same class as our "bf16": CB and state are rounded to bf16 per tile). Our prefill
   state feeds decode and the RL parity chain (`train-inference-mismatch-localization.md`), so the precision choice
   is Julian's call. The table above prices it: bf16 -250 ms per 10k at 1.5e-3, split -90 ms at 5.7e-5.

### 1d. Replication spec (what the generated kernels must look like)

Per 1024-token piece, chunk Q = 256, heads 96 x P 80, groups 8 x N 128. Five kernels instead of about 14; a
generated kernel may merge K1/K2:
- **K1 conv + dt + cumsum** (vLLM 0+1): read the raw bf16 xBC plus a 3-row carried tail in-kernel (no cat
  materialization). Write x, B, C in bf16 (or the operand precision) in (L, head, P) layout, plus dt = softplus(dt+bias)
  and dA_cumsum (heads, chunks, Q) in fp32. Memory-bound: about 45 MB, so about 30 us.
- **K2 chunk_state** (grid (P/BM * N/BN, chunks, heads)): acc[P,N] += x_tile^T @ (B_tile * exp(dA_last - dA_k) * dt_k),
  with the scale applied to the B tile in registers once per K step, the tile cast to bf16, mma bf16 -> fp32, and
  states written in fp32. This is the "operand-prologue" lowering tinygrad lacks: an elementwise function of the K
  index applied to an operand tile, not per output.
- **K3 state passing**: one program per (head, 256-wide slice of P*N), a sequential loop over chunks in fp32 (for
  4 chunks, our (chunks+1)^2 matmul at 20 us is also fine).
- **K4 CB** = C @ B^T per (chunk, group), bf16 mma, fp32 out (0.54 GF).
- **K5 chunk_scan** (grid (Q/BM * P/BN, chunks, heads)): acc = (C_tile @ bf16(prev_state)) * exp(dA_m); then
  for k < (m_block+1)*BM, acc += bf16(CB * exp(min(dA_m - dA_k, 0)) * dt_k, causal-masked) @ x_tile, then
  acc += D * x and store y (optionally times silu(z), fusing the gate). The decay is recomputed per BN tile only
  (80/BN = 3x), never per output column, and the causal loop skips the upper triangle (halving the diag FLOPs).
- **K6 gated group RMSNorm**, fused with the bf16 cast / operand prep for ssm_out.
- Budget: about 130 us per piece for K2-K5 (vLLM measured), so the SSD part goes 1770 -> about 190 us per piece, or
  **-330 ms per 10k** at vLLM parity. The bf16 precision switch alone, with no new lowering, is about -250 ms.
- **Lowering asks** (none exist in today's space): (a) operand-prologue elementwise fused into the mma operand load
  (decay scaling, dt scaling, causal mask); (b) a causal K bound in the reduce loop; (c) two contractions summed into
  one accumulator in one kernel (previous + diag); (d) bf16 tensor-core mma for batched 3-D/4-D contractions whose
  operands are computed, not loaded.

## 2. Move 4: decode non-GEMM kernels

### 2a. Per class, ours vs vLLM (ms/step; launches/step), with the attribution corrections

| class | B=32 ours / vLLM (launches) | B=64 ours / vLLM | B=128 ours / vLLM (launches) | delta B=32 / 64 / 128 |
|---|---|---|---|---|
| attention | 0.553 / 0.180 (73 / 20) | 1.688 / 0.936 | 2.663 / 1.632 (64 / 20) | +0.37 / +0.75 / +1.03 |
| mamba (excl. flush and ssm_out prep) | 3.269 / 3.010 (422 / 63) | 6.071 / 6.662 | 11.113 / 13.665 (359 / 63) | +0.26 / -0.59 / -2.55 |
| ssm_out input prep (GEMM aux, was in mamba) | in mamba (different name) | in mamba | 1.031 / 0 (21) | - / - / +1.03 |
| ring flush (amortized; was split across flush + sampling) | 0.288 / 0 | 0.583 / 0 | 1.024 / 0 | +0.29 / +0.58 / +1.02 |
| norm / residual | 0.622 / 0.084 (151 / 44) | 0.655 / 0.078 | 0.677 / 0.092 (151 / 44) | +0.54 / +0.58 / +0.59 |
| sampling + bookkeeping | 0.583 / 0.352 (12 / 77) | 0.886 / 0.314 | 1.818 / 0.435 (12 / 77) | +0.23 / +0.57 / +1.38 |
| mlp_act | 0 / 0.012 | 0 / 0.018 | 0 / 0.032 | fused on our side |

The per-kernel lists are in `per_kernel.json`. vLLM's decode non-GEMM per step: `flash_fwd_splitkv_kernel` x4
(+combine at B=32), `_selective_scan_update_kernel` x21 (reads and writes the fp32 state: 640 us at B=128,
1.57 TB/s), `_causal_conv1d_update_kernel` x21, a fused gated norm x21, `fused_add_rms_norm` x42, relu2 x17, then
`_gumbel_sample_kernel` + `_topk_log_softmax_kernel` + a radix top-k for logprobs=1, plus the Mamba-align copies.

### 2b. Top 5 by recoverable time, with the test and a replication spec for each

**1. Sampler tail: -0.32 / -1.36 ms (B=32 / 128), measured.** Ours at B=128 (trace):
`r_16_2048_8_16_4_256_64_4` 1140 us, `r_16_16_8_16_128_4` 221, `r_16_4_8_16_4_256_128_4` 2x143,
`E_8192_32_4_4_4` 110 (logits + bias, contiguous), `E_16_2048_8_16_4` 58 (threefry bits), `r_..._128_4` 50. The 1140-us
kernel writes the full fp32 log-softmax (128 x 131072). tinygrad split the logsumexp into 256 partials and then fused
the final 256-way combine into the per-element consumer, so every one of 16.7M elements re-reduces the partials
(the same failure class as the SSD decay recompute). vLLM needs about 170 us in its two main kernels and never
materializes log-probs. Tested (`tail_iso.py`, graph wall; tokens identical in all variants):

| variant | B=32 | B=128 |
|---|---|---|
| V0 production | 0.732 ms | 2.164 ms |
| V1 normalizer materialized ([B,1] logsumexp, `.contiguous()`) | 0.409 | 0.960 |
| V2 no full log-prob tensor (gumbel argmax on scaled logits, chosen = logit[tok] - lse) | 0.424 | 0.800 |

Chosen log-probs differ from V0 in the last ulp (-4.8708625 vs -4.870863), so `tail_logprobs` must use the same
form to stay bit-exact. Replication spec: (K1) a per-row split over the vocab (S slices per row, S fixed for
determinism). One pass over the fp32 logits computes, per slice, the online (max, sumexp) and the gumbel-argmax
(value, index) with counter-based RNG keyed by (seed, row, col), and writes partials [B, S]. (K2) combine per row:
lse, token, chosen = logit[token]/T - lse. Floor: one read of 64 MB, about 40 us at B=128, against 2.3 ms today
(V2 gets to 0.8 ms). Fold the +bias into K1's load.

**2. Decode attention: -0.37 / -1.2 (up to -3.1) ms, measured in isolation.** `two_segment_attention` routes every
dot through `_exact_dot` = `a.float().dot(b.float())`, fp32 CUDA-core math. At B=128 the prefix QK / PV kernels
(`r_2_2_8_20_32_8_16_2_2_32_4` 236 us, `r_2_2_8_2_20_2_8_16_2_2_256_4` 227 us, per layer) run 2.6 GF each at about
11 TF/s. Test (`att_iso.py`, TC=1 swaps in `a.matmul(b, dtype=float)` on bf16 operands): one layer's graph wall
B=32/P=200: 0.296 -> 0.203 ms; B=128/P=2000: 1.330 -> 0.479 ms. Per kernel: prefix QK 631 -> 69 us, prefix PV
362 -> 216 us. **Numerics caveat:** bf16 x bf16 products are exact in an fp32 mma, so only the accumulation order
should change (about 1e-6), but the measured difference is 1.3e-3 / 2.1e-3 of max. Something in the TC path rounds
(suspect: the output or the P operand goes through bf16). Find this before using the path. Replication spec
(flash-decode, like vLLM's `flash_fwd_splitkv` + combine): one kernel per (kv head, KV split) that takes the
B*group query rows of the shared prefix (our layout already reads the prefix once for the whole batch, which vLLM
cannot) plus a per-sequence suffix split, with QK^T as bf16 mma -> fp32, online max/sum in registers, P rounded to
bf16 (the storage dtype, as today) and PV as bf16 mma -> fp32. It writes (m, l, acc) partials, and one combine
kernel follows. It replaces today's about 16 kernels per layer (scores, peaks, weights, sums, dots, 3-way combine).
Floor at B=128: suffix KV 134 MB plus prefix 8 MB, about 90 us per layer (vs 665 today, 401 for vLLM). The gain
beyond vLLM comes from the prefix sharing.

**3. Mamba decode aux kernels: -1.2 / -4.3 ms, estimated from bandwidth, not tested.** Per layer, besides the state
read `r_*_8_4_4_16` (B=128: 323 us, 93% of DRAM; vLLM's read+write update takes 640 us), we run about 19 small
kernels (B=128 per layer, trace): ring recent-sum `r_2_64_6_5_2_16_4_2_2_8` 68 us (reads the fp32 ring_x, 63 MB);
pack `E_64_6_5_2_16_4_4` 49 us at 0.2 TB/s (a strided 3-way cat); conv write `E_64_152_2_3_16_4` 14; plus
`E_16_608`, `r_4_608`, `r_32_8_2_8`, `r_8_32_4_3` (0.08 TB/s), `E_4_8_12`, norm reduce, 2x `E_128_2`, 2x
`E_16_16`, ... = 255 us (72 us at B=32), vs vLLM's conv update 7.9 us + gated norm 2.9 us. Floor (ring read plus
about 20 MB of small operands) is about 50 us per layer at B=128. Replication spec: 3 kernels per layer. (a) conv
update + SiLU + dt softplus + exp(dA) + the x*dt / B pack, writing the ring slot in place (vLLM's
`_causal_conv1d_update_kernel` shape: grid (B, conv_dim/256)). (b) readout: one pass per (seq, head) that reads the
state rows, the ring slots and C, applies the checkpoint decay and the ring weights in registers, adds D*x, and
writes y. (c) gated group RMSNorm + bf16/hi-lo operand prep for ssm_out (this also absorbs the 1.03-ms
`E_2_640_32_4_3_4`, which runs at 0.14 TB/s). Our Mamba class is already 2.55 ms under vLLM at B=128; this is
headroom beyond parity.

**4. Ring flush: -0.1 / -0.37 ms, estimated.** Per flush layer at B=128: the state update `E_128_48_5_2_2_4_16_4_4`
677 us (r+w 1 GB, about 95% of DRAM: fine), `r_16_6_427_8_16_3_16` 355 us at 0.36 TB/s (21%: the decayed-pack
build over the ring), `r_64_6_2...` 5 us. Spec: build the packed operand in one bandwidth-bound pass (about 130 MB,
about 80 us). The flush is inherent to the replay design; flush plus state read (7.8 ms) is still below vLLM's
13.4 ms.

**5. Residual + RMSNorm: -0.54 / -0.59 ms, from the trace.** Per block: the residual add `E_*_49_8_16_4` (63/step),
the mean-square reduce `r_*_8_4_4_392` (43/step, 9.7 us at both B=32 and B=128: one thread loops over 392
elements, 146 GB/s, latency-bound), and the apply + bf16/hi-lo `E_*_49_8_16_4_4` (43/step). vLLM:
`fused_add_rms_norm` (grid = B rows, 1.5-2.8 us), 42/step. Isolated, the reduce is unchanged under BEAM=4 (11.3 ->
10.9 us, same kernel), so the search space lacks the shape. Spec: one kernel per block, one CTA per row with 256+
threads and a warp-shuffle tree reduce. It does residual add (written back in place), sum of squares, rsqrt, scale
by the weight, and writes the next GEMM's operand (bf16, or hi+lo). Target about 2 us: 43 x (9.7 + 2 x 2.3 - 2)
is about 0.53 ms/step, the same at every B because the kernels are latency-bound.

Totals: items 1, 2 and 5 (measured or trace-derived) come to -1.2 / -3.2 ms per step at B=32 / 128, above the scan's
non-GEMM delta (+1.68 / +2.47). Most of that is attention and sampling moving past vLLM. Items 3 and 4 add -1.3 /
-4.7 (estimated).

## 3. BoltBeam importer defect (fix-first; not committed, pending Julian's push decision)

- Where: `boltbeam/profiler/ncu_counters.py:139-144` (`mark_gemm`: `r["aux"] = r is not gemm`, only the single longest
  launch is GEMM), used at `:176`. The consumer is `boltbeam/profiler/ncu_audit.py:93-97` (`_side`: `gemm = next(...)`;
  `gemm_us = gemm["duration_us"]`).
- Fix: `r["aux"] = r["kernel"] != gemm["kernel"]` (every launch of the GEMM kernel is GEMM), and in `_side`:
  `gemm_us = sum` over the non-aux rows, with the longest as the representative for counters and grid.
- Evidence: re-importing `/home/ubuntu/storage/scan-20260926/scan/oncu.raw.csv` changes ssm_in 512 from 468/524 to
  928/64 us gemm/aux, ssm_in 1024 from 469/1681 to 1860/291, ffn_up 1024 from 290/1044 to 1155/179; decode rows
  are unchanged. BoltBeam's ncu tests (15) pass. The added regression test fails on unpatched code (mutation-checked).
- After the fix, the scan's research-queue item 4 (prefill "unexplained" -129 to -488 us) should be re-derived.

## 4. Next (the solve half, and the confirmations that are left)

1. Julian: the prefill SSD precision choice (bf16 -250 ms per 10k at 1.5e-3; split -90 ms at 5.7e-5), same class as
   the hi/lo decision.
2. Sampler V1/V2 is the cheapest measured decode win (-1.36 ms at B=128), with `tail_logprobs` kept in the same form.
3. Attention on tensor cores: find the 1e-3 rounding in the TC path first, then build the flash-decode kernel.
4. ncu (BoltBeam `ncu-collect`) on the three replication targets (SSD bf16 diag, attention TC, sampler V2), to
   confirm limiters before any generated-kernel work.

## Reproduce

```
cd tinygrad-self-training; B=docs/nemotron-vllm-parity/bench/nongemm-audit-20260926
python3 $B/per_kernel.py                                  # per-kernel tables from the scan's traces (CPU only)
gpu-run time $B/run_ssd.sh; gpu-run time $B/run_dec.sh; gpu-run time $B/run_mdec.sh   # scripts write to /home/ubuntu/storage/audit-nongemm
```
