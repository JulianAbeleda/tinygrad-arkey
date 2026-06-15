# Q0a cooperative-fusion fix RESULT (2026-06-15): fast STANDALONE (409 GB/s), regresses END-TO-END.

After diagnosing the Q0a slowdown (NOT a lowering hoist bug -- a THREAD-ASSIGNMENT CONFLICT: the
LOCAL opt's threads were consumed by phase-1 quant, so phase-2's dot ran redundantly across all
threads), built the HAND-managed cooperative kernel `q4k_q8_1_coop_fused_kernel` (amd_copy_matmul
pattern): workgroup = block_m rows; the block_m local threads do BOTH phase-1 cooperative quant (each
thread quantizes q8b/block_m blocks of x into LDS) and phase-2 their own row's int-dot. Kept in
`extra/q4_k_gemv_primitive.py`.

## Results
- Correctness: PASS (rel_err 0.004).
- STANDALONE microbench (ffn_gate): **409 Q4-GB/s** -- vs broken-fused 10, separate-int-dot 242, fp 173,
  and APPROACHING llama.cpp's ~470. tinygrad CAN express a fast fused decode GEMV.
- END-TO-END decode: 24-25 tok/s (block_m 16/32/64 all ~same) -- REGRESSED vs fp 58.

## Why standalone-fast but e2e-slow (the structural finding)
The fused kernel must re-quantize x PER WORKGROUP into LDS + a BARRIER (phase1->phase2 sync). Two costs
that don't show standalone but dominate end-to-end:
- LDS (~4.5KB/workgroup) CAPS occupancy (fewer workgroups/CU vs the LDS-free fp path).
- the BARRIER + low occupancy breaks the INTER-KERNEL PIPELINING the decode regime lives on: fp is
  barrier-free, so consecutive GEMVs overlap (e2e 278 GB/s > per-kernel 173); the coop kernel is the
  opposite (e2e 117 << standalone 409). Small-GEMV decode is occupancy/latency-bound, and LDS+barrier
  is exactly the wrong structure for it.

## The fix this reveals (the right structure = llama.cpp's)
The LDS+barrier is ONLY needed because the kernel re-quantizes x per workgroup. llama.cpp does NOT:
it quantizes the activation ONCE per token (a cheap separate pass), writes q8 to GLOBAL, then the
int-dot GEMVs read global q8 -- BARRIER-FREE, so they pipeline. So the right next step is AMORTIZED
GLOBAL QUANT + the barrier-free int-dot (q4k_q8_1_intdot_partial_kernel): quantize the shared
activation ONCE (attn-input -> q/k/v; ffn-input -> gate/up), reuse across the linears. D0 measured the
barrier-free int-dot at 242 GB/s standalone but 28 e2e WITH PER-LINEAR quant (7x/layer launches);
amortizing the quant to ~2x/layer (per shared activation) is the untested path that could finally beat
fp -- and it needs model-forward surgery (quantize x once in the attention/ffn block, pass q8 to the
linears), not a fused kernel.

## Net
We PROVED tinygrad can express a 409-GB/s fused decode GEMV (near llama.cpp), refuting "the kernel is
the wall." The real wall is that fusion (LDS+barrier) is structurally wrong for occupancy/pipelining-
bound small-GEMV decode; the right structure is amortized-global-quant + barrier-free int-dot (the
llama.cpp shape), reachable via model-forward surgery. fp (58) remains best UNTIL that is built. This
is the most concrete, promising decode lead found -- and the first that is NOT a dead end.
