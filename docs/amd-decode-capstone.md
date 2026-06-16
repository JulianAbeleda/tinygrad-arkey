# AMD decode arc — capstone

Date: 2026-06-15. Hardware: RX 7900 XTX (gfx1100), HBM peak 859 GB/s. Model: Qwen3-8B Q4_K_M.
Bar: llama.cpp = 105.7 tok/s (57% of peak) on this exact GPU.

## Headline
**Decode: 23 → 60.9 tok/s (2.65×), 22% → 58% of llama.cpp — with byte-identical output for the big win and
two clean machine-search results: the kernel beats llama standalone, and the per-tensor quant assignment
beats llama's fixed recipe.**

## The two mission results
1. **Kernel beats llama (standalone).** The int-dot (`v_dot4`) Q4_K GEMV sustains **76% of HBM peak** cold/
   full-clock vs llama.cpp's 57% end-to-end. Machine search *exceeds* the reference at the kernel level.
2. **Search beats llama's fixed scheme (shipped).** Two ways:
   - **Coverage**: Qwen3-8B-Q4_K_M is mixed-quant; the Q6_K matmuls had no primitive and ran a slow fallback
     (59% of GPU work). Enabling the Q6_K primitive → **23 → 53.5 tok/s (2.2×), byte-identical output**.
   - **Bit-width**: Q4_K_M over-provisions `ffn_down` (Q6 where Q4 is ~free). A built-from-scratch Q4_K
     quantizer (bit-exact vs llama) demotes the 18 Q6 ffn_down → Q4 → **+14% (53.4 → 60.9 tok/s)** at
     dNLL −0.0028 (free). A faster operating point llama's fixed recipe doesn't offer.

## Lever ledger (every lever, honest outcome)
| lever | outcome |
|---|---|
| Q6_K primitive coverage (the big one) | **WON** 23→53.5 (2.2×), exact output, default-on |
| Q6_K attn_v + lm_head coverage (COVER_MORE) | **WON** +5% → 53.5, exact, default-on |
| B3 ffn_down Q6→Q4 demotion (+ Q4_K quantizer) | **WON** +14% → 60.9, free quality, gated |
| B1 in-graph int-dot GEMV | NULL — per-kernel GEMV at batch-1 occupancy ceiling (~llama's 57%) |
| split-K / horizontal fusion | within noise / hurts |
| B5 speculative decoding | exact + verify-fast (S3), but net-negative (1.7B draft too costly) |
| S3 batched GEMM primitive | BUILT + verified; validated via the speculative verify (fast); dormant otherwise |
| P2 fused attention | re-profiled → ~8% over a diffuse target; **next** |
| B2 overlap | caps below llama (weight-read bound); deferred |

## The measurement-discipline thread (what made it trustworthy)
Every wrong turn came from an uncontrolled confound; every correction came from re-measuring the *real*
in-graph kernel. The three confounds (Infinity Cache, launch overhead, **memory-clock ramp**) each produced a
false conclusion that was overturned. Notable corrections, all logged: "in-graph GEMV is 12%" (wrong kernel),
"weight read is 95% of the token" (circular), "small kernels can't saturate" (clock ramp), "decode is host-
bound" (single-graph artifact; it's 97% GPU-busy), "attn_v/output lose to the fused graph" (stale). The
**re-profile-before-building** rule repeatedly saved effort (it shrank P2 and redirected speculative).

## Where it stands
The token is now **~71% GEMV (weight read, at the B1 occupancy ceiling) + ~29% diffuse non-GEMV**. We are
near the practical per-kernel decode ceiling. Remaining levers are modest + build-heavy:
- **P2 (attention)**: ~8%, needs a flash-attention decode kernel — *next*.
- **B3 extensions**: cache the requant (kill the ~3min load cost); assess Q4→Q3 on the tensors phase-0
  couldn't resolve below noise.
- **B2 (overlap)**: caps below llama; low ROI.

## Artifacts
- Code: `tinygrad/llm/model.py` (Q6_K primitive default-on, COVER_MORE, Q6K_DEMOTE_FFNDOWN, batched GEMM),
  `extra/q6_k_gemv_primitive.py` (Q6_K GEMV+GEMM), `extra/q4_k_gemv_primitive.py` (Q4_K GEMM),
  `extra/qk_quantize.py` (the Q4_K quantizer), `extra/qk_speculative.py`.
- Tests: `test/external/test_qk_gemm_batched.py`, `test_qk_quantize.py`.
- Results: `bench/amd-decode-flywheel-proof-20260614/{KERNEL_BEATS_LLAMACPP, prefetch-gemv/{PERLAYER,
  BREAKDOWN, Q6K_FIX, B1_INTDOT, B5_S0, S1_SPECULATIVE, S3_BATCHED, B3_SENSITIVITY, B3_DEMOTE}_RESULT}.md`.
- Synthesis: `amd-decode-arc-synthesis.md` (primitive lens), `amd-decode-beyond-llama-roadmap.md` (levers).
- Memories: `amd-decode-{kernel-beats-llamacpp, measurement-confounds, real-bottleneck}`.
