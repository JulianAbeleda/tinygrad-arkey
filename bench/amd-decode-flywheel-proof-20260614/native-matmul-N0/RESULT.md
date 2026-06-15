# Phase N0a RESULT -- matmul_decoded is the competitive batched path (2026-06-15)

`extra/qk_matmul_decoded.py`, `n0a_summary.json`. dequant pass (compressed Q4_K -> fp16) + NATIVE
tinygrad matmul, vs the W2 fused split-K kernel (reads compressed). Real 8B Q4_K shape (4096x4096),
batch N in {16..2048}. All correct.

  N     native_mm   %peak   dequant   matmul   per-call   fused(approx)   per-call / fused
  16    2.0 TF      2.4%    114us     264us    378us      ~1694us         4.5x faster
  64    5.5 TF      6.5%    114us     394us    508us      ~2045us         4.0x faster
  256   17.9 TF     21%     112us     480us    593us      ~3843us         6.5x faster
  512   27.3 TF     33%     112us     630us    742us      ~7129us         9.6x faster
  2048  33.0 TF     39%     112us    2085us   2197us      ~12483us        5.7x faster

Findings:
- matmul_decoded (per-call, INCLUDING the dequant round-trip) is 4.5-9.6x FASTER than the fused
  split-K kernel at EVERY batch size, including small N=16. Amortized (fp16 resident) it is 5-11x.
- The dequant pass is ~112us regardless of N (M*K work): ~30% of per-call at N=16, ~5% at N=2048.
  This is the honest, modest "price of dropping fusion" -- fully amortized across the batch.
- Native matmul is memory-bound at small N (2.4% peak) and compute-bound at large N (39% peak); it is
  tinygrad's heuristic schedule (TC+UPCAST*2+LOCAL), the SAME rich opt space BEAM searches.

Conclusion: H-N0 holds -- matmul_decoded is the competitive path for the batched/prefill regime, and
the native-matmul opt space is now established as the real, instrumentable search substrate for N1
(cost-model learnability + cross-kernel transfer). Next: N0b -- log BEAM (config -> device_time)
trials over the model's matmul shapes as the N1 dataset.
