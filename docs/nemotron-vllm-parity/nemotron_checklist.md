# Nemotron-H sampler speed checklist
1. [x] Regression check: Qwen3-8B NV d512 238.9-239.5 vs 238.4-239.2; unit failset == HEAD
2. [x] 1d4be7aa4 realize GGUF weights + bf16-cast matvec detection
3. [x] e166365c9 small-M batched matvec (M<=16), skip TC for small-M bf16
4. [x] e166365c9 warp-wide bf16 matvec + bf16 16B load folding: 558 -> 1339 GB/s (79%) M=1
5. [x] 4d29728b4 residual contiguous (799->647 kernels); 6b831fb91 W3 replay Mamba; 9d22e2b55 W2 attention;
       0203cebc9 device-chained loop; b8081eb73 chunk-rounded capacities
6. [-] prime stall: coordinator/W-prefill (NemotronHPrefill); measured 10k warm 7.0s, cold ~88s
7. [ ] final report
Open: B>=16 needs small-M TC GEMM (TC agent); bf16 KV parity (max 0.014-0.024) decision; test_prefill_overlay_roles census failure from 9a2c02bdc (not mine)
