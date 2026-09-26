| lever | B=32 | B=64 | B=128 | 10k prefill |
|---|---|---|---|---|
| ragged_k | 0.45 | 0.00 | 0.22 | 0.00 |
| stream_k | 0.08 | 0.33 | 0.55 | 14.86 |
| n_pad | 0.05 | 0.07 | 0.10 | 7.98 |
| single_stage | 0.00 | 0.00 | 0.00 | 0.00 |
| fragment_prefetch | 0.32 | 0.38 | 0.00 | 25.66 |
| barrier_64k | 0.53 | 0.32 | 0.64 | 6.87 |
| hilo_rows | 0.25 | 0.94 | 5.41 | 124.32 |
| aux(hi/lo split+sum, split-K reduce) | 0.41 | 0.38 | 0.70 | 140.16 |
| gemm_total_gap | 1.05 | 3.68 | 7.49 | 756.08 |
