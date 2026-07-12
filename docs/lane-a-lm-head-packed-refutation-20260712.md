# Lane A LM-head packed route — measured refutation

Date: 2026-07-12

## Result

The packed Q6_K direct-out LM-head kernel is **refuted as a prefill speedup**. It is
correctness-PASS but speed-FAIL, mirroring the 14B MMQ coop atom.

Single-variable whole-prefill A/B on Qwen3-8B-Q4_K_M, ctx512@start_pos=0, K=8,
4 warmups, 3 rounds, pinned clocks, canonical four-role authority env
(`PREFILL_V2=1 PREFILL_GRAPH_GEMM=1` + multirole-buffer2 candidate set). Both arms
differ only in LM-head routing via `PREFILL_LM_HEAD_DIRECT`:

| LM-head route | ctx512 total | attributed LM-head cost |
|---|---:|---:|
| resident fp16 (baseline, flag off) | 155.8 ms | ~17 ms |
| packed q6k kernel (candidate, flag on) | 243.2 ms | ~104 ms |

Delta: **+87.4 ms (+56%)** when the LM head is switched to the packed kernel.

## Evidence

`DEBUG=2` probe (same candidate env, 0 warmups / 1 round) confirms the packed kernel
actually dispatched — the artifact does not log kernel names, so the earlier
name-grep was uninformative:

```
*** AMD  q6k_gen_prefill_direct_out_151936_4096_512  mem 20.79 GB  tm 104.46 ms
         (13058 GFLOPS, ~11 GB/s effective)
```

The kernel is memory-bound at ~11 GB/s because it re-dequantizes the full
151936x4096 Q6_K weight in-kernel every call. Numerics were already proven
(`extra/qk/lm_head_prefill_m512_correctness.py`: max_abs=2.44e-02 PASS at
M=512,N=151936,K=4096).

## Conclusions

1. The LM head is **not** where the ctx512 30.7 ms residual lives. Under resident
   fp16 it is ~17 ms and near its memory-bound floor.
2. The packed direct-out q6k LM-head route is refuted for speed. Do not promote.
   `PREFILL_LM_HEAD_DIRECT=1` is retained only as the measurement knob that selects
   this (currently slow) route for re-testing if the kernel is ever made
   bandwidth-efficient.
3. The committed LM-head wiring (route `self.output` through the resident-fp16
   prefill path, `_lm_head_wants_pf16`) is the better of the two measured options
   and stays.

## Next

Residual attribution moves to the remaining lanes (attention score/value GEMMs,
non-GEMM norm/rope/softmax/mask/residual, memory transport, low-occupancy
`attn_kv`). Use the now-committed semantic role tagging
(`tinygrad.tensor.role_metadata`) + `extra/qk/graph_profile_attribution.py` on a
fresh ctx512 capture to rank those lanes by measured device-time share before
selecting the next lever.

## Open question (not measured here)

Whether the committed resident-fp16 LM-head wiring is itself a win vs the
pre-wiring per-call fp16-recast path (the recorded 147.05 ms authority predates the
wiring). Needs a controlled 3-way, not the single-run 155.8 vs 147.05 comparison
(different run config).
