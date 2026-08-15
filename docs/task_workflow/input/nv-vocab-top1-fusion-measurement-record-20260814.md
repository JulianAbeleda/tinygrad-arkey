# NV vocab top-1 fusion measurement record - fused GEMV epilogue wall verdict

Date: 2026-08-14
Status: measurement record (NO_GO_WALL). Research-only route, closed-default and
lease-gated; no promotion record is touched.
Branch: tinygrad `nvidia-bringup-20260731`.
Corrected: 2026-08-15. The first measurement was dominated by a single-thread custom
reduce; the corrected wall is measured below and keeps the NO_GO_WALL verdict.

## 1. What was measured

The vocab-head argmax tail is four kernels today (`E_1187_32_4`, `r_32_4_1187`,
`r_128_16_8_1187`, `r_16_8`) after the Q6K vocab GEMV. This experiment folds them into a
P1 packed per-tile `(max, index)` epilogue: the coop GEMV stores one packed u64 key per
warp tile and in-kernel warp-reduces it, then an ordinary scheduler u64 cross-tile reduce
unpacks the token.
The route is `q6k_vocab_top1_call` in `tinygrad/llm/decode_routes.py`, reachable only from
`forward_greedy` under `model._decode_vocab_top1_lease`.

Harness: `extra/llm_research/decode/nv_vocab_top1_fusion_ab.py` (fresh-process child +
timing child + wall A/B). Same RTX 5090 / sm_120 box, Qwen3-8B-Q4_K_M, depth 512, all GPU
runs flocked on `/tmp/gpu-bench.lock`.

## 2. Correctness gate

- CPU microgate (`nv_vocab_top1_fusion_cpu_microgate.py`): all-normal, all-zero, and tie
  cases bit-exact.
- `pytest test/unit/test_vocab_top1_fusion.py`: 7 passed.
- GPU topology: candidate removes the four tail kernels and the legacy vocab GEMV,
  emits one `q6k_gen_coop_151936_4096_inkernel_epi_vocabtop1`, then the scheduler reduce
  chain (`E_32_32_4_5a5673...`, `r_16_4_1187_77cd...`, `r_16_4_ff72...`), not the old
  `q6k_vocab_top1_reduce_151936_4096` (program count 592 vs control 594).
- Token stream: candidate sha matches control exactly
  (`7e2ff56bb98978c6656a49628acae430db10b295c75aa9fc69aa1e30eae5f4e6`).

## 3. JIT replay bug found and fixed

Returning `token.reshape(1, 1)` directly made the JIT replay the token one step behind the
eager stream (`4710,32313,32313,...` instead of `4710,32313,11,...`). The same held copy
firewall is required with the ordinary reduce output, not only the old custom-kernel
buffer. Returning a held copy (`token.reshape(1, 1).clone()`) restores the correct stream;
this is the same fresh replay-written-allocation firewall the full-logit diagnostic path
uses.

## 4. Wall verdict

| arm | median ms/token | tok/s |
| --- | ---: | ---: |
| control (legacy tail) | 5.102 | ~196.0 |
| candidate (fused epilogue, corrected) | 5.159 | ~193.8 |

Verdict `NO_GO_WALL`: candidate is +0.0572 ms/token, -1.11% wall after correcting the
single-thread custom reduce.

The first measurement was contaminated by `q6k_vocab_top1_reduce_151936_4096` lowering to
one thread and serializing 75,968 u64 keys (~0.94 ms/token). After moving that step to the
ordinary scheduler path, the remaining cost is real rather than a bug: the scheduler u64
reduce `r_16_4_1187` is ~83.3 us/token while the entire legacy four-kernel float32 tail is
~56 us/token. A u64 max halves the reduction parallelism and doubles the element size, so
the packed key costs more to finish than the tail it removes. The legacy tail stays.

## 5. Next target

Keep the legacy vocab tail. Do not promote this route. A future vocab win needs the
cross-tile max to happen inside the vocab GEMV (single pass, no 607 KB u64 intermediate),
or a cheaper 32-bit-comparable tie-break scheme. The packed u64 carry is a useful substrate
for a single-pass epilogue, but it is not a win as a two-program chain.
