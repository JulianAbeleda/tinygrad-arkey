# NV vocab top-1 fusion measurement record - fused GEMV epilogue wall verdict

Date: 2026-08-14
Status: measurement record (NO_GO_WALL). Research-only route, closed-default and
lease-gated; no promotion record is touched.
Branch: tinygrad `nvidia-bringup-20260731`.

## 1. What was measured

The vocab-head argmax tail is four kernels today (`E_1187_32_4`, `r_32_4_1187`,
`r_128_16_8_1187`, `r_16_8`) after the Q6K vocab GEMV. This experiment folds them into a
P1 packed per-tile `(max, index)` epilogue: the coop GEMV stores one packed u64 key per
warp tile and in-kernel warp-reduces it, then a tiny cross-tile reduce unpacks the token.
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
  emits one `q6k_gen_coop_151936_4096_inkernel_epi_vocabtop1` and one
  `q6k_vocab_top1_reduce_151936_4096` (program count 592 vs control 594).
- Token stream: candidate sha matches control exactly
  (`7e2ff56bb98978c6656a49628acae430db10b295c75aa9fc69aa1e30eae5f4e6`).

## 3. JIT replay bug found and fixed

Returning `token.reshape(1, 1)` directly made the JIT replay the token one step behind the
eager stream (`4710,4710,32313,...` instead of `4710,32313,11,...`). The reduce output is
a custom-kernel buffer, and the reshape view let the JIT memory plan reuse the producer
after the custom CALL. Returning a held copy (`token.reshape(1, 1).clone()`) restores the
correct stream; this is the same fresh replay-written-allocation firewall the full-logit
diagnostic path uses.

## 4. Wall verdict

| arm | median ms/token | tok/s |
| --- | ---: | ---: |
| control (legacy tail) | 5.088 | ~196.5 |
| candidate (fused epilogue) | 5.983 | ~167.1 |

Verdict `NO_GO_WALL`: candidate is +0.895 ms/token, -14.96% wall.

The packed per-tile epilogue plus its dependent tiny reduce and the held-copy barrier
serialize on the critical path. The legacy four tail kernels are tiny enough that the
scheduler hides them under the next token's GEMV, so fusing them out costs more than it
saves at this geometry.

## 5. Next target

Keep the legacy vocab tail. Do not promote this route. Revisit vocab support fusion only
if a launch-hiding substrate lets the packed reduce overlap the following GEMV, or as
part of a broader single-pass GEMV epilogue rather than a two-program chain.
