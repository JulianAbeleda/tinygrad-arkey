# NV decode gap decomposition record

Date: 2026-08-03 (measured 2026-08-03, one flocked session, same RTX 5090 box)
Status: measurement record for the parity wall-gap decomposition. Authorizes the
kv-store-chain fusion scope (`decode-kv-store-chain-fusion-scope-20260803.md`) and
supersedes nothing; the parity wall authority remains
`nv-decode-parity-final-20260802.md` and the canonical forward scope remains
`nv-parity-and-beyond-forward-scope-20260803.md`.
Branch: tinygrad `nvidia-bringup-20260731`, HEAD `894c08c48` (w1w3 scalar shape
landed). All numbers below carry evidence class OBSERVED unless marked INFERRED.

## 1. Question

At the same depth, on the same box, why is our decode wall 1.3-1.6x behind llama
(benchmark: Qwen3-8B-Q4_K_M, RTX 5090 / sm_120, n_batch 2048 / n_ubatch 512, flash
attn on, `--depth D --n_gen 10`, llama-bench CUDA build `ac4cddeb0`)? Is the gap
GEMV compute, sequential non-GEMV kernels, graph overlap, or host overhead -- and
which piece is buildable next?

## 2. Same-session parity refresh (as-committed w1w3 scalar, OBSERVED)

Files: ours `/tmp/tg_d{512,2048,4096}_final.json` (census harness, `--w1w3-scalar`
as-committed route, fixed-depth prompt `[1]*depth`, 20 measured tokens x 3 reps);
llama `/tmp/llama_d{512,2048,4096}_bench.json` (`samples_ts` median).

| depth | ours tok/s | llama tok/s | ratio | ours token sha / first token |
| --- | ---: | ---: | ---: | --- |
| d512 | 177.2 | 251.8 | 0.704x | `9d6b3787...` / 151936 |
| d2048 | 164.8 | 237.8 | 0.693x | `9d6b3787...` / 151936 |
| d4096 | 152.1 | 228.3 | 0.666x | `9d6b3787...` / 151936 |

Depth scaling (INFERRED from the OBSERVED rows): ours -14.2% d512 -> d4096, llama
-9.3%. Ours degrades faster with context; the flash score kernel alone grows
7.78 -> 32.4 us/median over that range (OBSERVED, section 4 census rows).

## 3. llama node trace and the overlap finding (OBSERVED)

Fresh nsys profile of the same llama-bench run at d512
(`/tmp/llama_nsys_d512.nsys-rep` + `.sqlite`, graphId 2, 762 nodes, 29 replays).
Node-sum by class (CUPTI kernel rows, duration = end-start):

| class | nodes | node-sum ms |
| --- | ---: | ---: |
| mmq | 217 | 3.578 |
| quantize_q8_1 | 217 | 0.551 |
| flash_attn | 72 | 0.363 |
| rms_norm | 145 | 0.308 |
| rope | 72 | 0.126 |
| kv_ops | 39 | 0.080 |
| total | 762 | 5.006 |

CRITICAL FINDING: the graph's replay wall span (median ~3.89 ms per node span over
replays) is BELOW the 5.006 ms node-sum by ~22%. llama's decode graph has node-internal
OVERLAP: nodes on the single stream 43 run concurrently with each other (graph nodes
are not serialized the way a stream-ordered kernel sequence is). llama's non-GEMV
classes (norm, rope, quantize, kv_ops, flash) are largely HIDDEN behind its mmq chain;
only the mmq class itself is wall-visible. Caveat: under-nsys llama-bench measured
~7% slower than unprofiled (228.6 vs 246.2 avg ts), so node durations are inflated
vs the unprofiled wall.

## 4. Our per-layer attribution (OBSERVED)

Census harness with `--log-out` (`/tmp/tg_prime_log.txt`, DEBUG=2 prime token, 948
kernels, 6021 us total at d512):

| role | n | us | note |
| --- | ---: | ---: | --- |
| ffn_gate_up (w1w3 fused) | 36 | 1428 | |
| ffn_down | 36 | 1120 | |
| attn_qo | 72 | 676 | |
| attn_kv_gemv | 72 | 584 | |
| kv_cache_chain | 270 | 568 | 7.5 kernels/layer, the lever |
| vocab head + scatter | 5 | 383 | |
| rmsnorm | 145 | 431 | 2 kernels/layer vs llama's 1 |
| flash score + combine | 72 | 406 | score 7.78 us vs llama ext_vec 4.80 us; combine 3.39 vs 5.28 us; net delta only ~43 us |
| residual add + ffn | 144 | 252 | |
| x_cast_f16, post_combine, other | 96 | 174 | |

The kv_cache_chain role is the q/k rope + casts/layouts + `Tensor.stack(k, v)` +
cache `uop.store` sequence in `model.py:640-700`; it materializes as separate kernels
per layer (270 = 7.5 x 36 layers).

## 5. Wall-gap decomposition (INFERRED from the OBSERVED rows)

d512 wall gap = 1/177.2 - 1/251.8 = 1.67 ms/token. The decomposition:

- GEMV like-for-like delta ~0.50 ms: ours bare GEMV-class 3.775 ms vs llama mmq 3.274
  ms (bare, pre-overlap). The cap on this piece is ~0.6 ms and it is nearly spent.
- Sequential non-GEMV ~1.2 ms: our chain is stream-serialized (948 kernels, 6.02 ms
  node-sum), while llama overlaps ~22% of its node-sum behind its mmq chain. Our
  overlap opportunity is limited because our non-GEMV classes are not hidden behind
  anything; they are a sequential tail.
- Host overhead: not measured this session; the B3 characterization
  (b3-prefill scope) is prefill-specific and not reused here.

Buildable lever ranking (by measured kernel-sum, all three are sequential):

1. kv_cache_chain: 270 kernels / 568 us. A fused kv-store kernel (rope k + cast +
   stack + store in one kernel) removes ~180 kernels / ~380 us (see the fusion
   scope).
2. rmsnorm pair fusion: 431 us, 2 kernels/layer vs llama's 1. Secondary.
3. flash score: 109 us delta vs llama's ext_vec. Secondary; llama's flash nodes are
   also partially overlapped.

## 6. Open questions (NOT causes)

- llama's overlap mechanism is inside its CUDA graph scheduling (nodes on one stream
  run concurrently); whether tinygrad can express equivalent overlap through its own
  graph is not established here.
- Whether the kv-store fusion's kernel-sum saving converts 1:1 to wall is not known
  until measured (it is a sequential chain, so the conversion is expected to be high,
  but that expectation is INFERRED).
- q-rope and q-cast (part of the same 270-kernel chain) are left for a follow-up;
  this scope covers only the k/v store side.
