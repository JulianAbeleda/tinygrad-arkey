# NV decode final composed same-session record

Date: 2026-08-05
Target: Qwen3-8B-Q4_K_M, d512, RTX 5090 / driver 595.84
Native composition: default-on P1 + callify redirect + direct greedy + feedback
ping-pong + bounded cooperative-Q4/shared-Q8 g12 lease

## Verdict

The final experimental composition is **not at llama.cpp parity**. In one
flocked llama/native/llama session, steady llama latency was
`4.005677 ms/token` and native NV was `5.324244 ms/token`. Native throughput is
`0.75235x` llama (187.82 versus 249.65 token/s), leaving a fresh steady-state
gap of **1.318567 ms/token**.

The llama executable's reported `avg_ts` includes one cold/high sample in each
seven-repeat arm. Retaining that official average instead gives a llama A/C
midpoint of `4.070055 ms/token`, native/llama `0.76444x`, and a
`1.254189 ms/token` gap. The steady median is the primary comparison because
the native arm also excludes eager/capture setup; both forms are reported so
the authority is reproducible rather than selected after the fact.

## Commands and samples

Llama A and C used:

```sh
GGML_CUDA_GRAPH_OPT=0 llama-bench \
  -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  -ngl 99 -fa 1 -p 0 -n 10 -d 512 -r 7 -o json
```

Per-token llama samples (each `samples_ns / 10 / 1e6`):

| arm | samples (ms/token) | steady median | reported `avg_ts` |
| --- | --- | ---: | ---: |
| llama A | 4.504989, 4.011826, 4.013376, 4.005583, 4.003819, 4.003990, 4.002300 | 4.005583 | 245.632319 |
| llama C | 4.500490, 4.007236, 4.002488, 4.004474, 4.003474, 4.006475, 4.005770 | 4.005770 | 245.761535 |

Native used the settled-continuous qualification path with both ping-pong
captures completed before timing, five uninterrupted 32-token windows, and no
reset/capture in any timer:

```sh
DEV=NV CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1 \
python3 extra/llm_research/decode/nv_shared_q8_progressive_qualification.py \
  --mode timing-child --cooperative-q4 --composed --settled-continuous \
  --depth 512 --count 32 --max-context 1024 --groups 0 \
  --fused-groups 12 --reps 5 --out /tmp/nv-parity-final-native-g12.json
```

Native samples were `5.321381, 5.317449, 5.324244, 5.328420,
5.329130 ms/token`; all were accepted, with median `5.324244`. The complete
160-token stream SHA-256 was
`f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`,
the same stream used by the g12 A/B/A control arms. The separate g12 semantic
gate retained exact tokens/argmax/top-10 and relative L2 `8.36963e-4`.

## Accounting interpretation

This same-session row supersedes any attempt to derive a current parity ratio
by subtracting ledger entries from the old `5.612310 / 3.966140` authority.
The strict causal ledger still books its disjoint native recoveries against
that fixed authority; the fresh row is the absolute current measurement. The
remaining gap is still dominated by native quantized-GEMV achieved bandwidth
and serialized support work. The cooperative Q4 route adds only
`24.676 us/token` through g12 and stops at g18 on the predeclared numerical
contract, so it cannot close the remaining 1.319 ms.
