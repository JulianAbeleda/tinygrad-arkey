# NV decode parity P1 baseline and P1.5 group-boundary record

Date: 2026-08-04. Branch `nvidia-bringup-20260731`, tinygrad
`1084270bc`, llama.cpp `ac4cddeb0`. Authority:
`nv-decode-parity-causal-trace-execution-scope-20260804.md` P1/P1.5.
Status: **P1 PERFORMANCE SETTLED; P1 BLOCKED ON CORRECTNESS; P1.5 SINGLE-GRAPH NO-GO**.

## 1. Findings

The stable reverse-order bracket measures the current d512 wall gap at about
**1.646-1.647 ms/token**:

| arm | median tok/s | median ms/token | ratio vs llama midpoint |
| --- | ---: | ---: | ---: |
| llama A | 252.130 | 3.96621 | control |
| tinygrad CUDA S1 | 178.150 | 5.61323 | 0.70657x |
| tinygrad native NV | 178.180 | 5.61231 | 0.70669x |
| llama B | 252.139 | 3.96607 | control |

Llama bracket drift is **0.00357%**, below the 1% gate. Native and CUDA are
wall-equivalent: native leads by only `0.0165%`, or `0.00093 ms/token`.
The route itself is therefore not the owner of the parity gap in this session.

P1 is nevertheless **BLOCKED_CORRECTNESS**. Follow-up P1-B evidence in
`nv-decode-parity-p1b-shared-logits-record-20260804.md` shows that native NV
and CUDA full prompt-final logits are bitwise identical and both select
llama's argmax `13876`; llama's full row remains outside the predeclared strict
`atol=0.01` gate. Native production temperature-zero sampling then returns
invalid one-past-vocabulary sentinel `151936`, localizing the native failure to
the post-logits sampling/scalar path. These timing rows remain valid wall
diagnostics but cannot qualify parity until both gates are resolved.

The cheap six-group collapse is decisively refuted:

| CUDA arm | graph groups | kernels | median tok/s | median ms/token |
| --- | ---: | ---: | ---: | ---: |
| control | 6 (`32/64/128/256/512/29`) | 1021 | 178.150 | 5.61323 |
| `JIT_BATCH_SIZE=0` | 1 (`1021`) | 1021 | 166.465 | 6.00725 |

The one-graph arm preserves the exact CUDA token hash but regresses
**6.30%**, adding **0.378 ms/token**. `DEBUG=2` records
`JIT GRAPHing batch with 1021 kernels`, proving the intended decode topology
was constructed. Fewer graph boundaries are not automatically faster; the
current doubling split is beneficial on this workload.

## 2. Protocol

Every GPU arm was serialized under `/tmp/gpu-bench.lock`. The admitted bracket
ran in one uninterrupted acquisition in reverse order, as required after the
first forward bracket exceeded the drift gate:

```text
llama -> tinygrad CUDA S1 -> tinygrad native NV -> llama
```

Llama command:

```bash
GGML_CUDA_GRAPH_OPT=0 /home/ubuntu/env/llama.cpp/build-cuda/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  -ngl 99 -fa 1 -p 0 -n 10 -d 512 -r 7 -o json
```

Tinygrad command shape:

```bash
DEV={CUDA|NV} CUDA_GRAPH_STREAMS=1 PYTHONPATH=. .venv/bin/python \
  extra/llm_research/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --ckpts 512 --max-context 4608 --nmeas 10 --reps 5 \
  --warmup-decode 3 --chunk-size 32 --skip-dispatch-diagnostic --out <unique>
```

Native NV requires the established `/tmp/b3_runner.py` setup wrapper, which
disables the currently broken fused prefill attention route before the decode
measurement. The measured decode route reports `sdpa`. This is an explicit
setup deviation, not a production-route edit.

Model: Qwen3-8B-Q4_K_M, `5,027,783,488` file bytes, SHA-256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
GPU: RTX 5090, driver 595.84. Llama CUDA binary SHA-256
`947eb29052871f151719762c2fc265024e14f833b98df7801af9eb09da1625a8`.

## 3. Dispersion and drift

The compact JSON contains every retained sample plus p5, p95, and MAD. Llama's
first sample in each process remains its characteristic slow sample; the
median and MAD are used as the robust authority. The initial uninterrupted
forward bracket had `3.89%` llama drift (`242.514 -> 252.123 tok/s`) and is
excluded by the declared gate. The required reverse bracket then measured
`252.130 -> 252.139 tok/s`, so no historical row was substituted.

## 4. Construction and tooling blockers

The campaign uncovered five concrete blockers:

1. Native NV and CUDA fixed-depth streams disagree because native's post-logits
   production sampling path emits invalid sentinel `151936`. Shared-input
   prompt-final logits and argmax are identical across the two tinygrad
   backends; the sampler/scalar boundary is the narrow native blocker.
2. `extra/llm/bench/llama_bench.py` defaults to `/build/bin/llama-bench`, which
   is currently a HIP build with no usable GPU; the explicit pinned CUDA build
   at `/build-cuda/bin/llama-bench` was used.
3. Native default setup hits the already documented `PACKED_FRAGMENT_LOAD`
   verification failure; the established fused-prefill-off wrapper is needed
   merely to reach decode.
4. The canonical graph-admission export asserts that its second warmup did not
   capture when `JIT_BATCH_SIZE=0`, although the unobserved wall run succeeds.
5. The untracked aligned-census helper fails on one-group topology with
   `unhashable type`. Runtime `DEBUG=2` supplied the construction proof instead;
   no user-owned helper was edited.

Items 1-3 block a clean qualified native setup. The shared llama/native oracle
also retains a strict full-logit numerical failure despite exact argmax. Items 4-5 block only
optional topology instrumentation and do not weaken the one-graph wall verdict.

## 5. Decisions

- Use `252.1345 tok/s` as this session's bracket-midpoint llama wall control.
- Use `1.646-1.647 ms/token` as the current observed parity gap.
- Treat native and CUDA wall cost as equivalent within this session, but never
  treat CUDA tokens as the native correctness control. Do not describe the
  diagnostic CUDA route as carrying a material route tax.
- Close whole-token six-to-one graph collapse as `SINGLE_GRAPH_NO_GO` on the
  current kernel mix.
- Fix and qualify native's post-logits temperature-zero sampling/scalar path,
  and separately resolve the predeclared strict llama/native full-logit gate,
  before composing exact-kernel oracle results into a native parity claim;
  keep CUDA oracle substitutions diagnostic-only.

Compact payload:
`docs/task_workflow/output/nv-decode-parity-p1-baseline-p15-boundary-20260804.json`.
All raw artifacts remain in `/tmp`; their SHA-256 values are embedded in the
compact payload. No production route, runtime, lowering, or default changed.
