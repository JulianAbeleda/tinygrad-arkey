# NV decode native epilogue causal reconciliation

Date: 2026-08-05. Workload: Qwen3-8B-Q4_K_M d512, native `DEV=NV`, RTX
5090 / 595.84. Status: **closed for the currently admissible generic epilogue
constructions; no native residual credit.**

## What the 240 us row does and does not say

The exact non-quant partition assigns six serialized native role populations
to residual/cast/activation-epilogue/contiguous work:

| comparison | us/token |
| --- | ---: |
| native serialized role sum | 240.762 |
| llama exposed elementwise Shapley ownership | 0.443 |
| accounting difference | **240.319** |

This is an overlap-safe *location* result, not a claim that fusing generic
epilogues recovers 240 us. Llama's 0.443 us is a symmetric allocation of
overlapping non-MMQ timeline intervals. In particular, llama's fused O/down
residuals live inside MMQ nodes, while the tinygrad roles above are standalone
serialized nodes. Neither raw class time nor Shapley ownership is an A/B
saving estimate.

## Reconciliation of every prior applicable construction

| construction | result | consequence |
| --- | --- | --- |
| M4 native attention-O residual route | +69.0 us kernel time, +36 nodes, -1.15% wall | closed: it changes the emitter boundary and loses |
| P6-C all-35 llama fused attention-O substitution | +21.213 us/token | closed: fp16/fp32 and Q8 adapters plus added nodes cost more than removed adds; CUDA diagnostic only |
| P6-D Q4 FFN-down substrate vs full residual epilogue | full semantic is +0.348 us versus substrate | residual absorption is wall-neutral; the 65.8 us signal is MMVQ substrate |
| M5 typed flash-combine boundary | removes 36 copies; +0.22% in its older session | small boundary win, already outside the present generic role claim; no current same-session residual credit |
| KV store fusion | 948 -> 948; 177.8 vs 178.4 tok/s | closed wall-neutral: the baseline had already fused each layer's store chain |

This also reconciles the apparent contradiction in older copy counts: they
described different experimental or pre-promotion graphs. The current exact
native token has 948 programs and owns 37.513 us of block-output contiguous
work, not an unmeasured five-kernel KV chain per layer.

## Verdict and next admissible experiment

There is no cheap native GPU test left for “generic epilogue fusion.” Re-running
an M4-like custom kernel, llama substitution, typed boundary, or KV route would
only repeat a closed construction. No GPU A/B was therefore run.

The exact blocker is a construction, not missing timing: tinygrad needs an
ordinary-UOp, native-owned in-core projection epilogue that accepts the native
fp32 GEMV result and residual directly, while avoiding all of the already
measured taxes: custom-program transport, fp16/fp32 adapters, Q8 quantization,
activation recomputation, and a changed output-buffer contract. Once it exists,
the first valid test is a native-NV all-role A/B with token/logit pin, node
census, reverse bracket, and no overlap/Shapley arithmetic credit.

Machine ledger: `docs/task_workflow/output/nv-decode-native-epilogue-causal-ledger-20260805.json`.
Analyzer: `extra/llm_research/decode/native_epilogue_causal_ledger.py`.
