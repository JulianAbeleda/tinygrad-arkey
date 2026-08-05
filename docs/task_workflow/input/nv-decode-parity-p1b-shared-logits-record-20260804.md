# P1-B shared-input prompt-final logits record

Date: 2026-08-04. Diagnostic only; no production route, lowering, or default
was changed. This closes the original vague correctness blocker with an exact
shared-token experiment, while keeping the numeric gate fail-closed.

## Construction

The corpus is the decode authority's exact d512 prompt construction:

- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, SHA-256
  `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`;
- 512 raw token IDs (not retokenized text), SHA-256
  `e536e95f0307d714bc77b286f8ee4a6f7f38dc5ebc3c7321eda2170149308c49`;
- llama arm: pinned CUDA llama.cpp C API, all layers GPU, raw integer IDs;
- tinygrad arms: `DEV=NV` and `DEV=CUDA`, same integer IDs and `start_pos=0`.

`scratchpad/llama_prompt_final_logits_dump.cpp` obtains llama's final-prompt
row. `scratchpad/shared_prompt_final_logits_probe.py` obtains the tinygrad row,
its backend `argmax`, and independently materializes the host row. GPU calls
were serialized with `/tmp/gpu-bench.lock`.

## Results

| check | llama CUDA | tinygrad NV | tinygrad CUDA |
|---|---:|---:|---:|
| prompt-final argmax | 13876 | 13876 (backend + host) | 13876 (backend + host) |
| top-10 set overlap vs llama | — | 10/10 | 10/10 |
| top-100 set overlap vs llama | — | 95/100 | 95/100 |
| full-row cosine vs llama | — | 0.9996496662 | 0.9996496662 |
| full-row Pearson vs llama | — | 0.9994834122 | 0.9994834122 |
| full-row MAE / RMSE / max abs vs llama | — | .126559 / .158589 / .795796 | same |

The NV and CUDA full rows are **bitwise identical**, not merely close:
151,936 f32 entries, SHA-256
`efb907ccb1cf19b2cf238aa54dc4fb5f132d7e61c2416f1554749363fd86aac2` for
both. The common top-1/top-2 margin is 15.255, so this argmax match is not a
near tie.

The declared strict gate (`atol=0.01`, exact argmax) remains **FAIL**: full-row
max absolute difference is .795796. Do not relax that tolerance after seeing
the result. The result is evidence of close but non-identical cross-engine
quantized numerics, not a strict llama numeric qualification.

## New localized correctness finding

On the same native-NV process and IDs, direct production sampling

```python
model(Tensor([tokens], dtype="int32"), 0, Tensor([0.0]), use_flash=False).item()
```

returns **151936**, one past the 151936-entry vocabulary, whereas both
`logits.argmax(-1).item()` and host `numpy().argmax()` return 13876. This was
run through the established `/tmp/b3_runner.py` fused-prefill-off diagnostic
wrapper; without it, the known packed-fragment prefill verifier failure prevents
the production-forward control from compiling.

Therefore the previous claim that native's fixed-depth sentinel stream might
reflect a prompt/prefill numerical divergence is refuted. The reproducible
owner is the **post-logits temperature-zero sampling / argmax / scalar-output
path**, not native model kernels, KV construction, or prompt-final logits.
`151936` is an invalid sentinel and must not be described as the prompt-final
argmax.

## Next cheapest decisive test

Capture the exact UOp/program census for the two expressions below under NV
and CUDA, then compare the scalar output and buffer bounds:

```python
logits.argmax(-1).item()
model(tokens, 0, Tensor([0.0]), use_flash=False).item()
```

The first is already correct; the second is faulty. Their graph difference is
the minimal owner search space (temperature clamp/division, random/Gumbel
expression, reduction, and scalar boundary). Do not resume decode-parity
qualification until an in-range sample and the shared-logits strict numeric
policy are separately resolved.
