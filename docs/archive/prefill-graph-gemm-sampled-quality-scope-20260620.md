# Prefill Graph GEMM Sampled Quality Scope - 2026-06-20

Verdict target: `PASS_PREFILL_GRAPH_GEMM_SAMPLED_QUALITY`

The graph GEMM route passed one-role numeric correctness and full-route performance, then blocked on the
existing teacher-forced NLL harness because it realizes a full `(512, vocab)` logits tensor at high VRAM.
That is a harness memory problem, not yet evidence of a model-quality problem.

## Online Context

The low-memory shape of the next gate is consistent with external practice:

- vLLM issue `#5907` shows logprobs can add enough memory during prefill to OOM even when the model itself fits:
  https://github.com/vllm-project/vllm/issues/5907
- vLLM's logprobs path aggregates prompt logprobs during prefill chunks instead of keeping all prompt logits:
  https://docs.vllm.ai/en/v0.22.1/api/vllm/v1/engine/logprobs/
- Cut Cross-Entropy states the loss only needs the ground-truth logit plus log-sum-exp, not a retained full
  logits matrix: https://arxiv.org/html/2411.09009v1
- Hugging Face's perplexity docs frame the metric as next-token likelihood, so sampled teacher-forced positions
  are a valid narrow smoke before a full corpus pass: https://huggingface.co/docs/transformers/en/perplexity

## Gate

Run baseline `PREFILL_V2` and `PREFILL_GRAPH_GEMM=1` in separate subprocesses. Each child:

- loads the Qwen3-8B Q4_K_M model with `PREFILL_V2=1`,
- uses a concrete `T=512` window so the graph route is eligible,
- computes `model.logits(tokens, 0)[:, -2, :]` and realizes only that vocab vector,
- scores `-log p(win[-1] | win[:-1])`.

Acceptance:

| check | threshold |
|---|---:|
| baseline finite | true |
| graph finite | true |
| max sampled `abs(dNLL)` | `<= 0.01` |

This does not replace a future full corpus/perplexity gate. It closes the immediate promotion blocker: the
graph route must not perturb teacher-forced logits at an observable sampled position while exercising the real
512-token in-model path.
