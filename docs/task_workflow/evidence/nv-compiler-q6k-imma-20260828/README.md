# NVIDIA compiler-native Q6_K IMMA evidence

- `qualification-r9.json`: authoritative adversarial, real V, and real
  FFN-down correctness/resource/timing record.
- `artifacts/adversarial.{cu,cubin,sass}`: bounded scale/bit/sign discriminator.
- `artifacts/v.{cu,cubin,sass}`: real `blk.0.attn_v.weight` candidate.
- `artifacts/down.{cu,cubin,sass}`: real `blk.0.ffn_down.weight` candidate.

Reproduce from the repository root:

```sh
PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q6k_imma_gate.py \
  --rounds 9 --roles v,down \
  --out docs/task_workflow/evidence/nv-compiler-q6k-imma-20260828/qualification-r9.json \
  --artifacts docs/task_workflow/evidence/nv-compiler-q6k-imma-20260828/artifacts
```

The harness is research-only and makes no model-routing change.
