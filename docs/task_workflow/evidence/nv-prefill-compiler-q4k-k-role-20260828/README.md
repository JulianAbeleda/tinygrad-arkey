# Compiler Q4_K K/V role evidence

Canonical artifacts:

- `strict-cold-r9.json`: exhaustive geometry/correctness/resource cold gate.
- `strict-hot-r9.json`: independent hot R9 confirmation.
- `population-strict-r9.json`: 36-real-K dependency-ordered population proxy.
- `q4-v-strict-r9.json`: actual Q4_K V full-output qualification.
- `model-k-candidate-r9-v3.json`: default-off K model arm, exact census,
  replay, and fresh R9.
- `model-k-control-r9-v3.json`: matched current gate/up compiler control.
- `model-k-compare-v3.json`: candidate/control full-logit and wall comparison.
- `strict-{cold,hot}-artifacts/`: emitted CUDA, cubins, and SASS.
- `q4-v-strict-artifacts/`: corresponding actual-V artifacts.

The earlier `cold-r9.json`, `hot-r9.json`, and `population-serial-r9.json`
record the useful 64x64 discovery step.  That 128-CTA geometry is deliberately
superseded by the strict 256-CTA result and is not a production claim.

Reproduce under the shared GPU lock:

```sh
flock /tmp/gpu-bench.lock env PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_k_role_gate.py --rounds 9 \
  --out docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/strict-hot-r9.json \
  --artifacts docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/strict-hot-artifacts

flock /tmp/gpu-bench.lock env PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_k_population_proxy.py --rounds 9 \
  --out docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/population-strict-r9.json

flock /tmp/gpu-bench.lock env NV_COMPILER_Q4_IMMA_PP512=1 \
  NV_COMPILER_Q4_IMMA_K_PP512=1 PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_k_model_arm.py \
  --arm candidate --warmups 3 --rounds 9 \
  --out docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-candidate-r9-v3.json \
  --logits-npz docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-candidate-logits-v3.npz

flock /tmp/gpu-bench.lock env NV_COMPILER_Q4_IMMA_PP512=1 PYTHONPATH=. \
  .venv/bin/python extra/llm_research/prefill/nv_compiler_q4k_k_model_arm.py \
  --arm control --warmups 3 --rounds 9 \
  --out docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-control-r9-v3.json \
  --logits-npz docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-control-logits-v3.npz

PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_k_model_arm.py --arm compare \
  --candidate-json docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-candidate-r9-v3.json \
  --candidate-npz docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-candidate-logits-v3.npz \
  --control-json docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-control-r9-v3.json \
  --control-npz docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-control-logits-v3.npz \
  --out docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-compare-v3.json
```
