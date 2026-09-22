# Compiler-native Q4_K/Q8_1 IMMA evidence

Primary retained artifacts:

- `raw-imma-gate.{json,cu,cubin,sass}`: typed logical Q4 nibble-provider gate.
- `tiny_group_accumulator_staged_k{32,64,128}.json`: bounded exact correction
  and K-substep ownership gates.
- `production_gate_staged_k64.{json,cu,cubin,sass}`: canonical full real-shape
  Gate-A result and executed binary.
- `production_gate_guarded_k64.{json,cu,cubin,sass}`: post-review full-shape
  confirmation with the explicit K32 fragment-boundary guard.
- `production_gate_staged_k32.json` and `production_gate_staged_k128.json`:
  barrier/service and spill-cliff controls.
- `proxy_72real_k64_reverse.json`: exact 72-call generated real-weight capture.
- `proxy_72real_v4_matched.json`: matched native-v4 proxy comparator.

Reproduce the canonical production gate:

```sh
flock /tmp/gpu-bench.lock env PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_production_gate.py \
  --tile-k 64 --rounds 9 \
  --out docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/production_gate_staged_k64.json
```

Reproduce the generated 72-real-weight capture:

```sh
flock /tmp/gpu-bench.lock env PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_72real_proxy.py --rounds 7 \
  --out docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/proxy_72real_k64.json
```

Whole-model Gate B artifacts:

- `model_candidate_r9.json` and `model_candidate_logits.npz`: clean,
  unprofiled compiler-packed arm.  The captured graph has exactly 72 compact
  Q8 producers, 72 compiler-generated direct-output mains, no old fixups, no
  packed-weight transport copies, and 72 canonical weight bases.
- `model_fp16_r9.json` and `model_fp16_logits.npz`: fresh-process resident-FP16
  control with zero compiler-route calls.
- `model_compare.json`: full-vocabulary correctness and synchronized wall
  comparison across the two fresh-process artifacts.
- `model_candidate_profile.json`: instrumented three-replay device profile and
  the retained full call/copy/weight census.  Its host wall includes profiler
  overhead and is not the wall authority.

Reproduce the authoritative arms without changing route environment variables
inside a process (tinygrad caches environment reads):

```sh
NV_COMPILER_Q4_IMMA_PP512=1 PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_model_arm.py \
  --arm candidate --warmups 3 --rounds 9 \
  --out docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/model_candidate_r9.json \
  --logits-npz docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/model_candidate_logits.npz

PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_model_arm.py \
  --arm fp16 --warmups 3 --rounds 9 \
  --out docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/model_fp16_r9.json \
  --logits-npz docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/model_fp16_logits.npz
```

Gate A and the default-off whole-model integration gate pass.  The production
default remains unchanged; promotion is intentionally out of scope here.
