# Compiler-native Q4_K Q/O qualification evidence

Decision: primitive PASS, 72-real-weight population PASS, separate default-off
whole-model PASS with the exact captured Flash-dependency queue cut.  Unguarded
NV ready placement remains a replay NO_GO.

Primary artifacts:

- `q_gate_r9.{json,cu,cubin,sass}`: real `blk.0.attn_q.weight` full-output gate.
- `o_gate_r9.{json,cu,cubin,sass}`: real `blk.0.attn_output.weight` full-output gate.
- `proxy_72real_r9.json`: captured all-real-Q/O compiler versus FP16 population.
- `model_candidate_r9.json` and `model_candidate_logits.npz`: structurally
  clean default-off candidate wall and output; status FAIL on exact replay.
- `model_fp16_r9.json` and `model_fp16_logits.npz`: fresh FP16 authority.
- `model_compare.json`: full-logit/token comparison and wall arithmetic.
- `model_candidate_stage_replay_detail.json`: block-0 versus block-35 Q8 and
  main-output replay localization.
- `model_candidate_replay_probe.json`, `model_candidate_stage_replay_probe.json`,
  and `model_fp16_replay_probe.json`: retained diagnosis runs.
- `candidate_graph_profile.jsonl`, `candidate_queue_census.jsonl`, and
  `attention_safe_cut.json`: captured graph authority and conservative cut.
- `model_candidate_safe_cut_20cycle_replay.json` versus
  `model_candidate_default_20cycle_replay.json`: decisive replay stress.
- `model_candidate_attention_safe_cut_r9.json`,
  `model_fp16_attention_safe_cut_fresh_r9.json`, and
  `model_attention_safe_cut_compare.json`: admitted fresh wall/quality bracket.
- `attention_safe_cut_dep*.json` and `model_candidate_cut_dep*_probe.json`:
  per-edge causal ablations; dependency position 2 is the only single-position
  cut that passed the retained 20-cycle stress.

Reproduce the isolated role gates:

```sh
flock /tmp/gpu-bench.lock env PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_qo_gate.py \
  --role q --rounds 9 \
  --out docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/q_gate_r9.json

flock /tmp/gpu-bench.lock env PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_qo_gate.py \
  --role o --rounds 9 \
  --out docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/o_gate_r9.json
```

Reproduce the population gate:

```sh
flock /tmp/gpu-bench.lock env PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_qo_72real_proxy.py --rounds 9 \
  --out docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/proxy_72real_r9.json
```

The model arm is intentionally default-off and separate from the gate/up
research variable.  Do not set either gate/up variable while reproducing it.

```sh
flock /tmp/gpu-bench.lock env NV_COMPILER_Q4_IMMA_QO_PP512=1 PYTHONPATH=. \
  .venv/bin/python extra/llm_research/prefill/nv_compiler_q4k_qo_model_arm.py \
  --arm candidate --warmups 3 --rounds 9 \
  --out docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/model_candidate_r9.json \
  --logits-npz docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/model_candidate_logits.npz

flock /tmp/gpu-bench.lock env -u NV_COMPILER_Q4_IMMA_QO_PP512 PYTHONPATH=. \
  .venv/bin/python extra/llm_research/prefill/nv_compiler_q4k_qo_model_arm.py \
  --arm fp16 --warmups 3 --rounds 9 \
  --out docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/model_fp16_r9.json \
  --logits-npz docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/model_fp16_logits.npz
```

Reproduce the admitted candidate with the exact graph cut:

```sh
PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/nv_qo_attention_safe_cut.py \
  --profile docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/candidate_graph_profile.jsonl \
  --census docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/candidate_queue_census.jsonl \
  --out docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/attention_safe_cut.json

flock /tmp/gpu-bench.lock env HCQ_NV_READY_PLACEMENT=0 \
  HCQ_NV_MULTI_QUEUE_CUT_POLICY=docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/attention_safe_cut.json \
  NV_COMPILER_Q4_IMMA_QO_PP512=1 PYTHONPATH=. .venv/bin/python \
  extra/llm_research/prefill/nv_compiler_q4k_qo_model_arm.py \
  --arm candidate --deep-replay --replay-cycles 20 --warmups 3 --rounds 9 \
  --out docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/model_candidate_attention_safe_cut_r9.json
```
