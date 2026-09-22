# Q4 FFN-down live R31 gate

Command:

```sh
env DEV=NV PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/nv_q4down_matched_ab.py \
  --z docs/task_workflow/evidence/nv-q4down-capture-20260829/z.npy \
  --out docs/task_workflow/evidence/nv-q4down-live-r31-20260905/result.json --rounds 9
```

The harness constructs candidate and live llama projection graphs with dynamic activations, realizes each graph, and interleaves 31 timed calls. It records PROGRAM identities and producer/main/fixup counts from each captured graph. The result is finite and within the declared tolerance on two activations: candidate generated main 18, producer 18; llama main 18, producer 18, fixup 18. `max_abs=0.01215595` on the first activation. The candidate median is 8.793532 ms and llama is 3.685446 ms. The candidate is therefore correctness-qualified for this fixture but rejected on performance; it remains opt-in.
