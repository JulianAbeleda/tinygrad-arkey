# Q/O Q4_K direct gates and production-arm failure (2026-09-05)

Commands (fresh NV process, current HEAD):

```sh
env DEV=NV PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/nv_compiler_q4k_qo_gate.py --role q --rounds 9 --out /tmp/nv-p4-q/q9.json
env DEV=NV PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/nv_compiler_q4k_qo_gate.py --role o --rounds 9 --out /tmp/nv-p4-q/o9.json
env DEV=NV NV_COMPILER_Q4_IMMA_QO_PP512=1 PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/nv_compiler_q4k_qo_model_arm.py --arm candidate --warmups 1 --rounds 3 --replay-cycles 1 --out /tmp/nv-p4-qo/candidate.json --logits-npz /tmp/nv-p4-qo/candidate.npz
```

The direct gates passed finite output, canonical uint32 weights, exact candidate identity, signed IMMA and numerical comparison. Q and O each use the real GGUF type-12 tensor (`blk.0.attn_q.weight` and `blk.0.attn_output.weight`), shape M512 N4096 K4096, and measured median candidate time 301.247 us (Q) and 300.185 us (O) against the static oracle's 499.904 us and 500.352 us. These are single-role direct gates, not production promotion evidence.

The fresh whole-model candidate arm fails before graph capture at `CompilerQOBinding.project`: `ValueError: compiler Q/O residual route requires residual`. The model route currently has no residual argument at the Q projection boundary, so this arm does not qualify Q. The O production-shaped residual lifecycle remains unqualified; the direct O gate does not substitute for it. The default route remains unchanged.
