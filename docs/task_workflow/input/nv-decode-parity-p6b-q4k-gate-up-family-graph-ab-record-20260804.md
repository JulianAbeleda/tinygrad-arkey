# NV decode parity P6-B — Q4_K gate/up full-family llama graph A/B

Date: 2026-08-04
Status: **PASS causal CUDA diagnostic; zero native-NV ledger credit**

## Result

All 36 CUDA-route FFN gate/up chains were replaced in the live token graphs.
The diagnostic contracts tinygrad's two Q4_K projections plus separate SiLU
and multiply/cast into:

```text
shared fp16 activation
  -> fp32 adapter
  -> llama q8_1 once
  -> llama fused Q4_K MMVQ(up, gate), producing up*silu(gate)
  -> fp16 adapter into the original FFN-down consumer buffer
```

No production route, default, or model source changed. The original external
dependencies and consumer buffer identities are retained. Four graph nodes
replace four graph nodes, so the measured effect is not a launch-count claim.
One FFN crosses TinyJit's group-2/group-3 boundary; its fused f32 result crosses
the same already-sequential boundary and the old root multiply/cast is removed
from group 3. The six graphs retain node counts `32, 64, 128, 256, 512, 29`.

Two opposite-order 30-steady-sample brackets replicated a small improvement:

| bracket | control reference (ms/token) | candidate (ms/token) | delta |
| --- | ---: | ---: | ---: |
| control / candidate / control | midpoint 5.6070695 | 5.5522450 | **-54.8245 us** |
| candidate / control / candidate | 5.6187840 | midpoint 5.5514658 | **-67.3183 us** |

The six 31-token sequences are exactly identical. A separate decode-logit arm
also produced bitwise-identical full vocabulary rows: 151,936/151,936 f32
values equal, max absolute error 0, shared SHA-256
`0ce7bc66ac91d41a1a1bb481f162c4970d2d5d2e039400942baa65950501e0c1`,
and shared argmax 271.

## What this explains—and what it does not

This closes the **CUDA unfused gate/up population** as a real but small
54.8–67.3 us/token opportunity. It does not debit the 1.646170 ms native-NV
residual. Native NV already promotes tinygrad's own one-kernel w1w3 fusion,
whereas DEV=CUDA deliberately does not. Therefore the 36-core topology removal
cannot be counted as a reason native trails llama.

The native-relevant comparison is fused-versus-fused. Existing in-loop records
put tinygrad's native fused primitive near 39.4 us/layer, while llama's fused
MMVQ plus its q8 producer is approximately 37.856 + 3.072 = 40.928 us/layer
in the current trace. Those cross-session figures are direction only, but they
make gate/up unlikely to be a positive native gap contributor; a same-session
native fused class attach is the decisive closure if finer accounting is needed.

## Artifacts

- Harness: `scratchpad/cuda_decode_q4k_gate_up_llama_graph_ab.py`
- Adapter: `scratchpad/cuda_decode_q4k_gate_up_graph_adapter.cu`
- Hermetic pins: `test/unit/test_cuda_decode_q4k_gate_up_llama_graph_ab.py`
- Compact payload:
  `docs/task_workflow/output/nv-decode-q4k-gate-up-llama-family-graph-ab-20260804.json`

The six raw timing payloads and two raw logit rows remain under `/tmp`; their
hashes are pinned in the compact payload.
