# CUDA d512 decode group-span ledger

Date: 2026-08-04. Route: `DEV=CUDA CUDA_GRAPH_STREAMS=1`; model: Qwen3-8B-Q4_K_M; depth 512; RTX 5090 / driver 595.84.

## Result

The production CUDA route replays six strictly stream-ordered graph groups. CUDA events placed immediately before and after each unmodified `CUDAGraph.__call__` show a median **5.3787 ms** device graph window.

| group (programs) | median span |
| --- | ---: |
| 32 | 0.186-0.188 ms |
| 64 | 0.339-0.341 ms |
| 128 | 0.591-0.593 ms |
| 256 | 1.174-1.176 ms |
| 512 | 2.434-2.438 ms |
| 29 | 0.645-0.647 ms |
| total spans | **5.3763 ms** |
| inter-group gaps | **0.0024 ms** |
| device window | **5.3787 ms** |
| production-token wall | **5.6380 ms** |
| outside graph window | **0.2631 ms** |

The arithmetic closes exactly per replay: `window = sum(group spans) + sum gaps`, with zero residual to event precision, and `wall = device window + outside-window remainder` by direct elapsed-wall observation. This is not profiler arithmetic and does not infer an unmeasured overlap fraction.

## Controls and pins

- Five alternating fresh-request marker-off/marker-on arms emitted the same token (`330`) and six groups / 1021 programs each time.
- Stable ordered-program topology hashes were observed for all six groups.
- Marker-off median wall was 5.6348 ms; marker-on was 5.6380 ms: **+3.2 us (0.057%)**, below the 2% / 50-us perturbation threshold.
- The event wrappers only execute `cuEventRecord` on the same default stream around the original graph launch. They do not modify graph nodes, edges, streams, input buffers, or synchronization points.

## Relation to the llama ledger

The independently captured llama reference is 3.8898 ms graph span plus 0.0820 ms outside graph. On comparable definitions, tinygrad's graph window is therefore about **1.489 ms larger** and its outside-window remainder about **0.181 ms larger**. This is a decomposition, not a same-session parity claim: the wall rows were collected in different runs. It establishes that almost all CUDA-route time is device work inside the six graphs, not launch gaps: the six inter-group gaps total only 2.4 us.

The remaining causal work is consequently kernel/fused-subgraph attribution inside those 5.376 ms of group spans, rather than a six-launch host-gap theory.

Raw evidence: `docs/task_workflow/output/nv-decode-cuda-d512-group-span-ledger-20260804.json` (`tinygrad.cuda_decode_group_span_ledger.v1`). Tool: `scratchpad/cuda_decode_group_span_ledger.py`; hermetic arithmetic test: `test/unit/test_cuda_decode_group_span_ledger.py`.
