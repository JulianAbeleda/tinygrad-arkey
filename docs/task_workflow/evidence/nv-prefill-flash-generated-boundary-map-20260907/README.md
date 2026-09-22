# Generated Flash finalized boundary map

The exact generated families removed by the native-Flash diagnostic form one contiguous per-layer boundary:

- `E_2048...`: 36 calls, median `9.984 us`, previous `E_512_8...`, next first cache/layout writer; buffers are two 23,592,960-byte char bases plus one 262,144-byte float input.
- `E_4096...659...`: 36 calls, median `3.168 us`; one 2,097,152-byte fp16 input and one 2,377,728-byte char output/base.
- `E_4096...5a8...`: 36 calls, median `3.168 us`; the same buffer contract, immediately before generated Flash.
- generated fused Flash: 36 calls, consumes two 23,592,960-byte char bases and two 2,377,728-byte char bases.
- `E_512...284...`: 36 calls, median `3.488 us`, immediately after Flash; two 23,592,960-byte char bases.

The three measured pre-Flash services total `16.320 us/layer`; the immediate output handoff adds `3.488 us/layer`. Exact source-operation names are absent from finalized ProgramInfo metadata, so dtype/buffer direction and adjacency are authoritative while semantic labels require source instrumentation. The first bounded generated substitution should target the immediate `E_512...` output handoff because it is isolated, contiguous with Flash, and does not perturb Q/K/V production.

Source extraction closes the semantic labels. `E_2048...` is the Q rotary complex multiply fused with fp32-to-fp16 conversion. The two `E_4096...` programs are half4 slice copies: they copy the first and second 524,288-half regions of one 1,048,576-half combined KV allocation into separate contiguous K and V buffers. A typed combined-KV Flash input can therefore remove both copies without changing arithmetic; its measured ceiling is `6.336 us/layer`, or `0.228096 ms/token`.
