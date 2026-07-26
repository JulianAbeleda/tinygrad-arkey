# LUNA-012: G=5 attention and KV source map

## Status

`PASS (static map and byte formula)`. Applies to a Qwen3-14B geometry assumption of 48 transformer layers, 40 query heads, 8 KV heads, and head dimension 128: `G = n_head / n_head_kv = 40 / 8 = 5`. LUNA-002 must verify these GGUF metadata values and KV types before trace results are treated as model-specific.

## Ownership map

| Concern | Source owner | Static behavior | Expected trace family |
|---|---|---|---|
| Q/K/V and GQA grouping | `src/llama-graph.cpp` attention construction; model hyperparameters in `src/llama-model.*` | Q has 40 heads; K/V retain 8 heads and attention maps five Q heads to each KV head. | Projection `mul_mat`; then RoPE / reshape / copy kernels as applicable. |
| KV allocation/layout | `src/llama-kv-cache.cpp`, `llama_kv_cache` initialization and layer buffers | Per-layer K and V tensors are stored in cache layers. Views use `n_head_kv`, per-head K/V dimensions and cache sequence/cell indexing. | Allocation/setup is outside measured token kernels; record buffer pointers/sizes from backend logs where available. |
| KV writes | `llama_kv_cache_context::cpy_k` / `cpy_v`; cache `cpy_k` / `cpy_v` implementations | Current-token K/V are copied to cache positions selected by K/V index inputs. | `cpy` / set-style kernels during prompt and every decode token. |
| KV reads | `llama_kv_cache_context::get_k` / `get_v`; cache `get_k` / `get_v` | Attention consumes cache views covering active cells/KV length. | Attention score/value kernels; memory traffic grows with context. |
| Mask and causal bounds | `llama_kv_cache::set_input_kq_mask`; graph attention construction | Host fills KQ mask from ubatch positions and causal flag. A query may only attend to permitted prior/current positions. | Mask materialization or fused attention-mask behavior. |
| Flash admission | `src/llama-graph.cpp` flash-attention branch and ggml `FLASH_ATTN_EXT` node; backend `ggml/src/ggml-cuda/fattn.cu` | Enabled only when context option, graph shape/types/layout, and backend support admit it. Otherwise graph uses explicit QK matmul, scale/mask/softmax, then V matmul. | Fused `fattn` family if admitted; otherwise `mul_mat` + softmax + `mul_mat` families. |
| Split/combine | `llama-graph.cpp` graph construction and backend scheduler | ubatching splits prompt graphs; backend scheduler may split graph execution across assigned backends. GQA reshapes/views map Q heads to KV heads; no five-way duplicated KV allocation is implied. | Multiple submissions for prompt ubatches; attention kernels per layer. |

## KV bytes

For F16 K and F16 V, one token stores:

`bytes/token = n_layer * (n_head_kv * head_dim * sizeof(K) + n_head_kv * head_dim * sizeof(V))`

`= 48 * 2 * 8 * 128 * 2 = 196,608 bytes = 192 KiB/token`.

| Active context | KV payload (F16 K + F16 V) |
|---:|---:|
| 128 | 25,165,824 bytes = 24 MiB |
| 512 | 100,663,296 bytes = 96 MiB |
| 4096 | 805,306,368 bytes = 768 MiB |

This is logical K/V payload only: it excludes allocator padding, metadata/cell arrays, staging, graph temporaries, and any KV quantization. If GGUF/context options select different `type_k` or `type_v`, replace `sizeof(K/V)` with their row storage costs; `llama_kv_cache::size_k_bytes` and `size_v_bytes` are the authoritative implementation-level counters.

## Context expectations and controls

- At 128/512/4096, each decode still writes one K and one V vector per layer; read span increases with active KV cells.
- Flash path: expect a fused `fattn` candidate per attention layer only when admission is positive. Otherwise expect decomposed score/mask/softmax/value families.
- Positive control for LUNA-021: correlate one attention launch to a `FLASH_ATTN_EXT` graph node or to the decomposed node sequence. Do not infer flash from the CLI flag alone.
- Causal control: use a standard causal prompt and verify no trace/source record expands the attended K range past the current position.
