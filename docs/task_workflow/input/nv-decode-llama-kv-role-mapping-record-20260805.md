# Llama Qwen3 d512 K/V MMVQ role mapping record

Date: 2026-08-05. Evidence class: source plus the pinned 762-node CUDA graph
trace. Status: **PASS - exact for all 36 Qwen3 layers; no GPU experiment was
needed.**

## Result

The two equal-shape `1024 x 4096` llama MMVQ projections in each layer are:

| captured order after Q | exact model role | median llama MMVQ us | native matched role us | diagnostic delta us |
| --- | --- | ---: | ---: | ---: |
| first | attention V | 165.053 | 382.465 | +217.412 |
| second | attention K | 117.376 | 152.381 | +35.005 |

This corrects the former launch-order-only pairing, which had assigned the
first projection to K and second to V. The aggregate quantized-core accounting
does not change: it remains `3882.604 - 3579.816 = +302.788 us`. Only its exact
K/V ownership changes.

## Proof

For the pinned llama.cpp commit `ac4cddeb0dbd778f650bf568f6f08344a06abe3a`:

1. `src/models/qwen3.cpp` calls `build_qkv(...)` and passes the resulting
   `Qcur, Kcur, Vcur` to `build_attn(...)` for each `il` in `0..35`.
2. The separate-QKV branch in `src/llama-graph.cpp::build_qkv` constructs the
   matmuls in Q/K/V order: `layer.wq`, `layer.wk`, `layer.wv`.
3. `src/llama-graph.cpp::build_attn` then expands them in **Q/V/K** order:
   `q_cur`, `v_cur`, `k_cur`. Its adjacent comment explicitly says the nodes
   are added together so they are not reordered, and that K is expanded later
   to enable RoPE fusion directly into the KV cache.
4. A graph expansion recursively appends new dependency nodes and preserves
   already-visited nodes (`ggml_build_forward_expand` in `ggml/src/ggml.c`).
   Hence the captured MMVQ order after Q is V then K, not construction order.

The pinned trace is consistent at both ends of the 36-layer census: layer 0
has Q/V/K MMVQ node IDs `8589934594/8589934598/8589934600`; layer 35 has
`8589935329/8589935333/8589935335`. The regenerated manifest contains exactly
36 `attn_v` and 36 `attn_k` rows and fails closed if its expected 762-node
census or Q8-to-MMVQ adjacency changes.

## Bound and scope

This establishes role identity and launch order, not buffer addresses, strides,
or an independently additive per-role wall contribution. The role timings are
independently medianed diagnostics and must not be summed into the aggregate
MMQ union. No production route, llama source, benchmark configuration, or
default changed.

Artifacts: `nv-decode-llama-kv-role-mapping-20260805.json`, the regenerated
`nv-decode-llama-tinygrad-semantic-call-manifest-20260804.json`, and
`nv-decode-native-semantic-profile-ledger-20260805.json`.

Validation:

```text
PYTHONPATH=. .venv/bin/python -m pytest -q \
  test/unit/test_llama_tinygrad_role_manifest.py \
  test/unit/test_native_semantic_profile_ledger.py
```
