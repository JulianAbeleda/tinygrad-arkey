# 14B Role/Facts Inventory - 2026-07-10

Model: `/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf`

Generated from the actual GGUF header with existing in-repo helpers: `gguf_load_metadata`, `model_facts_from_gguf_metadata`, and `build_model_route_plan`.

## Summary

- tensors inventoried: 282
- route-plan entries: 281
- current direct-packed prefill entries: 281
- model facts: architecture=qwen3, hidden=5120, intermediate=17408, heads=40, kv_heads=8, head_dim=128
- first facts-backed MMQ target: Q4_K `ffn_gate_up` (80 tensors, example `blk.0.ffn_gate.weight`)

## Role / Quant Counts

| role | quant | tensors | packed bytes |
| --- | --- | ---: | ---: |
| attn_kv | Q4_K | 60 | 176947200 |
| attn_kv | Q6_K | 20 | 86016000 |
| attn_qo | Q4_K | 80 | 1179648000 |
| ffn_down | Q4_K | 20 | 1002700800 |
| ffn_down | Q6_K | 20 | 1462272000 |
| ffn_gate_up | Q4_K | 80 | 4010803200 |
| lm_head | Q6_K | 1 | 638131200 |
| unresolved | Q4_K | 1 | 437575680 |

## Route-Plan Examples

| tensor name | inferred role | quant type | typ | rows | cols | packed bytes | route-plan decision | current direct-packed prefill |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `blk.0.attn_k.weight` | attn_kv | Q4_K | 12 | 1024 | 5120 | 2949120 | install_primitive_route_entry family=q4_k_packed_u32 parts=1 opts=LOCAL:0:64 | current_prefill_route_direct_packed route_id=prefill_q4k_direct_packed_load_direct_out parts=1 opts=LOCAL:0:16,LOCAL:1:16,UPCAST:0:4,UPCAST:1:4 |
| `blk.0.attn_output.weight` | attn_qo | Q4_K | 12 | 5120 | 5120 | 14745600 | install_primitive_route_entry family=q4_k_packed_u32 parts=1 opts=LOCAL:0:64 | current_prefill_route_direct_packed route_id=prefill_q4k_direct_packed_load_direct_out parts=1 opts=LOCAL:0:16,LOCAL:1:16,UPCAST:0:4,UPCAST:1:4 |
| `blk.0.attn_q.weight` | attn_qo | Q4_K | 12 | 5120 | 5120 | 14745600 | install_primitive_route_entry family=q4_k_packed_u32 parts=1 opts=LOCAL:0:64 | current_prefill_route_direct_packed route_id=prefill_q4k_direct_packed_load_direct_out parts=1 opts=LOCAL:0:16,LOCAL:1:16,UPCAST:0:4,UPCAST:1:4 |
| `blk.0.attn_v.weight` | attn_kv | Q6_K | 14 | 1024 | 5120 | 4300800 | install_primitive_route_entry family=q6_k_packed_u16 parts=4 opts=LOCAL:0:32 | current_prefill_route_direct_packed route_id=prefill_q6k_direct_generated parts=1 opts=LOCAL:0:32,UPCAST:1:4 |
| `blk.0.ffn_down.weight` | ffn_down | Q6_K | 14 | 5120 | 17408 | 73113600 | install_primitive_route_entry family=q6_k_packed_u16 parts=1 opts=LOCAL:0:64 | current_prefill_route_direct_packed route_id=prefill_q6k_direct_generated parts=1 opts=LOCAL:0:64,UPCAST:1:4 |
| `blk.0.ffn_gate.weight` | ffn_gate_up | Q4_K | 12 | 17408 | 5120 | 50135040 | install_primitive_route_entry family=q4_k_packed_u32 parts=1 opts=LOCAL:0:64 | current_prefill_route_direct_packed route_id=prefill_q4k_direct_packed_load_direct_out parts=1 opts=LOCAL:0:16,LOCAL:1:16,UPCAST:0:4,UPCAST:1:4 |
| `blk.0.ffn_up.weight` | ffn_gate_up | Q4_K | 12 | 17408 | 5120 | 50135040 | install_primitive_route_entry family=q4_k_packed_u32 parts=1 opts=LOCAL:0:64 | current_prefill_route_direct_packed route_id=prefill_q4k_direct_packed_load_direct_out parts=1 opts=LOCAL:0:16,LOCAL:1:16,UPCAST:0:4,UPCAST:1:4 |
| `blk.1.attn_k.weight` | attn_kv | Q4_K | 12 | 1024 | 5120 | 2949120 | install_primitive_route_entry family=q4_k_packed_u32 parts=1 opts=LOCAL:0:64 | current_prefill_route_direct_packed route_id=prefill_q4k_direct_packed_load_direct_out parts=1 opts=LOCAL:0:16,LOCAL:1:16,UPCAST:0:4,UPCAST:1:4 |

Full per-tensor inventory is in `docs/14b-role-facts-inventory-20260710.json` with fields: tensor name, inferred role, quant type, typ, rows, cols, packed bytes, route-plan decision, and current direct-packed prefill parts/opts.
