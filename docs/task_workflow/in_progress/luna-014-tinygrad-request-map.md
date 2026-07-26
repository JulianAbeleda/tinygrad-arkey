# LUNA-014: Tinygrad Equivalent Source Map

Status: CPU/static source map only. No GPU behavior is asserted.

## Canonical fixed-depth route

`extra/qk/decode/decode_runtime_overhead.py:main` loads through
`extra.llm.generate.load_model_and_tokenizer`, creates `prefix() + encode("the quick brown fox jumps. " * 800)`, repeats/truncates it to each checkpoint, then calls `_warm_depth` before measurement. `_prefill` calls `Transformer.generate`; the first `next(gen)` performs prompt setup and returns the first sampled token. `_measure_w` continues that generator with per-token host sync. `_measure_d` instead invokes the selected captured model JIT and one final sync.

`tinygrad.llm.model.Transformer.generate` chunks the prompt (canonical `--chunk-size 32`) and calls `Transformer.__call__`. `__call__` selects `prefill_jit` for non-single-token calls and `rollout_jit` or `rollout_jit_flash` for `T == 1`. `forward -> logits` runs embedding, blocks, output norm, LM head, then temperature-zero argmax sampling.

## Context boundaries

| Boundary | Owner | 128 | 512 / 4096 | Decision class | Observable control |
|---|---|---|---|---|---|
| Fixed depth | `decode_runtime_overhead._make_prompt` | exactly 128 input IDs | exactly requested input IDs | workload | prompt evidence hash in output JSON |
| Prefill chunks | `Transformer.generate` | 32-token chunks | 32-token chunks; special concrete-ubatch paths require their own config | shape/runtime | generated call sizes / JIT captures |
| Custom prefill attention | `model._should_use_custom_kernel_prefill_attn`, `fused_attention.prefill_grid_spec` | declined: admission table only has `q_tokens=512` | admitted only if exact `(Hq,Hkv,T)` is `(32,8,512)` or `(40,8,512)` on AMD gfx1100 | shape/backend | `custom_kernel_attention_trace_snapshot()` and route identity |
| Prefill attention fallback | `fused_attention.route_prefill_attention` | SDPA after custom route is declined | custom-kernel injection if all gates pass; otherwise SDPA | shape/backend | custom route dispatch counter; otherwise SDPA graph |
| Decode flash selection | `route_policy.should_use_flash_decode` | default auto declines while context is below 512 | default auto admits when `start_pos + T >= 512`, `T == 1`, and `flash_decode` is enabled | runtime/shape policy | `decode_runtime_overhead._route` / artifact `route_sequence` |
| Decode flash implementation | `decode_routes.flash_decode_attention_route` | not selected by default auto | G5 candidate for `Hq=40`, G4 candidate for `Hq=32`; both require `B=1,Hkv=8,Hd=128` | geometry/backend | candidate binding or raised unsupported-shape error |

## Prefill-linear routing and direct-packed boundary

`Transformer.__call__` marks each Q4_K/Q6_K linear as prefill or decode. The linear path reaches `tinygrad.llm.prefill_routes.route_prefill_linear`.

1. `_attached_production_route` requires an immutable `PrefillRouteAttachment`; environment variables cannot select a production route.
2. For attached `direct_packed` / `bounded_packed` routes, `route_packed_wmma_prefill` runs first when `packed_wmma_prefill_enabled()` is true. It requires `_attached_direct_packed_spec` and a selected candidate from `route_ops.select_packed_wmma_prefill_candidate`.
3. A missing/declined packed-WMMA candidate returns `None`; `route_prefill_linear` then calls `route_direct_packed_prefill`.
4. `route_direct_packed_prefill` rechecks attachment, `PrefillDirectPackedBinding`, active prefill scope, exact `(m,n,k)`, role, and Q4_K/Q6_K storage. It then emits the direct packed kernel. If that also declines, route falls through to attached FP16 graph-GEMM or ordinary `Tensor.linear`.

The explicit rollback gate is `packed_wmma_prefill_enabled`, controlled by `TINYGRAD_PREFILL_PACKED_WMMA` (default enabled). `validate_packed_wmma_prefill_mode(40,8)` rejects explicit disable before model construction. That gate does **not** prove a candidate cannot decline per shape; the implicit boundary is the `None` result from `select_packed_wmma_prefill_candidate` / `candidate.run`. LUNA-034 must instrument those exact decisions and demonstrate both a selected packed-WMMA positive control and an allowed direct-packed positive control before attributing ctx128 to this fallback family.

## 8B versus 14B decode geometry

`decode_routes._FlashDecodeCandidate.bind` encodes `Hq=32`; `FLASH_DECODE_G5_CANDIDATE` encodes `Hq=40`. Both use `split_size=48`, `staging=KV_BOTH`, and feed `route_ops.flash_decode_live_split_block_tile`. The source map establishes route identity only; it does not establish GPU resource use, occupancy, correctness, or throughput.

## Verdict

LUNA-014 source-map acceptance is met: every LUNA-034 decision point has a named owner and a positive-control seam. The actual ctx128 route remains unverified and requires LUNA-031/LUNA-034 evidence.
