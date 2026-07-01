# Attention-Combine Reduce — FLASH_L Knob Search (14B) — REFUTED

Lever: shrink the attention score/combine reduce (~12-24% of decode, the biggest
removable reduce bucket) WITHOUT a handwritten kernel. Constraint: codegen/search
does everything; no handwritten kernel.

## Key fact: the flash decode is already a generated route

`_attention` -> `extra/qk_flash_decode.flash_decode_attention`, default variant
`gqa_coop_vec` — a GENERATED/scheduler route (tinygrad ops), NOT handwritten (the
hand-AMDGCN "owned tile" is `DECODE_ATTN_AMDGCN_TILE`, default-OFF). It splits the
KV cache into S chunks (S = ctx / FLASH_L) -> Hq*S workgroups, each a partial, then
an online-softmax COMBINE across chunks. That combine IS the attention_combine
reduce. Its size is set by the existing search knob `FLASH_L`. So this is a pure
knob search — no kernel to write.

## Search — REFUTED

Authority W==D (14B, ctx512) across FLASH_L, plus the attention_combine bucket
(BoltBeam `resolve_trace` + `bucket_delta` on the reduce traces):

| FLASH_L | chunks | tok/s | attention_combine bucket |
|---------|--------|-------|--------------------------|
| 128 (default) | 4 | **50.20** | 13.57% |
| 256 | 2 | 47.60 | ~ |
| 512 | 1 | 43.50 | 12.01% |

Raising L **does** shrink the combine (13.57% -> 12.01%, total reduce 19.0% ->
16.8% — mechanism confirmed), but tok/s regresses monotonically: fewer chunks =
less parallelism, and the parallelism loss dominates the combine saved.

## BoltBeam verdict

Evaluated with the new `reduce_eliminated` mechanism guardrail:
**reduce_eliminated = PASS** (the combine really shrank 1.56pp) but verdict =
**refute** (protected-context speed regression -13.3%). The guardrail confirms the
mechanism while the evaluator still refuses to credit a net regression — exactly
the intended behavior.

## Conclusion / reopen

The attention combine cannot be cheaply removed at the knob level (chunk count
trades directly against parallelism). The real capability is an **in-kernel
combine** inside the generated flash route's combine expression (LDS/atomics across
the KV chunks so there is no external reduce AND no parallelism loss) — GENERIC
codegen in `extra/qk_flash_decode`, NOT a handwritten kernel. This is the SAME
unifying capability as `decode_q4k_split_k_kv`'s reopen (in-kernel combine): build
it once and it addresses both L2 (split-K KV) and the attention combine.

No tinygrad code changed (FLASH_L is a pre-existing knob); this is a pure search
result. BoltBeam candidate `decode_attention_combine_reduce_fusion`.
