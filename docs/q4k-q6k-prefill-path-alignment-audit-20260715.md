# Q4_K/Q6_K prefill path-alignment audit

Status: scoped audit; documentation only. No runtime behavior or route policy was changed.

## Conclusion

The current `direct_packed` path is a correct packed-format execution family, but it is not yet structurally aligned with the llama MMQ path. The largest gap is lifecycle: llama stages one cooperative Q4/Q8 tile and reuses it across many output elements, while tinygrad's current generated direct-packed kernels load/decode packed data inside a per-output-token GEMM body. Existing Q4 MMQ/DS4 modes are research-selectable experiments, not the default, and Q6 has no corresponding MMQ-style staged grammar.

The authoritative baseline remains `direct_packed`; no MMQ route should be promoted from this audit.

## Reference and current paths

| Concern | llama-style reference | Current tinygrad path | Gap / implication |
|---|---|---|---|
| Route selection | `extra/qk/mmq_llama_research_source.py:17-55` records `mmq.cuh` anchors: `load_tiles_q4_K`, `vec_dot_q4_K_q8_1_impl_mmq`, `mmq_write_back_mma`, `mul_mat_q_process_tile` | `tinygrad/llm/prefill_routes.py:435-455` selects `direct_packed` for `auto` when no fp16 overlay exists; `:400` binds the Q4/Q6 direct candidate | Reference is a cooperative tile lifecycle; production selection is a packed GEMM family selector with no MMQ lifecycle contract |
| Q4/Q8 mode | Q4 weights plus Q8_1/DS4 activation panels; two Q8 panels are loaded around the Q4 tile and separated by uniform barriers | Q4 research modes `wmma`, `wmma_tiled`, `packed_ds4`, `packed_row_major`, `packed_fused` are selected by `PREFILL_Q4K_Q8` at `prefill_routes.py:364-398`; default empty mode uses direct packed | DS4 packing exists, but staging and barrier reuse are not represented as a production candidate contract |
| Q6 mode | No Q6 claim should be inferred from Q4 MMQ evidence | `Q6KDirectPackedPrefillCandidate.run` at `prefill_routes.py:253-282` uses packed Q6 halfwords and scalar/generated packed-load kernels | Q6 has format-specific decode, but no separate staged Q6 tile/reuse/ownership path |
| Default / rollback | llama reference is evidence for structure, not a tinygrad dispatch default | `PREFILL_ROUTE_CHOICES` and `route_prefill_linear` at `prefill_routes.py:13-16, 430-454` preserve direct-packed | Correct safety posture; alignment work must remain research-only |

## Quant inventory and role coverage

The inventory is fact-driven, not profile-label-driven:

- `tinygrad/llm/model_facts.py:8-11` maps GGML types 12 and 14 to Q4_K and Q6_K.
- `QwenDenseRoleResolver.resolve` at `model_facts.py:54-68` resolves role from architecture and exact matrix shape.
- `test/unit/test_model_route_plan.py:17-70` exercises Qwen-like 8B and 14B shapes, including Q4 roles (`ffn_gate_up`, `ffn_down`, `attn_qo`, `attn_kv`) and Q6 `ffn_down`, `attn_v`, and `lm_head`.
- `model_route_plan.py:42-53` groups module leaves into role families, while `:58-70` installs different Q4/Q6 defaults. Q4 and Q6 coverage is therefore not interchangeable.

Alignment gaps:

1. `ModelFacts` records exact tensor quant, but the prefill route receives quant from linear-object attributes (`prefill_routes.py:67-73, 291-294`). Add a test comparing installed linear quant, fact inventory, route spec, and packed-byte/halfword storage for every role.
2. `PrimitiveRouteEntry` (`model_route_plan.py:11-25`) carries parts/options but not tile geometry, staging lifetime, or ownership identity. It cannot describe whether an entry is direct-packed or a cooperative MMQ candidate.
3. `Q6K_COVER_MORE` and role-specific defaults at `model_route_plan.py:64-70` are coverage policy, not evidence that the Q6 role has a valid MMQ path.

## Tile geometry

### Llama geometry

`extra/qk/mmq_llama_oracle.py:LlamaMMQOracleGeometry` defaults to:

| Field | Value | Meaning |
|---|---:|---|
| `mmq_x` | 128 | output-column tile |
| `mmq_y` | 128 | output-row tile |
| `iter_k` | 256 | one complete Q4_K K block |
| `nwarps` | 8 | cooperative row stripes |
| `warp_size` | 32 in the oracle | must be checked against the actual target path |
| `tile_C` | 16x16 | fragment/writeback geometry |

`LlamaMMQOracleGeometry.validate` requires `nwarps * tile_c_i == mmq_y`, `mmq_x % tile_c_j == 0`, and `iter_k % 256 == 0`. The source deconstruction also records the reference's wave64/RDNA3 assumptions in `docs/14b-mmq-wave-process-deconstruction-20260710.md:92-121`; these must not be copied into a gfx1100 candidate without a target-capability check.

### Current geometry

- Q4 direct-packed spec: `extra/qk/q4k_prefill_route_spec.py:24-73` validates only rows, K alignment, parts, and output layout. Its kernel uses logical `row`, `bb`, `blk`, and `lane4` at `:94-108`; there is no cooperative M/N output tile descriptor.
- Q6 direct-packed spec: `extra/qk/q6k_prefill_route_spec.py:37-75` has the same shape/parts boundary, with `lane2` in `:99-111`.
- Runtime defaults apply Q4 `LOCAL:0:16`, `LOCAL:1:16`, and two `UPCAST:4` choices in `prefill_routes.py:146-154`; Q6 uses a single B upcast through the shared fallback at `:155`.
- The Q4 MMQ descriptor (`extra/qk/q4k_q8_mmq_prefill_spec.py:18-116`) declares 16x16x256 and 64/256 workgroup choices, but its emitter is explicitly `NotImplementedError` at `:128-129`.

Gap: the current 16x16 direct/generated shape is a useful legality and correctness canary, but it is not the llama 128x128 cooperative tile. A future candidate must carry M/N/K geometry and target mapping as candidate data and prove edge tiles; it must not silently reinterpret direct `row`/`bb` loops as cooperative MMQ axes.

## Staging and reuse

The llama lifecycle is explicit in `docs/14b-mmq-wave-process-deconstruction-20260710.md:63-91`:

1. stage `tile_x` (decoded Q4_K weights) into CTA-local storage;
2. stage `tile_y` (Q8_1 activation panel);
3. synchronize;
4. run the dot;
5. stage a second activation panel and reuse the same weight tile;
6. synchronize and continue K.

The current path differs:

- `Q4KDirectPackedPrefillCandidate.run` allocates partials and invokes `emit_q4k_packed_prefill_kernel` (`prefill_routes.py:225-250`).
- `_q4k_block_dot_packed_load_gemm` is called from the K/lane body in `extra/qk/quant/q4_k_gemv_primitive.py`; its reuse is compiler CSE within the unrolled token dimension, not a declared shared tile lifecycle.
- Q6 follows the same shape in `prefill_routes.py:253-282`, calling `_q6k_block_dot_packed_load_gemm` from `extra/qk/quant/q6_k_gemv_primitive.py`.
- Q8 DS4 values/scales/sums can be cached across calls through `_MMQ_DS4_LAST_PACKED` (`prefill_routes.py:18, 390-396`), but this is a host-side packed-activation cache, not device tile reuse. Cache identity and preparation cost must be included in any comparison.

Gap: no current production descriptor proves where decoded weights or Q8/Q6 panels live, their lifetime, barrier scope, or reuse count. `Staging` exists in `extra/qk/mmq_logical_vocabulary.py`, but the Q4 descriptor does not lower it, and Q6 only has the separate semantic grammar in `extra/qk/q6k_mmq_vocabulary.py`.

## Ownership and writeback

The llama contract separates cooperative computation from final ownership:

- `mmq_write_back_mma` and the `sum[(j0/tile_C::J + n)*tile_C::ne + l]` mapping are captured by `llama_mma_sum_slot_mapping` in `extra/qk/mmq_llama_oracle.py:70-151`.
- The oracle reports missing and duplicate output owners and explicitly marks itself `research_only` / `production_dispatch_changed=False`.
- The reference target is one owner per output, after all waves cooperate on the tile.

Current path:

- Direct-out Q4 writes `out[bb, row]` in `q4k_prefill_route_spec.py:104-108`; Q6 does the analogous write at `q6k_prefill_route_spec.py:106-111`.
- Partial paths write `partials[row, bb, part]` and later reduce in `prefill_routes.py:246-250, 278-282`.
- Current ownership is output-index/partial-axis ownership, not fragment-to-lane ownership. No emitted-lane proof connects it to the llama fragment mapping.

Gap: a passing direct-output numerical test does not prove cooperative MMQ writeback coverage. Required evidence is exact owner coverage for bounded 16x16 and 128x128 tiles, including edge M/N tiles and partial-K behavior.

## Scoped gap matrix and required tests

| Area | Current status | Missing proof | Test / artifact needed | Primary owner files |
|---|---|---|---|---|
| Fact inventory | Q4/Q6 labels and shape-derived roles exist | installed storage and route quant agree with GGUF facts | fixture with mixed Q4/Q6 tensors; assert fact → linear → route spec → kernel format | `model_facts.py`, `model_route_plan.py`, `test_model_route_plan.py` |
| Route safety | direct-packed is default; MMQ modes are opt-in | MMQ candidate cannot masquerade as direct-packed | route census asserting route ID, candidate identity, rollback, and default under each env mode | `prefill_routes.py`, `test_llm_prefill_routes.py` |
| Q4 geometry | direct kernel shape and MMQ declaration exist | 128x128x256 cooperative geometry lowers and handles edges | generated 16x16x256 and 128x128x256 compile/correctness tests, plus M/N tails | `q4k_prefill_route_spec.py`, `q4k_q8_mmq_prefill_spec.py` |
| Q6 geometry | 256 K block alignment only | Q6 tile geometry is independent of Q4 and target capability | Q6 descriptor candidates for 16x16x256 and role shapes; reject Q4 decode/WMMA assumptions | `q6k_prefill_route_spec.py`, `q6k_mmq_vocabulary.py` |
| Q8 preparation | DS4 packers and cache exist | preparation, reuse, and sums policy are explicit in candidate identity | same-session packed-vs-unpacked comparison including pack/cache hit/miss costs | `route_ops.py`, `prefill_routes.py`, `test_llm_prefill_routes.py` |
| Staging | direct packed and compiler CSE | declared register/LDS/direct lifetime and uniform barriers | compile artifact with staging, barrier, bytes, and reuse counters; reject hidden choices | `mmq_logical_vocabulary.py`, emitter owner |
| Q6 staging | packed halfword loads; no MMQ tile | Q6 decode-to-tile path without full dequant fallback | bounded Q6 packed tile reference and generated compile gate; assert no full `[N,K]` materialization | `q6k_mmq_vocabulary.py`, `q6_k_gemv_primitive.py` |
| Ownership | direct output or partial-axis ownership | exact lane/fragment owner mapping | compare generated store map to `llama_mma_sum_slot_mapping`; assert no duplicates/missing stores | `mmq_llama_oracle.py`, `mmq_owner_coverage.py` |
| Hardware | target strings and capability fields exist | gfx1100 compile/resource/ISA/health/timing evidence | same-session AMD artifact for each role/quant; host-only results remain blocked | `mmq_compile_evidence.py`, `mmq_resource_snapshot.py` |

## Non-goals and gates

This audit does not authorize:

- changes to `tinygrad/llm/prefill_routes.py` or model installation;
- a full dequantized Q4/Q6 fallback;
- reuse of Q4 MMQ evidence as Q6 evidence;
- route promotion, manifest changes, or production ownership claims;
- treating the llama source/oracle as proof of tinygrad lowering.

The alignment work is complete only when each role/quant pair has independent numeric correctness, staging/resource, exact ownership, candidate identity, and same-session timing evidence. Until then, direct-packed remains the authoritative comparator and rollback path.
