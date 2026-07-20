# Exhaustive implementation plan: generated fp16-dequant-in-register Q4_K prefill kernel (AMD)

Date: 2026-07-20. Supersedes the summary scope `amd-fp16-dequant-q4k-primitive-scope-20260720.md`. Derives from handoff §1.15–1.16 (decision) and the certification method doc (C0–C8). Read-only research; no code written yet.

**Goal.** Generate a lean, AMD-supportable Q4_K prefill kernel implementing the decided primitive (handoff §1.16): dequant Q4_K → fp16 **in registers**, fp16 WMMA (`v_wmma_f32_16x16x16_f16`), stream K at BK=32 through ~16 KB LDS. The proven reference is the hand kernel `build_gemm_lds2_q4k`. This replaces the crashing 57 KB int8-MMQ generated kernel on AMD; the int8-MMQ family is retained for NVIDIA (routed by the §1.15 occupancy axis).

**Structural win found during scoping:** core tinygrad **already supports fp16 WMMA** — `tinygrad/codegen/opt/tc.py:140-147` `amd_rdna3` includes `(dtypes.half, dtypes.float)` next to `(dtypes.char, dtypes.int)`, and `tinygrad/codegen/opt/kernel_lds.py` `validate_rdna3_wmma_descriptor`/`validate_precontract_wmma_abi` already admit both. So the WMMA op itself needs no core change; all int8-specificity lives in `extra/qk`.

---

## PART I — The target algorithm (spec to generate)

From `extra/qk/prefill/wmma.py:501-654` (`build_gemm_lds2_q4k`), shared base `build_gemm_lds2`/`LDS2PrimitiveEmitter` `:201-499`. Shipped config `WAVES_M=WAVES_N=2, WM=WN=4`.

### I.1 Geometry (`wmma.py:509-517`)
```
BK=32 (one Q4_K sub-group)   KT=BK//16=2   PAD=0
THREADS=WAVES_M*WAVES_N*32=128   BM=BN=128   assert BN==THREADS  (one weight-row/thread)
SA=SB=BK*2+PAD=64 B/row
LDS: A tile [0,8192)  (64B/row × 128 rows) ; B tile [8192,16384) (64B/row × 128 rows)
BUFSZ = 16384 B, single-buffered (DBUF=0, no double buffer)
NSB = K//256 super-blocks ; BKPR = NSB*144 packed bytes/weight-row
```
VGPR map `wmma.py:516-522`: FA=10, FB=42, ACCb=74, CTA=202, HDR=218, QW=222, OUT=230; decode temps v10–v24 deliberately aliased into the low FA region (dead during decode; compute0 reloads FA/FB from LDS each group).

### I.2 Q4_K decode — the numerically load-bearing part (`wmma.py:554-592`)
Must match ggml `GGML_Q4_K` bit-for-bit (reference `tinygrad/llm/gguf.py:57-67`). Block = 144 B: `d`(fp16) + `dmin`(fp16) + `scales`(12 B, packed 6-bit) + `qs`(128 B nibbles).
- **`get_scale_min_k4(g)`** (`:555-567`): for g<4 `sc=scale[g]&63`, `mn=scale[4+g]&63`; for g≥4 (gg=g-4) `sc=(scale[gg]>>6)*16 + (scale[8+gg]&0xf)`, `mn=(scale[4+gg]>>6)*16 + (scale[8+gg]>>4)`. Yields integer sc,mn ∈ [0,63].
- **`expand_f16(H)`** (`:568-574`): fp16→fp32 via **integer bit-twiddle** (mantissa `(H&0x3ff)<<13`, exponent `((H>>10)&0x1f)+112)<<23`, sign `(H>>15)<<31`), **not** a native cvt — deliberately, because scalar fp16 arith is unreliable on this ISA (comment `:521,575`). Assumes normal fp16 (exp≠0).
- **`decode_group(g)`** (`:575-592`): `d=expand_f16(HDR.lo16)`, `dmin=expand_f16(HDR.hi16)` (recomputed per group), `Tdsc=d*sc`, `Tdmn=dmin*mn` (f32); per nibble `code=(QW[l//4]>>((l%4)*8+(g%2)*4))&0xf`, value = **`d*sc*code − dmin*mn`** in f32, then **exactly one** `v_cvt_f16_f32`. 32 fp16 values/group → `ds_store_b128` ×4 into the B LDS row. `s_delay_alu(1)` after every f32-producing op feeding a dependent VALU (gfx1100 RAW hazards are not interlocked on this raw-ISA path; `s_nop` does not satisfy the scoreboard).

### I.3 K-streaming loop (`wmma.py:615-631`)
Outer over NSB super-blocks; inner 8 sub-groups Python-unrolled: `global_load` HDR (16 B) once/block; per group `global_load` QW (32 B), `coop_load_A()` (plain fp16), `decode_group(g)` → B LDS, `coop_store_A()` → A LDS, `waitcnt_lgkm(0)+s_barrier()`, `compute0()`, `s_barrier()`; advance A ptr +BK*2=64 B/group, B ptr +144 B/block. A uses cooperative striping (RSTRIDE=32, loadsA=4); B is one-row-per-thread (BN==THREADS), no B striping.

### I.4 WMMA compute + sync (`compute0`, `wmma.py:600-614`)
Per kt∈{0,1}: `ds_load_b128` A/B fp16 fragments (8 VGPR each per mi/ni), `waitcnt_lgkm(0)`, then `v_wmma_f32_16x16x16_f16` per (mi,ni) into fp32 accumulators (WM*WN=16 tiles × 8 fp32 = 128 acc VGPR). Double `s_barrier` around single-buffered LDS reuse (before: writes visible; after: reads done before next overwrite).

### I.5 Epilogue (`wmma.py:632-650`)
Row-major `C[row][col]`, byte offset `(row*N+col)*2`; per accumulator element one `v_cvt_f16_f32` → `global_store_b16`. **No post-WMMA dm/ds correction** — all Q4_K scale/min folded into decode; the accumulator is a plain fp16×fp16→fp32 sum.

### I.6 Input contract
Kernarg = (A_ptr, W_ptr, C_ptr). **A** plain fp16 `[M,K]` (never quantized, no ids/q8). **W** packed Q4_K bytes `[N, NSB*144]` (never densified; decoded per 32-slice into LDS). **C** written as fp16 (`global_store_b16`) `[M,N]`. Grid `(N//128, M//128, 1)`, 128 threads. Constraints: `BN==THREADS`, `DBUF=0`, `K%256==0`.

### I.7 gfx1100 ISA specifics a generated version must reproduce (or re-verify)
1. `expand_f16` integer trick (or prove a native cvt safe on this target). 2. `s_delay_alu(1)` RAW hazards (or rely on renderer scheduling). 3. `v_wmma_f32_16x16x16_f16` operand-to-lane packing (B pre-transposed in LDS). 4. exact `s_waitcnt` vm/lgkm encodings (`:190-199`). 5. double-barrier cadence (consequence of DBUF=0). 6. single-f16-rounding rule (f32 accumulate, one final cast) — determines the numeric authority. 7. VGPR<238 on the raw-ISA/ELF path (likely not binding for a renderer-allocated kernel, but verify).

---

## PART II — Change-map of the current int8-MMQ stack

The current kernel is a **hand-authored Python UOp graph** → `AMDISARenderer` (not scheduler-lowered). KEEP = reuse as-is; MODIFY = targeted change; REPLACE = rewrite.

### II.1 KEEP (algorithm-agnostic scaffolding)
- `extra/qk/kernel_pipeline.py` — `DotUpdateRecurrencePlan`, `HierarchicalKernelPipelinePlan`, lifecycle/proof machinery (dtype/op-parametrized).
- `extra/qk/kernel_writeback.py` — `WMMAWritebackDescriptor`/`build_wmma_writeback` take `accumulator_dtype`/`tc` as params; RDNA3 output-coord math is dtype-independent.
- `tinygrad/codegen/opt/kernel_lds.py` validators + `tinygrad/codegen/opt/tc.py` `amd_rdna3` — already admit `(half,float)`. **No change.**
- `extra/qk/kernel_lds.py` generic parts (`PackedComponent*`/`PackedRecord*`, `validate_packed_component_templates`, `cooperative_lds_stores`, `wmma_output_owners`, `wmma_fragment_loads`) — reusable when called with the fp16 `element_bytes`/dtype.
- `RecordProducerInstanceWitness` ordering machinery (`mmq_llama_record_producers.py:22-68`).

### II.2 MODIFY
- **`extra/qk/kernel_lds.py`** `build_hierarchical_packed_record_stage` (~898-1055) + `prove_hierarchical_packed_record_stage` (~1056-1170) + `_hierarchical_record_roles` (~858-895): parametrize the hardcoded `dtypes.char` at `:886,911,1020,1022,1131` to `tc.dtype_in`.
- **`mmq_llama_candidate_plan.py`** `_rdna3_i8_tc()` (`:30-31`): select `dtype_in==half, dtype_out==float`. `_geometry()` (`:44-56`): the **authoritative LDS layout** — replace ids/q8/q4 regions with two fp16 regions (~16 KB). Rest of `LlamaMMQCandidatePlan` (schemas, transforms, `recurrence=DotUpdateRecurrencePlan(float.vec8,int.vec8,...)` `:153`, source anchors): re-declare per §II.5.
- **`mmq_llama_oracle_epoch.py`**: dtype/size asserts `:33,35` (q4 uint32 / q8 uint8) and the two-template call shape change; `_contracts()` (`:20-27`) stays.
- **`mmq_llama_five_buffer_full_kernel.py`**: KEEP grid/topology/`to_program(AMDISARenderer)` plumbing (`:243-284`); FIX the **duplicate** hardcoded `FullGridTopology.lds_bytes=57856` (`:35`, not derived from geometry); shrink `_full_grid_sink` q4/values/scales/sums wiring (`:192-241`) with the ABI.
- **`mmq_llama_five_buffer_graph.py`**: `five_buffer_parameters` (`:85-95`) is one of two ABI declaration sites — shrink to 2-3 buffers; `__post_init__` `range(5)` assert (`:74-79`) relaxes.
- **`mmq_llama_full_kernel.py`**: `ScannedTargetFacts.signed_i8_wmma` gate (`:171,185,274`) → f16 capability gate or drop.

### II.3 REPLACE (rewrite)
- **`mmq_llama_oracle_recurrence.py` (FULL FILE)** — the int8-dot + float-correction recurrence. `build_llama_oracle_recurrence` (`:176-229`): `zero` int32.vec8 → float.vec8; the two chained `Ops.WMMA` stay structurally but dtype int→float; **the whole `scale=dm*ds.x, bias=dm*ds.y, update=prev+scale*C+bias` block (`:207-219`) is deleted** (native fp32-accumulate needs no rescale); `update=previous+second.gep(i)`. Drop `_sidecars()` (`:101-137`, dm/ds correction loads), `_fragment_at` char hardcode (`:91-98`), `_renderer_signed_operand_contract()` (`:80-88`, the `v_wmma_i32...neg=3` AST check — meaningless for fp16), and the `dtype_in!=char` assert (`:183-184`). Rewrite `prove_llama_oracle_recurrence` (`:232-290`) for the new fp32-accumulate identity.
- **`mmq_llama_packed_operands.py`** — `Q4_K_GLOBAL_BLOCK`, `Q8_1_DS4_ROW`, `Q4_K_DECODED_LDS_ROW` (`:35-50`) + arena constants (`:53-121`, the **second** hardcoded copy of the 57856 layout, self-`validate()`d): describe int4/int8 wire format; replace with fp16 weight/activation layout.
- **`mmq_llama_record_producers.py`** — `Q4KOracleSchedule.__call__` (`:216-240`) + `q4_k_dm/qs_record_callback`/`_scale_or_min`/`_packed_byte` (`:138-187`): the nibble-unpack + 6-bit-scale decode → **must produce a dequantized `half` value (`d*sc*code−dmin*mn`) written to the fp16 LDS row**, not int32 nibbles. `_q8_split_*`/`_linear_q8_ds4_schedule` (`:89-135,190-209`) + `build_split_q8_ds4_record_template` (`:301-322`) deleted (activation becomes plain fp16).
- **`mmq_llama_differential.py`, `mmq_llama_oracle.py`** — structurally diff the plan vs vendored `mmq.cuh` (int8 vocabulary: `"v_wmma_i32_16x16x16_iu8"`, `"q8_ds_semantics"`). A from-scratch fp16 kernel has no `mmq.cuh` to diff — drop this apparatus.

### II.4 The three (really five) int8-WMMA dtype sites to switch
`mmq_llama_candidate_plan.py:30-31` (`_rdna3_i8_tc`), `mmq_llama_oracle_recurrence.py:183-184` (assert), `:80-88` (`_renderer_signed_operand_contract`), plus `mmq_llama_full_kernel.py:171/274` (`signed_i8_wmma` gate) and `kernel_lds.py:886/911/1020/1022/1131` (char hardcodes in the "generic" hierarchical builder). Core tinygrad needs none.

### II.5 The 57856-LDS number lives in ≥7 places (change all)
Authoritative: `mmq_llama_candidate_plan.py:_geometry()` (`:44-56`). Independent copies/readers: `mmq_llama_packed_operands.py:66-71,113-121` (self-validated 2nd copy); `mmq_llama_five_buffer_full_kernel.py:35` (duplicate default); `mmq_llama_differential.py:57,60-61,94-95` (drops out); `mmq_machine_search.py:1095,1374,1384,1397` (admission — silent mis-admit if stale); plus derived readers `mmq_llama_full_kernel.py:274`, `mmq_llama_oracle_epoch.py:51,100`, `kernel_lds.py:907`, `kernel_writeback.py:129` (verify).

### II.6 ABI shrink (5 → 2-3 buffers) — declaration sites
Two in-stack: `mmq_llama_five_buffer_graph.py:85-95` (`five_buffer_parameters`) + `mmq_llama_five_buffer_full_kernel.py:67,192-241,287-302`. Producer adapter to delete: `mmq_llama_record_producers.py:301-322` (3-q8-buffers → 1 interleaved row). Downstream canonical ABI constants (independently re-hardcoded, all must be re-declared): `extra/qk/prefill/frozen_exact_role_runtime.py:37-39` (`Q4_WORDS_PER_EPOCH_ROW=36`, `ABI_NAMES`, `ABI_DTYPES`), re-asserted in `mmq_frozen_epoch_program_set.py:56`, `mmq_frozen_epoch_memory_certificate.py:27`, `mmq_frozen_staged_low_level_session.py:78,96`, `mmq_staged_c7_c8_contract.py`. Shrink: keep `output`(0) + `q4`(1, decoded in-register now); collapse `q8_values/scales/sums`(2-4) → one `dtypes.half` activation buffer.

### II.7 Identity/schema/source-pin constants a new family must re-declare (not mutate)
Source-pins `LLAMA_SOURCE_COMMIT`/`LLAMA_MMQ_CUH` (`mmq_llama_oracle_recurrence.py:19-20` + duplicated across stack; every `__post_init__` raises on drift) — a from-scratch fp16 kernel has no mmq.cuh source; define its own identity or drop the pin. Schema strings (`PLAN_SCHEMA`, `DESCRIPTOR_ID`, per-module `SCHEMA`, transform ids) and `SOURCE_ANCHORS` tuples → new family strings so identity hashes don't collide/false-diff. `RESOURCE_BLOCKER` string (`five_buffer_full_kernel.py:26`) → adjust to the new register-pressure profile.

---

## PART III — New frozen family + C0–C8 certification (register + certify checklist)

Ordered; each step marks **[reuse]** scaffolding vs **[author]** new. (Detail from the gate-machinery map.)

1. **[author]** New ABI constant tuple (names/dtypes/element-count for 2-3 buffers), sibling to `frozen_exact_role_runtime.py:37-39`.
2. **[author]** New `ExactRoleSpec.abi_elements` formula (`mmq_exact_role_spec.py:36-37` is the int8 5-buffer formula) + register role/candidate identity in the inventory JSON (`bench/prefill-pure-full-kernel/qwen3-14b-mixed-quant-candidate-inventory-v1.json`, admission `q4k_q8_five_buffer_role_gate.py`). [reuse] the `ExactRoleSpec` shape/grid math.
3. **[author]** Sibling frozen-staged-family module (`_abi_contract`/`_staging_contract` for 2-3 slots), mirroring `mmq_frozen_staged_family.py` structure. [reuse] the identity/hash/atomic-publish pattern (`produce/load_frozen_staged_family_manifest`, `:366-427`) + provenance recording.
4. **[author]** Sibling memory certificate (`_abi_certificate` with `range(N)`, N∈{2,3}), mirroring `mmq_frozen_staged_memory_certificate.py` (`:92-109,187-237` hardcode 5). [reuse] `certify_native_program_memory`/`certify_source_sink_memory` primitives.
5. **[reuse] C2 resource gate** — call `mmq_resource_checks.py:check_mmq_resource_evidence` directly with `max_lds_bytes≈16384` (already parametrized; also carries the dormant `min_occupancy` field the §1.15 axis can populate).
6. **[author]** New GPU execution harness for the 2-3-buffer f16 program, analogous to `mmq_llama_five_buffer_gpu_harness.py` (which bakes in the int8 DS4 NumPy oracle). [reuse] `mmq_target_epoch_orchestrator.py` (health/fault-window/isolation).
7. **[author]** C5 phase-isolated prefix runners analogous to `mmq_ffn_gate_up_pm4_{no_doorbell,prefix1,reduced_grid}_runner.py`, supplying the **new reference function** (Part IV). [reuse] the attestation/comparison envelope in `mmq_ffn_gate_up_guarded_correctness.py` (PM4 pre-submit snapshot, `_validate_numeric_comparison`).
8. **[reuse/author] C6** — `_validate_full_comparison` pattern reusable; redeclare `OUTPUT_SHAPE`/`OUTPUT_ELEMENTS`/tolerance constants for the role.
9. **[reuse] C7** — `mmq_staged_c7_authority.py` + `mmq_frozen_staged_c7_census.py` work against any `FrozenStagedFamily`-shaped object.
10. **[author] C8** — new runtime module analogous to `mmq_ffn_gate_up_c8_runtime.py`/`_c8_paired_sessions.py`/`_c8_session_worker.py`.
11. **[author]** New `route_manifest.py` ROUTES row mirroring `prefill_q4k_int8_wmma_generated_research` (`:248-265`): keys `workload/profile_id/status="research"/roles/quant/shape_guards/env/rollback/baseline_route_id/strict_fallback/expected_kernels/forbidden_kernels/authority_gate/promotion_artifacts/purity_status/provenance="machine_authored_generated"/selector/route_attribution/note`. Wire a real `env` guard in `tinygrad/llm/prefill_routes.py`. `baseline_route_id = prefill_q4k_direct_tile4x4_default`, `strict_fallback=True` until C8 wins.

---

## PART IV — New correctness authority (§2 constraint 5)

The fp16-dequant kernel computes differently (dequant-then-accumulate) than int8-MMQ (quantize→int-dot→correct), so it **cannot** reuse the int8 authority `mmq_q4k_q8_reference.py:467-496` (int8 activation quant + int32 dot + rescale — wrong rounding path) or the five-buffer harness's baked DS4 NumPy oracle.

- **Base reference (correct formula):** `tinygrad/llm/gguf.py:76-84` (`ggml_data_to_tensor`, ggml_type 12) computes `(d*sc*q − dmin*mn)` in all-fp32 tinygrad ops — exactly the target decode. `extra/qk/layout.py:157` `q4_k_reference` delegates to it.
- **Closest existing analogue to build on:** `mmq_ffn_gate_up_guarded_correctness.py:357-375` `ffn_gate_up_direct_dense_reference` — "independent dense fp16-activation/Q4_K dequant oracle", dequant-then-accumulate, no int8. Caveats to close: it accumulates in NumPy fp32 (`@`) not literal fp16×fp16→fp32 WMMA order, and trusts `resident_fp16_activation` is already fp16-rounded — so it's a starting point, not byte-exact; the new authority must record the residual rounding-order RMSE against it.
- **Exact swap point:** author `ffn_gate_up_fp16_dequant_reference(fixture) -> np.ndarray fp32 [M,N]` with the same signature as `ffn_gate_up_consumer_prefix_reference` (`:302-354`) and pass it into the existing `_validate_numeric_comparison`/`_validate_full_comparison` (`:223-299`) call sites in `run_candidate_prefix_child`/`run_ffn_reduced_grid_child` (`:1413,1645`). Comparison/tolerance/attestation infra is reused unchanged; **only the reference function is new.**
- **C0A sign-off (method doc §C0A):** retain separately (1) producer-vs-spec — prove the kernel's Q4_K dequant matches the ggml `d*sc*code−dmin*mn` formula (or record exact drift), and (2) target-vs-retained-producer — GPU output vs same-session producer bytes; (3) the named authority + tolerance recorded explicitly, as a content-addressed durable artifact (cf. the attn_kv closeout pattern), not merely a passing run.
- **Acceptance:** `rtol=atol=3e-3`, zero mismatches across full output-element count, all finite; producer-vs-spec RMSE/mismatch recorded even if nonzero (drift declared, not hidden) — matching `_validate_full_comparison` (`:280-299`).

---

## PART V — Recommended implementation order (correctness-first, checkpointed)

1. **Bring-up the f16 recurrence in isolation.** Write the new (small) recurrence module modeled on hand `compute0`/`decode_group` (§I.2/I.4) — do NOT generalize `mmq_llama_oracle_recurrence.py`. Verify it emits `v_wmma_f32_16x16x16_f16` + a clean spill-free program via `AMDISARenderer` on a toy shape. Checkpoint: ISA dump shows f16 WMMA, LDS≈16 KB, 0 spills.
2. **Author the ABI + geometry** (Part II.2/II.5/II.6): two fp16 LDS regions, 2-3 buffers, new constants. Checkpoint: `check_mmq_resource_evidence(max_lds_bytes=16384)` passes; C3 memory certificate (sibling) clean.
3. **Wire the decode-to-fp16-in-register producer** (Part II.3, `record_producers`) — nibble-unpack + 6-bit-scale → `half` into LDS. Checkpoint: CPU C1 deterministic generation + frozen family manifest.
4. **Stand up the correctness authority** (Part IV) and the CPU parity check vs `ffn_gate_up_direct_dense_reference`/ggml. Checkpoint: C0A producer-vs-spec recorded (RMSE bound).
5. **New frozen family + C4 canary** (Part III steps 1-6). Checkpoint: no-target runtime preconstruction clean.
6. **C5 guarded dispatch** — reduced-grid ladder first (reuse the runner pattern), then prefix-1/3. **Critically: run the full 544-wg grid** — this is the whole point; expect it to PASS (16 KB LDS → healthy occupancy) where the int8 kernel wedged at 64. Checkpoint: zero-mismatch at full grid, clean health.
7. **C6 full-role correctness → C7 memory admission → C8 timing** vs the direct-packed baseline (PM4 + AQL separately). Checkpoint: `CERTIFIED_WIN`/`CERTIFIED_FALLBACK`.
8. **Route manifest research row** (Part III step 11), env-gated, strict fallback, until promotion.

---

## PART VI — Risks

- **Rounding authority (§2.5).** dequant-then-MAC ≠ quantize-then-dot-then-correct. Must stand up a *new* justified authority (ggml/hand-kernel), record the drift, get sign-off — not reuse/bypass the int8 one. Load-bearing.
- **fp16 dequant numeric match.** Replicate the hand decode order exactly (`d*sc*code−dmin*mn`, **f32 intermediate, single final f16 cast**). Scalar fp16 arith is unreliable on this ISA (`expand_f16` integer trick) — a generated version must reproduce it or re-verify a native cvt against the remu/hardware oracle.
- **fp16 WMMA accumulate** is fp32-native, so precision risk sits in decode not accumulate — but it's untested in the generated stack; needs its own C5/C6 gate.
- **Renderer hazards.** `s_delay_alu` RAW hazards and the double-barrier cadence are hand-inserted in the reference; a UOp/renderer path must either get these from the renderer's scheduler or insert equivalents. Verify no VGPR>238 / occupancy regression in the generated ISA.
- **Scope realism.** Bounded (~400–600 authored lines + sibling family/gate modules) *because* the scaffolding (pipeline, writeback, LDS validators, C2/C7 gates, orchestrator, comparison envelope) is reused and core tinygrad already has f16 WMMA. Becomes a research project only if the alternative scheduler-Tensor-graph route is attempted (it would dense-materialize fp16 weights, violating §2.4 — rejected).

---

## PART VII — First files to open (execution start)
Spec: `wmma.py:501-654`. Recurrence rewrite: `mmq_llama_oracle_recurrence.py` (full). Geometry/ABI/dtype: `mmq_llama_candidate_plan.py:30-31,44-56`, `mmq_llama_five_buffer_graph.py:85-95`, `mmq_llama_five_buffer_full_kernel.py:35,192-241`, `frozen_exact_role_runtime.py:37-39`. Decode: `mmq_llama_record_producers.py:138-240`. Authority: `mmq_ffn_gate_up_guarded_correctness.py:302-375,223-299`, `gguf.py:76-84`. Family/gates: `mmq_exact_role_spec.py:18-62`, `mmq_frozen_staged_family.py:366-427`, `route_manifest.py:248-265`. Constraints: handoff §1.15-1.16, §2, §5-6; method doc C0-C8, §C0A, §11.
