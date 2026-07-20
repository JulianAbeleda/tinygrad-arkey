# Scope: generated fp16-dequant-in-register Q4_K prefill primitive (AMD)

> **Superseded by the exhaustive plan: [`amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md`](amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md)** — the bit-exact target algorithm, full keep/modify/replace change-map, C0–C8 register-and-certify checklist, and correctness-authority swap. This doc is the summary.

Date: 2026-07-20. Derives from handoff §1.15–1.16 (the AMD-primitive decision) and §2/§5–6 (constraints).

## Objective

Generate a lean, AMD-supportable Q4_K prefill kernel to replace the crashing int8-MMQ generated kernel, implementing the **decided** AMD primitive (handoff §1.16): dequantize Q4_K weights to fp16 **in registers**, accumulate with **fp16 WMMA** (`v_wmma_f32_16x16x16_f16`), stream K at BK=32 through a **~16 KB LDS** tile. The proven reference is the hand kernel `build_gemm_lds2_q4k`; this scope is about *generating* the same algorithm so it satisfies the generated-route claim (a hand kernel cannot per §2). The crashing int8-MMQ family is retained for NVIDIA (routed there once the §1.15 occupancy axis exists).

## Target spec (the hand kernel to hit)

`build_gemm_lds2_q4k`, `/home/ubuntu/tinygrad-arkey-hand-asm-bisect/extra/qk/prefill/wmma.py:501-654`:
- `BK=32`, `KT=BK//16=2`, `SA=SB=64 B/row`, two fp16 LDS regions only (no `ids`, no int8 `q8`), `BUFSZ≈16,384 B`.
- Per Q4_K superblock (256 elems), 8 sub-groups Python-unrolled: `decode_group` (:575-592) is the core — expand fp16 d/dmin via integer bit-twiddle (`expand_f16`, scalar fp16 arith is unreliable on this ISA), compute `d*sc` / `dmin*mn` in **f32**, per nibble `d*sc*code − dmin*mn` → `v_cvt_f16_f32` → 32 fp16 regs → `ds_store_b128` into the B LDS region. A (activations) is **plain fp16**, cooperatively loaded, never quantized.
- `compute0` (:600-614): `ds_load` fp16 A/B fragments, `waitcnt_lgkm(0)`, `v_wmma_f32_16x16x16_f16` per subtile into **fp32 accumulators**; double `s_barrier` around LDS reuse. Epilogue identical to the plain fp16 GEMM — **corrections folded into the decode, no post-WMMA correction**.

Key: weight stays Q4_K-compressed in VRAM; decode happens per 32-wide K-slice immediately before staging — **never a dense fp16 weight buffer** (so §2.4 no-dense-dequant is satisfied).

## Delta from the current generated int8-MMQ kernel

The current stack is a **hand-authored Python UOp graph** (`mmq_llama_oracle_epoch.py` builds `UOp.range`/`placeholder`; `mmq_llama_oracle_recurrence.py:174-224` builds `Ops.WMMA` nodes; `mmq_llama_five_buffer_full_kernel.py:192-277` sinks to `AMDISARenderer`) — **not** scheduler-lowered. Changes:

| aspect | current int8-MMQ | target fp16-dequant |
|---|---|---|
| LDS | 57,856 B (`ids`+`q8`+full-K `q4`) | ~16,384 B, two fp16 regions |
| activations | external int8 + scales + sums (5-buffer ABI) | plain fp16, no quant, no ids/q8 |
| weight residency | full-K decoded panel, corrections post-WMMA | streamed BK=32, decoded to fp16 pre-WMMA, corrections folded in-register |
| WMMA | `v_wmma_i32_16x16x16_iu8` (hardwired `_rdna3_i8_tc`) | `v_wmma_f32_16x16x16_f16` |
| recurrence | int32 dot + `(prev + scale*C) + bias` correction | straight fp32-accumulate, no correction chain |
| ABI | 5 buffers | 2–3 buffers (out, q4_packed, activation_fp16) |

## Approaches considered

**(i) Author in the existing Python UOp-graph MMQ stack — RECOMMENDED.** Reuse the AMD-hardened scaffolding (LDS staging `kernel_lds.py`, `kernel_writeback.py`, PM4/AQL dispatch, resource/ISA proof gates, frozen-family machinery) and replace the algorithm core. Files: `mmq_llama_candidate_plan.py:44-56` (`_geometry` → two fp16 regions; `:30-31` `_rdna3_i8_tc` → f16 TC lookup), `mmq_llama_record_producers.py:138-240` (Q4 decode → f32-then-f16 mirroring `decode_group`), **`mmq_llama_oracle_recurrence.py` (full rewrite** — drop int32+correction, straight fp16-WMMA/fp32-accumulate; replace the dtype/operand contracts at `:180-181`, `:80-88`), `mmq_llama_five_buffer_*` (ABI 5→2-3 buffers, `FullGridTopology.lds_bytes` 57856→~16384, BK-streamed addressing), `mmq_llama_packed_operands.py` (delete the Q8/Q4-decoded LDS-row transforms). Estimate **~400–600 authored lines + a new frozen-family/gate registration**. Bounded and precedented — this stack already emits clean, spill-free AMD ISA.

**(ii) Route via the tinygrad scheduler (Tensor-graph, `route_pf16_graph_gemm`-style) — NOT RECOMMENDED.** Much less code, but the route `_install_candidate_matmul` (`prefill_graph_gemm_route.py:108`) forces `.contiguous()` on the fp16 weight operand, and `ggml_data_to_tensor`'s Q4_K path (`gguf.py:62`) starts with a `.contiguous()` realize boundary — so chaining a dequant producer would very likely **materialize the full dense fp16 weight** before the matmul, exactly the §2.4 violation that disqualifies the fp16 overlay for 14B. No existing streamed dequant-into-WMMA precedent in the repo. This is a research spike (needs `.schedule()`/kernel-dump proof it can stay streamed), not an implementation task.

## Recommended path (correctness-first)

1. Write a **new, small recurrence module** modeled directly on the hand `compute0`/`decode_group` (~90 lines of logic) — do NOT generalize `mmq_llama_oracle_recurrence.py` (its int32+correction shape doesn't apply).
2. Keep the algorithm-agnostic scaffolding (staging, writeback, dispatch, gates).
3. Register a **new frozen family** with the new 2–3-buffer ABI (parallel to, not a variant of, the int8-MMQ `ffn_gate_up` family).
4. **New correctness authority (load-bearing, see risk):** the fp16-dequant kernel is a structurally different computation from int8-MMQ, so it cannot reuse the pinned int8 rounding authority (`mmq_q4k_q8_reference.py`). Its C5/C6 numeric gate must check against a *new* authority — the hand kernel's already-proven-0-RMSE output or direct fp32 GGML Q4_K dequant math — explicitly stood up and signed off under §2.5.
5. Add as a **research route row** in `route_manifest.py` (mirror `prefill_q4k_int8_wmma_generated_research`, `:248-264`), env-gated, `baseline_route_id = prefill_q4k_direct_tile4x4_default`, `strict_fallback=True`, until it clears C1–C8 and matches/beats the baseline on the §3 whole-prefill promotion gates.
6. Only then does the §1.15 occupancy-admission axis matter for automatic AMD-vs-NVIDIA device selection (tracked separately).

## Risks

- **Rounding-authority divergence (§2.5).** dequant-then-MAC ≠ quantize-then-dot-then-correct. Must stand up a new, independently-justified rounding reference and get sign-off — not silently bypass the existing one.
- **fp16 dequant numeric match.** Replicate the hand kernel's exact decode order (`d*sc*code − dmin*mn`, **f32 intermediate**, single final f16 cast) — scalar fp16 arith is unreliable on this ISA; drift breaks parity with both llama's C reference and the hand kernel.
- **fp16 WMMA accumulate.** Accumulates in fp32 already, so precision risk is in the decode, not the accumulate — but it's untested in the generated stack and needs its own C5/C6 gate.
- **Scope realism.** A real, bounded implementation task under (i) given scaffolding reuse (~400–600 lines + gate wiring). Becomes a research project only if (ii) is attempted.

## First files to open

`wmma.py:501-654` (spec); `mmq_llama_candidate_plan.py:44-56,30-31`; `mmq_llama_oracle_recurrence.py` (full, esp. :174-224,:180-181,:80-88); `mmq_llama_record_producers.py:138-240`; `mmq_llama_five_buffer_full_kernel.py:35,192-277`; `route_manifest.py:248-275`; handoff §1.15-1.16, §2, §5-6.
