# Machine-Code Translation Roadmap — Result (2026-06-23)

## 1. Verdict: `MACHINE_CODE_TRANSLATION_ROADMAP_READY` + `BUFFER_IDENTITY_ABI_RULE_RECORDED` + `MACHINE_SEARCH_STILL_NOT_READY_FOR_8B_SPEED` / `MACHINE_SEARCH_READY_FOR_CODEGEN_MICROPRIMITIVES_ONLY`
This is the exhaustive map of project learnings → machine-code-aware artifacts. **No kernels implemented, no search
started.** Source unchanged except docs/tooling metadata. Artifacts under `bench/qk-machine-code-translation/`.

## 2. Primitive inventory (`primitive_inventory.json`)
| primitive | impl | status | translate → target |
|---|---|---|---|
| Q4K GEMV warp | tinygrad-native UOp schedule | W==D pass, env-gated | schedule template + ISA guard |
| owned attention tile | hand HIP code object | W==D pass, **default-on** | native codegen / owned template |
| whole-cache KV read | owned tile ABI + cache layout | W==D pass, **default-on** | **buffer-identity ABI rule** |
| KV append raw ISA | probe only | NOT active (lane retired) | diagnostic template |
| ISA audit wrapper | tool | ready | **mandatory guard** |
| prefill LDS GEMM (hand-asm) | hand AMDGCN | Tensile-class ~61 TFLOPS (prefill) | ISA template + native-codegen target |
| small-op fusion | unproven | optional | fusion/search template (gated) |

## 3. Artifact types (`artifact_types.json`)
Eight representation forms: hand-owned HIP kernel · hand-owned AMDGCN ISA · tinygrad-native schedule · codegen
capability · ISA template · ABI/layout rule · machine-search knob · regression guard. Each primitive is classified
into one or more (e.g. owned tile = HIP kernel + ISA template + ABI rule + regression guard).

## 4. Owned attention machine-code facts (`owned_attention_machine_code_facts.json`) — `OWNED_ATTENTION_MACHINE_FACTS_READY`
- **Kernels:** `owned_flash_tile_gqa_whole` (tile) + `owned_flash_combine` (combine; `hd`/`sr` variants exist).
- **Tile ABI:** `[Q, CACHE, part(f32), meta(f32)]` + scalar `start_pos` (S/scale/`n_valid=start_pos+1` baked); grid
  `(Hkv,S,1)` block `(128,1,1)`; ins `(0,1)` outs `(2,3)`.
- **dtype:** fp16 cache/Q/K/V, fp32 accumulation (online-softmax m,l + fp32 partials).
- **Layout:** whole `cache_kv` `[2,1,Hkv,MAXC,Hd]` flat; `kbase = CACHE + kvh·MAXC·Hd`, `vbase = CACHE + (Hkv+kvh)·MAXC·Hd` (`Hkv=8` baked).
- **Decomposition:** split-KV over S workgroups (S=48); Hkv x-workgroups; 4 warps (wave32); warp→q-head (G=4); TK=16
  LDS-staged positions; `__shfl_xor` 32-lane reduce.
- **ISA:** v_dot2 (`fdot2`) + LDS (8192 B) + cross-lane + fp16 vector loads; **60 VGPR** (range 56–64), 0 scratch,
  **0 spill** (invariant).

## 5. Q4K GEMV warp machine-code facts (`q4k_gemv_warp_machine_code_facts.json`) — `Q4K_GEMV_MACHINE_FACTS_READY`
tinygrad-native UOp schedule (not a hand kernel); per-output-row reduction split across a warp at 256-element Q4_K
superblock granularity; reduction lowers to a **native LDS tree-reduce** (NOT v_dot2, NOT ds_bpermute — confirmed by
the native-codegen experiment). At/below llama MMVQ (weight-GEMV parity). Differs from llama's integer-MMVQ path
(tinygrad has no integer dot primitive). Must not regress: lossless byte-identical weight-GEMV parity.

## 6. Buffer-identity ABI rule (`buffer_identity_abi_rule.json`) — `BUFFER_IDENTITY_ABI_RULE_RECORDED`
> **Precompiled graph-node kernels must receive buffer-identity inputs unless the kernel itself supports
> base-buffer + offset ABI. Do not pass sliced/cache views across the precompiled-call boundary when whole-buffer
> offset math can be done in the kernel.**

Mechanism: `callify.transform_precompiled_call` force-`.contiguous()`s every input except `AFTER`/`BIND`;
`_precompiled_output_redirect` reads a `BUFFER`/`MULTI` with `has_buffer_identity()` directly but **materializes** a
`SLICE`/`RESHAPE`. A `RESHAPE` *on top of* the `AFTER` still materializes (the redirect only accepts `BUFFER`/`MULTI`).
Bad: `cache_kv[0,layer]`, `cache_kv.reshape(flat).after(store)`. Good: whole `cache_kv.after(store)` (no reshape) +
in-kernel K/V offsets. Recorded in `structure/Development/performance-primitive-research-principles.md`.

## 7. Native-codegen translation targets (`native_codegen_targets.json`) — `NATIVE_CODEGEN_TRANSLATION_TARGETS_READY`
High priority: **v_dot2 lowering**, **cross-lane reduce (ds_bpermute)**, **LDS tile template** (the two codegen gaps
+ the one already-native LDS), and the **buffer-offset ABI** (already solved) + **Q4K warp schedule template**
(active). ISA evidence: tinygrad emits LDS natively but **not** v_dot2 or ds_bpermute — those are the renderer gaps.
None need W==D now (learning targets), except the two already-solved/active ones.

## 8. Machine-search readiness (`search_readiness.json`)
**`MACHINE_SEARCH_STILL_NOT_READY_FOR_8B_SPEED`** — attention & GEMV are ≥ llama (search risks non-transfer);
whole-cache ABI is solved; small-ops fusion is unproven. **`MACHINE_SEARCH_READY_FOR_CODEGEN_MICROPRIMITIVES_ONLY`**
— bounded ISA-audited microbenches (dot / cross-lane / LDS pattern variants) are searchable for *learning*, not W==D.
**Prefill LDS GEMM** is the one compute-bound lane with real searchable headroom (knobs: BK, PAD, PLR/PGR prefetch
depth, wg-occupancy, WGM) — a separate prefill task, not decode.

## 9. What remains hand-owned vs what can become tinygrad-native
- **Stays hand-owned (for now):** owned attention tile (v_dot2 + cross-lane + LDS + split-KV — the renderer can't
  emit v_dot2/cross-lane yet); prefill LDS GEMM (hand-asm, deeper-K pipeline is asm-only).
- **Already tinygrad-native:** Q4K GEMV warp schedule; LDS reductions.
- **Can become native (codegen targets):** v_dot2 lowering, cross-lane reduce, LDS tile template — each gated behind
  a bounded ISA-audited micro-proof, no W==D requirement.
- **Pure principle (no code):** the buffer-identity ABI rule.

## 10. Files changed
New: this doc + `docs/decode-campaign-final-synthesis-20260623.md`; 7 artifacts under
`bench/qk-machine-code-translation/` + 3 under `bench/qk-post-parity-hardening/`; principles + README + handoff +
`candidates.json` superseding note. **No `tinygrad/` source or default changes** (docs/tooling/metadata only).

## 11. Git status
Clean before this task (HEAD `032506bbf`). Adds two result docs + 10 artifacts + doc/principle/registry updates. No
kernels, no machine search, no attention/GEMV reopen, no 14B/32B, no default flip, no source change.
