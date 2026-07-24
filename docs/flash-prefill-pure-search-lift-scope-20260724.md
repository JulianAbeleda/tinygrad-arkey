# FlashPrefillAttentionSpec — Pure-Machine-Search Prefill Lift Scope (2026-07-24)

## Goal
Make prefill fused attention a genuine `machine_authored_generated` route — a `FlashPrefillAttentionSpec`
descriptor over a shape-generic emitter that composes the (now-shared, post-wmma-refactor) primitives as
DATA — matching how decode became pure via `FlashDecodeAttentionSpec`. NOT the naming trap (wrapping the
fixed hand kernel in a dataclass). The hand-kernel custom_kernel route stays only as the correctness oracle.

## Purpose
The custom-kernel route is `hand_authored_uop_template` and must never ship (violates the pure_machine_search
principle: audit `extra/audit/pure_machine_search_default_path_census.py`, guard `extra/qk/pure_search_guard.py`).
The only principled default is a descriptor-generated route. This scope sizes and sequences that.

## Verdict from the fixed-vs-parametrizable analysis (grounded)
Line-by-line analysis of `amd_gfx1100_q16_grid_hd128_loop_attention` (tinygrad/schedule/wmma/kernels.py:232-283):

- **CLOSE — already data-driven, no rewrite:** q_tokens (grid-launch via `grid.q_tiles=q_tokens//16`),
  kv_tokens (a REAL IR `UOp.range((kv_tokens+15)//16, ...)`, kernels.py:244), Hq, Hkv, causal, valid_kv,
  query_start, acc_blocks, output_block_base, scale, phase_abi_v1. All real kwargs / validated production data
  (fused_attention.py:117-121, postrange.py:352-357).
- **GENUINELY FIXED — validated hardware constants (keep as constants, like AMDAttentionGridSpec.wave_size):**
  WMMA 16x16x16 fp16->fp32 wave=32 (`warg`, kernels.py:243), lane/col wave32 math, the C-fragment axis-rewrite
  convention, `float.vec(8)`/`half.vec(16)` fragment widths (also validated in AMDRowSoftmaxRepackSpec.qk_c_lanes=8,
  pv_a_lanes=16, ops.py:1616-1617), fp16 PARAM ABI.
- **THE ONE WELDED DOF — head_dim=128**, hardcoded in THREE independent places:
  1. kernels.py:263 `for b in range(8)` (= head_dim//16) — the QK Hd-block accumulation, a Python trace-time-
     unrolled loop, plus derived literals at kernels.py:236 (`128` inline in sizes), 241 (`8-acc_blocks` bound),
     244 (`acc_blocks*8` creg), 282 (`acc_blocks==8`).
  2. SUBSTRATE (uop/ops.py): `AMDAttentionGridSpec.validate()` `head_dim!=128 -> raise` (1725);
     `AMDLoopStateSpec.validate()` `block < 8` (1814); `AMDPackedFragmentLoopSpec.validate()` `head_block < 8` (1833).
  3. `AMDAttentionOutputDrainSpec` `address_expr="e*256+halfwave*128+j*16+col"` — a literal formula baked for
     Hd=128 (ops.py:1773).
- **MAGIC-BUT-DERIVABLE (mechanical):** register/owner ids 94xx/95xx/96xx/97xx/98xx (deterministic per-kernel
  100-blocks, no hardware meaning), phase-LDS size 512.

## THE BAR (what "generated, not naming-trap" requires) — from decode
`flash_kernels.py` (flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel) is judged generated because the
emitter COMPUTES every Hd/Hq-adjacent extent from the spec fields INSIDE the builder (`G=Hq//Hkv`, `R=Hd//LANES`,
`RP=Hd//64`, `NB=ceildiv(L,TK)`) and every geometry loop is a real `UOp.range` over a derived value — never a
bare literal. `FlashPrefillAttentionSpec` meets the bar only if its emitter likewise derives all extents from
fields (e.g. `head_dim//16`) rather than hardcoding `8`/`128`/`256`.

## SCOPE DECISION (the fork)
- **Track A (NEAR, recommended):** pin `head_dim=128` as a validated constant IN the spec (identical posture to
  `AMDAttentionGridSpec.validate()` today), BUT de-literalize the emitter so every Hd-derived value is written as
  `head_dim//16` / `head_dim` / `2*head_dim` from a spec field. The FORM is generic + auditable; the single legal
  value is a documented validated-scope constant. Every OTHER DOF is already real. This is bounded plumbing.
- **Track B (FAR):** make `head_dim` a genuine DOF — requires editing the compiler-owned AMD*Spec.validate()
  methods + the AMDAttentionOutputDrainSpec address formula in uop/ops.py (larger blast radius, the substrate the
  wmma-modularization scope deliberately did NOT touch). Only pursue if a reviewer rejects Track A's pinned-Hd as
  a naming trap. If Track A is rejected AND Hd-DOF is too costly, that is the signal to pursue the composite-path
  (Option B / tinygrad_scheduler_generated) instead — an open-ended compiler project.

This scope specifies **Track A**. Track B / composite are noted as the fallback fork, not planned here.

## PHASES (Track A; dependency-ordered; gate each)

### P1 — De-literalize head_dim in the emitter (kernels.py) [byte-identical]
Replace every bare Hd-derived literal in amd_gfx1100_q16_grid_hd128_loop_attention (and the qk_stats/pv_slice
stages) with a value computed from `head_dim`: `HD_BLOCKS = head_dim//16` for the `range(8)` loops (263, and the
`8` in 236/241/244/282), `head_dim` for the inline `128` (236), `2*head_dim` for the drain `256` if reached here.
head_dim comes from the existing `grid.head_dim`/`AMDAttentionGridSpec`. NO behavior change (8 == 128//16), so
captures stay byte-identical. This is what makes the kernel honestly parametrized-in-form.
GATE: isolated captures 254/0 byte-identical (all 4 routes); unit tests 6/93; a4/varkv numerics 6.1e-5.

### P2 — FlashPrefillAttentionSpec (new: extra/qk/flash_prefill_attention_spec.py)
A frozen dataclass mirroring FlashDecodeTileSpec (flash_decode_attention_spec.py:44-85):
- Fields: `Hq, Hkv, Hd(=128 validated), q_tokens, kv_tokens, causal, valid_kv, query_start, acc_blocks,
  output_block_base, state_strategy(register|lds, = phase_abi_v1), scale, target="amd_gfx1100"`.
- `validate()`: Hd==128; q_tokens%16==0; kv_tokens%16==0 and <=4096; Hq%Hkv==0; acc_blocks in {1,2,4,8};
  output_block_base aligned; kv_tokens==start_pos+q_tokens invariant (the fail-safe from fused_attention.py:105).
- `emit()` / `emit_grid()` / `emit_qk_stats()` / `emit_pv_slice()`: thin methods that call the (P1-genericized)
  kernels.py builders with fields as kwargs — wrapping the current grid-full-loop and the qk_stats/pv_slice SPLIT
  as spec-composable data (a `mode` field or two emit methods) instead of 3 separately-named Python functions
  route dispatch imports by name today.
- `emitted_kernel_names` property (like decode's) for the manifest expected_kernels.
GATE: import; construct for 8B/14B geometries; `.emit()` produces the same UOp as the direct builder call
(structural UOp-equality check, no GPU).

### P3 — Spec-driven route + eligibility (tinygrad/llm/fused_attention.py + postrange.py)
- Replace the hardcoded `ADMITTED_GRIDS` frozenset (fused_attention.py:61) with spec-driven admission:
  build a FlashPrefillAttentionSpec from (q,k) shapes + ctx, call `.validate()`, admit on success.
- `custom_kernel_attention` (and the postrange native-swap, postrange.py:345) bind the SPEC and call
  `spec.emit(...)` instead of importing `amd_gfx1100_q16_grid_hd128_loop_attention` by name. (Keep the
  hand-builder callable — the spec just composes it; the point is the SPEC owns the topology as data.)
GATE: a4/varkv numerics unchanged; whole-model schedules clean; captures byte-identical.

### P4 — Manifest row + authority gate (extra/qk/route_manifest.py + a new gate)
- Add a `prefill_flash_attention_generated` (or similar) ROUTES row: `workload=prefill`, `roles=[attention_tile]`,
  `quant=[fp16]`, shape_guards for (32,8,512)/(40,8,512) [and any KV lengths], `expected_kernels`=spec's names,
  `forbidden_kernels=[fallback_graph]`, `provenance: machine_authored_generated` (JUSTIFIED: emitter derives all
  extents from spec fields, Hd pinned-but-derived like AMDAttentionGridSpec, reused across 8B/14B; NOT a fixed
  one-off), `purity_status` via derive_purity_status, `selector`, `route_attribution` through the Spec chain
  (mirror the decode row's attribution style, route_manifest.py:142-143), `authority_gate`=the new gate.
- New authority gate (model on extra/qk/prefilled_route_parity.py / benchmark_shared_attention): proves the route
  is route-bound (no hidden SDPA fallback), token-parity vs SDPA on REAL 8B/14B (reuse the e2e A/B + a4_numerics
  against attention_harness_common.reference_attention), and correct+fast. Cite the numeric artifact.
GATE: `assert_pure_machine_search({'PURE_MACHINE_SEARCH_ONLY':'1'})` still PASS with the new row present, AND the
route resolves as machine_authored_generated (not flagged impure). validate_manifest() passes.

### P5 — End-to-end + enable
- Wire the route so `prefill_custom_kernel_attn` (rename -> `prefill_flash_attention` or via policy) selects the
  spec-generated route; confirm PURE_MACHINE_SEARCH_ONLY governs it (now visible to the guard, unlike today's
  config-flag-invisible route — closes the gap found earlier).
- Real-model e2e token-match (already demonstrated: 49855==49855) re-run through the spec route.
GATE: full guard + unit + capture + numerics + e2e green.

## INVARIANTS
1. P1 is byte-identical (de-literalize only; 8==128//16). Any capture change is a bug.
2. Do NOT touch uop/ops.py AMD*Spec.validate() (that's Track B). Track A keeps Hd=128 pinned.
3. The hand builder (kernels.py) is composed BY the spec, not deleted — the spec owning the topology as data is
   what earns the provenance; the builder becoming derived-in-form (P1) is what avoids the naming trap.
4. No change to the machine-searched default path (matmul/gemv routes) — verified separately by pure_search_guard.

## RISKS / OPEN JUDGMENT CALL
- **Naming-trap acceptance:** whether a reviewer accepts Track A (pinned-but-derived Hd, spec-composed) as
  genuinely machine_authored_generated vs demands Hd-DOF. Mitigation: P1 de-literalization + documenting Hd=128 as
  a validated-scope constant consistent with the already-accepted AMDAttentionGridSpec posture and decode's own
  pinned fields (token_block=16). If rejected -> Track B (substrate) or composite (Option B).
- The authority gate must run on REAL 8B/14B, not synthetic — the handoff notes the injected path was only ever
  validated on synthetic 61-node graphs + our recent real e2e; P4's gate must use the real model.

## SIZE
Track A: P1 (small, byte-identical) + P2 (new spec file, bounded) + P3 (route rewire, bounded) + P4 (manifest +
gate, precedented by decode/prefill-GEMM) + P5 (enable). No new kernel technology; the kernel exists and is proven.
This is the "near" path. Track B (Hd-DOF substrate) and Option B (composite scheduler-generated) remain the far
forks if Track A's provenance is contested.
