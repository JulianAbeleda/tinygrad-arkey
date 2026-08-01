# WC0 — WMMA carrier census (dtype-carrier surface)

Task: `docs/dtype-carrier-census-task-deepseek-20260731.md` §2. Branch `exp` @ `e3f2173ad` (2026-07-31).
Read-only inspection plus two scratchpad probes in `/tmp` (not committed). No code changed.

## Stop-condition finding

**Category B is a handful, not a diffuse pattern.** The load-bearing dtype-identity sites are
eight rewrites/creations, enumerated in full below (§ Category B). This materially shrinks the
shape of the follow-on work: a carrier shape-aware migration needs to touch these seven sites
first, not a sprawling pattern family.

## Verified mechanism (probe-confirmed root chain)

The failing graph (test `test_gfx1100_model_grid_static_loop_body_is_invariant[32-8-512]`, uop 824)
contains `Ops.MUL dtypes.float` whose src0 is `STACK float.vec(8)` and whose src1 is an
`AMD_ROW_SOFTMAX_SLOT` typed `dtypes.float` with **shape `(8,)`** (probe 2, `/tmp/wc0_probe2.py`).

Chain, each link verified:

1. `tinygrad/schedule/wmma/kernels.py:321` (`amd_gfx1100_q16_grid_hd128_loop_attention`):
   `oc.alu(Ops.MUL, alpha)` where `alpha` is the slot returned by
   `amd_gfx1100_row_softmax_state` (also :79, :131, :195, :242, :375 in the other five
   gfx1100 attention builders).
2. `tinygrad/schedule/wmma/softmax.py:54` creates the slot: `AMD_ROW_SOFTMAX_SLOT dtypes.float`
   with shape `(8,)` — a carrier-shaped value typed as scalar float.
3. `tinygrad/uop/ops.py:625-633` `UOp.alu`: `out_dtype = all_srcs[-1].dtype` (:631) — the MUL is
   typed by the last operand's dtype, so the scalar-typed slot makes the whole MUL scalar.
4. `tinygrad/renderer/isa/amd_attention_abi.py:189` `expand_native_row_softmax_repack` (installed
   at `tinygrad/codegen/__init__.py:106-108`, runs before the spec check at :117) replaces the
   slot with a real `STACK float.vec(8)`; the consumer MUL's dtype stays `float`.
5. `tinygrad/uop/spec.py:69` (`type_verify`, called at `tinygrad/codegen/__init__.py:117`) rejects
   it via the ALU rule at `spec.py:164`: `all(x.dtype.base == y.dtype.base for y in x.src)`.
   `float` vs `float.vec(8)` (whose `.base` is the vec itself, probe 1) fails.

The same creation pattern exists at `kernels.py:79` (`amd_gfx1100_q16_kv32_attention`),
`:131` (`amd_gfx1100_q16_kv32_hd128_attention`), `:195` (`amd_gfx1100_q16_kv64_hd128_loop_attention`),
`:242` (`amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_attention`), `:321` (grid loop), `:375`
(`amd_gfx1100_q16_grid_pv_slice_stage`). Six builders, one mechanism.

## Category A — the carrier spelled as a literal

`tc` = tensor-core descriptor; "tc-derivable" means the site could compute its width from
`tc.elements_per_thread` instead of hardwiring. None of the literal sites below derive from `tc`.

| file:line | expression / role | tc-derivable | scope |
| --- | --- | --- | --- |
| `schedule/wmma/kernels.py` 33 sites (`:31,:60,:62,:108,:110,:165,:179,:223,:230,:264,:297,:344,:350,:366-367,:371` and WMMA carriers `:37,:43,:70,:83,:121,:135,:188,:196,:237,:242,:303,:321,:373,:375`) | `dtypes.float.vec(8)` consts `(0.0,)*8` / `(-inf,)*8`, WMMA result dtypes | no | AMD-only |
| `schedule/wmma/kernels.py:35,60,108,165,223,264,344-345,366-367` | `axes=((),(),tuple((-120-i,2) for i in range(3)))` upcast axes literal | no | AMD-only |
| `schedule/wmma/kernels.py:305-306,327` | `STACK float.vec(8)` reassembly of m/l loop reads | no | AMD-only |
| `schedule/wmma/kernels.py:168-171,225-227` | `placeholder((8,), float)` m/l and `placeholder(((hd//16)*8,), float)` creg | no | AMD-only |
| `schedule/wmma/softmax.py:35-37` | QK-C `float32.vec(8)` check; `state_dt = float32 if legacy else float32.vec(8)` | no | AMD-only |
| `schedule/wmma/softmax.py:58,82-83,96-97` | initial-state / block-transition / PV-C exact `float.vec(8)` checks | no | AMD-only |
| `schedule/wmma/loop_state.py:35` | `STACK float.vec(8)` state read | no | AMD-only |
| `uop/ops.py:1696-1707` | `AMDPVCLaneSpec.lane_count = 8`; "exact gfx1100 float.vec(8) ABI" | no | shared (AMD gated) |
| `uop/spec.py:81,87,94` | drain/stats rules: `all(s.dtype == float.vec(8) ...)`, `dtype.size == 2048` | no | shared |
| `uop/spec.py:335-348` | descriptor rules: exact `float32.vec(8)` on score/m/l | no | shared |
| `uop/spec.py:356,360` | AMD_PV_C_LANE / AMD_ATTENTION_LOOP_STATE GEP rules `float.vec(8)` | no | shared (AMD gated) |
| `renderer/isa/amd_attention_abi.py:73,186,203-204,207,324-326,403` | A/B `half.vec(16)` STACK; QK-C `float.vec(8)`; state `(float.vec(8), (8,))`; alpha `STACK float.vec(8)` | no | AMD-only |
| `renderer/isa/amd.py:544,555,1206,1273-1278,1358` | drain/stats ownership checks; `NOOP float.vec(8)` 8-lane aliases; WMMA lane check | no | AMD-only |
| `renderer/isa/amd_register_allocator.py:32,43` | `half.vec(16)` lane width; eight packed VGPRs | no | AMD-only |
| `renderer/isa/amd_wmma_residency.py:144-148,253` | "C is independently fixed at eight"; `half.vec(16)` carrier + `len(carrier.src) == 16` | no | AMD-only |
| `schedule/rangeify.py:230-233` | raw QK lowering requires exact `(half.vec(16), half.vec(16), float32.vec(8))` tuple | no | shared |
| `codegen/opt/gemm_consumer.py:114-115,132` | `fragment_dtype half.vec(16)`, `accumulator_dtype float.vec(8)` | no | shared |
| `codegen/opt/kernel_pipeline.py:181,296` | `accumulator_dtype.vec(8)` | no | shared |
| `renderer/llvmir.py:225-228,272-284` | LLVM WMMA `half.vec(8)`/`half.vec(16)`/`float.vec(8)`/`uint16.vec(8)` cast set | no | shared (LLVM) |
| `renderer/cstyle.py:680` | fp8 `vec(8)` WMMA source dtypes | no | shared |
| `renderer/isa/x86.py:449` | `vshufps` on `float32.vec(4)`/`float32.vec(8)` STACK | no | shared (x86) |
| `extra/llm_research/prefill/pure_register_evaluation_gate.py:175` | `"accumulator_carrier", "float.vec(8)"` | no | extra/llm_research |
| `codegen/opt/postrange.py:505-559` | `tc_upcast_axes` from `int(math.log2(tc.elements_per_thread[i]))`; WMMA `dtype_out.vec(tc.elements_per_thread[2])` | **yes** | shared |
| `codegen/opt/kernel_lds.py:58-62` | `binary_axis_count(tc, i) = log2(elements_per_thread[i])` | **yes** | shared |

## Category B — rewrites keyed on dtype identity (load-bearing)

These are the sites §1 names. None of them contain the string `vec(8)` except where noted.

| # | site | what it keys on | scope |
| --- | --- | --- | --- |
| B1 | `uop/ops.py:625-633` (`UOp.alu`, esp. `out_dtype = all_srcs[-1].dtype` :631) | result typed by last operand's dtype; a carrier-shaped slot typed scalar poisons the whole ALU | shared |
| B2 | `schedule/wmma/softmax.py:47-54` (`amd_gfx1100_row_softmax_state`) | creates `AMD_ROW_SOFTMAX_SLOT` typed `dtypes.float` (scalar) with shape `(8,)`; also :45,:60-61 (slot 0 typed half) | AMD-only |
| B3 | `uop/spec.py:71-74` (`validate_scalar_gep`) | `gep.dtype == src.dtype.scalar()` decides canonical lane extraction by dtype identity | shared |
| B4 | `uop/spec.py:164` (ALU rule) | `all(x.dtype.base == y.dtype.base for y in x.src)` | shared |
| B5 | `uop/symbolic.py:451-453` (reorder ALU/VECTORIZE) | scalarization: `UOp(alu.op, alu.dtype.scalar(), (x,y))` repeated `alu.dtype.count` times — the §1 EXP scalarization trial rule | shared |
| B6 | `uop/symbolic.py:213-216` (GEP-through-ALU) | `alu.dtype.scalar().vec(gep.dtype.count)` | shared |
| B7 | `uop/spec.py:349-353` (slot rules) | `x.dtype == x.arg.scalar_dtype`; `slot.dtype == dtypes.float and slot.shape == (8,)` | AMD-only |
| B8 | `renderer/isa/amd_attention_abi.py:207` | `state_dt, state_shape = (float.vec(8), (8,)) if stateful else (float, ())` — dtype+shape pair as the mode marker (contains a literal) | AMD-only |

B5 is the exact rule §1's quote points at ("associative symbolic rewrites currently use dtype
identity to keep a complete `(8,)` WMMA value separate from one scalar projected lane").

## Category C — consumers that assume a width

| file:line | what assumes the width | scope |
| --- | --- | --- |
| `renderer/isa/amd_attention_abi.py:221` | `for e in range(8)` expansion loop | AMD-only |
| `renderer/isa/amd_attention_abi.py:315-317` | `for i in range(8)` state reads | AMD-only |
| `renderer/isa/amd_attention_abi.py:165-186` | 16-lane A/B offset loops and `half.vec(16)` STACK | AMD-only |
| `uop/spec.py:353,360` | `0 <= x.arg[0] < 8` | shared (AMD gated) |
| `schedule/wmma/softmax.py:96` | `not 0 <= e < 8` | AMD-only |
| `uop/ops.py:1698,1706` | `lane_count = 8`; `lane_count != 8` | shared (AMD gated) |
| `schedule/wmma/kernels.py:171,227,181,231,321` | creg width `(hd//16)*8`; `block*8` offsets | AMD-only |
| `schedule/wmma/loop_state.py:29-38` | default `lanes=8` in write/read helpers | AMD-only |
| `codegen/opt/postrange.py:505-559`, `codegen/opt/kernel_lds.py:58-62` | the only tc-derived width consumers | shared |

## Unclassified

Items inspected but not assigned to A/B/C:

- 13 `test/unit` files hard-code `float.vec(8)` (`test_amd_isa_wmma.py`, `test_amd_wave_lds_fence.py`,
  `test_devectorizer_output_safety.py`, `test_devectorizer_reconstruction.py`,
  `test_kernel_lds_mapping.py`, `test_kernel_pipeline_expansion.py`, `test_logits_only_reg_store.py`,
  `test_online_softmax_tile.py`, `test_precontract_int8_lds_contract.py`,
  `test_pure_register_compile_capture.py`, `test_pure_register_evaluation_gate.py`,
  `test_state_phase_abi.py`, `test_wmma_gep_spec.py`). Tests are consumers, not implementation
  sites; enumerated here because they will constrain any NFC claim about the carrier.
- `renderer/llvmir.py:225-284` — LLVM WMMA width set (8 for half/bf16/float, 16 for half casts).
  Not exercised on this box; its target-width relationship to `tc` is not established here.
- `renderer/cstyle.py:680` — fp8 `vec(8)` WMMA carriers (different dtype family; not the float C
  carrier). Not classified as part of the 8-wide float C carrier.
- `renderer/isa/x86.py:449` — `vshufps` STACK pattern with a `float32.vec(8)` alternative.
- `renderer/isa/amd.py:681` — a comment describing the ABI, not a check.
- `extra/llm_research/prefill/pure_register_evaluation_gate.py:175` — llm_research gate, not core.

Search performed: `rg` for `vec(8)`, `float.vec(8)`, `half.vec(16)`, `range(8)`, `range(16)`,
`dtype.scalar()`, `elements_per_thread`, `\.gep\(` across `tinygrad/`, `test/unit/`, and the
`extra/llm_research` prefill subtree; plus `rg` for `dtype ==` / `dtype is` in the symbolic and
spec pattern matchers.

## Derivation summary

Exactly two sites derive the width from `tc` (`postrange.py:505-559`, `kernel_lds.py:58-62`).
Every AMD kernel/ABI/spec site is hardwired to gfx1100's 8 (C carrier) and 16 (A/B carrier).
Shared rules (`spec.py`, `symbolic.py`, `rangeify.py`, `llvmir.py`) are hardwired to 8/16 with no
`tc` at the rewrite site.

## Stop condition

This is the census artifact WC2/D1/D2/D3 are gated on. Per task §5: Category B is a handful
(eight sites, § Category B), stated prominently. No fix was designed and none was made.
