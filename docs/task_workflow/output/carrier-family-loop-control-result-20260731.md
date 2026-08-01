# Carrier family-loop control — result

Implements `docs/task_workflow/input/carrier-family-loop-control-scope-20260731.md` (which
supersedes WC3's four-render minimum): one compile-only control that loops every registered
tensor-core family in `tinygrad/codegen/opt/tc.py` and asserts the emitted WMMA carrier equals
the descriptor-derived dtype. This is the "actually-agnostic" control Claude called for, and the
one that would have caught `float.vec(8)` being wrong for the non-RDNA families (WC2,
`3ec8557f1`).

Control: `scratchpad/fc13_carrier_family_control.py`. Run from the repo root; exit 0 only if
every family row, alias check, and the AMDISA golden check pass.

## Per-family table (HEAD `ae347e02d`, all 14 registered names)

Assertions 1-3 run on the pre-devectorizer TC-opt window (the WMMA node as the generic TC opt
emits it: `WMMA` dtype == `dtype_out.vec(ept[2])`, operand carriers == `dtype_in.vec(ept[0/1])`,
UNROLL folds `binary_axis_count(tc, 2)` binary axes). Rendering is PG2's non-ISA pipeline
(`full_rewrite_to_sink` -> `do_linearize` -> `do_estimates` -> `do_render`), never `do_compile`.
`desc` = descriptors asserted 1-3; `render` = descriptors that rendered without raising; SHA is
the first rendered descriptor's source hash (`sha256(src + "\n")`, PG2 convention).

| family | carrier (C) | desc | render | arch | SHA |
| --- | --- | ---: | ---: | --- | --- |
| `amd_rdna3` | `float.vec(8)` | 4/4 | 4/4 | gfx1100 | `2758933204af` |
| `amd_rdna4` | `float.vec(8)` | 4/4 | 4/4 | gfx1200 | `5d59cae01646` |
| `amd_cdna_161616` | `float.vec(4)` | 2/2 | 2/2 | gfx942 | `f0c4a0447dd4` |
| `amd_cdna_161632` | `float.vec(4)` | 4/4 | 4/4 | gfx942 | `66eabd66c577` |
| `amd_cdna_1616128` | `float.vec(4)` | 2/2 | 2/2 | gfx950 | `87a6760a82af` |
| `amd_cdna3` (alias) | `float.vec(4)` | 4/4 | 4/4 | gfx942 | `66eabd66c577` |
| `amd_cdna4` (alias) | `float.vec(4)` | 8/8 | 8/8 | gfx950 | `87a6760a82af` |
| `cuda_81616` | `float.vec(4)` | 3/3 | 3/3 | sm_80 | `d2c1c27250e2` |
| `cuda_81632_f8` | `float.vec(4)` | 2/2 | 2/2 | sm_89 | `020d1fe845e6` |
| `cuda_8168_f16` | `float.vec(4)` | 2/2 | 2/2 | sm_80 | `a79a73341024` |
| `cuda_8168_tf32` | `float.vec(4)` | 1/1 | 1/1 | sm_80 | `10295ff12fa6` |
| `cuda_sm75` (alias) | `float.vec(4)` | 2/2 | 2/2 | sm_75 | `a79a73341024` |
| `cuda_sm80` (alias) | `float.vec(4)` | 6/6 | 6/6 | sm_80 | `d2c1c27250e2` |
| `metal` | `float.vec(2)` | 5/5 | 5/5 | Apple7 | `11e117c8a263` |

14 registered names = 10 base lists + 4 composition aliases (the scope doc's "13" undercounted
the alias rows). Aliases verify by object identity: `cuda_sm75 is cuda_8168_f16`, and
`cuda_sm80` / `amd_cdna3` / `amd_cdna4` are elementwise-identical to their declared
compositions. Composition fidelity is visible in the hashes: `amd_cdna3`'s half row matches
`amd_cdna_161616`'s (`f0c4a0447dd4`), `cuda_sm75` matches `cuda_8168_f16`, and `cuda_sm80`
repeats `cuda_81616`'s and `cuda_8168_f16`'s rows.

The `amd_rdna3` AMDISA fixture golden is asserted exactly via `_emit_fixture(_tc_matmul_ast)`:
binary `4a558d215767...`, mnemonic `f415079ccd15...`, 972 bytes, 149 instructions, 1 wmma --
matching `FIXTURES["tc_16x16x16_unrolled"]` at HEAD.

## Follow-up: fp8 WMMA now fails closed on renderers without native fp8

The original finding was that `amd_cdna_1616128` (K=128 fp8) crashed the gfx942 renderer with a
bare `ValueError` from `cstyle.py:fp8_index`: gfx942 has no native fp8_ocp support
(`HIPRenderer.supported_dtypes` admits fp8_ocp only on gfx950), so the dtype decomposer emulated
fp8 as half, and the CDNA K=128 string pattern then called `fp8_index(half)`. K=32 fp8 on gfx942
(`amd_cdna3` does bind those descriptors) was the quieter sibling: it rendered with half operands
handed to an fp8 builtin -- the same mismatch, without the crash.

Per the owner's direction ("fail loudly if we run into it"), this is now a fail-closed guard in
`postrange.py::_apply_generic_tensor_core_opt`: a descriptor whose operand dtype the renderer
cannot express natively is refused at TC selection with a `KernelOptError` naming the capability
and target, instead of either historical failure mode. `cstyle.py:fp8_index` additionally raises
a descriptive `RuntimeError` (only reachable now by hand-built graphs). gfx950 and CUDA sm_89 are
fp8-native and unaffected; gfx942's half/bf16 descriptors are unaffected (the guard is
dtype-specific, and the control's `amd_cdna_161632` / `amd_cdna3` rows now report the fp8 pair as
`blocked` -- the intended behavior). The gfx942 K=128 cross-check row verifies the guard fires
there; the natural gfx950 row is the production path (`get_amd("gfx950")` = `amd_cdna4` contains
1616128; `get_amd("gfx942")` = `amd_cdna3` does not).

Coverage: `test/unit/test_tc_fp8_wmma_admission.py` (4 tests: forced-select and search-mode
fail-closed on gfx942, half-descriptor regression, gfx950 native fp8 carriers). The mutation
check above still fails all 14 families when the oracle dtype lines are changed.

## Mutation check (the gate can fail)

Per scope section 4.3, both occurrences of the oracle expression
`tc.dtype_out.vec(tc.elements_per_thread[2])` in `postrange.py` were mutated to
`dtypes.float.vec(8)` (the scope cites line 552, the pipeline path; the plain-matmul path this
control exercises uses the duplicate at line 605, so both were changed for the check). Result:
all 14 families fail assertion 1 -- every CDNA/CUDA/Metal descriptor reports
`WMMA=float.vec(8) expect float.vec(4)/vec(2)`, and the RDNA rows whose `dtype_out` is not
float (half/bf16/int) fail the same way; the two RDNA float rows pass by the width-8
coincidence, exactly as the scope predicts. Restored byte-identical; the control passes again at
HEAD.

## Regression evidence

- PG2 (`scratchpad/pg2_amd_all_routes_rendered_source_equality.py`) still emits the six known
  hashes byte-identical: `0e4c2e9218a7 8e01063e3c8f ce03d94bb58a 5ced48b9fa7c b0df79b8bb58
  349a2c8c521f`.
- `test/unit/test_amd_isa_wmma.py` + `test/unit/test_amd_isa_extraction_fixtures.py`: 41 passed,
  3 subtests passed.
- No emitter, dtype-authority, or test code changed by this packet; the control is additive.
