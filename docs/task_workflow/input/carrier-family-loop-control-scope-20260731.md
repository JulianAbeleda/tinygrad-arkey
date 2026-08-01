# Carrier family-loop control — scope

Date: 2026-07-31
Status: scoped, not implemented. Branch boundary: tinygrad `exp`. Compile-only.

Companion to `wmma-carrier-shape-awareness-scope-20260731.md` **WC3**. WC3 says "at least one
family from each carrier class — RDNA3 (8), CDNA (4), CUDA (4), Metal (2)" — four render checks.
**This scope supersedes that minimum: one control that loops every registered tensor-core family
in `tinygrad/codegen/opt/tc.py` and asserts the emitted carrier equals the descriptor-derived
dtype.** It is the "actually-agnostic" control Claude called for, and the one that would have
caught `float.vec(8)` being wrong for the non-RDNA families.

## 1. What the control asserts

For every registered family, build a plain half matmul AST, force its tensor-core selection, run
the generic TC opt compile-only, and check the emitted WMMA graph:

1. `Ops.WMMA` node dtype == `tc.dtype_out.vec(tc.elements_per_thread[2])` — the descriptor-derived
   carrier, never a literal.
2. The two operand carriers == `tc.dtype_in.vec(tc.elements_per_thread[0/1])` (A/B widths).
3. `binary_axis_count(tc, 2)` (the existing fact source, `kernel_lds.py:58`) matches the number of
   binary accumulator axes the graph actually folds.
4. Render with the family's real renderer (HIP / CUDA / Metal / AMD ISA) and record the source SHA.
   Rendering must not raise — that alone catches a welded path (the FA0 class of failure).

Assertions 1–3 catch **wrongness** directly; assertion 4 catches **breakage**. No golden hashes are
needed for 1–3 — the descriptor is the oracle. This is the difference from PG2/FA-CTRL, which hash
bytes and therefore only detect *change*.

## 2. Registry (measured from `tc.py`)

Named lists, with C-carrier width:

| family list | C width | carrier | compile-only renderer |
| --- | ---: | --- | --- |
| `amd_rdna3` | 8 | `float.vec(8)` | HIP gfx1100 / AMD ISA |
| `amd_rdna4` | 8 | `float.vec(8)` | HIP gfx1200 |
| `amd_cdna3` (alias) | 4 | `float.vec(4)` | HIP gfx942 |
| `amd_cdna4` (alias) | 4 | `float.vec(4)` | HIP gfx950 |
| `amd_cdna_161616` | 4 | `float.vec(4)` | HIP |
| `amd_cdna_161632` | 4 | `float.vec(4)` | HIP |
| `amd_cdna_1616128` | 4 | `float.vec(4)` | HIP |
| `cuda_sm75` (alias) | 4 | `float.vec(4)` | CUDA sm_75 |
| `cuda_sm80` (alias) | 4 | `float.vec(4)` | CUDA sm_80 |
| `cuda_81616` | 4 | `float.vec(4)` | CUDA |
| `cuda_81632_f8` | 4 | `float.vec(4)` | CUDA |
| `cuda_8168_f16` | 4 | `float.vec(4)` | CUDA |
| `cuda_8168_tf32` | 4 | `float.vec(4)` | CUDA |
| `metal` | 2 | `float.vec(2)` | Metal (Apple >= 7) |

The aliases (`amd_cdna3`, `amd_cdna4`, `cuda_sm75`, `cuda_sm80`) are compositions of the base
lists; the control iterates the base lists and verifies the aliases resolve to the same carriers.

## 3. Method (follows WC1's established technique)

Reuse the proven compile-only pipeline: `Target.parse(...)` + `to_program(ast, renderer)` under
`ALLOW_DEVICE_USAGE=0`, never `do_compile`, no GPU. AST: the plain `_tc_matmul_ast`-family builder
from `test_amd_isa_wmma.py` (forced `Opt(OptOps.TC, axis=0, arg=(tc_select, 0, 1))`), sized to each
family's dims so the selection is legal. Renderers are the real ones — `HIPRenderer` /
`CUDARenderer` / `MetalRenderer` / `AMDISARenderer` — each with `tensor_cores` bound to that
family's list, exactly as production binds them.

Output: one line per family — name, carrier, assert result, source SHA, instruction/WMMA count.
Exit non-zero on any failed assertion with the family named.

## 4. Acceptance

1. All 13 families render and pass assertions 1–3 at HEAD.
2. `amd_rdna3` SHA matches the fixture golden family (the `_tc_matmul_ast` 16x16x16 path) and the
   six PG2 hashes stay byte-identical (this control adds no code to the emitter).
3. The control fails loudly if a literal carrier is reintroduced — mutation-checked once: change
   `tc.dtype_out.vec(tc.elements_per_thread[2])` to `dtypes.float.vec(8)` in `postrange.py:552`,
   rerun, confirm the CDNA/CUDA/Metal rows fail, restore.

## 5. Deliverable + HARD STOP

One scratchpad control committed (`scratchpad/`), one `[test]`-prefix commit, output doc recording
the per-family table. No emitter changes. No dtype authority work (D1/D2/D3) interleaved. Do not
start the Metal port or any NVIDIA adapter work from this packet.

## 6. One-line job

**Loop every registered tensor-core family, assert its WMMA carrier equals the descriptor-derived
dtype, render each compile-only, and record the table.**
