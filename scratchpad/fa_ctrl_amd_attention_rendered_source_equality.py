#!/usr/bin/env python3
"""FA-CTRL -- AMD non-regression control for the fused prefill-attention kernel (compile-only).

`scratchpad/pg2_amd_all_routes_rendered_source_equality.py` covers the six `PACKED_WMMA_ROUTES`
dense-GEMM rows. It does NOT cover the fused attention path -- there is no equivalent instrument
for `tinygrad.llm.fused_attention.custom_kernel_attention` / `amd_gfx1100_q16_grid_hd128_loop_attention`.
This script is the analogue PG2 does not provide: render the fused-attention kernel for both
`ADMITTED_GRIDS` (`tinygrad/llm/fused_attention.py:68`) -- `(32,8,512)` = Qwen3-8B, `(40,8,512)` =
Qwen3-14B (q_heads, kv_heads, q_tokens) -- and hash the rendered output, one line per grid.

TECHNIQUE
---------
Same family as PG2/`scratchpad/m1d_confirm_c_fragment.py`: `Target.parse(...)` + `to_program(ast,
renderer)`, no `Device[...]` opened, no GPU touched (`to_program` runs `do_to_program` under
`Context(ALLOW_DEVICE_USAGE=0)`). The ast is built via the SAME production seam the doc calls out
(`tinygrad/schedule/wmma/flash_prefill.py:FlashPrefillAttentionSpec.emit()`, the target-keyed
dispatch `_PREFILL_EMITTERS["amd_gfx1100"]` resolves to in `fused_attention.custom_kernel_attention`)
-- PARAM placeholders in production's exact slot convention (out=0, Q=1, K=2, V=3, per
`amd_gfx1100_q16_grid_hd128_loop_attention`'s own `tuple(x.arg.slot for x in owners)!=(1,2,3,0)`
check), causal=True (the only value `tinygrad/llm/model.py` ever passes), start_pos=0 / kv_tokens=
q_tokens (the pp512 configuration `docs/task_workflow/input/metal-fused-attention-port-scope-
20260731.md` section 1 measures).

RENDERER: `tinygrad.renderer.isa.amd.AMDISARenderer` (`AMD:ISA:gfx1100`), NOT `HIPRenderer`/
`__WMMA` -- the scope doc is explicit that attention "goes through AMDISARenderer (ISA), not the
HIP C-style renderer the packed routes use" (section 5, FA-CTRL). `AMDISARenderer.asm()` assembles
via `tinygrad/renderer/amd/elf.py:assemble_linear`, a pure-Python ELF encoder -- no comgr/native
compiler subprocess, so this is safe to run without an AMD GPU or ROCm toolchain (same posture PG0's
docstring records for HIP: cross-compile-only, no native-bus-error risk here because nothing native
is invoked). `do_assemble` produces the rendered text as `"\\n".join(str(u.arg) for u in lin.src)`
-- one real rdna3 instruction per line (confirmed against
`test/unit/test_shared_attention_compiler_capture.py`'s synthetic ISA fixture, which uses
`v_wmma_f32_16x16x16_f16` as its WMMA line). The marker counted here is therefore `v_wmma` (an ISA
mnemonic substring), never `__WMMA` (a HIP C-style renderer macro `MetalRenderer`/`AMDISARenderer`
never emit) and never `simdgroup_matrix` (Metal's oracle-only search string, not tinygrad's).

STOP CONDITION -- REPORTED, NOT WORKED AROUND
-----------------------------------------------
At commit fa134bb38 (branch exp), THIS RENDER FAILS FOR BOTH ADMITTED GRIDS. `to_program` raises
inside `full_rewrite_to_sink`'s SPEC gate (`tinygrad/codegen/__init__.py:117`,
`tinygrad/uop/spec.py:69`) before any INS/SOURCE is produced:

  RuntimeError: UOp verification failed at 854 on Ops.MUL dtypes.float 2
  [(Ops.STACK, dtypes.float.vec(8), None), (Ops.STACK, dtypes.float.vec(8), None)] None

(index differs slightly by grid/geometry; the shape of the failure -- a MUL whose OWN declared
dtype is scalar `dtypes.float` while both of its rewritten sources are `dtypes.float.vec(8)` STACK
nodes -- is identical for every grid/renderer combination tried). Traced to
`tinygrad/schedule/wmma/softmax.py:54` (`amd_gfx1100_row_softmax_state`): the `new_m`/`new_l`/
`alpha` projections are constructed as `Ops.AMD_ROW_SOFTMAX_SLOT` with declared dtype scalar
`dtypes.float`, but every consumer treats them as 8-wide row state (`kernels.py:313`'s `nm.gep(i)`/
`nl.gep(i)`, `kernels.py:321`'s `oc.alu(Ops.MUL,alpha)`, and `amd_attention_abi.py`'s own expander
which requires `dtypes.float.vec(8)` for stateful m/l, `amd_attention_abi.py:207`). Once
`native_repack_matcher` substitutes the real `STACK(vec8)` value in, the MUL that consumed the
placeholder keeps its stale scalar dtype -- exactly the mismatch `type_verify` reports. This is
NOT specific to this script's construction: the repo's OWN test suite reproduces it identically at
this same commit, unmodified --

    test/unit/test_online_softmax_tile.py::test_gfx1100_model_grid_static_loop_body_is_invariant
      [32-8-512] and [40-8-512] (THE two admitted grids) -- both fail, same signature
    test/unit/test_amd_attention_kv_tile_oob_guard.py -- 17 of 65 tests fail, same signature
    test/unit/test_shared_attention_compiler_capture.py::
      test_constructor_uses_actual_scheduled_call_and_final_hip_amdisa_programs -- fails, same signature

  ($ .venv/bin/python3 -m pytest test/unit/test_online_softmax_tile.py
     test/unit/test_amd_attention_kv_tile_oob_guard.py test/unit/test_shared_attention_compiler_capture.py -q
     -> 39 failed, 66 passed / 17 failed, 48 passed, all RuntimeError at the same MUL/STACK signature)

Both `AMDISARenderer` and `HIPRenderer` hit it identically (verified directly: `type_verify` runs
in `_full_rewrite_to_sink` before the renderer-kind branch, and both renderer classes define
`native_repack_matcher`, so both substitute the same stale-dtype MUL). Diagnostic only (NOT part of
this control -- it deliberately disables the same correctness gate the control exists to enforce,
so it is never used to manufacture a passing hash): with `SPEC=0` bypassing the type_verify gate
entirely, rendering fails a SECOND, independent way further down the AMDISARenderer pipeline --
`NotImplementedError: AMD:ISA unsupported WMMA operand carrier dtypes.half` -- confirming the block
is not a single isolated check but multiple points in the current attention-lowering pipeline.

Per the scope doc's own stop condition: "if the attention path cannot be rendered compile-only the
way the packed routes can, report that -- it changes the safety model for everything downstream."
It cannot, today. No hash exists to record. This script is still the correct rerunnable instrument:
once FA0/FA2 fix the underlying dtype declaration, re-running this file is what turns green and
produces the two baseline hashes this docstring will then be updated to record. Until then, its
FAILURE is itself the control's true, current, honestly-reported state -- PG2's six packed-WMMA
rows remain unaffected and byte-identical (re-verified same session:
`0e4c2e9218a7 8e01063e3c8f ce03d94bb58a 5ced48b9fa7c b0df79b8bb58 349a2c8c521f`), so this is an
attention-specific gap, not a general rendering regression.

Usage: `python3 scratchpad/fa_ctrl_amd_attention_rendered_source_equality.py`
Exits nonzero while any grid fails to render (today: both).
"""
from __future__ import annotations
import hashlib, sys
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.codegen import to_program
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.schedule.wmma.flash_prefill import FlashPrefillAttentionSpec
from tinygrad.llm.fused_attention import ADMITTED_GRIDS

HEAD_DIM = 128  # the only head_dim ADMITTED_GRIDS' rows are proven at (AMDAttentionGridSpec's hd128 native_abi)


def render_grid(q_heads: int, kv_heads: int, q_tokens: int) -> tuple[str, int]:
  """Render amd_gfx1100_q16_grid_hd128_loop_attention for one admitted (Hq,Hkv,q_tokens) grid via
  the production FlashPrefillAttentionSpec.emit() seam, AMDISARenderer, Target.parse + to_program.
  kv_tokens=q_tokens, start_pos=0 (query_start=0): the pp512-style full-prefill-from-empty-cache
  configuration the scope doc's depth table measures at depth=512. causal=True: the only value
  tinygrad/llm/model.py ever passes to route_prefill_attention. Returns (rendered_text, v_wmma_count).
  """
  spec = FlashPrefillAttentionSpec(Hq=q_heads, Hkv=kv_heads, q_tokens=q_tokens, kv_tokens=q_tokens,
    causal=True, scale=1.0 / (HEAD_DIM ** 0.5), Hd=HEAD_DIM, valid_kv=q_tokens, query_start=0)
  spec.validate()
  fxn = spec.emit()

  # slots MUST be (out=0, Q=1, K=2, V=3) -- amd_gfx1100_q16_grid_hd128_loop_attention hard-checks
  # tuple(x.arg.slot for x in owners)!=(1,2,3,0), matching custom_kernel_attention's own ABI comment.
  out_ph = UOp.placeholder((q_heads * q_tokens * HEAD_DIM,), dtypes.half, 0)
  q_ph = UOp.placeholder((q_heads * q_tokens * HEAD_DIM,), dtypes.half, 1)
  k_ph = UOp.placeholder((kv_heads * q_tokens * HEAD_DIM,), dtypes.half, 2)
  v_ph = UOp.placeholder((kv_heads * q_tokens * HEAD_DIM,), dtypes.half, 3)
  ast = fxn(out_ph, q_ph, k_ph, v_ph)

  renderer = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  program = to_program(ast, renderer)  # raises today -- see module docstring's STOP CONDITION
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  return source, source.count("v_wmma")


def main() -> None:
  grids = sorted(ADMITTED_GRIDS)  # (q_heads, kv_heads, q_tokens); (32,8,512) then (40,8,512)
  results = []
  for q_heads, kv_heads, q_tokens in grids:
    label = f"Hq={q_heads:<3d} Hkv={kv_heads:<3d} q_tok={q_tokens:<5d} Hd={HEAD_DIM}"
    try:
      src, wmma_calls = render_grid(q_heads, kv_heads, q_tokens)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the sweep over a single bad grid
      print(f"{label} FAILED: {type(exc).__name__}: {exc}")
      results.append(((q_heads, kv_heads, q_tokens), None, None))
      continue
    digest = hashlib.sha256((src + "\n").encode()).hexdigest()
    print(f"{label} sha256={digest[:12]} v_wmma={wmma_calls} instructions={len(src.splitlines())} src_len={len(src)}")
    results.append(((q_heads, kv_heads, q_tokens), digest, wmma_calls))

  failed = [grid for grid, digest, _ in results if digest is None]
  if failed:
    print(f"\n{len(failed)} of {len(results)} admitted grids FAILED to render: {failed}", file=sys.stderr)
    print("This IS the reportable finding, not a harness bug -- see module docstring STOP CONDITION.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
