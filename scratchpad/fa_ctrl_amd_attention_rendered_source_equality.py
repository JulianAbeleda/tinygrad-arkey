#!/usr/bin/env python3
"""FA-CTRL -- AMD non-regression control for the fused prefill-attention kernel (compile-only).

`scratchpad/pg2_amd_all_routes_rendered_source_equality.py` covers the six `PACKED_WMMA_ROUTES`
dense-GEMM rows. This script is the attention analogue: render the fused-attention kernel for both
`ADMITTED_GRIDS` (`tinygrad/llm/fused_attention.py:68`) -- `(32,8,512)` = Qwen3-8B, `(40,8,512)` =
Qwen3-14B (q_heads, kv_heads, q_tokens) -- and hash the rendered output, one line per grid.

TWO RENDERER ARMS (required by nv-fused-prefill-attention-port-scope-20260801.md section 10.1)
------------------------------------------------------------------------------------------------
The control renders through BOTH AMD renderers, because the NV port's P1/P2 edit HIP-path code
(`cstyle.py:139` bpermute-fmax rule, `:155-176` drain expansion, `:630` matcher binding):

  - `AMDISARenderer` (`AMD:ISA:gfx1100`): the ISA capture/reference path. `asm()` assembles via
    `tinygrad/renderer/amd/elf.py:assemble_linear`, a pure-Python ELF encoder -- safe to run without
    an AMD GPU or ROCm toolchain (same posture PG0 records). Rendered text is one rdna3 instruction
    per line; the WMMA marker counted here is `v_wmma`.
  - `HIPRenderer` (`AMD:HIP:gfx1100`): the production AMD runtime consumer -- the model compiles the
    custom-kernel program through the device's C-style renderer, where `cstyle.py:630` binds the
    native attention matchers under `target.arch.split(":")[0] == "gfx1100"`. Its marker is `__WMMA`
    (the renderer's device-function prefix; 17 = 16 call sites + 1 definition, matching the ISA arm's
    16 instructions).

Both arms build the AST through the SAME production seam (`FlashPrefillAttentionSpec.emit()`, the
target-keyed dispatch `_PREFILL_EMITTERS["amd_gfx1100"]` resolves to in
`fused_attention.custom_kernel_attention`) with PARAM placeholders in production's exact slot
convention (out=0, Q=1, K=2, V=3), causal=True (the only value `tinygrad/llm/model.py` ever passes),
and start_pos=0 / kv_tokens=q_tokens (the pp512 configuration the scope doc measures).

PINNED BASELINES (recorded 2026-08-01 at dcc1bc778; rerun after every commit)
-----------------------------------------------------------------------------
AMD:ISA  (32,8,512) sha256=19829976aa55 v_wmma=16 instructions=1752
AMD:ISA  (40,8,512) sha256=7efc22cdda57 v_wmma=16 instructions=1753
AMD:HIP  (32,8,512) sha256=ea8cdefba409 __WMMA=17 src_len=57930
AMD:HIP  (40,8,512) sha256=daefec1dd70a __WMMA=17 src_len=57930

PG2's six packed-WMMA rows are a separate instrument and remain byte-identical
(`0e4c2e9218a7 8e01063e3c8f ce03d94bb58a 5ced48b9fa7c b0df79b8bb58 349a2c8c521f`).

Usage: `PYTHONPATH=/path/to/tinygrad-arkey python3 scratchpad/fa_ctrl_amd_attention_rendered_source_equality.py`
Exits nonzero while any grid fails to render on either arm.
"""
from __future__ import annotations
import hashlib, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.codegen import to_program
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.schedule.wmma.flash_prefill import FlashPrefillAttentionSpec
from tinygrad.llm.fused_attention import ADMITTED_GRIDS

HEAD_DIM = 128  # the only head_dim ADMITTED_GRIDS' rows are proven at (AMDAttentionGridSpec's hd128 native_abi)


def _build_ast(q_heads: int, kv_heads: int, q_tokens: int) -> UOp:
  """Build the production fused-attention AST for one admitted grid via FlashPrefillAttentionSpec.emit().

  kv_tokens=q_tokens, start_pos=0 (query_start=0): the pp512-style full-prefill-from-empty-cache
  configuration the scope doc's depth table measures at depth=512. causal=True: the only value
  tinygrad/llm/model.py ever passes to route_prefill_attention.
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
  return fxn(out_ph, q_ph, k_ph, v_ph)


def _render(ast: UOp, renderer) -> str:
  program = to_program(ast, renderer)  # compile-only; no GPU touched
  return next(u.arg for u in program.src if u.op is Ops.SOURCE)


def main() -> None:
  grids = sorted(ADMITTED_GRIDS)  # (q_heads, kv_heads, q_tokens); (32,8,512) then (40,8,512)
  arms = [
    ("AMD:ISA", lambda: AMDISARenderer(Target.parse("AMD:ISA:gfx1100")), "v_wmma", "instructions"),
    ("AMD:HIP", lambda: HIPRenderer(Target.parse("AMD:HIP:gfx1100")), "__WMMA", "src_len"),
  ]
  failed = []
  for q_heads, kv_heads, q_tokens in grids:
    label = f"Hq={q_heads:<3d} Hkv={kv_heads:<3d} q_tok={q_tokens:<5d} Hd={HEAD_DIM}"
    try:
      ast = _build_ast(q_heads, kv_heads, q_tokens)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the sweep over a single bad grid
      print(f"{label} AST FAILED: {type(exc).__name__}: {exc}")
      failed.append((q_heads, kv_heads, q_tokens, "AST"))
      continue
    for arm_name, make_renderer, marker, len_field in arms:
      try:
        src = _render(ast, make_renderer())
      except Exception as exc:  # noqa: BLE001
        print(f"{label} {arm_name} FAILED: {type(exc).__name__}: {exc}")
        failed.append((q_heads, kv_heads, q_tokens, arm_name))
        continue
      digest = hashlib.sha256((src + "\n").encode()).hexdigest()
      detail = len(src.splitlines()) if len_field == "instructions" else len(src)
      print(f"{label} {arm_name} sha256={digest[:12]} {marker}={src.count(marker)} {len_field}={detail}")

  if failed:
    print(f"\n{len(failed)} render(s) FAILED: {failed}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
