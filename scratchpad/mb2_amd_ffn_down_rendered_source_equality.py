#!/usr/bin/env python3
"""MB2 AMD non-regression probe for the bc=2 branch (compile-only, no AMD hardware).

`scratchpad/pg0_amd_rendered_source_equality.py` renders `ffn_gate_up` at `bc=1` (the plain-LDS
precontract branch, `pipeline_plan=None`). MB2 changes code inside the `candidate_pipeline is not
None` branch (`tinygrad/codegen/opt/postrange.py`, the buffer2 accumulator-contract binary-axis
count and the accumulator-elements/offset derivation), which PG0's `ffn_gate_up` control never
walks into (`bc=1`). `PACKED_WMMA_ROUTES` contains exactly one row with `bc=2`:
`PackedWmmaRoute("Q4_K", "ffn_down", (512, 5120, 17408), (256, 128, 32, 8, 2, 2), ...)`. AMD
genuinely executes this branch, so this is the control that can actually catch a regression here.

Same rendered-source-text-equality boundary as PG0 (no ROCm compiler on this Mac -- see PG0's own
docstring for the native-bus-error detail): replays `to_program`'s non-ISA pipeline
(`full_rewrite_to_sink` -> `do_linearize` -> `do_estimates` -> `do_render`), never calls
`do_compile`/`ctx.compiler`.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from tinygrad import Tensor, dtypes
from tinygrad.codegen import do_estimates, do_linearize, do_render, full_rewrite_to_sink
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, ProgramInfo, UOp
from tinygrad.codegen.opt.postrange import warmstart_candidate_state
from tinygrad.llm.packed_wmma_prefill import warmstart_entry, packed_half_carrier

ROUTE = ("Q4_K", "ffn_down", (512, 5120, 17408))  # real production row, gfx1100 geometry, bc=2


def render_only(ast: UOp, ren: Renderer) -> str:
  """Replay to_program's non-ISA pipeline up to (and including) do_render, never do_compile."""
  full_sink = full_rewrite_to_sink(ast, ren, optimize=ast.tag is None)
  prg = UOp(Ops.PROGRAM, src=(full_sink, UOp(Ops.DEVICE, arg=ren.target.device)), arg=ProgramInfo.from_sink(full_sink))
  prg = do_linearize(ren, prg, full_sink)
  updated = do_estimates(prg, full_sink, prg.src[2])
  if updated is not None: prg = updated
  prg = do_render(ren, prg, prg.src[2])
  return prg.src[3].arg


def main() -> None:
  entry = warmstart_entry(*ROUTE)
  m, n, k = entry["m"], entry["n"], entry["k"]
  transform, context = entry["transform"], entry["context"]

  raw_words = Tensor.empty(transform.packed_bytes // transform.storage_width, dtype=transform.storage_dtype, device="AMD")
  b = packed_half_carrier(raw_words, transform, n, k)
  x_batch = Tensor.empty(m, k, dtype=dtypes.float16, device="AMD")
  out = (x_batch @ b.transpose()).contiguous().reshape(1, m, n)
  calls = [c for c in out.schedule_linear().src if c.op is Ops.CALL]
  if len(calls) != 1: raise RuntimeError(f"expected exactly one CALL in the schedule, got {len(calls)}")
  ast = calls[0].src[0]

  ren = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  opts = {entry["key"]: entry["opt"]}
  candidate_contexts = {entry["key"]: context}
  with warmstart_candidate_state(opts, candidate_contexts):
    src = render_only(ast, ren)
  wmma_calls = src.count("__WMMA")
  print(f"# route={ROUTE} lds_bytes={context.geometry.lds_bytes} src_len={len(src)} wmma_calls={wmma_calls}", file=sys.stderr)
  print(src)


if __name__ == "__main__":
  main()
