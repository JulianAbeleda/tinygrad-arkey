#!/usr/bin/env python3
"""PG0 AMD non-regression probe (compile-only, no AMD hardware).

Renders the real production `("Q4_K", "ffn_gate_up", (512, 17408, 5120))` packed-WMMA precontract
route through `HIPRenderer(Target.parse("AMD:HIP:gfx1100"))`, using the exact production
`warmstart_entry`/`warmstart_candidate_state` machinery (same idea as the M1 Metal probe, but AMD/HIP
and the real route table -- this route already exists in `PACKED_WMMA_ROUTES`, so it is not a
locally-constructed row).

LIMITATION (recorded, not worked around): this machine has no AMD/ROCm hardware or compiler. Calling
`tinygrad.codegen.to_program` all the way through for `HIPRenderer` invokes the real HIP compiler
(`compile_hip` -> `ctypes` into libamdhip64/comgr) and crashes with a native bus error on this Mac --
confirmed to happen identically on the pre-change tree, i.e. it is an environment limitation, not a
regression from this change. So this probe stops one step earlier than `to_program`: it replays
`to_program`'s own non-ISA pipeline (`full_rewrite_to_sink` -> `do_linearize` -> `do_estimates` ->
`do_render`) and never calls `do_compile`/`ctx.compiler`. That is exactly the boundary TG1 established
for AMD non-regression evidence: rendered-source (text) equality, never a compiled binary, because no
binary can be produced here.
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

ROUTE = ("Q4_K", "ffn_gate_up", (512, 17408, 5120))  # real production row, gfx1100 geometry


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
