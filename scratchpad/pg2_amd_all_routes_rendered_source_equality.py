#!/usr/bin/env python3
"""PG2 -- AMD non-regression control for all six `PACKED_WMMA_ROUTES` rows (compile-only).

`scratchpad/pg0_amd_rendered_source_equality.py` covers only `ffn_gate_up` (bc=1).
`scratchpad/mb2_amd_ffn_down_rendered_source_equality.py` covers only `ffn_down` (bc=2).
Together those guard 2 of 6 production rows. This script extends the same technique --
`warmstart_entry` + `HIPRenderer` + `to_program`'s non-ISA pipeline (`full_rewrite_to_sink` ->
`do_linearize` -> `do_estimates` -> `do_render`), never `do_compile` (no ROCm compiler on this
Mac, same native-bus-error limitation PG0's docstring records) -- to loop all six rows in
`tinygrad.llm.packed_wmma_prefill.PACKED_WMMA_ROUTES` and print one line per row: the row's
identity, its rendered-source SHA-256, and its `__WMMA` occurrence count. One command, one
process, all six.

Usage: `python3 scratchpad/pg2_amd_all_routes_rendered_source_equality.py`
Each line's hash can be independently re-derived: `... | grep <role> | ...`, or by isolating a
single row with `ROUTES = ROUTES[i:i+1]` if ever needed -- not required for normal use.
"""
from __future__ import annotations
import hashlib, sys
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from tinygrad import Tensor, dtypes
from tinygrad.codegen import do_estimates, do_linearize, do_render, full_rewrite_to_sink
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, ProgramInfo, UOp
from tinygrad.codegen.opt.postrange import warmstart_candidate_state
from tinygrad.llm.packed_wmma_prefill import PACKED_WMMA_ROUTES, warmstart_entry, packed_half_carrier


def render_only(ast: UOp, ren: Renderer) -> str:
  """Replay to_program's non-ISA pipeline up to (and including) do_render, never do_compile."""
  full_sink = full_rewrite_to_sink(ast, ren, optimize=ast.tag is None)
  prg = UOp(Ops.PROGRAM, src=(full_sink, UOp(Ops.DEVICE, arg=ren.target.device)), arg=ProgramInfo.from_sink(full_sink))
  prg = do_linearize(ren, prg, full_sink)
  updated = do_estimates(prg, full_sink, prg.src[2])
  if updated is not None: prg = updated
  prg = do_render(ren, prg, prg.src[2])
  return prg.src[3].arg


def render_route(row) -> tuple[str, int]:
  """Render one `PackedWmmaRoute` on AMD/HIP, exactly PG0/MB2's per-row body. Returns (src, wmma_calls)."""
  entry = warmstart_entry(row.quant, row.role, row.shape)
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
  return src, src.count("__WMMA")


def main() -> None:
  results = []
  for row in PACKED_WMMA_ROUTES:
    try:
      src, wmma_calls = render_route(row)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the sweep over a single bad row
      print(f"{row.quant:5s} {row.role:10s} {str(row.shape):20s} {str(row.geometry):24s} FAILED: {exc!r}")
      results.append((row, None, None))
      continue
    # Hash `src + "\n"`, matching `print(src) | shasum -a 256` (PG0/MB2's own convention: the
    # piped stdout includes the trailing newline `print` adds), so hashes here are directly
    # comparable to PG0's/MB2's documented values without a second invocation.
    digest = hashlib.sha256((src + "\n").encode()).hexdigest()
    print(f"{row.quant:5s} {row.role:10s} {str(row.shape):20s} {str(row.geometry):24s} "
          f"sha256={digest[:12]} wmma={wmma_calls} src_len={len(src)}")
    results.append((row, digest, wmma_calls))

  failed = [row for row, digest, _ in results if digest is None]
  if failed:
    print(f"\n{len(failed)} of {len(results)} rows FAILED to render", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
