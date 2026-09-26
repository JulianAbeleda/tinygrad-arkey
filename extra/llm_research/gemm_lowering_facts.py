#!/usr/bin/env python3
"""Export what tinygrad's precontract tile lowering can emit for one GEMM, as BoltBeam's gemm_lowering_facts.v1.

BoltBeam derives the tile-GEMM strategy space from two fact sets: the GPU's (its target registry scan) and the
compiler's (this document).  Every value here is read from the renderer that would compile the kernel and from the
lowering's own derivations -- the tensor-core descriptor, ``derive_matrix_fragment_layout``, ``xor_swizzle_conflict_free``
-- never restated.  Research only; nothing under tinygrad/ imports this.

  python3 extra/llm_research/gemm_lowering_facts.py --backend NV --arch sm_120 [--out facts.json]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCHEMA = "boltbeam.gemm_lowering_facts.v1"
# Row widths probed for the XOR swizzle: whole 16-byte chunks up to 512-byte rows (tile_k 256 at 2 bytes).
_SWIZZLE_PROBE_ROW_BYTES = tuple(16 << i for i in range(6))


def _renderer_class(backend:str):
  if backend in ("NV", "CUDA"):
    from tinygrad.renderer.cuda import CUDARenderer
    return CUDARenderer
  raise ValueError(f"no precontract tile lowering facts for backend {backend!r}")


def _tensor_cores(backend:str, arch:str):
  from tinygrad.codegen.opt import tc
  if backend in ("NV", "CUDA"): return tc.get_cuda(arch)
  raise ValueError(f"no tensor-core descriptor table for backend {backend!r}")


def lowering_facts(backend:str="NV", arch:str="sm_120", dtype_in:str="bfloat16", dtype_out:str="float") -> dict:
  from tinygrad.dtype import dtypes
  from tinygrad.codegen.opt.kernel_lds import derive_matrix_fragment_layout, xor_swizzle_conflict_free
  from tinygrad.llm.dense_candidate_gemm import WEIGHT_ROW_TILE
  from extra.llm_research.runtime_specs import _CAPABILITY_ROWS
  renderer = _renderer_class(backend)
  row = _CAPABILITY_ROWS[(backend, arch)]["two_buffer_stage1"]   # the declared LDS tile row for this target
  din, dout = getattr(dtypes, dtype_in), getattr(dtypes, dtype_out)
  # The descriptor the precontract lowering picks: the largest-K atom for this dtype pair (m16n8k16 for bf16 on NV).
  tcs = [t for t in _tensor_cores(backend, arch) if t.dtype_in == din and t.dtype_out == dout]
  if not tcs: raise ValueError(f"{backend}:{arch} has no {dtype_in}->{dtype_out} tensor core")
  tc = max(tcs, key=lambda t: t.dims[2])
  n_atom, m_atom, k_atom = tc.dims            # descriptor dims are (N, M, K)
  ept_a, ept_b, ept_c = tc.elements_per_thread
  try:
    matrix = all(derive_matrix_fragment_layout(tc, i) is not None for i in (0, 1))
  except ValueError: matrix = False
  banks = getattr(renderer, "lds_bank_dwords", None)
  swizzle_rows = [] if banks is None else [rb for rb in _SWIZZLE_PROBE_ROW_BYTES
    if _conflict_free(xor_swizzle_conflict_free, SimpleNamespace(base=0, end=rb * 16, stride_bytes=rb, xor_swizzle=True), banks, din.itemsize)]
  runtime = getattr(renderer, "max_runtime_local_bytes", None) if getattr(renderer, "runtime_local_prefix", None) else None
  return {"schema": SCHEMA, "backend": backend, "arch": arch, "mma_m": m_atom, "mma_n": n_atom, "mma_k": k_atom,
          "a_fragment_regs": ept_a * din.itemsize // 4, "b_fragment_regs": ept_b * din.itemsize // 4,
          "accumulator_regs": ept_c * dout.itemsize // 4, "operand_bytes": din.itemsize, "accumulator_bytes": dout.itemsize,
          "vector_bytes": row.vector_bytes, "static_lds_bytes": renderer.max_static_local_bytes, "runtime_lds_bytes": runtime,
          "lds_padding_bytes": row.lds_padding,
          "async_copy": getattr(renderer, "async_copy_ops", None) is not None, "matrix_fragments": matrix,
          "swizzle_row_bytes": swizzle_rows, "weight_row_granule": WEIGHT_ROW_TILE,
          "ragged_split_k": False, "stream_k": False,
          "provenance": {"renderer": f"{renderer.__module__}.{renderer.__name__}", "tensor_core": f"{tc.dims} {din}->{dout}",
                         "capability_row": row.capability_id,
                         "matrix_fragments": "kernel_lds.derive_matrix_fragment_layout", "swizzle": "kernel_lds.xor_swizzle_conflict_free",
                         "weight_row_granule": "tinygrad.llm.dense_candidate_gemm.WEIGHT_ROW_TILE",
                         "ragged_split_k": "precontract split-K slices are whole K tiles (dense_bf16_geometry_search.feasible)",
                         "stream_k": "no dense precontract stream-K lowering"}}


def _conflict_free(check, window, banks:int, item_bytes:int) -> bool:
  try: return check(window, banks, item_bytes=item_bytes)
  except ValueError: return False


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--backend", default="NV")
  ap.add_argument("--arch", default="sm_120")
  ap.add_argument("--dtype-in", default="bfloat16")
  ap.add_argument("--dtype-out", default="float")
  ap.add_argument("--out", type=Path)
  args = ap.parse_args()
  text = json.dumps(lowering_facts(args.backend, args.arch, args.dtype_in, args.dtype_out), indent=1) + "\n"
  if args.out: args.out.write_text(text)
  else: sys.stdout.write(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
