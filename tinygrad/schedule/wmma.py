"""Reusable WMMA authoring helpers for scheduler-owned generated kernels."""
from __future__ import annotations

from dataclasses import dataclass
from tinygrad.dtype import DType
from tinygrad.uop.ops import Ops, UOp


@dataclass(frozen=True)
class OnlineSoftmaxTile:
  """Declarative register-tile contract for fused attention.

  ``qk`` and ``pv`` are deliberately separate SHAPED_WMMA nodes.  The
  nonlinear normalization is represented by the caller between them, so a
  backend can lower the complete tile without materializing score/probability
  buffers.  This is only an authoring contract; admission remains fail-closed
  until a backend proves the lane ABI.
  """
  qk: UOp
  pv: UOp
  m: UOp
  l: UOp
  acc: UOp


def online_softmax_tile(q_frag:UOp, k_frag:UOp, v_frag:UOp, *,
                        qk_acc:UOp, pv_acc:UOp, m:UOp, l:UOp,
                        dims:tuple[int, int, int], device:str, threads:int,
                        dtype_out:DType|None=None) -> OnlineSoftmaxTile:
  """Build a tile-level QK -> online-softmax -> PV primitive.

  ``qk_acc`` is the score-tile accumulator and ``pv_acc`` is the output
  accumulator.  The caller owns the online max/sum-exp update (including the
  rescaling of ``pv_acc``); keeping that state explicit makes this usable for
  fp16 and non-fp16 routes without duplicating the scheduler path.
  """
  qk = shaped_wmma(q_frag, k_frag, qk_acc, dims=dims, device=device, threads=threads,
                   dtype_out=dtype_out)
  pv = shaped_wmma(qk, v_frag, pv_acc, dims=dims, device=device, threads=threads,
                   dtype_out=dtype_out)
  return OnlineSoftmaxTile(qk=qk, pv=pv, m=m, l=l, acc=pv)


def shaped_wmma(a_frag:UOp, b_frag:UOp, acc_frag:UOp, *, dims:tuple[int, int, int],
                device:str, threads:int, dtype_out:DType|None=None) -> UOp:
  """Construct a declarative SHAPED_WMMA tensor-graph node.

  The rangeify pass owns lowering this to Ops.WMMA. Callers must pass already-shaped per-thread fragments; this helper
  exists so route code does not construct route-local Ops.WMMA or duplicate the SHAPED_WMMA argument convention.
  """
  return UOp(Ops.SHAPED_WMMA, dtype_out or acc_frag.dtype.scalar(), (a_frag, b_frag, acc_frag),
             arg=(dims, device, threads))
