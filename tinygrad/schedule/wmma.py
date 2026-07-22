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
  weights: UOp|None = None

  def validate(self) -> None:
    """Validate the backend-neutral tile boundary before backend lowering.

    This intentionally does not admit the primitive for code generation; it
    only guarantees that diagnostics describe a complete QK/PV tile contract.
    """
    if self.qk.op is not Ops.SHAPED_WMMA or self.pv.op is not Ops.SHAPED_WMMA:
      raise ValueError("online softmax tile requires SHAPED_WMMA QK and PV nodes")
    if self.qk.arg != self.pv.arg:
      raise ValueError("online softmax tile QK/PV descriptors must match")
    if self.acc is not self.pv:
      raise ValueError("online softmax tile acc must be the PV accumulator result")
    if self.m.shape is None or self.l.shape is None:
      raise ValueError("online softmax tile state must have logical shapes")

  def abi_report(self) -> dict:
    """Return stable source/ISA diagnostic metadata without claiming emission."""
    self.validate()
    dims, device, threads = self.qk.arg
    return {"primitive": "online_softmax_tile", "qk": "SHAPED_WMMA", "pv": "SHAPED_WMMA",
            "dims": tuple(dims), "device": device, "threads": threads,
            "renderer": "fail-closed", "isa": "not-emitted"}


def online_softmax_tile(q_frag:UOp, k_frag:UOp, v_frag:UOp, *,
                        qk_acc:UOp, pv_acc:UOp, m:UOp, l:UOp,
                        dims:tuple[int, int, int], device:str, threads:int,
                        dtype_out:DType|None=None, normalize:bool=False) -> OnlineSoftmaxTile:
  """Build a tile-level QK -> online-softmax -> PV primitive.

  ``qk_acc`` is the score-tile accumulator and ``pv_acc`` is the output
  accumulator.  The caller owns the online max/sum-exp update (including the
  rescaling of ``pv_acc``); keeping that state explicit makes this usable for
  fp16 and non-fp16 routes without duplicating the scheduler path.
  """
  qk = shaped_wmma(q_frag, k_frag, qk_acc, dims=dims, device=device, threads=threads,
                   dtype_out=dtype_out)
  # The default preserves the original declarative contract. Primitive
  # callers may opt into the mathematically complete tile update: normalize
  # each score tile against the running m/l state before feeding PV. This is
  # register-only and never creates a score/probability buffer.
  weights = None
  pv_input = qk
  if normalize:
    block_m = qk.max(axis=-1, keepdim=True)
    new_m = m.maximum(block_m)
    corr = (m - new_m).exp()
    probs = (qk - new_m).exp()
    block_l = probs.sum(axis=-1, keepdim=True)
    new_l = l * corr + block_l
    weights = probs / new_l
    pv_input = weights
  pv = shaped_wmma(pv_input, v_frag, pv_acc, dims=dims, device=device, threads=threads,
                   dtype_out=dtype_out)
  tile = OnlineSoftmaxTile(qk=qk, pv=pv, m=m, l=l, acc=pv, weights=weights)
  tile.validate()
  return tile


def shaped_wmma(a_frag:UOp, b_frag:UOp, acc_frag:UOp, *, dims:tuple[int, int, int],
                device:str, threads:int, dtype_out:DType|None=None) -> UOp:
  """Construct a declarative SHAPED_WMMA tensor-graph node.

  The rangeify pass owns lowering this to Ops.WMMA. Callers must pass already-shaped per-thread fragments; this helper
  exists so route code does not construct route-local Ops.WMMA or duplicate the SHAPED_WMMA argument convention.
  """
  return UOp(Ops.SHAPED_WMMA, dtype_out or acc_frag.dtype.scalar(), (a_frag, b_frag, acc_frag),
             arg=(dims, device, threads))
