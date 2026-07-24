"""Online-softmax primitives."""
from __future__ import annotations

from typing import NamedTuple
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, RowSoftmaxRepackSpec, AMDRowSoftmaxRepackSpec, AMDRowSoftmaxSlotSpec, AMDPVCLaneSpec

def row_softmax_lds_repack(score: UOp, m: UOp, l: UOp, *,
                           spec: RowSoftmaxRepackSpec|None = None) -> UOp:
  """Construct the typed scheduler-only QK-C -> PV-A bridge.

  No reshape, state broadcast, LDS allocation, or barrier is synthesized here.
  The descriptor records those mandatory lowering semantics, while all live
  score/state dependencies remain normal UOp sources. Unsupported geometry or
  dtype fails before a backend can observe the node.
  """
  spec = RowSoftmaxRepackSpec() if spec is None else spec
  spec.validate()
  if score.shape != spec.fragment_shape or score.dtype.base != dtypes.float32:
    raise ValueError("row-softmax repack requires one fp32 16x16 QK-C fragment")
  if any(x.shape != spec.state_shape or x.dtype.base != dtypes.float32 for x in (m, l)):
    raise ValueError("row-softmax repack requires fp32 16x1 m/l row state")
  return UOp(Ops.ROW_SOFTMAX_REPACK, dtypes.half, (score, m, l), arg=spec)

def amd_gfx1100_row_softmax_repack(score: UOp, m: UOp, l: UOp, *,
                                   spec: AMDRowSoftmaxRepackSpec|None = None, kv_tile:UOp|None=None, grid_id:UOp|None=None) -> UOp:
  """Build the exact native wave32 QK-C -> PV-A scheduler carrier.

  ``score`` is one physical QK WMMA C fragment owned by a wave lane. ``m``
  and ``l`` are the scalar row-state values for that lane. No target, lane,
  LDS, barrier, or reload property is inferred by this primitive.
  """
  spec = AMDRowSoftmaxRepackSpec() if spec is None else spec
  spec.validate()
  if score.dtype != dtypes.float32.vec(8) or score.shape != (8,):
    raise ValueError("gfx1100 row-softmax repack requires native QK-C float.vec(8)")
  state_dt = dtypes.float32 if spec.mode == "legacy_normalized" else dtypes.float32.vec(8)
  state_shape = () if spec.mode == "legacy_normalized" else (8,)
  if any(x.dtype != state_dt or x.shape != state_shape for x in (m, l)):
    raise ValueError(f"gfx1100 {spec.mode} repack requires exact {state_dt} m/l row state")
  if (kv_tile is not None) != spec.dynamic_kv_v1 or (kv_tile is not None and kv_tile.op is not Ops.RANGE):
    raise ValueError("dynamic row-softmax repack requires its exact RANGE tile source")
  if (grid_id is not None) != (spec.grid is not None): raise ValueError("grid repack requires its exact group source")
  owner = UOp(Ops.AMD_ROW_SOFTMAX_REPACK, dtypes.half.vec(16), (score, m, l)+(() if kv_tile is None else (kv_tile,))+(() if grid_id is None else (grid_id,)), arg=spec)
  return UOp(Ops.AMD_ROW_SOFTMAX_SLOT, dtypes.half.vec(16), (owner,), arg=AMDRowSoftmaxSlotSpec(slot=0))

def amd_gfx1100_row_softmax_state(score:UOp, m:UOp, l:UOp, *, spec:AMDRowSoftmaxRepackSpec|None=None,
                                  kv_tile:UOp|None=None, grid_id:UOp|None=None) -> tuple[UOp, UOp, UOp, UOp]:
  """Return typed views of one native repack execution: P, new_m, new_l, alpha."""
  spec = AMDRowSoftmaxRepackSpec(mode="stateful_unnormalized_v1") if spec is None else spec
  if spec.mode not in {"stateful_unnormalized_v1", "loop_state_v1"}: raise ValueError("native state projections require a stateful native mode")
  p = amd_gfx1100_row_softmax_repack(score, m, l, spec=spec, kv_tile=kv_tile, grid_id=grid_id)
  owner = p.src[0]
  return (p, *(UOp(Ops.AMD_ROW_SOFTMAX_SLOT, dtypes.float.vec(8), (owner,), arg=AMDRowSoftmaxSlotSpec(slot=i)) for i in range(1, 4)))

def amd_gfx1100_row_softmax_initial(score:UOp, *, spec:AMDRowSoftmaxRepackSpec) -> tuple[UOp, UOp, UOp, UOp]:
  spec.validate()
  if spec.mode != "initial_state_v1" or score.dtype != dtypes.float.vec(8): raise ValueError("invalid native initial-state repack")
  owner=UOp(Ops.AMD_ROW_SOFTMAX_REPACK,dtypes.half.vec(16),(score,),arg=spec)
  return (UOp(Ops.AMD_ROW_SOFTMAX_SLOT,dtypes.half.vec(16),(owner,),arg=AMDRowSoftmaxSlotSpec(slot=0)),
    *(UOp(Ops.AMD_ROW_SOFTMAX_SLOT,dtypes.float.vec(8),(owner,),arg=AMDRowSoftmaxSlotSpec(slot=i)) for i in range(1,4)))

class OnlineSoftmaxBlockTransition(NamedTuple):
  """Typed online-softmax state at one completed KV tile boundary."""
  new_m: UOp
  alpha: UOp
  probability_scale: UOp
  pv_c: UOp
  new_l: UOp
  new_acc: UOp

def online_softmax_block_transition(old_m:UOp, old_l:UOp, old_acc:UOp,
                                    block_m:UOp, block_l:UOp, block_acc:UOp) -> OnlineSoftmaxBlockTransition:
  """Merge one completed score/PV tile into resident fp32 state.

  ``block_acc`` is PV over probabilities relative to ``block_m``. The returned
  ``pv_c`` is the exact C seed for a subsequent PV contraction, while
  ``probability_scale`` is the multiplier applied to that tile's P fragment.
  """
  if any(x.dtype != dtypes.float or x.shape != () for x in (old_m, old_l, block_m, block_l)):
    raise ValueError("online softmax block transition requires scalar fp32 m/l state")
  if old_acc.dtype != dtypes.float.vec(8) or block_acc.dtype != dtypes.float.vec(8):
    raise ValueError("online softmax block transition requires native float.vec(8) accumulators")
  new_m = old_m.alu(Ops.MAX, block_m)
  log2e = UOp.const(dtypes.float, 1.4426950408889634)
  alpha = (old_m-new_m).alu(Ops.MUL, log2e).exp2()
  probability_scale = (block_m-new_m).alu(Ops.MUL, log2e).exp2()
  pv_c = old_acc.alu(Ops.MUL, alpha.broadcast(8))
  new_l = old_l.alu(Ops.MUL, alpha).alu(Ops.ADD, block_l.alu(Ops.MUL, probability_scale))
  new_acc = pv_c.alu(Ops.ADD, block_acc.alu(Ops.MUL, probability_scale.broadcast(8)))
  return OnlineSoftmaxBlockTransition(new_m, alpha, probability_scale, pv_c, new_l, new_acc)

def amd_gfx1100_pv_c_lane(acc:UOp, e:int, *, spec:AMDPVCLaneSpec|None=None) -> UOp:
  spec = AMDPVCLaneSpec() if spec is None else spec
  spec.validate()
  if acc.dtype != dtypes.float.vec(8) or not isinstance(e, int) or not 0 <= e < 8:
    raise ValueError("PV-C lane projection requires float.vec(8) and e in [0,8)")
  return UOp(Ops.AMD_PV_C_LANE, dtypes.float, (acc,), arg=spec._replace(element=e))

def amd_gfx1100_broadcast_row_state(state:UOp, lane:UOp) -> UOp:
  if state.dtype != dtypes.float or state.shape != () or lane.dtype.scalar() not in dtypes.ints+(dtypes.weakint,):
    raise ValueError("gfx1100 row-state broadcast requires scalar fp32 state and integer lane")
  addr = lane.cast(dtypes.int).alu(Ops.AND, UOp.const(dtypes.int, 16)).alu(Ops.MUL, UOp.const(dtypes.int, 4))
  return UOp(Ops.CUSTOMI, dtypes.float, (addr, state), "bpermute")
