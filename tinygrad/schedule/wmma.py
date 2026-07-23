"""Reusable WMMA authoring helpers for scheduler-owned generated kernels."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple
from tinygrad.dtype import DType, dtypes, PtrDType, AddrSpace
from tinygrad.uop.ops import Ops, UOp
from tinygrad.uop.ops import CompositeReduce, CompositeTileCarrier, TileGatherSpec, RowSoftmaxRepackSpec, AMDRowSoftmaxRepackSpec, AMDRowSoftmaxSlotSpec, AMDPVCLaneSpec, Ops

def grouped_tile_load(source: UOp, spec: TileGatherSpec, *indices: UOp) -> UOp:
  """Build an ownership-preserving INDEX/LOAD tile carrier.

  ``indices`` are supplied by the scheduler for exactly ``source_axes``;
  this helper never invents, flattens, or broadcasts an index.  The resulting
  LOAD is wrapped in TILE_GATHER so the later shaped-fragment lowering can
  consume the explicit axis/lane ABI.
  """
  spec.validate()
  if len(indices) != len(spec.source_axes):
    raise ValueError("grouped tile load requires one index per owned source axis")
  if source.shape is None or any(a < 0 or a >= len(source.shape) for a in spec.source_axes):
    raise ValueError("tile gather source axes are out of range")
  if any(not dtypes.is_int(i.dtype.base) or i.shape not in ((), (1,),
          (spec.fragment_shape[t],)) for i, t in zip(indices, spec.tile_axes)):
    raise ValueError("tile gather indices must be scalar or lane-vector integer UOps")
  indexed = source.index(*indices)
  loaded = indexed.load()
  # The load must already have the logical fragment shape; no reshape belongs
  # in this primitive, because that would erase lane ownership.
  if loaded.shape != spec.fragment_shape:
    raise ValueError("grouped tile load does not produce the declared fragment shape")
  return tile_gather(loaded, spec)

def tile_gather(source: UOp, spec: TileGatherSpec) -> UOp:
  """Create an opt-in ownership-preserving tile carrier.

  This is deliberately scheduler-only: it records how q/kv/hd axes map into
  a shaped fragment but performs no flattening or backend lowering.
  """
  spec.validate()
  if source.shape is None or any(a < 0 or a >= len(source.shape) for a in spec.source_axes):
    raise ValueError("tile gather source axes are out of range")
  if spec.base_offsets and any(o >= source.shape[a] for o, a in zip(spec.base_offsets, spec.source_axes)):
    raise ValueError("tile gather base offset is outside source shape")
  return UOp(Ops.TILE_GATHER, source.dtype, (source,), arg=spec)

def build_owned_fragment_index_map(source_shape: tuple[int, ...], spec: TileGatherSpec) -> tuple[tuple[int, ...], ...]:
  """Build concrete source coordinates for one owned logical 16x16 tile.

  This is a renderer-neutral primitive: it does not reshape, broadcast, or
  infer lanes. ``source_axes`` explicitly names the source dimensions owned by
  the two fragment axes in ``tile_axes``; all remaining dimensions are fixed
  by ``base_offsets`` (zero when omitted).  Only exact 16x16 tiles are
  accepted, and value tiles must expose an Hd source axis with a compatible
  lane group.
  """
  spec.validate()
  if tuple(spec.fragment_shape) != (16, 16):
    raise ValueError("owned fragment builder requires an exact 16x16 fragment")
  if len(source_shape) == 0 or any(not isinstance(x, int) or x <= 0 for x in source_shape):
    raise ValueError("source shape must be a positive concrete rank")
  if any(a >= len(source_shape) for a in spec.source_axes):
    raise ValueError("tile gather source axes are out of range")
  offsets = spec.base_offsets or (0,) * len(spec.source_axes)
  if len(offsets) != len(spec.source_axes):
    raise ValueError("tile gather base offsets must match source axes")
  # A value tile's second fragment axis is the logical Hd lane.  Keep this
  # explicit rather than guessing a packed/vector layout.
  if spec.role == "value" and 1 not in spec.tile_axes:
    raise ValueError("value fragment must map Hd ownership to tile axis 1")
  if 16 % spec.lane_group:
    raise ValueError("lane group must divide the 16-lane fragment")
  axis_to_tile = {axis: tile for axis, tile in zip(spec.source_axes, spec.tile_axes)}
  out = []
  for row in range(16):
    for col in range(16):
      coord = [0] * len(source_shape)
      for axis in range(len(source_shape)):
        if axis in spec.source_axes:
          i = spec.source_axes.index(axis)
          tile_axis = axis_to_tile[axis]
          coord[axis] = offsets[i] + (row if tile_axis == 0 else col)
        elif spec.base_offsets:
          coord[axis] = 0
        if coord[axis] >= source_shape[axis]:
          raise ValueError("owned fragment coordinate exceeds source shape")
      out.append(tuple(coord))
  return tuple(out)

def construct_hd16_tile_carriers(score: UOp, value: UOp, acc: UOp, *,
                                 batch: int = 1, heads: int = 1,
                                 provenance: tuple[str, ...] = ()) -> tuple[UOp, UOp, UOp]:
  """Construct the first exact QK/PV/acc tile handoff.

  This is intentionally limited to the geometry whose ownership map is
  proven: score ``(B,H,16,16,1)``, value ``(B,H,1,16,16)``, and accumulator
  ``(B,H,16,16)``.  The constructor does not reshape or infer lanes; callers
  must provide already-shaped logical tile sources.  It is therefore safe to
  use as a scheduler primitive while broader Hd packing remains fail-closed.
  """
  if any(x.shape is None for x in (score, value, acc)):
    raise ValueError("Hd16 tile carriers require concrete source shapes")
  if score.shape != (batch, heads, 16, 16, 1):
    raise ValueError("Hd16 score carrier requires (B,H,16,16,1) ownership")
  if value.shape != (batch, heads, 1, 16, 16):
    raise ValueError("Hd16 value carrier requires (B,H,1,16,16) ownership")
  if acc.shape != (batch, heads, 16, 16):
    raise ValueError("Hd16 accumulator carrier requires (B,H,16,16) ownership")
  score_spec = TileGatherSpec("score", (16, 16), (2, 3), (0, 1))
  value_spec = TileGatherSpec("value", (16, 16), (3, 4), (0, 1))
  acc_spec = TileGatherSpec("acc", (16, 16), (2, 3), (0, 1))
  build_owned_fragment_index_map(score.shape, score_spec)
  build_owned_fragment_index_map(value.shape, value_spec)
  # Accumulator ownership follows query/Hd lanes; Hd=16 makes this exact.
  return (tile_gather(score, score_spec), tile_gather(value, value_spec),
          tile_gather(acc, acc_spec))

def composite_reduce_hd16_carriers(red: UOp) -> tuple[UOp, UOp, UOp] | None:
  """Return an owned QK/PV/acc carrier triple for one exact composite REDUCE.

  This is an opt-in scheduler primitive, not a production admission hook.  It
  deliberately requires rankful sources with the proven ``(B,H,16,16,1)`` /
  ``(B,H,1,16,16)`` / ``(B,H,16,16)`` ownership map.  Any ordinary reduction,
  missing metadata, or different geometry returns ``None`` so the existing
  scalar online-softmax reducer remains authoritative.
  """
  if red.op is not Ops.REDUCE or not red.arg or not isinstance(red.arg[0], CompositeReduce):
    return None
  comp = red.arg[0]
  # Never let a vector-typed logical reduction enter this experimental handoff:
  # the scalar online-softmax reducer is still the only production-safe path.
  if red.dtype.count != 1 or red.src[0].dtype.count != 1:
    return None
  if len(red.src[0].shape or ()) != 5 or tuple(red.arg[1] or ()) != (3,):
    return None
  carrier = comp.tile_carrier
  if carrier is None:
    return None
  try:
    carrier.validate()
  except ValueError:
    return None
  if carrier.score_shape != (16, 16, 16) or carrier.value_shape != (16, 16, 16) or carrier.output_shape != (16, 16, 16):
    return None
  # The reducer's primary source is the score tensor; the declared auxiliary
  # source is V.  Accumulator shape is taken from slot_shapes, never inferred
  # from a vector dtype.
  score = red.src[0]
  aux = tuple(x for x in red.src[1:] if x.op is not Ops.RANGE)
  if len(aux) != 1 or not comp.slot_shapes or len(comp.slot_shapes) < 3:
    return None
  value = aux[0]
  if value.dtype.count != 1 or len(value.shape or ()) != 5:
    return None
  acc_shape = comp.slot_shapes[2]
  if acc_shape is None:
    return None
  acc = UOp.placeholder(acc_shape, comp.slots[2].dtype, -1)
  try:
    return construct_hd16_tile_carriers(score, value, acc,
                                        batch=score.shape[0], heads=score.shape[1])
  except (AttributeError, IndexError, TypeError, ValueError):
    return None

def lower_tile_gather(source: UOp, *, role: str, dtype: DType) -> UOp:
  """Resolve a TILE_GATHER only when an upstream pass already shaped it.

  No flattening, broadcast, or index synthesis is permitted here.  This
  fail-closed resolver is the handoff point for the future grouped LOAD pass.
  """
  if source.op is not Ops.TILE_GATHER:
    raise ValueError("expected TILE_GATHER carrier")
  spec = source.arg
  spec.validate()
  declared_role = "v" if spec.role == "value" else spec.role
  if declared_role != role or source.shape != spec.fragment_shape:
    raise ValueError("tile gather is not a shaped fragment")
  # A scheduler-owned carrier may retain the rankful source that established
  # ownership (the Hd16 constructor does this). Keep the opaque carrier at
  # the WMMA boundary rather than flattening that source prematurely.
  if source.src[0].shape != spec.fragment_shape:
    allowed = ((len(source.src[0].shape) == 5 and spec.role == "score" and source.src[0].shape[-3:] == (16, 16, 1)) or
               (len(source.src[0].shape) == 5 and spec.role == "value" and source.src[0].shape[-3:] == (1, 16, 16)) or
               (len(source.src[0].shape) == 4 and spec.role == "acc" and source.src[0].shape[-2:] == (16, 16)))
    if not allowed:
      raise ValueError("tile gather is not a shaped fragment")
    return lower_attached_tile_gather(source, role=role, dtype=dtype)
  return adapt_wmma_fragment(source, role=role, dtype=dtype, shape=spec.fragment_shape)

def lower_attached_tile_gather(source: UOp, *, role: str, dtype: DType) -> UOp:
  """Materialize one proven rankful carrier as an exact logical fragment.

  The coordinate map is the ownership proof.  Only the three bounded Hd=16
  layouts are accepted; unsupported packing remains opaque so the scalar
  reducer stays authoritative.  This helper is intentionally not installed
  in the production rangeify matcher.
  """
  if source.op is not Ops.TILE_GATHER:
    raise ValueError("expected TILE_GATHER carrier")
  spec = source.arg
  spec.validate()
  if spec.role != ("value" if role == "v" else role):
    raise ValueError("tile gather role mismatch")
  if source.shape != spec.fragment_shape:
    raise ValueError("tile gather is not a shaped fragment")
  base = source.src[0]
  if base.shape == spec.fragment_shape:
    return adapt_wmma_fragment(source, role=role, dtype=dtype, shape=spec.fragment_shape)
  allowed = ((spec.role == "score" and base.shape[-3:] == (16, 16, 1)) or
             (spec.role == "value" and base.shape[-3:] == (1, 16, 16)) or
             (spec.role == "acc" and base.shape[-2:] == (16, 16)))
  if not allowed:
    raise ValueError("tile gather source geometry is unsupported")
  if spec.role in ("score", "value") and len(base.shape) < 5:
    raise ValueError("tile gather source rank is unsupported")
  build_owned_fragment_index_map(base.shape, spec)
  fragment = base.reshape(spec.fragment_shape)
  # Keep the original axis contract in ``spec`` even though the materialized
  # child is now rank-2; reusing ``tile_gather`` would (correctly) reject the
  # detached source axes against the new rank.
  gathered = UOp(Ops.TILE_GATHER, fragment.dtype, (fragment,), arg=spec)
  return adapt_wmma_fragment(gathered, role=role, dtype=dtype, shape=spec.fragment_shape)

def emit_tile_gather_shaped_wmma(a_frag: UOp, b_frag: UOp, acc_frag: UOp, *,
                                 roles: tuple[str, str, str] = ("score", "value", "acc"),
                                 dims: tuple[int, int, int] = (16, 16, 16),
                                 device: str = "AMD", threads: int = 32,
                                 dtype_out: DType | None = None) -> UOp:
  """Emit an existing SHAPED_WMMA node from exact tile carriers only.

  This is the deliberately small handoff for scheduler-owned fragments.  The
  carriers must already be logical 16x16 tiles; no reshape, broadcast, index
  synthesis, or lane inference is performed here.  Consequently malformed
  composite sources fail before backend code generation.
  """
  if len(roles) != 3:
    raise ValueError("tile WMMA emitter requires three carrier roles")
  carriers = (a_frag, b_frag, acc_frag)
  abi_roles = tuple("v" if r == "value" else r for r in roles)
  lowered = tuple(lower_tile_gather(x, role=r, dtype=x.dtype.base)
                  for x, r in zip(carriers, abi_roles))
  if any(x.shape != (16, 16) for x in lowered):
    raise ValueError("tile WMMA emitter requires exact 16x16 fragments")
  return shaped_wmma(*lowered, dims=dims, device=device, threads=threads, dtype_out=dtype_out)

def emit_hd16_dual_tile_wmma(score: UOp, value: UOp, acc: UOp, *,
                             dims: tuple[int, int, int] = (16, 16, 16),
                             device: str = "AMD", threads: int = 32,
                             dtype_out: DType | None = None) -> tuple[UOp, UOp]:
  """Route one proven Hd=16 carrier triple into separate QK/PV nodes.

  This is an authoring primitive only.  ``score`` is retained as the QK-side
  owned tile and ``value``/``acc`` as the PV-side operands; both nodes share
  the exact carrier validation path and no lane packing is inferred here.
  Production admission remains fail-closed until source and ISA evidence
  exists for the resulting fused loop.
  """
  qk = emit_tile_gather_shaped_wmma(score, score, acc, roles=("score", "score", "acc"),
                                   dims=dims, device=device, threads=threads, dtype_out=dtype_out)
  pv = emit_tile_gather_shaped_wmma(score, value, acc, roles=("score", "value", "acc"),
                                   dims=dims, device=device, threads=threads, dtype_out=dtype_out)
  return qk, pv

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
                                   spec: AMDRowSoftmaxRepackSpec|None = None, kv_tile:UOp|None=None) -> UOp:
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
  owner = UOp(Ops.AMD_ROW_SOFTMAX_REPACK, dtypes.half.vec(16), (score, m, l)+(() if kv_tile is None else (kv_tile,)), arg=spec)
  return UOp(Ops.AMD_ROW_SOFTMAX_SLOT, dtypes.half.vec(16), (owner,), arg=AMDRowSoftmaxSlotSpec(slot=0))

def amd_gfx1100_row_softmax_state(score:UOp, m:UOp, l:UOp, *, spec:AMDRowSoftmaxRepackSpec|None=None,
                                  kv_tile:UOp|None=None) -> tuple[UOp, UOp, UOp, UOp]:
  """Return typed views of one native repack execution: P, new_m, new_l, alpha."""
  spec = AMDRowSoftmaxRepackSpec(mode="stateful_unnormalized_v1") if spec is None else spec
  if spec.mode not in {"stateful_unnormalized_v1", "loop_state_v1"}: raise ValueError("native state projections require a stateful native mode")
  p = amd_gfx1100_row_softmax_repack(score, m, l, spec=spec, kv_tile=kv_tile)
  owner = p.src[0]
  return (p, *(UOp(Ops.AMD_ROW_SOFTMAX_SLOT, dtypes.float.vec(8), (owner,), arg=AMDRowSoftmaxSlotSpec(slot=i)) for i in range(1, 4)))

def amd_gfx1100_row_softmax_initial(score:UOp, *, spec:AMDRowSoftmaxRepackSpec) -> tuple[UOp, UOp, UOp, UOp]:
  spec.validate()
  if spec.mode != "initial_state_v1" or score.dtype != dtypes.float.vec(8): raise ValueError("invalid native initial-state repack")
  owner=UOp(Ops.AMD_ROW_SOFTMAX_REPACK,dtypes.half.vec(16),(score,),arg=spec)
  return (UOp(Ops.AMD_ROW_SOFTMAX_SLOT,dtypes.half.vec(16),(owner,),arg=AMDRowSoftmaxSlotSpec(slot=0)),
    *(UOp(Ops.AMD_ROW_SOFTMAX_SLOT,dtypes.float.vec(8),(owner,),arg=AMDRowSoftmaxSlotSpec(slot=i)) for i in range(1,4)))

def amd_gfx1100_q16_attention(q:UOp, k:UOp, v:UOp, out:UOp, *, scale:float, kernel_info,
                             causal:bool=False, valid_kv:int=16, query_start:int=0) -> UOp:
  """Build the exact live-owner q16 native attention kernel graph."""
  owners = (q, k, v, out)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype, PtrDType) or x.ptrdtype.size != 256 for x in owners):
    raise ValueError("q16 native attention requires four live 256-element PARAM owners")
  if tuple(x.ptrdtype.base for x in owners) != (dtypes.half,)*4:
    raise ValueError("q16 native attention requires fp16 Q/K/V/output owners")
  if tuple(x.arg.slot for x in owners) != (1, 2, 3, 0):
    raise ValueError("q16 native attention requires PARAM slots Q=1 K=2 V=3 output=0")
  if not isinstance(scale, float) or not math.isfinite(scale) or scale <= 0:
    raise ValueError("q16 native attention requires one positive finite score scale")
  if not isinstance(causal,bool) or not isinstance(valid_kv,int) or not 0 <= valid_kv <= 16 or not isinstance(query_start,int):
    raise ValueError("q16 native attention requires typed causal/KV validity metadata")
  lane = UOp.special(32, "lidx0")
  col, halfwave = lane & 15, lane >> 4
  qfrag = UOp(Ops.STACK, dtypes.half.vec(16), tuple(q.index(col*16+i).load() for i in range(16)),
    tag=("amd_gfx1100_fragment_load_v1","Q",0,q,lane,col))
  kfrag = UOp(Ops.STACK, dtypes.half.vec(16), tuple(k.index(col*16+i).load() for i in range(16)),
    tag=("amd_gfx1100_fragment_load_v1","K",0,k,lane,col))
  zero = UOp.const(dtypes.float.vec(8), (0.0,)*8)
  # A/B are already physical half16 fragments and must not be permuted by
  # logical upcast-axis rewriting. C retains its exact three binary axes so
  # eight native accumulator lanes can be projected with GEP.
  fragment_axes = ((), (), tuple((-120-i, 2) for i in range(3)))
  warg = ("WMMA_16_16_16_half_float", (16,16,16), dtypes.half, dtypes.float, "AMD:gfx1100", 32, fragment_axes, ())
  qk = UOp(Ops.WMMA, dtypes.float.vec(8), (qfrag, kfrag, zero), warg)
  weights,sm,sl,_ = amd_gfx1100_row_softmax_initial(qk, spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),
    mode="initial_state_v1",validity_mode="causal_v1" if causal else "all_v1",query_start=query_start,
    kv_start=0,valid_kv=valid_kv))
  vfrag = UOp(Ops.STACK, dtypes.half.vec(16), tuple(v.index(i*16+col).load() for i in range(16)),
    tag=("amd_gfx1100_fragment_load_v1","V",0,v,lane,col))
  pv = UOp(Ops.WMMA, dtypes.float.vec(8), (weights, vfrag, zero), warg)
  stores=[]
  for e in range(8):
    value=pv.gep(e)
    if stores: value=value.bitcast(dtypes.uint).after(UOp.group(stores[-1])).bitcast(dtypes.float)
    den=sl.gep(e); recip=den.ne(UOp.const(dtypes.float,0)).where(UOp.const(dtypes.float,1)/den,UOp.const(dtypes.float,0))
    stores.append(out.index((UOp.const(dtypes.weakint,2*e)+halfwave)*16+col).store((value*recip).cast(dtypes.half)))
  return UOp.sink(*stores, arg=kernel_info)

def amd_gfx1100_q16_kv32_attention(q:UOp, k:UOp, v:UOp, out:UOp, *, scale:float, kernel_info) -> UOp:
  owners=(q,k,v,out)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype,PtrDType) for x in owners): raise ValueError("q16-kv32 requires PARAM owners")
  if tuple(x.arg.slot for x in owners)!=(1,2,3,0) or tuple(x.ptrdtype.size for x in owners)!=(256,512,512,256):
    raise ValueError("q16-kv32 requires Q1/K2/V3/out0 sized 256/512/512/256")
  if tuple(x.ptrdtype.base for x in owners)!=(dtypes.half,)*4 or not isinstance(scale,float) or not math.isfinite(scale) or scale<=0:
    raise ValueError("q16-kv32 requires fp16 owners and positive finite scale")
  lane=UOp.special(32,"lidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15)); half=lane.alu(Ops.SHR,UOp.const(dtypes.weakint,4))
  zero=UOp.const(dtypes.float.vec(8),(0.0,)*8); axes=((),(),tuple((-120-i,2) for i in range(3)))
  warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,())
  sm=UOp.const(dtypes.float.vec(8),(-float("inf"),)*8); sl=UOp.const(dtypes.float.vec(8),(0.0,)*8); acc=zero
  for tile in range(2):
    base=UOp.const(dtypes.weakint,tile*256)
    q_owner,k_owner,v_owner=(q,k,v) if tile == 0 else (q.after(acc),k.after(acc),v.after(acc))
    qf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(q_owner.index(col*16+i).load() for i in range(16)),
      tag=("amd_gfx1100_fragment_load_v1","Q",tile,q,lane,col))
    kf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(k_owner.index(base+col*16+i).load() for i in range(16)),
      tag=("amd_gfx1100_fragment_load_v1","K",tile,k,lane,col))
    qk=UOp(Ops.WMMA,dtypes.float.vec(8),(qf,kf,zero),warg)
    if tile == 0:
      p,sm,sl,alpha=amd_gfx1100_row_softmax_initial(qk,
        spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="initial_state_v1"))
    else:
      p,sm,sl,alpha=amd_gfx1100_row_softmax_state(qk,sm,sl,
        spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="stateful_unnormalized_v1"))
    corrected=zero
    if tile:
      corrected=acc.alu(Ops.MUL,alpha)
    v_ready=v_owner.after(corrected if tile else p)
    vf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(v_ready.index(base+i*16+col).load() for i in range(16)),
      tag=("amd_gfx1100_fragment_load_v1","V",tile,v,lane,col))
    acc=UOp(Ops.WMMA,dtypes.float.vec(8),(p,vf,corrected),warg)
  stores=[]
  for e in range(8):
    value=acc.gep(e)
    if stores: value=value.bitcast(dtypes.uint).after(UOp.group(stores[-1])).bitcast(dtypes.float)
    den=sl.gep(e)
    recip=den.ne(UOp.const(dtypes.float,0)).where(UOp.const(dtypes.float,1)/den,UOp.const(dtypes.float,0))
    dst=out.index((UOp.const(dtypes.weakint,2*e)+half)*16+col)
    stores.append(dst.store(value.alu(Ops.MUL,recip).cast(dtypes.half)))
  return UOp.sink(*stores,arg=kernel_info)

def amd_gfx1100_q16_kv32_hd128_attention(q:UOp, k:UOp, v:UOp, out:UOp, *, scale:float, kernel_info) -> UOp:
  """Exact B=H=1, Q=16, KV=32, Hd=128 online-softmax kernel graph."""
  owners=(q,k,v,out)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype,PtrDType) for x in owners): raise ValueError("q16-kv32-hd128 requires PARAM owners")
  if tuple(x.arg.slot for x in owners)!=(1,2,3,0) or tuple(x.ptrdtype.size for x in owners)!=(2048,4096,4096,2048):
    raise ValueError("q16-kv32-hd128 requires Q1/K2/V3/out0 sized 2048/4096/4096/2048")
  if tuple(x.ptrdtype.base for x in owners)!=(dtypes.half,)*4 or not isinstance(scale,float) or not math.isfinite(scale) or scale<=0:
    raise ValueError("q16-kv32-hd128 requires fp16 owners and positive finite scale")
  lane=UOp.special(32,"lidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15)); half=lane.alu(Ops.SHR,UOp.const(dtypes.weakint,4))
  zero=UOp.const(dtypes.float.vec(8),(0.0,)*8); axes=((),(),tuple((-120-i,2) for i in range(3)))
  warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,())
  sm=UOp.const(dtypes.float.vec(8),(-float("inf"),)*8); sl=UOp.const(dtypes.float.vec(8),(0.0,)*8)
  acc=[zero]*8
  for tile in range(2):
    qk=zero
    for hd_block in range(8):
      q_owner=q if tile == 0 else q.after(*acc); k_owner=k if tile == 0 else k.after(*acc)
      qf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(q_owner.index(col*128+hd_block*16+i).load() for i in range(16)),
        tag=("amd_gfx1100_fragment_load_hd128_v1","Q",tile,hd_block,q,lane,col))
      kbase=UOp.const(dtypes.weakint,tile*2048+hd_block*16)
      kf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(k_owner.index(kbase+col*128+i).load() for i in range(16)),
        tag=("amd_gfx1100_fragment_load_hd128_v1","K",tile,hd_block,k,lane,col))
      qk=UOp(Ops.WMMA,dtypes.float.vec(8),(qf,kf,qk),warg)
    if tile == 0:
      p,sm,sl,alpha=amd_gfx1100_row_softmax_initial(qk,
        spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="initial_state_v1"))
    else:
      p,sm,sl,alpha=amd_gfx1100_row_softmax_state(qk,sm,sl,
        spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="stateful_unnormalized_v1",kv_start=16,
          validity_mode="causal_v1",query_start=16,valid_kv=32))
    next_acc=[]
    for hd_block in range(8):
      corrected=zero if tile == 0 else acc[hd_block].alu(Ops.MUL,alpha)
      v_owner=v.after(corrected if tile else p); vbase=UOp.const(dtypes.weakint,tile*2048+hd_block*16)
      vf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(v_owner.index(vbase+i*128+col).load() for i in range(16)),
        tag=("amd_gfx1100_fragment_load_hd128_v1","V",tile,hd_block,v,lane,col))
      next_acc.append(UOp(Ops.WMMA,dtypes.float.vec(8),(p,vf,corrected),warg))
    acc=next_acc
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec
  drain=UOp(Ops.AMD_ATTENTION_OUTPUT_DRAIN,dtypes.void,(out,sl,*acc),arg=AMDAttentionOutputDrainSpec())
  return UOp.sink(drain,arg=kernel_info)

def amd_gfx1100_q16_kv64_hd128_loop_attention(q:UOp, k:UOp, v:UOp, out:UOp, *, scale:float, kernel_info,
                                               causal:bool=False, valid_kv:int=64, query_start:int|None=None) -> UOp:
  """Scheduler-only Q16/KV64/Hd128 recurrence with one runtime KV tile body.

  This deliberately has no AMD/HIP lowering yet.  The typed state and dynamic
  fragment carriers prevent it from silently falling back to the static KV32
  implementation while retaining the exact graph that a future backend must
  consume.
  """
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec, AMDLoopStateSpec, AMDPackedFragmentLoopSpec, AxisType
  owners=(q,k,v,out)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype,PtrDType) for x in owners): raise ValueError("q16-kv64-hd128 requires PARAM owners")
  if tuple(x.arg.slot for x in owners)!=(1,2,3,0) or tuple(x.ptrdtype.size for x in owners)!=(2048,8192,8192,2048):
    raise ValueError("q16-kv64-hd128 requires Q1/K2/V3/out0 sized 2048/8192/8192/2048")
  if tuple(x.ptrdtype.base for x in owners)!=(dtypes.half,)*4 or not isinstance(scale,float) or not math.isfinite(scale) or scale<=0:
    raise ValueError("q16-kv64-hd128 requires fp16 owners and positive finite scale")
  if not isinstance(valid_kv,int) or isinstance(valid_kv,bool) or not 0 <= valid_kv <= 64: raise ValueError("valid_kv must be in [0,64]")
  if query_start is None: query_start=valid_kv-q_tokens
  if not isinstance(query_start,int) or isinstance(query_start,bool): raise ValueError("query_start must be integral")
  lane=UOp.special(32,"lidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15))
  zero=UOp.const(dtypes.float.vec(8),(0.0,)*8); axes=((),(),tuple((-120-i,2) for i in range(3)))
  warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,())
  rng=UOp.range(4,9400,AxisType.REDUCE)
  mreg=UOp.placeholder((8,),dtypes.float,9401,addrspace=AddrSpace.REG)
  lreg=UOp.placeholder((8,),dtypes.float,9402,addrspace=AddrSpace.REG)
  creg=UOp.placeholder((64,),dtypes.float,9403,addrspace=AddrSpace.REG)
  state_owner=9404
  def state_write(reg, role, value, block=0, offset=0, access="write"):
    return tuple(UOp(Ops.AMD_ATTENTION_LOOP_STATE,dtypes.void,
      (reg.index(UOp.const(dtypes.weakint,offset+i)).store(value.gep(i)),),
      arg=AMDLoopStateSpec(role=role,access=access,block=block,lane=i,owner=state_owner)) for i in range(8))
  m_init=UOp.group(*state_write(mreg,"m",UOp.const(dtypes.float.vec(8),(-float("inf"),)*8),access="init"))
  l_init=UOp.group(*state_write(lreg,"l",zero,access="init"))
  c_init=UOp.group(*(x for block in range(8) for x in state_write(creg,"acc",zero,block,block*8,access="init")))
  def state_read(reg, init, role, block=0, offset=0, final=False):
    lanes=tuple(UOp(Ops.AMD_ATTENTION_LOOP_STATE,dtypes.float,
      (reg,init) if final else (reg,init,rng),
      arg=AMDLoopStateSpec(role=role,access="final_read" if final else "read",block=block,lane=i,owner=state_owner)) for i in range(8))
    return UOp(Ops.STACK,dtypes.float.vec(8),lanes)
  def fragment(owner, role, block):
    return UOp(Ops.AMD_PACKED_FRAGMENT_LOAD,dtypes.half.vec(16),(owner,lane,col,rng),arg=AMDPackedFragmentLoopSpec(role=role,head_block=block))
  old_m,old_l=state_read(mreg,m_init,"m"),state_read(lreg,l_init,"l")
  qk=zero
  for block in range(8): qk=UOp(Ops.WMMA,dtypes.float.vec(8),(fragment(q,"Q",block),fragment(k,"K",block),qk),warg)
  p,new_m,new_l,alpha=amd_gfx1100_row_softmax_state(qk,old_m,old_l,
    spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="loop_state_v1",validity_mode="causal_v1" if causal else "tail_v1",
      query_start=query_start,kv_start=-1,valid_kv=valid_kv,dynamic_kv_v1=True),kv_tile=rng)
  writes=[*state_write(mreg,"m",new_m),*state_write(lreg,"l",new_l)]
  for block in range(8):
    old_c=state_read(creg,c_init,"acc",block,block*8)
    corrected=old_c.alu(Ops.MUL,alpha)
    pv=UOp(Ops.WMMA,dtypes.float.vec(8),(p,fragment(v,"V",block),corrected),warg)
    writes.extend(state_write(creg,"acc",pv,block,block*8))
  end=UOp.group(*writes).end(rng).replace(tag=("amd_gfx1100_attention_kv64_loop_end_v1",rng))
  final_l=state_read(lreg,end,"l",final=True)
  final_c=tuple(state_read(creg,end,"acc",block,block*8,final=True) for block in range(8))
  drain=UOp(Ops.AMD_ATTENTION_OUTPUT_DRAIN,dtypes.void,(out,final_l,*final_c),arg=AMDAttentionOutputDrainSpec())
  return UOp.sink(m_init,l_init,c_init,end,drain,arg=kernel_info).replace(tag=("amd_gfx1100_q16_kv64_hd128_loop_v1",))

def amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_attention(q:UOp, k:UOp, v:UOp, out:UOp, *, scale:float, kernel_info,
                                                          causal:bool=False, valid_kv:int=64, query_start:int|None=None) -> UOp:
  """Grid-native Q32/Hq4/Hkv2/G2 attention; one wave32 per Q-head tile."""
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec, AMDAttentionGridSpec, AMDLoopStateSpec, AMDPackedFragmentLoopSpec, AxisType
  grid=AMDAttentionGridSpec(); grid.validate(); owners=(q,k,v,out)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype,PtrDType) for x in owners): raise ValueError("q32-hq4-hkv2 requires PARAM owners")
  if tuple(x.arg.slot for x in owners)!=(1,2,3,0) or tuple(x.ptrdtype.size for x in owners)!=(16384,16384,16384,16384):
    raise ValueError("q32-hq4-hkv2 requires Q1/K2/V3/out0 sized 16384")
  if tuple(x.ptrdtype.base for x in owners)!=(dtypes.half,)*4 or not isinstance(scale,float) or not math.isfinite(scale) or scale<=0:
    raise ValueError("q32-hq4-hkv2 requires fp16 owners and positive finite scale")
  if not isinstance(valid_kv,int) or isinstance(valid_kv,bool) or not 0 <= valid_kv <= 64: raise ValueError("valid_kv must be in [0,64]")
  if query_start is None: query_start=valid_kv-16
  if not isinstance(query_start,int) or isinstance(query_start,bool): raise ValueError("query_start must be integral")
  lane=UOp.special(32,"lidx0"); group=UOp.special(8,"gidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15))
  zero=UOp.const(dtypes.float.vec(8),(0.0,)*8); axes=((),(),tuple((-120-i,2) for i in range(3)))
  warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,())
  rng=UOp.range(4,9500,AxisType.REDUCE); mreg=UOp.placeholder((8,),dtypes.float,9501,addrspace=AddrSpace.REG)
  lreg=UOp.placeholder((8,),dtypes.float,9502,addrspace=AddrSpace.REG); creg=UOp.placeholder((64,),dtypes.float,9503,addrspace=AddrSpace.REG); state_owner=9504
  def state_write(reg,role,value,block=0,offset=0,access="write"):
    return tuple(UOp(Ops.AMD_ATTENTION_LOOP_STATE,dtypes.void,(reg.index(UOp.const(dtypes.weakint,offset+i)).store(value.gep(i)),),
      arg=AMDLoopStateSpec(role=role,access=access,block=block,lane=i,owner=state_owner)) for i in range(8))
  m_init=UOp.group(*state_write(mreg,"m",UOp.const(dtypes.float.vec(8),(-float("inf"),)*8),access="init")); l_init=UOp.group(*state_write(lreg,"l",zero,access="init"))
  c_init=UOp.group(*(x for block in range(8) for x in state_write(creg,"acc",zero,block,block*8,access="init")))
  def state_read(reg,init,role,block=0,offset=0,final=False):
    return UOp(Ops.STACK,dtypes.float.vec(8),tuple(UOp(Ops.AMD_ATTENTION_LOOP_STATE,dtypes.float,(reg,init) if final else (reg,init,rng),
      arg=AMDLoopStateSpec(role=role,access="final_read" if final else "read",block=block,lane=i,owner=state_owner)) for i in range(8)))
  def fragment(owner,role,block):
    return UOp(Ops.AMD_PACKED_FRAGMENT_LOAD,dtypes.half.vec(16),(owner,lane,col,rng,group),arg=AMDPackedFragmentLoopSpec(role=role,head_block=block,grid=grid))
  old_m,old_l=state_read(mreg,m_init,"m"),state_read(lreg,l_init,"l"); qk=zero
  for block in range(8): qk=UOp(Ops.WMMA,dtypes.float.vec(8),(fragment(q,"Q",block),fragment(k,"K",block),qk),warg)
  p,new_m,new_l,alpha=amd_gfx1100_row_softmax_state(qk,old_m,old_l,spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="loop_state_v1",
    validity_mode="causal_v1" if causal else "tail_v1",query_start=query_start,kv_start=-1,valid_kv=valid_kv,dynamic_kv_v1=True),kv_tile=rng)
  writes=[*state_write(mreg,"m",new_m),*state_write(lreg,"l",new_l)]
  for block in range(8):
    old_c=state_read(creg,c_init,"acc",block,block*8); pv=UOp(Ops.WMMA,dtypes.float.vec(8),(p,fragment(v,"V",block),old_c.alu(Ops.MUL,alpha)),warg)
    writes.extend(state_write(creg,"acc",pv,block,block*8))
  end=UOp.group(*writes).end(rng).replace(tag=("amd_gfx1100_attention_grid_kv64_loop_end_v1",rng)); final_l=state_read(lreg,end,"l",final=True)
  final_c=tuple(state_read(creg,end,"acc",block,block*8,final=True) for block in range(8))
  drain=UOp(Ops.AMD_ATTENTION_OUTPUT_DRAIN,dtypes.void,(out,group,final_l,*final_c),arg=AMDAttentionOutputDrainSpec(grid=grid))
  return UOp.sink(m_init,l_init,c_init,end,drain,arg=kernel_info).replace(tag=("amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_v1",))

def amd_gfx1100_q16_grid_hd128_loop_attention(q:UOp,k:UOp,v:UOp,out:UOp,*,q_tokens:int,q_heads:int,kv_heads:int,kv_tokens:int,scale:float,kernel_info,causal:bool=False,valid_kv:int|None=None,query_start:int|None=None)->UOp:
  """Fixed 16-WMMA attention wave with compile-time model geometry."""
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec, AMDAttentionGridSpec, AMDLoopStateSpec, AMDPackedFragmentLoopSpec, AxisType
  grid=AMDAttentionGridSpec(q_tokens=q_tokens,q_heads=q_heads,kv_heads=kv_heads,group_ratio=q_heads//kv_heads,kv_tokens=kv_tokens); grid.validate()
  owners=(q,k,v,out); sizes=(q_heads*q_tokens*128,kv_heads*kv_tokens*128,kv_heads*kv_tokens*128,q_heads*q_tokens*128)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype,PtrDType) for x in owners) or tuple(x.arg.slot for x in owners)!=(1,2,3,0) or tuple(x.ptrdtype.size for x in owners)!=sizes: raise ValueError(f"grid loop requires Q1/K2/V3/out0 sized {sizes}")
  if tuple(x.ptrdtype.base for x in owners)!=(dtypes.half,)*4 or not isinstance(scale,float) or not math.isfinite(scale) or scale<=0: raise ValueError("grid loop requires fp16 and finite scale")
  valid_kv=kv_tokens if valid_kv is None else valid_kv
  if not isinstance(valid_kv,int) or isinstance(valid_kv,bool) or not 0<=valid_kv<=kv_tokens: raise ValueError("valid_kv is outside KV geometry")
  if query_start is None: query_start=valid_kv-16
  lane=UOp.special(32,"lidx0"); group=UOp.special(q_heads*grid.q_tiles,"gidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15)); zero=UOp.const(dtypes.float.vec(8),(0.0,)*8); axes=((),(),tuple((-120-i,2) for i in range(3))); warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,())
  rng=UOp.range((kv_tokens+15)//16,9600,AxisType.REDUCE); mreg=UOp.placeholder((8,),dtypes.float,9601,addrspace=AddrSpace.REG); lreg=UOp.placeholder((8,),dtypes.float,9602,addrspace=AddrSpace.REG); creg=UOp.placeholder((64,),dtypes.float,9603,addrspace=AddrSpace.REG)
  def wr(reg,role,value,b=0,o=0,a="write"): return tuple(UOp(Ops.AMD_ATTENTION_LOOP_STATE,dtypes.void,(reg.index(UOp.const(dtypes.weakint,o+i)).store(value.gep(i)),),arg=AMDLoopStateSpec(role=role,access=a,block=b,lane=i,owner=9604)) for i in range(8))
  mi=UOp.group(*wr(mreg,"m",UOp.const(dtypes.float.vec(8),(-float("inf"),)*8),a="init")); li=UOp.group(*wr(lreg,"l",zero,a="init")); ci=UOp.group(*(x for b in range(8) for x in wr(creg,"acc",zero,b,b*8,"init")))
  def rd(reg,init,role,b=0,o=0,final=False): return UOp(Ops.STACK,dtypes.float.vec(8),tuple(UOp(Ops.AMD_ATTENTION_LOOP_STATE,dtypes.float,(reg,init) if final else (reg,init,rng),arg=AMDLoopStateSpec(role=role,access="final_read" if final else "read",block=b,lane=i,owner=9604)) for i in range(8)))
  def fr(owner,role,b): return UOp(Ops.AMD_PACKED_FRAGMENT_LOAD,dtypes.half.vec(16),(owner,lane,col,rng,group),arg=AMDPackedFragmentLoopSpec(role=role,head_block=b,grid=grid))
  om,ol=rd(mreg,mi,"m"),rd(lreg,li,"l"); qk=zero
  for b in range(8): qk=UOp(Ops.WMMA,dtypes.float.vec(8),(fr(q,"Q",b),fr(k,"K",b),qk),warg)
  p,nm,nl,alpha=amd_gfx1100_row_softmax_state(qk,om,ol,spec=AMDRowSoftmaxRepackSpec(score_scale=scale,mode="loop_state_v1",validity_mode="causal_v1" if causal else "tail_v1",query_start=query_start,kv_start=-1,valid_kv=valid_kv,dynamic_kv_v1=True),kv_tile=rng); writes=[*wr(mreg,"m",nm),*wr(lreg,"l",nl)]
  for b in range(8):
    oc=rd(creg,ci,"acc",b,b*8); pv=UOp(Ops.WMMA,dtypes.float.vec(8),(p,fr(v,"V",b),oc.alu(Ops.MUL,alpha)),warg); writes.extend(wr(creg,"acc",pv,b,b*8))
  end=UOp.group(*writes).end(rng).replace(tag=("amd_gfx1100_attention_grid_loop_end_v1",rng)); fl=rd(lreg,end,"l",final=True); fc=tuple(rd(creg,end,"acc",b,b*8,final=True) for b in range(8)); drain=UOp(Ops.AMD_ATTENTION_OUTPUT_DRAIN,dtypes.void,(out,group,fl,*fc),arg=AMDAttentionOutputDrainSpec(grid=grid))
  return UOp.sink(mi,li,ci,end,drain,arg=kernel_info).replace(tag=("amd_gfx1100_q16_grid_hd128_loop_v1",))

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

def amd_tile_wmma_boundary_report(*, qk_score: UOp, pv_value: UOp, pv_acc: UOp) -> dict:
  """Describe whether AMD can consume the composite tile at the WMMA boundary.

  This is intentionally diagnostic only.  The renderer must not synthesize
  lane packing or emit an instruction from a logical composite reduction.  A
  report is promotable only when all three operands are explicit TILE_GATHER
  carriers with exact 16x16 ownership and the expected score/value/acc roles.
  """
  reasons = []
  nodes = (("score", qk_score), ("value", pv_value), ("acc", pv_acc))
  for role, node in nodes:
    if node.op is not Ops.TILE_GATHER:
      reasons.append(f"{role} is not a TILE_GATHER carrier")
      continue
    spec = node.arg
    try: spec.validate()
    except ValueError as e:
      reasons.append(f"{role} carrier invalid: {e}")
      continue
    if spec.role != role:
      reasons.append(f"{role} carrier declares role {spec.role}")
    if spec.fragment_shape != (16, 16) or node.shape != (16, 16) or node.src[0].shape != (16, 16):
      reasons.append(f"{role} carrier is not an exact 16x16 fragment")
  return {"backend": "amd", "qk": "score", "pv": "value", "acc": "acc",
          "promotable": not reasons, "renderer": "ordinary_wmma" if not reasons else "fail-closed",
          "isa": "eligible" if not reasons else "not-emitted", "reasons": tuple(reasons)}

def composite_reduce_tile_report(red: UOp) -> dict:
  """Diagnose whether a real composite REDUCE has reached fragment lowering.

  The bounded semantic attention route intentionally remains scalar until the
  scheduler can construct owned 16x16 score/value/accumulator fragments.  This
  resolver never reshapes or broadcasts a logical reduction; it reports the
  exact missing edge and keeps production admission fail-closed.
  """
  reasons = []
  if red.op is not Ops.REDUCE:
    reasons.append("node is not a REDUCE")
    return {"promotable": False, "renderer": "fail-closed", "isa": "not-emitted", "reasons": tuple(reasons)}
  comp = red.arg[0] if red.arg else None
  if not isinstance(comp, CompositeReduce):
    reasons.append("REDUCE does not carry CompositeReduce metadata")
  carrier = getattr(comp, "tile_carrier", None)
  if carrier is None:
    reasons.append("composite REDUCE has no tile carrier")
  else:
    try: carrier.validate()
    except ValueError as e: reasons.append(f"tile carrier invalid: {e}")
  carriers = [u for u in red.toposort() if u.op is Ops.TILE_GATHER]
  if len(carriers) < 3:
    reasons.append(f"real reduction exposes {len(carriers)} TILE_GATHER fragments; need score/value/acc")
  return {"promotable": not reasons, "renderer": "ordinary_wmma" if not reasons else "fail-closed",
          "isa": "eligible" if not reasons else "not-emitted", "reasons": tuple(reasons)}


def adapt_wmma_fragment(source: UOp, *, role: str, dtype: DType, shape: tuple[int, int] = (16, 16)) -> UOp:
  """Validate/adapt one logical tile at the SHAPED_WMMA boundary.

  Composite lowering must perform the real range ownership and packing before
  this point.  This primitive deliberately does not reshape or broadcast: it
  accepts only an exact 16x16 carrier, making invalid score/V lane mappings
  fail immediately instead of reaching backend codegen with corrupted lanes.
  """
  if role not in ("q", "k", "score", "v", "acc"):
    raise ValueError(f"unknown WMMA fragment role: {role}")
  if source.shape != shape:
    raise ValueError(f"{role} fragment must be a logical {shape[0]}x{shape[1]} tile")
  if source.dtype.base != dtype:
    raise ValueError(f"{role} fragment dtype does not match the tile ABI")
  return source

def adapt_composite_tile_fragments(carrier: CompositeTileCarrier, *, score: UOp, value: UOp,
                                   acc: UOp, dtype: DType) -> tuple[UOp, UOp, UOp]:
  """Validate the logical carriers before constructing QK/PV WMMA nodes.

  This is intentionally a zero-copy boundary: grouped Hd lanes must already
  be owned by the scheduler.  Flattening or broadcasting here would silently
  destroy lane provenance, so malformed composite sources fail closed.
  """
  carrier.validate()
  expected = (("score", score, carrier.score_fragment or carrier.score_shape[:2]),
              ("v", value, carrier.value_fragment or (carrier.value_shape[0], carrier.value_shape[2])),
              ("acc", acc, carrier.output_fragment or (carrier.output_shape[0], carrier.output_shape[2])))
  out = []
  for role, src, shape in expected:
    out.append(adapt_wmma_fragment(src, role=role, dtype=dtype if role != "acc" else src.dtype.base, shape=shape))
  return tuple(out)


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

  def ordinary_wmma_ready(self) -> bool:
    """Return whether both contractions have the ordinary fragment ABI.

    This is deliberately only a descriptor check.  It does not change
    admission policy; callers still need backend/source/ISA evidence before
    enabling a production attention shape.
    """
    self.validate()
    dims, _device, threads = self.qk.arg
    if tuple(dims) != (16, 16, 16) or threads != 32: return False
    return all(len(n.src) == 3 and n.src[0].shape == (16, 16) and
               n.src[1].shape == (16, 16) and n.src[2].shape == (16, 16)
               for n in (self.qk, self.pv))

  def candidate_report(self) -> dict:
    """Describe bounded admission without claiming backend promotion.

    This is intentionally diagnostic: a shaped graph can satisfy the ordinary
    fragment descriptor while still lacking generated source/ISA evidence.
    Keeping those facts separate prevents an opt-in experiment from silently
    enabling the production attention route.
    """
    self.validate()
    ready = self.ordinary_wmma_ready()
    reasons = [] if ready else ["fragment ABI is not descriptor-shaped"]
    if self.weights is None:
      reasons.append("normalized score weights are not present")
    return {"descriptor_valid": True, "ordinary_fragment_abi": ready,
            "qk_wmma_candidate": ready, "pv_wmma_candidate": ready and self.weights is not None,
            "source_evidence": False, "isa_evidence": False,
            "production_promotion": False, "reasons": tuple(reasons)}


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
    # Preserve the descriptor-owned C(row,kv) layout through the nonlinear
    # boundary. Backend lowering must realize the declared row reductions and
    # LDS/barrier repack before PV consumes its native A fragment.
    weights = row_softmax_lds_repack(qk, m, l)
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
