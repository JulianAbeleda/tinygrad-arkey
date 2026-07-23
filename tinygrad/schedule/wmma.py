"""Reusable WMMA authoring helpers for scheduler-owned generated kernels."""
from __future__ import annotations

from dataclasses import dataclass
from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.uop.ops import CompositeReduce, CompositeTileCarrier, TileGatherSpec, Ops

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
