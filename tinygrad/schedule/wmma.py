"""Reusable WMMA authoring helpers for scheduler-owned generated kernels."""
from __future__ import annotations

from dataclasses import dataclass
from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.uop.ops import CompositeTileCarrier, TileGatherSpec

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

def lower_tile_gather(source: UOp, *, role: str, dtype: DType) -> UOp:
  """Resolve a TILE_GATHER only when an upstream pass already shaped it.

  No flattening, broadcast, or index synthesis is permitted here.  This
  fail-closed resolver is the handoff point for the future grouped LOAD pass.
  """
  if source.op is not Ops.TILE_GATHER:
    raise ValueError("expected TILE_GATHER carrier")
  spec = source.arg
  spec.validate()
  if spec.role != role or source.shape != spec.fragment_shape or source.src[0].shape != spec.fragment_shape:
    raise ValueError("tile gather is not a shaped fragment")
  return adapt_wmma_fragment(source, role=role, dtype=dtype, shape=spec.fragment_shape)

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
