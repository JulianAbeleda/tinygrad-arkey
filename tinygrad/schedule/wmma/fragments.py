"""Tile gather / fragment index-map / WMMA-shaped emission primitives."""
from __future__ import annotations

from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import Ops, UOp, TileGatherSpec

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

def shaped_wmma(a_frag:UOp, b_frag:UOp, acc_frag:UOp, *, dims:tuple[int, int, int],
                device:str, threads:int, dtype_out:DType|None=None) -> UOp:
  """Construct a declarative SHAPED_WMMA tensor-graph node.

  The rangeify pass owns lowering this to Ops.WMMA. Callers must pass already-shaped per-thread fragments; this helper
  exists so route code does not construct route-local Ops.WMMA or duplicate the SHAPED_WMMA argument convention.
  """
  return UOp(Ops.SHAPED_WMMA, dtype_out or acc_frag.dtype.scalar(), (a_frag, b_frag, acc_frag),
             arg=(dims, device, threads))
