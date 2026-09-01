"""Ordinary-UOp, finite-fp32 argmax used by the closed decode sampler experiment.

The normal :meth:`Tensor.argmax` is deliberately unchanged.  This helper is
only safe for finite float32 values: it turns each value and its *first* index
into one unsigned 64-bit reduction key.  It has no renderer source, custom
kernel, or model-shape knowledge.
"""
from __future__ import annotations

from tinygrad import Tensor, dtypes
from tinygrad.device import BufferSpec
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import cdiv
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from extra.llm_research.boltbeam_authority import lower_authorized_candidate


def make_native_argmax_host_mirror(device:str, memory:str="host") -> Tensor:
  """Allocate one pinned-host int32 that is also writable by an NV kernel."""
  if not device.startswith("NV"): raise ValueError(f"native argmax host mirror is NV-only, got {device}")
  uop = UOp.new_buffer(device, 1, dtypes.int32)
  if memory not in ("host", "mapped_vram"): raise ValueError(f"unsupported native argmax mirror memory {memory!r}")
  uop.buffer.options = BufferSpec(host=memory == "host", cpu_access=memory == "mapped_vram", nolru=True)
  uop.buffer.ensure_allocated()
  return Tensor(uop)


def read_native_argmax_host_mirror(mirror:Tensor) -> int:
  """Read a mirror after the caller has waited for its producing compute timeline."""
  if mirror.shape != (1,) or mirror.dtype != dtypes.int32 or not isinstance(mirror.device, str):
    raise ValueError("invalid native argmax host mirror")
  return int(mirror.uop.buffer.get_buf(mirror.device).cpu_view().view(size=4, fmt="i")[0])


def _argmax_pair(best_value:UOp, best_index:UOp, other_value:UOp, other_index:UOp) -> tuple[UOp, UOp]:
  """Select the larger fp32 value, breaking exact ties toward the first index."""
  better = (other_value > best_value) | (other_value.eq(best_value) & (other_index < best_index))
  return better.where(other_value, best_value), better.where(other_index, best_index)


def emit_native_finite_fp32_argmax(n:int, threads:int=1024, host_mirror:bool=False) -> callable:
  """One-CTA first-index argmax over one finite contiguous fp32 row.

  Each thread scans a fixed strided slice in registers, each warp reduces its
  value/index pair with shuffles, and warp zero reduces the 32 shared winners.
  Value comparisons remain fp32 throughout, so this preserves ordinary argmax
  ordering (including signed-zero ties) without a packed-u64 data path.
  """
  if not isinstance(n, int) or n < 1: raise ValueError(f"n must be a positive integer, got {n!r}")
  if threads not in (256, 512, 1024): raise ValueError(f"threads must be 256, 512, or 1024, got {threads}")
  warps = threads // 32
  def kernel(out:UOp, *args:UOp) -> UOp:
    mirror, x = args if host_mirror else (None, args[0])
    tid = UOp.special(threads, "lidx0")
    warp, lane = tid // 32, tid % 32

    best_value_reg = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    best_index_reg = UOp.placeholder((1,), dtypes.int32, 21, addrspace=AddrSpace.REG)
    value_init = best_value_reg[0].store(-float("inf"))
    index_init = best_index_reg.after(value_init)[0].store(n)
    best_value_reg = best_value_reg.after(value_init, index_init)
    best_index_reg = best_index_reg.after(value_init, index_init)

    step = UOp.range(cdiv(n, threads), 0, axis_type=AxisType.REDUCE)
    index = step * threads + tid
    valid = index < n
    value = valid.where(x.index(valid.where(index, 0)).load(dtype=dtypes.float32), -float("inf"))
    next_value, next_index = _argmax_pair(best_value_reg.after(step)[0], best_index_reg.after(step)[0],
                                           value, index.cast(dtypes.int32))
    value_update = best_value_reg[0].store(next_value)
    index_update = best_index_reg.after(value_update)[0].store(next_index).end(step)
    best_value, best_index = best_value_reg.after(index_update)[0], best_index_reg.after(index_update)[0]

    for slot, offset in enumerate((16, 8, 4, 2, 1), 90):
      other_value = _staged_shfl(best_value, offset, lane, slot)
      other_index = _staged_shfl(best_index, offset, lane, slot+5)
      best_value, best_index = _argmax_pair(best_value, best_index, other_value, other_index)

    shared_values = UOp.placeholder((warps,), dtypes.float32, 40, addrspace=AddrSpace.LOCAL)
    shared_indices = UOp.placeholder((warps,), dtypes.int32, 41, addrspace=AddrSpace.LOCAL)
    value_publish = shared_values[warp].store(best_value, lane.eq(0))
    index_publish = shared_indices.after(value_publish)[warp].store(best_index, lane.eq(0))
    ready = UOp.barrier(UOp.group(index_publish))

    valid_warp = lane < warps
    shared_index = valid_warp.where(lane, 0)
    block_value = valid_warp.where(shared_values.after(ready)[shared_index], -float("inf"))
    block_index = valid_warp.where(shared_indices.after(ready)[shared_index], n)
    for slot, offset in enumerate((16, 8, 4, 2, 1), 110):
      other_value = _staged_shfl(block_value, offset, lane, slot)
      other_index = _staged_shfl(block_index, offset, lane, slot+5)
      block_value, block_index = _argmax_pair(block_value, block_index, other_value, other_index)
    out_store = out[0].store(block_index, tid.eq(0))
    stores = (out_store, mirror[0].store(block_index, tid.eq(0))) if mirror is not None else (out_store,)
    return stores[0].sink(*stores[1:],
      arg=KernelInfo(name=f"native_finite_fp32_argmax_{n}_t{threads}{'_host_mirror' if host_mirror else ''}", opts_to_apply=()))
  return kernel


def native_argmax_finite_fp32(x:Tensor, threads:int=1024) -> Tensor:
  """Execute the one-kernel finite-fp32 argmax for one contiguous NV row."""
  if x.dtype != dtypes.float32 or x.ndim != 2 or x.shape[0] != 1 or not isinstance(x.shape[1], int):
    raise ValueError(f"native argmax needs one static fp32 row, got shape={x.shape} dtype={x.dtype}")
  if not str(x.device).startswith("NV"): raise ValueError(f"native argmax is NV-only, got {x.device}")
  n = x.shape[1]
  emitter,ticket=lower_authorized_candidate({"family":"finite_argmax.v1","n":n,"threads":threads,"host_mirror":False},
    (("decode_native_argmax","finite_fp32_argmax"),))
  program = KernelProgram("decode_native_finite_fp32_argmax", f"vocab_{n}_t{threads}",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emitter,
    output_spec=OutputSpec((1,), dtypes.int32),
    boltbeam_ticket=ticket)
  # The held decode return must not be a view of the custom program's internal
  # allocation: that allocation participates in the next replay's memory plan.
  return execute_promoted_program(None, x.reshape(n).contiguous(), program=program).reshape(1, 1).clone()


def native_argmax_finite_fp32_host_mirror(x:Tensor, mirror:Tensor, threads:int=1024) -> tuple[Tensor, Tensor]:
  """Research gate: write the exact native argmax to GPU output and a caller-owned mirror."""
  if x.dtype != dtypes.float32 or x.ndim != 2 or x.shape[0] != 1 or not isinstance(x.shape[1], int):
    raise ValueError(f"native argmax needs one static fp32 row, got shape={x.shape} dtype={x.dtype}")
  if mirror.shape != (1,) or mirror.dtype != dtypes.int32 or mirror.device != x.device:
    raise ValueError(f"native argmax mirror needs one int32 on {x.device}, got {mirror.shape=} {mirror.dtype=} {mirror.device=}")
  if not str(x.device).startswith("NV"): raise ValueError(f"native argmax is NV-only, got {x.device}")
  n = x.shape[1]
  emitter,ticket=lower_authorized_candidate({"family":"finite_argmax.v1","n":n,"threads":threads,"host_mirror":True},
    (("decode_native_argmax","finite_fp32_argmax"),))
  program = KernelProgram("decode_native_finite_fp32_argmax", f"vocab_{n}_t{threads}.host_mirror",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emitter,
    output_spec=OutputSpec((1,), dtypes.int32),
    boltbeam_ticket=ticket)
  out = Tensor.empty(1, dtype=dtypes.int32, device=x.device)
  results = out.uop_program(mirror, x.reshape(n).contiguous(), fxn=program.emitter)
  return results[0].reshape(1, 1).clone(), results[1]


def packed_argmax_finite_fp32(x:Tensor, axis:int=-1, keepdim:bool=False) -> Tensor:
  """Return first-index argmax for a finite fp32 tensor with one ordinary MAX.

  IEEE binary32 bit patterns have a monotonic unsigned ordering after flipping
  the sign bit for nonnegative values and complementing negative values.  Zero
  is canonicalized first so ``-0.0`` and ``+0.0`` tie exactly.  The low half of
  the key is ``(N-1-index)``; thus an unsigned max chooses the earliest index
  on equal values.

  This intentionally rejects non-fp32 inputs and dynamic/empty/too-wide axes.
  NaN policy is not silently invented here: callers must have a finite-value
  qualification before using the route.
  """
  if x.dtype != dtypes.float32: raise ValueError(f"packed argmax needs float32, got {x.dtype}")
  if x.ndim == 0: raise ValueError("packed argmax needs a non-scalar tensor")
  axis = axis + x.ndim if axis < 0 else axis
  if not 0 <= axis < x.ndim: raise ValueError(f"axis {axis} out of range for rank {x.ndim}")
  n = x.shape[axis]
  if not isinstance(n, int) or not 0 < n <= 0x1_0000_0000:
    raise ValueError(f"packed argmax needs static 1..2^32 axis, got {x.shape[axis]!r}")
  # Moving the reduced dimension to the end keeps the index construction
  # generic for all axes, then restore the requested output layout.
  y = x.transpose(axis, -1)
  bits = y.bitcast(dtypes.uint32)
  bits = y.eq(0.0).where(0, bits)  # signed zero must be one tie class
  ordered = (bits >> 31).where(~bits, bits ^ 0x80000000)
  inv_index = (n-1-Tensor.arange(n, dtype=dtypes.uint32).to(x.device)).cast(dtypes.uint64)
  key = (ordered.cast(dtypes.uint64) << 32) | inv_index
  index = (n-1-(key.max(axis=-1, keepdim=keepdim) & 0xffffffff)).cast(dtypes.int32)
  # Moving ``axis`` to the final position changes the order of every dimension
  # after it; restore that rotation (not merely a swap for rank > 3).
  if axis == x.ndim-1: return index
  if keepdim:
    order = tuple(range(axis)) + (x.ndim-1,) + tuple(range(axis+1, x.ndim-1)) + (axis,)
  else:
    order = tuple(range(axis)) + tuple(range(axis+1, x.ndim-1)) + (axis,)
  return index.permute(order)


def packed_argmax_tile_keys_fp32(x:Tensor, tile_rows:int, axis:int=-1) -> Tensor:
  """Per-tile (max, index) packed u64 keys for the fused vocab-head top-1 (P1).

  The vocab aux scatter-chain fusion (nv-vocab-aux-chain-fusion-scope-20260812.md) carries
  one (max, index) per GEMV warp tile instead of materialising the 151936-row logits chain.
  This is the ordinary-UOp mirror of that in-kernel carry: the reduced axis is grouped into
  ``tile_rows``-wide tiles, each tile becomes one u64 key ``(ordered fp32 bits) << 32 |
  (n-1-index)`` (the same monotonic IEEE reordering as :func:`packed_argmax_finite_fp32`,
  including the signed-zero canonicalisation), and a u64 MAX over the tile axis keeps the
  largest logit while the inverted index in the low half breaks ties to the FIRST index.
  The returned keys are the per-tile carry; the cross-tile reduce is one more u64 MAX
  (see :func:`packed_argmax_from_tile_keys`).  No float cast participates in the max+idx
  compare: the values only leave fp32 through a bit-exact BITCAST and integer ops.
  """
  if x.dtype != dtypes.float32: raise ValueError(f"packed argmax needs float32, got {x.dtype}")
  if x.ndim == 0: raise ValueError("packed argmax needs a non-scalar tensor")
  axis = axis + x.ndim if axis < 0 else axis
  if not 0 <= axis < x.ndim: raise ValueError(f"axis {axis} out of range for rank {x.ndim}")
  if not isinstance(tile_rows, int) or tile_rows < 1:
    raise ValueError(f"tile_rows must be a positive integer, got {tile_rows!r}")
  n = x.shape[axis]
  if not isinstance(n, int) or not 0 < n <= 0x1_0000_0000:
    raise ValueError(f"packed argmax needs static 1..2^32 axis, got {x.shape[axis]!r}")
  if n % tile_rows != 0: raise ValueError(f"axis size {n} must be divisible by tile_rows {tile_rows}")
  # Moving the reduced dimension to the end keeps the tile grouping generic for all axes,
  # then the tile axis is restored to the requested output position.
  y = x.transpose(axis, -1)
  bits = y.bitcast(dtypes.uint32)
  bits = y.eq(0.0).where(0, bits)  # signed zero must be one tie class
  ordered = (bits >> 31).where(~bits, bits ^ 0x80000000)
  inv_index = (n-1-Tensor.arange(n, dtype=dtypes.uint32).to(x.device)).cast(dtypes.uint64)
  keys = (ordered.cast(dtypes.uint64) << 32) | inv_index
  tiles = keys.reshape(y.shape[:-1] + (n // tile_rows, tile_rows))
  tile_keys = tiles.max(axis=-1)  # per-tile (max, index); first-index-wins on equal logits
  if axis == x.ndim-1: return tile_keys
  order = tuple(range(axis)) + (x.ndim-1,) + tuple(range(axis+1, x.ndim-1)) + (axis,)
  return tile_keys.permute(order)


def packed_argmax_from_tile_keys(tile_keys:Tensor, n:int, axis:int=-1, keepdim:bool=False) -> Tensor:
  """Final packed reduce over per-tile keys: one u64 MAX + unpack to the first-index token id.

  Consumes the per-tile (max, index) keys produced by :func:`packed_argmax_tile_keys_fp32`
  (or the vocab GEMV epilogue's in-kernel carry) and finishes the fused top-1: the single
  MAX over the tile axis keeps the largest packed key, and ``n-1-(key & 0xffffffff)`` unpacks
  the winning row.  Tie semantics are identical to :func:`packed_argmax_finite_fp32` and to
  today's ``r_16_8`` Tensor.argmax chain (first index wins).
  """
  if tile_keys.dtype != dtypes.uint64: raise ValueError(f"tile keys must be uint64, got {tile_keys.dtype}")
  if tile_keys.ndim == 0: raise ValueError("tile keys must be a non-scalar tensor")
  axis = axis + tile_keys.ndim if axis < 0 else axis
  if not 0 <= axis < tile_keys.ndim: raise ValueError(f"axis {axis} out of range for rank {tile_keys.ndim}")
  if not isinstance(n, int) or not 0 < n <= 0x1_0000_0000:
    raise ValueError(f"packed argmax needs static 1..2^32 axis, got n={n!r}")
  y = tile_keys.transpose(axis, -1)
  index = (n-1-(y.max(axis=-1, keepdim=keepdim) & 0xffffffff)).cast(dtypes.int32)
  if axis == tile_keys.ndim-1: return index
  if keepdim:
    order = tuple(range(axis)) + (tile_keys.ndim-1,) + tuple(range(axis+1, tile_keys.ndim-1)) + (axis,)
  else:
    order = tuple(range(axis)) + tuple(range(axis+1, tile_keys.ndim-1)) + (axis,)
  return index.permute(order)


def packed_argmax_tiles_fp32(x:Tensor, tile_rows:int, axis:int=-1, keepdim:bool=False) -> Tensor:
  """Fused per-tile (max, index) top-1 in one ordinary-UOp graph (vocab aux chain fusion, P1).

  Convenience composition of :func:`packed_argmax_tile_keys_fp32` and
  :func:`packed_argmax_from_tile_keys`: the exact two-stage reduce the fused epilogue
  performs (per-tile packed key, then one cross-tile MAX), bit-exact with
  :func:`packed_argmax_finite_fp32` and with today's Tensor.argmax chain.
  """
  keys = packed_argmax_tile_keys_fp32(x, tile_rows, axis=axis)
  return packed_argmax_from_tile_keys(keys, x.shape[axis], axis=axis, keepdim=keepdim)
