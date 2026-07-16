"""Small Tensor/UOp seam for scheduler-owned dynamic output tiles.

The important property here is that ``tile`` is a UOp RANGE value.  Index
arithmetic therefore remains in the generated program; callers must not make
one Python graph per output tile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt.kernel_pipeline import SchedulerOutputTileLoop, build_scheduler_output_tile_loop
from tinygrad.uop.ops import Ops, UOp


@dataclass(frozen=True)
class DynamicTile:
  tile: UOp
  weights: Tensor
  activation: Tensor
  scales: Tensor
  output: Tensor
  output_indices: Tensor


def _indices(tile: UOp, count: int, stride: int, device: str | None) -> Tensor:
  if count <= 0 or stride <= 0: raise ValueError("dynamic tile dimensions must be positive")
  # Tensor.arange is deliberately outside the loop: only its base is dynamic.
  base = Tensor.arange(count, dtype=dtypes.int32)
  if device is not None: base = base.to(device)
  return (base + tile * stride).cast(dtypes.int32)


def dynamic_tile_views(weights: Tensor, activation: Tensor, scales: Tensor, output: Tensor,
                       tile: UOp, *, weight_rows: int, activation_rows: int, scale_rows: int,
                       output_rows: int, row_width: int, weight_stride: int | None = None,
                       activation_stride: int | None = None, scale_stride: int | None = None,
                       output_stride: int | None = None) -> DynamicTile:
  """Address one tile of Q4 rows, activation rows, scales, and output rows."""
  if any(len(x.shape) != 1 for x in (weights, activation, scales, output)):
    raise ValueError("dynamic_tile_views expects flat storage tensors")
  dev = weights.device
  ws, xs, ss, os = (weight_stride or weight_rows * row_width, activation_stride or activation_rows * row_width,
                    scale_stride or scale_rows, output_stride or output_rows * row_width)
  return DynamicTile(tile, weights[_indices(tile, weight_rows * row_width, ws, dev)],
    activation[_indices(tile, activation_rows * row_width, xs, dev)],
    scales[_indices(tile, scale_rows, ss, dev)], output,
    _indices(tile, output_rows * row_width, os, dev))


def own_dynamic_tiles(plan: SchedulerOutputTileLoop, weights: Tensor, activation: Tensor, scales: Tensor,
                      output: Tensor, *, weight_rows: int, activation_rows: int, scale_rows: int,
                      output_rows: int, row_width: int,
                      weight_stride: int | None = None, activation_stride: int | None = None,
                      scale_stride: int | None = None, output_stride: int | None = None,
                      body: Callable[[DynamicTile], UOp]) -> UOp:
  """Run ``body`` once under a symbolic output-tile loop and return its sink."""
  def owned_body(tile: UOp) -> UOp:
    value = body(dynamic_tile_views(
      weights, activation, scales, output, tile, weight_rows=weight_rows, activation_rows=activation_rows,
      scale_rows=scale_rows, output_rows=output_rows, row_width=row_width, weight_stride=weight_stride,
      activation_stride=activation_stride, scale_stride=scale_stride, output_stride=output_stride))
    if not isinstance(value, UOp): raise TypeError("dynamic tile owner body must return a UOp")

    # Tensor indexing can structurally clone RANGE nodes while building a
    # nested producer graph.  Carry every exact loop-bearing STORE range into
    # the single outer END; closing only the scheduler spelling leaves a
    # producer range visible as a call input.
    store_ranges = {r for node in value.toposort() if node.op is Ops.STORE
                    for r in node.ranges if r.op is Ops.RANGE}
    # Mark the scheduler-owned spelling as closed too.  This prevents the
    # outer generic loop builder from adding a second ownership marker; the
    # exact store ranges below still close Tensor-created spellings.
    owned_ranges = tuple(dict.fromkeys((tile, *store_ranges)))
    missing = tuple(r for r in owned_ranges if r not in value.ended_ranges)
    if missing: value = value.end(*missing)
    return value

  return build_scheduler_output_tile_loop(plan, owned_body)


def dynamic_store(output: Tensor, indices: Tensor, values: Tensor) -> UOp:
  """The legal indexed writeback primitive used by dynamic tile owners."""
  # Range ownership belongs to own_dynamic_tiles, which closes the complete
  # AFTER/store effect after the callback has assembled all producer views.
  # Ending this store locally would leave END nested under the scheduler's
  # output effect and rangeify would treat it as an unsupported kernel.
  value = values.uop
  target = output.uop.index(indices.uop, ptr=True)
  store = target.store(value)
  return output.uop.after(store)


__all__ = ["DynamicTile", "dynamic_tile_views", "own_dynamic_tiles", "dynamic_store"]
