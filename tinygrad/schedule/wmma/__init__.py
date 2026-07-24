"""Reusable WMMA authoring helpers for scheduler-owned generated kernels.

This package is a concern-split of the former ``tinygrad/schedule/wmma.py``
monolith (see docs/wmma-modularization-scope-20260724.md). Every name that
was public on the old module is re-exported here unchanged so
``from tinygrad.schedule.wmma import X`` keeps working for all external
importers.
"""
from __future__ import annotations

# Re-exported third-party/stdlib/tinygrad-core names that were incidentally
# public on the old monolithic module (picked up by `dir()` on its imports).
import math
from dataclasses import dataclass
from typing import NamedTuple
from tinygrad.dtype import DType, dtypes, PtrDType, AddrSpace
from tinygrad.uop.ops import (Ops, UOp, CompositeReduce, CompositeTileCarrier, TileGatherSpec,
  RowSoftmaxRepackSpec, AMDRowSoftmaxRepackSpec, AMDRowSoftmaxSlotSpec, AMDPVCLaneSpec)

from tinygrad.schedule.wmma.fragments import (
  grouped_tile_load, tile_gather, build_owned_fragment_index_map,
  lower_tile_gather, lower_attached_tile_gather, emit_tile_gather_shaped_wmma,
  adapt_wmma_fragment, shaped_wmma)
from tinygrad.schedule.wmma.softmax import (
  row_softmax_lds_repack, amd_gfx1100_row_softmax_repack, amd_gfx1100_row_softmax_state,
  amd_gfx1100_row_softmax_initial, OnlineSoftmaxBlockTransition, online_softmax_block_transition,
  amd_gfx1100_broadcast_row_state, amd_gfx1100_pv_c_lane)
from tinygrad.schedule.wmma.kernels import (
  amd_gfx1100_q16_attention, amd_gfx1100_q16_kv32_attention, amd_gfx1100_q16_kv32_hd128_attention,
  amd_gfx1100_q16_kv64_hd128_loop_attention, amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_attention,
  amd_gfx1100_q16_grid_hd128_loop_attention, amd_gfx1100_q16_grid_qk_stats_stage,
  amd_gfx1100_q16_grid_pv_slice_stage)
from tinygrad.schedule.wmma.composite import (
  construct_hd16_tile_carriers, composite_reduce_hd16_carriers, emit_hd16_dual_tile_wmma,
  adapt_composite_tile_fragments, composite_reduce_tile_report, amd_tile_wmma_boundary_report,
  OnlineSoftmaxTile, online_softmax_tile)

__all__ = [
  "math", "dataclass", "NamedTuple", "DType", "dtypes", "PtrDType", "AddrSpace",
  "Ops", "UOp", "CompositeReduce", "CompositeTileCarrier", "TileGatherSpec",
  "RowSoftmaxRepackSpec", "AMDRowSoftmaxRepackSpec", "AMDRowSoftmaxSlotSpec", "AMDPVCLaneSpec",
  "grouped_tile_load", "tile_gather", "build_owned_fragment_index_map",
  "lower_tile_gather", "lower_attached_tile_gather", "emit_tile_gather_shaped_wmma",
  "adapt_wmma_fragment", "shaped_wmma",
  "row_softmax_lds_repack", "amd_gfx1100_row_softmax_repack", "amd_gfx1100_row_softmax_state",
  "amd_gfx1100_row_softmax_initial", "OnlineSoftmaxBlockTransition", "online_softmax_block_transition",
  "amd_gfx1100_broadcast_row_state", "amd_gfx1100_pv_c_lane",
  "amd_gfx1100_q16_attention", "amd_gfx1100_q16_kv32_attention", "amd_gfx1100_q16_kv32_hd128_attention",
  "amd_gfx1100_q16_kv64_hd128_loop_attention", "amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_attention",
  "amd_gfx1100_q16_grid_hd128_loop_attention", "amd_gfx1100_q16_grid_qk_stats_stage",
  "amd_gfx1100_q16_grid_pv_slice_stage",
  "construct_hd16_tile_carriers", "composite_reduce_hd16_carriers", "emit_hd16_dual_tile_wmma",
  "adapt_composite_tile_fragments", "composite_reduce_tile_report", "amd_tile_wmma_boundary_report",
  "OnlineSoftmaxTile", "online_softmax_tile",
]
