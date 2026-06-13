#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from extra.qk_layout import (
  GGML_Q4_K, GGML_Q6_K, GGUFInfo, GGUFMetadata, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS,
  Q4K_WORDS_PER_BLOCK, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK,
  packed_byte_range, role_from_name, tensor_shape,
)

Q4_STORAGE_DTYPE = "uint32"
Q6_STORAGE_DTYPE = "uint16"

@dataclass(frozen=True)
class PackedQKFormat:
  ggml_type: int
  name: str
  block_elems: int
  block_bytes: int
  storage_dtype: str
  storage_item_bytes: int
  storage_items_per_block: int
  quant_payload_offset_bytes: int
  quant_payload_bytes: int
  reference: str

@dataclass(frozen=True)
class PackedQKLoadTile:
  name: str
  storage_dtype: str
  lanes: int
  bytes: int
  alignment_bytes: int
  storage_items_per_load: int
  q_values_per_load: int | None
  requires_tail: bool
  mechanism: str

@dataclass(frozen=True)
class PackedQKTile:
  tensor: str
  role: str
  format: PackedQKFormat
  rows: int
  cols: int
  blocks: int
  byte_start: int
  packed_bytes: int
  data_start: int | None
  tensor_offset: int | None
  storage_aligned: bool
  legal_load_tiles: tuple[PackedQKLoadTile, ...]

Q4_K_FORMAT = PackedQKFormat(
  ggml_type=GGML_Q4_K, name="Q4_K", block_elems=Q4_K_BLOCK_ELEMS, block_bytes=Q4_K_BLOCK_BYTES,
  storage_dtype=Q4_STORAGE_DTYPE, storage_item_bytes=4, storage_items_per_block=Q4K_WORDS_PER_BLOCK,
  quant_payload_offset_bytes=16, quant_payload_bytes=128, reference="extra.qk_layout.q4_k_reference",
)

Q6_K_FORMAT = PackedQKFormat(
  ggml_type=GGML_Q6_K, name="Q6_K", block_elems=Q6_K_BLOCK_ELEMS, block_bytes=Q6_K_BLOCK_BYTES,
  storage_dtype=Q6_STORAGE_DTYPE, storage_item_bytes=2, storage_items_per_block=Q6K_HALFWORDS_PER_BLOCK,
  quant_payload_offset_bytes=0, quant_payload_bytes=192, reference="extra.qk_layout.q6_k_reference",
)

FORMATS = {GGML_Q4_K: Q4_K_FORMAT, GGML_Q6_K: Q6_K_FORMAT}

Q4_U32_SCALAR_LOAD = PackedQKLoadTile(
  name="u32_scalar", storage_dtype=Q4_STORAGE_DTYPE, lanes=1, bytes=4, alignment_bytes=4,
  storage_items_per_load=1, q_values_per_load=8, requires_tail=False,
  mechanism="load one Q4_K uint32 storage word; each quant payload word contains eight 4-bit values",
)

Q4_U32X4_ALIGNED_LOAD = PackedQKLoadTile(
  name="u32x4_aligned", storage_dtype=Q4_STORAGE_DTYPE, lanes=4, bytes=16, alignment_bytes=16,
  storage_items_per_load=4, q_values_per_load=32, requires_tail=False,
  mechanism="load four adjacent Q4_K uint32 storage words as one aligned vector load",
)

Q6_U16_SCALAR_LOAD = PackedQKLoadTile(
  name="u16_scalar", storage_dtype=Q6_STORAGE_DTYPE, lanes=1, bytes=2, alignment_bytes=2,
  storage_items_per_load=1, q_values_per_load=None, requires_tail=False,
  mechanism="load one Q6_K uint16 storage halfword; Q6_K payload has mixed 4-bit and 2-bit planes",
)

def format_for_type(ggml_type:int) -> PackedQKFormat:
  try:
    return FORMATS[ggml_type]
  except KeyError as exc:
    raise ValueError(f"unsupported packed QK ggml_type={ggml_type}") from exc

def _shape_from_info(info:GGUFInfo) -> tuple[int, int]:
  shape = tensor_shape(info)
  if len(shape) != 2: raise ValueError(f"{info.name} is not a 2D packed QK tensor: shape={shape}")
  return int(shape[0]), int(shape[1])

def _check_layout(fmt:PackedQKFormat, rows:int, cols:int, packed_bytes:int) -> int:
  elems = rows * cols
  if cols % fmt.block_elems != 0:
    raise ValueError(f"{fmt.name} K={cols} is not block-aligned to {fmt.block_elems}")
  if elems % fmt.block_elems != 0:
    raise ValueError(f"{fmt.name} element count {elems} is not block-aligned to {fmt.block_elems}")
  blocks = elems // fmt.block_elems
  expected = blocks * fmt.block_bytes
  if packed_bytes != expected:
    raise ValueError(f"{fmt.name} packed byte mismatch: got {packed_bytes}, expected {expected}")
  return blocks

def _legal_load_tiles(fmt:PackedQKFormat, byte_start:int, packed_bytes:int) -> tuple[PackedQKLoadTile, ...]:
  if fmt.ggml_type == GGML_Q4_K:
    tiles = [Q4_U32_SCALAR_LOAD]
    if byte_start % Q4_U32X4_ALIGNED_LOAD.alignment_bytes == 0 and packed_bytes % Q4_U32X4_ALIGNED_LOAD.bytes == 0:
      tiles.append(Q4_U32X4_ALIGNED_LOAD)
    return tuple(tiles)
  if fmt.ggml_type == GGML_Q6_K:
    return (Q6_U16_SCALAR_LOAD,)
  raise ValueError(f"unsupported packed QK format {fmt.name}")

def tile_from_info(meta:GGUFMetadata, info:GGUFInfo) -> PackedQKTile:
  fmt = format_for_type(info.typ)
  rows, cols = _shape_from_info(info)
  byte_start, packed_bytes = packed_byte_range(meta, info)
  blocks = _check_layout(fmt, rows, cols, packed_bytes)
  return PackedQKTile(
    tensor=info.name, role=role_from_name(info.name), format=fmt, rows=rows, cols=cols, blocks=blocks,
    byte_start=byte_start, packed_bytes=packed_bytes, data_start=meta.data_start, tensor_offset=info.off,
    storage_aligned=(byte_start % fmt.storage_item_bytes == 0 and packed_bytes % fmt.storage_item_bytes == 0),
    legal_load_tiles=_legal_load_tiles(fmt, byte_start, packed_bytes),
  )

def tile_from_semantic_row(row:dict[str, Any]) -> PackedQKTile:
  ggml_type = int(row["ggml_type"])
  fmt = format_for_type(ggml_type)
  layout = row.get("layout") or {}
  shape = row.get("shape") or {}
  rows, cols = int(shape["rows"]), int(shape["cols"])
  packed_bytes = int(layout["packed_bytes"])
  byte_start = int(layout["byte_start"])
  blocks = _check_layout(fmt, rows, cols, packed_bytes)
  block_bytes = int(layout.get("block_bytes") or fmt.block_bytes)
  block_elems = int(layout.get("block_elems") or fmt.block_elems)
  if block_bytes != fmt.block_bytes or block_elems != fmt.block_elems:
    raise ValueError(
      f"{row.get('tensor')} layout mismatch for {fmt.name}: "
      f"block_bytes={block_bytes} block_elems={block_elems}"
    )
  return PackedQKTile(
    tensor=str(row["tensor"]), role=str(row.get("role") or role_from_name(str(row["tensor"]))),
    format=fmt, rows=rows, cols=cols, blocks=blocks, byte_start=byte_start, packed_bytes=packed_bytes,
    data_start=layout.get("data_start"), tensor_offset=layout.get("tensor_offset"),
    storage_aligned=(byte_start % fmt.storage_item_bytes == 0 and packed_bytes % fmt.storage_item_bytes == 0),
    legal_load_tiles=_legal_load_tiles(fmt, byte_start, packed_bytes),
  )

def load_tile(tile:PackedQKTile, name:str) -> PackedQKLoadTile:
  for candidate in tile.legal_load_tiles:
    if candidate.name == name: return candidate
  legal = ", ".join(t.name for t in tile.legal_load_tiles)
  raise ValueError(f"{tile.tensor} does not support load tile {name!r}; legal tiles: {legal}")

def tile_to_json(tile:PackedQKTile) -> dict[str, Any]:
  out = asdict(tile)
  out["format"] = asdict(tile.format)
  out["legal_load_tiles"] = [asdict(t) for t in tile.legal_load_tiles]
  return out

def tile_summary(tile:PackedQKTile) -> dict[str, Any]:
  return {
    "tensor": tile.tensor,
    "role": tile.role,
    "format": tile.format.name,
    "shape": [tile.rows, tile.cols],
    "blocks": tile.blocks,
    "packed_bytes": tile.packed_bytes,
    "storage_dtype": tile.format.storage_dtype,
    "storage_items_per_block": tile.format.storage_items_per_block,
    "storage_aligned": tile.storage_aligned,
    "legal_load_tiles": [t.name for t in tile.legal_load_tiles],
    "reference": tile.format.reference,
  }

def qk_tile_search_axes(tile:PackedQKTile) -> dict[str, Any]:
  return {
    "semantic_object": "packed_qk_tile",
    "format": tile.format.name,
    "memory_axes": ["storage_dtype", "load_tile", "lane_mapping", "decode_lane_map"],
    "legal_load_tiles": [asdict(t) for t in tile.legal_load_tiles],
    "non_goals": ["parts_only", "local_only", "direct_output_only"],
    "next_blocker": "consume_vector_load_inside_qk_gemv" if any(t.name == "u32x4_aligned" for t in tile.legal_load_tiles) else None,
  }

def format_markdown() -> str:
  lines = [
    "# Packed QK Tile Formats",
    "",
    "| format | block elems | block bytes | storage dtype | items/block | legal initial load tiles |",
    "|---|---:|---:|---|---:|---|",
  ]
  for fmt in (Q4_K_FORMAT, Q6_K_FORMAT):
    dummy_start = 0
    dummy_bytes = fmt.block_bytes
    tiles = ", ".join(t.name for t in _legal_load_tiles(fmt, dummy_start, dummy_bytes))
    lines.append(
      f"| `{fmt.name}` | {fmt.block_elems} | {fmt.block_bytes} | `{fmt.storage_dtype}` | "
      f"{fmt.storage_items_per_block} | `{tiles}` |"
    )
  lines.append("")
  return "\n".join(lines)

def assert_known_qk_format_name(name:str) -> None:
  names = {fmt.name for fmt in FORMATS.values()}
  if name not in names:
    raise ValueError(f"unsupported packed QK format {name!r}; known formats: {sorted(names)}")

__all__ = [
  "PackedQKFormat", "PackedQKLoadTile", "PackedQKTile", "Q4_K_FORMAT", "Q6_K_FORMAT",
  "Q4_U32_SCALAR_LOAD", "Q4_U32X4_ALIGNED_LOAD", "Q6_U16_SCALAR_LOAD", "format_for_type",
  "tile_from_info", "tile_from_semantic_row", "load_tile", "tile_to_json", "tile_summary",
  "qk_tile_search_axes", "format_markdown", "assert_known_qk_format_name",
]
