"""Pure cooperative LDS ownership math for compiler-bound kernel geometry."""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad.uop.ops import KernelLDSWindow, KernelTileGeometry


@dataclass(frozen=True)
class CooperativeLDSStore:
  role: str
  thread: int
  iteration: int
  row: int
  vector: int
  byte_offset: int
  vector_bytes: int


def semantic_wave_coords(geometry:KernelTileGeometry, thread:int) -> tuple[int, int, int]:
  """Return (wave_m, wave_n, lane) using row-major wave ownership."""
  if not isinstance(thread, int) or isinstance(thread, bool) or not 0 <= thread < geometry.threads:
    raise ValueError(f"thread must be in [0, {geometry.threads})")
  wave_id, lane = divmod(thread, geometry.wave_size)
  wave_m, wave_n = divmod(wave_id, geometry.waves[1])
  return wave_m, wave_n, lane


def _window(geometry:KernelTileGeometry, role:str) -> KernelLDSWindow:
  if role not in ("A", "B"): raise ValueError(f"cooperative LDS role must be A or B, got {role!r}")
  return next(w for w in geometry.lds_windows if w.role == role)


def cooperative_lds_stores(geometry:KernelTileGeometry, role:str, *, element_bytes:int=2,
                           vector_bytes:int=16) -> tuple[CooperativeLDSStore, ...]:
  """Elect one thread for every non-padding vector in an A or B tile window."""
  if not isinstance(element_bytes, int) or isinstance(element_bytes, bool) or element_bytes <= 0:
    raise ValueError("element_bytes must be a positive int")
  if not isinstance(vector_bytes, int) or isinstance(vector_bytes, bool) or vector_bytes <= 0:
    raise ValueError("vector_bytes must be a positive int")
  window = _window(geometry, role)
  rows = geometry.tile[0] if role == "A" else geometry.tile[1]
  row_data_bytes = geometry.tile[2] * element_bytes
  if row_data_bytes % vector_bytes: raise ValueError("tile K row bytes must be divisible by vector_bytes")
  if window.stride_bytes < row_data_bytes or window.stride_bytes % vector_bytes:
    raise ValueError("LDS stride must contain the data row and be vector aligned")
  if window.end - window.base != rows * window.stride_bytes:
    raise ValueError("LDS window size must exactly equal rows * stride")
  vectors_per_row = row_data_bytes // vector_bytes
  vector_count = rows * vectors_per_row
  if vector_count % geometry.threads:
    raise ValueError("cooperative tile vectors must divide evenly across threads")
  stores = []
  for linear in range(vector_count):
    thread, iteration = linear % geometry.threads, linear // geometry.threads
    row, vector = divmod(linear, vectors_per_row)
    byte_offset = window.base + row * window.stride_bytes + vector * vector_bytes
    stores.append(CooperativeLDSStore(role, thread, iteration, row, vector, byte_offset, vector_bytes))
  return tuple(stores)


def cooperative_lds_padding_offsets(geometry:KernelTileGeometry, role:str, *, element_bytes:int=2,
                                    vector_bytes:int=16) -> tuple[int, ...]:
  """Return vector-aligned padding slots, which intentionally have no store owner."""
  window = _window(geometry, role)
  rows = geometry.tile[0] if role == "A" else geometry.tile[1]
  row_data_bytes = geometry.tile[2] * element_bytes
  if row_data_bytes % vector_bytes or window.stride_bytes < row_data_bytes or window.stride_bytes % vector_bytes:
    raise ValueError("LDS row data and stride must be valid vector-aligned intervals")
  return tuple(window.base + row * window.stride_bytes + offset
               for row in range(rows) for offset in range(row_data_bytes, window.stride_bytes, vector_bytes))
