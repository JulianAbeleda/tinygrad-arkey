"""Pure cooperative LDS ownership math for compiler-bound kernel geometry."""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad.uop.ops import KernelLDSWindow, KernelTileGeometry

_RDNA3_DIMS = (16, 16, 16)
_RDNA3_ELEMENTS = (16, 16, 8)
_RDNA3_OPTS = ("l0", "l0", "l0", "l0", "l1", "u1", "u1", "u1")
_RDNA3_SWIZZLE = ((('l4', 'u0', 'u1', 'u2', 'l0'), ('r1', 'r2', 'r3'), ('l1', 'l2', 'l3', 'r0')),
                  (('l0', 'l1', 'l2', 'l3', 'l4'), ('r1', 'r2', 'r3'), ('u0', 'u1', 'u2', 'r0')))
_RDNA3_REMAPS = ({'l0': 'l4', 'l1': 'u0', 'l2': 'u1', 'l3': 'u2', 'l4': 'l0', 'u0': 'r1', 'u1': 'r2', 'u2': 'r3',
                   'r0': 'l1', 'r1': 'l2', 'r2': 'l3', 'r3': 'r0'},
                  {'l0': 'l0', 'l1': 'l1', 'l2': 'l2', 'l3': 'l3', 'l4': 'l4', 'u0': 'r1', 'u1': 'r2', 'u2': 'r3',
                   'r0': 'u0', 'r1': 'u1', 'r2': 'u2', 'r3': 'r0'})


def validate_rdna3_wmma_descriptor(tc) -> None:
  """Admit only the exact tensor-core descriptor this mapping proves."""
  fields = (("dims", _RDNA3_DIMS), ("threads", 32), ("elements_per_thread", _RDNA3_ELEMENTS),
            ("opts", _RDNA3_OPTS), ("swizzle", _RDNA3_SWIZZLE))
  for name, expected in fields:
    if getattr(tc, name, None) != expected: raise ValueError(f"RDNA3 WMMA descriptor {name} drifted")
  if getattr(getattr(tc, "dtype_in", None), "name", None) != "half": raise ValueError("RDNA3 WMMA descriptor dtype_in drifted")
  if getattr(getattr(tc, "dtype_out", None), "name", None) != "float": raise ValueError("RDNA3 WMMA descriptor dtype_out drifted")
  try: remaps = tuple(tc.lane_map.remaps())
  except (AttributeError, AssertionError, TypeError, ValueError) as exc:
    raise ValueError("RDNA3 WMMA descriptor remaps are unavailable or invalid") from exc
  if remaps != _RDNA3_REMAPS: raise ValueError("RDNA3 WMMA descriptor remaps drifted")


@dataclass(frozen=True)
class CooperativeLDSStore:
  role: str
  thread: int
  iteration: int
  row: int
  vector: int
  byte_offset: int
  vector_bytes: int

@dataclass(frozen=True)
class WMMAFragmentLoad:
  role: str
  thread: int
  wave_m: int
  wave_n: int
  subtile: int
  k_substep: int
  element: int
  logical_row: int
  logical_k: int
  byte_offset: int

@dataclass(frozen=True)
class WMMAOutputOwner:
  thread: int
  wave_m: int
  wave_n: int
  subtile_m: int
  subtile_n: int
  element: int
  row: int
  col: int


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


def _rdna3_wmma_output_coord(lane:int, element:int) -> tuple[int, int]:
  if not isinstance(lane, int) or isinstance(lane, bool) or not 0 <= lane < 32: raise ValueError("lane must be in [0, 32)")
  if not isinstance(element, int) or isinstance(element, bool) or not 0 <= element < 8: raise ValueError("element must be in [0, 8)")
  return lane % 16, lane // 16 + element * 2

def rdna3_wmma_output_coord(lane:int, element:int, *, tc) -> tuple[int, int]:
  """RDNA3 fp32 16x16x16 output map used by the Python WMMA interpreter."""
  validate_rdna3_wmma_descriptor(tc)
  return _rdna3_wmma_output_coord(lane, element)


def wmma_fragment_loads(geometry:KernelTileGeometry, role:str, *, tc, element_bytes:int=2) -> tuple[WMMAFragmentLoad, ...]:
  """Enumerate per-wave RDNA3 A/B fragment loads from the staged tile windows."""
  validate_rdna3_wmma_descriptor(tc)
  window = _window(geometry, role)
  if geometry.wave_size != 32 or geometry.tile[2] % 16:
    raise ValueError("RDNA3 fragment mapping requires wave32 and K divisible by 16")
  if element_bytes != 2: raise ValueError("RDNA3 fp16 fragment mapping requires element_bytes=2")
  subtiles = geometry.tile[0] // (geometry.waves[0] * 16) if role == "A" else \
             geometry.tile[1] // (geometry.waves[1] * 16)
  if subtiles <= 0 or (geometry.tile[0] if role == "A" else geometry.tile[1]) != \
     subtiles * (geometry.waves[0] if role == "A" else geometry.waves[1]) * 16:
    raise ValueError("tile extent must divide exactly into wave 16x16 subtiles")
  loads = []
  for thread in range(geometry.threads):
    wave_m, wave_n, lane = semantic_wave_coords(geometry, thread)
    for subtile in range(subtiles):
      logical_row = (wave_m * subtiles + subtile) * 16 + lane % 16 if role == "A" else \
                    (wave_n * subtiles + subtile) * 16 + lane % 16
      for k_substep in range(geometry.tile[2] // 16):
        for element in range(16):
          logical_k = k_substep * 16 + element
          byte_offset = window.base + logical_row * window.stride_bytes + logical_k * element_bytes
          if not window.base <= byte_offset or byte_offset + element_bytes > window.end:
            raise ValueError("RDNA3 fragment load is outside its LDS window")
          loads.append(WMMAFragmentLoad(role, thread, wave_m, wave_n, subtile, k_substep, element,
                                        logical_row, logical_k, byte_offset))
  return tuple(loads)


def wmma_output_owners(geometry:KernelTileGeometry, *, tc) -> tuple[WMMAOutputOwner, ...]:
  """Enumerate RDNA3 output ownership for every wave and its 2-D WMMA subtile grid."""
  validate_rdna3_wmma_descriptor(tc)
  if geometry.wave_size != 32: raise ValueError("RDNA3 output mapping requires wave32")
  subtiles_m = geometry.tile[0] // (geometry.waves[0] * 16)
  subtiles_n = geometry.tile[1] // (geometry.waves[1] * 16)
  if (subtiles_m <= 0 or subtiles_n <= 0 or geometry.tile[0] != subtiles_m * geometry.waves[0] * 16 or
      geometry.tile[1] != subtiles_n * geometry.waves[1] * 16):
    raise ValueError("output tile must divide exactly into wave 16x16 subtiles")
  owners = []
  for thread in range(geometry.threads):
    wave_m, wave_n, lane = semantic_wave_coords(geometry, thread)
    for subtile_m in range(subtiles_m):
      for subtile_n in range(subtiles_n):
        for element in range(8):
          local_row, local_col = _rdna3_wmma_output_coord(lane, element)
          row = (wave_m * subtiles_m + subtile_m) * 16 + local_row
          col = (wave_n * subtiles_n + subtile_n) * 16 + local_col
          owners.append(WMMAOutputOwner(thread, wave_m, wave_n, subtile_m, subtile_n, element, row, col))
  return tuple(owners)
