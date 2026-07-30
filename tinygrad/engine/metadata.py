"""Target-neutral side metadata for concrete buffer allocations and calls."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import weakref

from tinygrad.helpers import Metadata, dedup
from tinygrad.uop.ops import Ops, UOp

PROGRAM_IDENTITY_FIELDS = ("phase", "tensor_name", "module_path", "role", "logical_m", "logical_n", "logical_k",
  "source_quant_storage", "source_layout", "module_representation", "input_dtype", "output_dtype", "accumulator_dtype")

@dataclass(frozen=True)
class _BufferMetadataRegion:
  byte_offset: int
  byte_length: int
  payload: Any

_buffer_metadata:weakref.WeakKeyDictionary[UOp, tuple[_BufferMetadataRegion, ...]] = weakref.WeakKeyDictionary()
_call_metadata_resolvers:list[Callable[[UOp], tuple[Metadata, ...]]] = []

def _as_uop(value:Any) -> UOp|None:
  uop = value if isinstance(value, UOp) else getattr(value, "uop", None)
  return uop if isinstance(uop, UOp) else None

def _buffer_region(value:Any) -> tuple[UOp, int, int]|None:
  uop = _as_uop(value)
  if uop is None or uop.op is Ops.PARAM: return None
  # Only erase wrappers which retain one exact logical byte region. CONTIGUOUS
  # may copy that region to a new allocation; this registry follows the exact
  # logical contents, not allocator aliasing. General movement, CAST, PAD and
  # EXPAND can change or sparsify the addressed bytes.
  while uop.op in {Ops.MEMORY_SEMANTIC, Ops.AFTER, Ops.RESHAPE, Ops.DETACH, Ops.BITCAST, Ops.CONTIGUOUS} and uop.src: uop = uop.src[0]
  byte_offset = 0
  length = None
  while uop.op is Ops.SLICE:
    try:
      byte_offset += int(uop.src[1].arg) * uop.src[0].dtype.itemsize
      # The first SLICE is the queried/bound region. Parent SLICE lengths must
      # not widen a nested subview back to the parent's extent.
      if length is None: length = int(uop.arg) * uop.dtype.itemsize
    except (AttributeError, TypeError, ValueError): return None
    uop = uop.src[0]
    while uop.op in {Ops.MEMORY_SEMANTIC, Ops.AFTER} and uop.src: uop = uop.src[0]
  if uop.op is not Ops.BUFFER: return None
  try: return uop, byte_offset, length if length is not None else int(uop.arg) * uop.dtype.itemsize
  except (AttributeError, TypeError, ValueError): return None

def bind_buffer_metadata(value:Any, payload:Any) -> None:
  """Bind opaque immutable payload to existing allocation identities only."""
  if (region := _buffer_region(value)) is None: return
  base, offset, length = region
  binding = _BufferMetadataRegion(offset, length, payload)
  current = _buffer_metadata.get(base, ())
  if binding not in current: _buffer_metadata[base] = (*current, binding)

def bind_buffer_metadata_region(value:Any, byte_offset:int, byte_length:int, payload:Any) -> bool:
  """Bind payload to one validated byte interval within an existing backing."""
  if not isinstance(byte_offset, int) or isinstance(byte_offset, bool) or byte_offset < 0:
    raise ValueError("byte_offset must be a non-negative integer")
  if not isinstance(byte_length, int) or isinstance(byte_length, bool) or byte_length <= 0:
    raise ValueError("byte_length must be a positive integer")
  if (region := _buffer_region(value)) is None: return False
  base, offset, length = region
  if byte_offset + byte_length > length: raise ValueError("metadata interval exceeds backing region")
  binding = _BufferMetadataRegion(offset + byte_offset, byte_length, payload)
  current = _buffer_metadata.get(base, ())
  if binding not in current: _buffer_metadata[base] = (*current, binding)
  return True

def buffer_byte_length(value:Any) -> int|None:
  """Return the exact contiguous byte extent represented by a buffer/view."""
  return None if (region := _buffer_region(value)) is None else region[2]

def buffer_metadata(value:Any) -> tuple[Any, ...]:
  if (region := _buffer_region(value)) is None: return ()
  base, offset, length = region
  found = []
  for binding in _buffer_metadata.get(base, ()):
    if offset >= binding.byte_offset and offset + length <= binding.byte_offset + binding.byte_length and binding.payload not in found:
      found.append(binding.payload)
  return tuple(found)

def propagate_buffer_metadata(source:Any, target:Any) -> None:
  for payload in buffer_metadata(source): bind_buffer_metadata(target, payload)

def register_call_metadata_resolver(resolver:Callable[[UOp], tuple[Metadata, ...]]) -> None:
  if resolver not in _call_metadata_resolvers: _call_metadata_resolvers.append(resolver)

def resolve_call_metadata(call:UOp) -> tuple[Metadata, ...]:
  return tuple(dedup(item for resolver in _call_metadata_resolvers for item in resolver(call)))
