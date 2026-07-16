"""Metadata-only construction of an exact memory ledger for one selected GGUF.

Model labels and profiles are deliberately audit metadata only.  Byte counts
come from the selected file's tensor table and explicit runtime geometry/facts.
Facts which the GGUF cannot establish (notably device allocation alignment and
resident copy policy) default to ``None`` so admission fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from tinygrad.helpers import prod
from tinygrad.llm.gguf import _GGML_NATIVE, _GGML_QUANT, gguf_load_metadata
from tinygrad.llm.memory_ledger import AllocationKind, AllocationProvenance, LedgerAllocation, SelectedModelMemoryLedger


MetadataLoader = Callable[[str | Path], tuple[dict, dict]]


@dataclass(frozen=True)
class RuntimeGeometry:
  """Geometry and measured runtime facts for the selected invocation.

  Byte widths are required to derive their corresponding allocation.  They
  intentionally have no conventional defaults: a missing runtime fact is not
  permission to assume fp16, fp32, batch one, or a backend scratch policy.
  """
  num_blocks: int
  n_kv_heads: int
  head_dim: int
  max_context: int
  prefill_ubatch: int
  batch_size: int | None = None
  kv_element_bytes: int | None = None
  kv_scale_element_bytes: int | None = None
  kv_scales_per_token: int = 0
  runtime_persistent_bytes: int | None = None
  peak_prefill_activation_bytes: int | None = None
  peak_prefill_output_bytes: int | None = None
  peak_prefill_scratch_bytes: int | None = None

  def __post_init__(self):
    for name in ("num_blocks", "n_kv_heads", "head_dim", "max_context", "prefill_ubatch"):
      if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool) or getattr(self, name) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    for name in ("batch_size", "kv_element_bytes", "kv_scale_element_bytes"):
      value = getattr(self, name)
      if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError(f"{name} must be a positive integer or None")
    if not isinstance(self.kv_scales_per_token, int) or isinstance(self.kv_scales_per_token, bool) or self.kv_scales_per_token < 0:
      raise ValueError("kv_scales_per_token must be a non-negative integer")
    for name in ("runtime_persistent_bytes", "peak_prefill_activation_bytes", "peak_prefill_output_bytes",
                 "peak_prefill_scratch_bytes"):
      value = getattr(self, name)
      if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None")

  @property
  def kv_bytes(self) -> int | None:
    if self.batch_size is None or self.kv_element_bytes is None: return None
    values = 2 * self.num_blocks * self.batch_size * self.n_kv_heads * self.max_context * self.head_dim * self.kv_element_bytes
    if self.kv_scales_per_token:
      if self.kv_scale_element_bytes is None: return None
      values += self.num_blocks * self.batch_size * self.max_context * self.kv_scales_per_token * self.kv_scale_element_bytes
    return values


@dataclass(frozen=True)
class CandidateWorkspace:
  candidate_id: str
  bytes: int | None
  detail: str

  def __post_init__(self):
    if not self.candidate_id or not self.detail: raise ValueError("candidate workspace requires an id and provenance detail")
    if self.bytes is not None and (not isinstance(self.bytes, int) or isinstance(self.bytes, bool) or self.bytes < 0):
      raise ValueError("candidate workspace bytes must be a non-negative integer or None")


@dataclass(frozen=True)
class TensorPayloadSpan:
  name: str
  ggml_type: int
  shape: tuple[int, ...]
  relative_offset: int
  absolute_offset: int
  payload_bytes: int | None
  span_bytes: int | None


@dataclass(frozen=True)
class GGUFMemoryScan:
  model_path: Path
  ledger: SelectedModelMemoryLedger
  tensor_spans: tuple[TensorPayloadSpan, ...]
  metadata: Mapping


def _payload_bytes(shape:tuple[int, ...], ggml_type:int) -> int | None:
  elements = prod(shape)
  if (dtype := _GGML_NATIVE.get(ggml_type)) is not None: return elements * dtype.itemsize
  if (block := _GGML_QUANT.get(ggml_type)) is not None:
    return None if elements % block[0] else elements // block[0] * block[1]
  return None


def selected_gguf_backing_bytes(model_path:str|Path, allocation_alignment:int|None) -> int | None:
  """Exact allocation size of tinygrad's path-based whole-file GGUF backing buffer."""
  if allocation_alignment is None: return None
  if not isinstance(allocation_alignment, int) or isinstance(allocation_alignment, bool) or allocation_alignment <= 0:
    raise ValueError("allocation_alignment must be a positive integer or None")
  try: size = Path(model_path).stat().st_size
  except OSError: return None
  return ((size + allocation_alignment - 1) // allocation_alignment) * allocation_alignment


def _tensor_spans(meta:Mapping, file_size:int|None) -> tuple[TensorPayloadSpan, ...]:
  data_start, infos = meta.get("data_start"), tuple(meta.get("tensor_infos", ()))
  if not isinstance(data_start, int) or data_start < 0: raise ValueError("GGUF metadata has no valid data_start")
  ordered = sorted(enumerate(infos), key=lambda item: item[1][3])
  if len({info[0] for info in infos}) != len(infos): raise ValueError("GGUF tensor names must be unique")
  out: list[TensorPayloadSpan] = []
  for pos, (_, (name, dims, typ, off)) in enumerate(ordered):
    if not isinstance(off, int) or off < 0: raise ValueError(f"GGUF tensor {name!r} has an invalid offset")
    end = data_start + ordered[pos+1][1][3] if pos+1 < len(ordered) else file_size
    span = None if end is None else end - (data_start + off)
    if span is not None and span < 0: raise ValueError(f"GGUF tensor {name!r} extends beyond the selected file")
    payload = _payload_bytes(tuple(dims), typ)
    if payload is not None and span is not None and payload > span:
      raise ValueError(f"GGUF tensor {name!r} payload ({payload}) exceeds its file span ({span})")
    out.append(TensorPayloadSpan(name, typ, tuple(reversed(dims)), off, data_start+off, payload, span))
  return tuple(out)


def scan_selected_gguf_memory(model_path:str|Path, geometry:RuntimeGeometry, candidates:Sequence[CandidateWorkspace], *,
                              allocation_alignment:int|None=None, resident_copies:int|None=None,
                              metadata_loader:MetadataLoader=gguf_load_metadata,
                              metadata:tuple[dict, dict]|None=None,
                              file_size:int|None=None) -> GGUFMemoryScan:
  """Read metadata for ``model_path`` and construct exact-ledger inputs.

  ``file_size`` exists for injectable unit metadata.  For the real loader it is
  obtained read-only from the selected path.
  """
  path = Path(model_path)
  # Production passes the metadata it already opened to derive the route inventory.  Keeping the injectable loader
  # is useful for standalone callers, but an explicit metadata tuple is authoritative and prevents a second scan.
  kv, meta = metadata if metadata is not None else metadata_loader(path)
  if file_size is None:
    try: file_size = path.stat().st_size
    except OSError: file_size = None
  spans = _tensor_spans(meta, file_size)
  selected = f"selected path={path}"
  allocations: list[LedgerAllocation] = []
  for span in spans:
    detail = (f"{selected}; ggml_type={span.ggml_type}; shape={span.shape}; relative_offset={span.relative_offset}; "
              f"absolute_offset={span.absolute_offset}; payload_span={span.span_bytes}")
    allocations.append(LedgerAllocation.gguf_tensor(span.name, span.payload_bytes, allocation_alignment, resident_copies,
                                                     AllocationProvenance("gguf_load_metadata tensor table", detail)))
  runtime_source = AllocationProvenance("runtime/config geometry", selected)
  allocations.extend((
    LedgerAllocation("kv_cache", AllocationKind.KV_CACHE, geometry.kv_bytes, runtime_source),
    LedgerAllocation("runtime_persistent", AllocationKind.RUNTIME_PERSISTENT, geometry.runtime_persistent_bytes, runtime_source),
    LedgerAllocation("peak_prefill_activations", AllocationKind.PREFILL_ACTIVATION, geometry.peak_prefill_activation_bytes, runtime_source),
    LedgerAllocation("peak_prefill_outputs", AllocationKind.PREFILL_OUTPUT, geometry.peak_prefill_output_bytes, runtime_source),
    LedgerAllocation("peak_prefill_scratch", AllocationKind.PREFILL_SCRATCH, geometry.peak_prefill_scratch_bytes, runtime_source),
  ))
  for candidate in candidates:
    allocations.append(LedgerAllocation(f"workspace:{candidate.candidate_id}", AllocationKind.CANDIDATE_WORKSPACE,
      candidate.bytes, AllocationProvenance("candidate workspace", f"{selected}; {candidate.detail}"), candidate_id=candidate.candidate_id))
  return GGUFMemoryScan(path, SelectedModelMemoryLedger(tuple(allocations)), spans, kv)


__all__ = ["CandidateWorkspace", "GGUFMemoryScan", "RuntimeGeometry", "TensorPayloadSpan",
           "scan_selected_gguf_memory", "selected_gguf_backing_bytes"]
