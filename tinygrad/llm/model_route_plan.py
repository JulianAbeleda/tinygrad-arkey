from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from tinygrad.llm.route_policy import q4k_policy, q6k_policy


@dataclass(frozen=True)
class PrimitiveRouteEntry:
  name: str
  module_path: str
  quant_label: str
  rows: int
  cols: int
  role: str
  parts: int
  opts: tuple[str, ...]
  family: str
  kernel_mode: str = "partial"


class ModelRoutePlan:
  def __init__(self, entries:Iterable[PrimitiveRouteEntry]=()):
    self._entries = {entry.name: entry for entry in entries}

  def primitive(self, name:str) -> PrimitiveRouteEntry|None:
    return self._entries.get(name)

  def __len__(self) -> int:
    return len(self._entries)

  def __iter__(self):
    return iter(self._entries.values())


def _module_path_from_tensor_name(name:str) -> str:
  return name[:-len(".weight")] if name.endswith(".weight") else name

def _role_from_module_path(module_path:str) -> str:
  return module_path.rsplit(".", 1)[-1]

def _shape_from_tensor_info(dims) -> tuple[int, int]|None:
  if len(dims) != 2: return None
  rows, cols = tuple(reversed(dims))
  return int(rows), int(cols)

def _facts_tensor_infos(model_facts:Any) -> Iterable[tuple[str, Any, int, int]]:
  if model_facts is None: return ()
  if hasattr(model_facts, "tensor_infos"): return getattr(model_facts, "tensor_infos")
  if hasattr(model_facts, "tensors"): return getattr(model_facts, "tensors")
  if isinstance(model_facts, dict): return model_facts.get("tensor_infos", model_facts.get("tensors", ()))
  return ()

def _entry_from_record(record:Any) -> PrimitiveRouteEntry|None:
  if isinstance(record, PrimitiveRouteEntry): return record
  if isinstance(record, dict):
    name = record.get("name") or record.get("tensor")
    typ = record.get("typ", record.get("ggml_type"))
    rows, cols = record.get("rows"), record.get("cols")
    dims = record.get("dims")
    module_path = record.get("module_path")
    quant_label = record.get("quant_label")
    role = record.get("role")
  else:
    name = getattr(record, "name", getattr(record, "tensor", None))
    typ = getattr(record, "typ", getattr(record, "ggml_type", None))
    rows, cols = getattr(record, "rows", None), getattr(record, "cols", None)
    dims = getattr(record, "dims", None)
    module_path = getattr(record, "module_path", None)
    quant_label = getattr(record, "quant_label", None)
    role = getattr(record, "role", None)
  if name is None or typ is None: return None
  if (rows is None or cols is None) and dims is not None:
    shape = _shape_from_tensor_info(dims)
    if shape is not None: rows, cols = shape
  if rows is None or cols is None: return None
  return primitive_route_entry_for_tensor(str(name), int(typ), int(rows), int(cols),
                                          module_path=None if module_path is None else str(module_path),
                                          quant_label=None if quant_label is None else str(quant_label),
                                          role=None if role is None else str(role))

def primitive_route_entry_for_tensor(name:str, typ:int, rows:int, cols:int, *, module_path:str|None=None,
                                     quant_label:str|None=None, role:str|None=None) -> PrimitiveRouteEntry|None:
  module_path = module_path or _module_path_from_tensor_name(name)
  role = role or _role_from_module_path(module_path)
  if typ == 12:
    policy = q4k_policy(name)
    if policy is None: return None
    parts, opts = policy
    return PrimitiveRouteEntry(name, module_path, quant_label or "Q4_K", rows, cols, role, parts, tuple(opts), "q4_k_packed_u32", "partial")
  if typ == 14:
    policy = q6k_policy(name)
    if policy is None: return None
    parts, opts = policy
    return PrimitiveRouteEntry(name, module_path, quant_label or "Q6_K", rows, cols, role, parts, tuple(opts), "q6_k_packed_u16", "partial")
  return None

def build_model_route_plan(meta:dict|None=None, model_facts:Any=None) -> ModelRoutePlan:
  entries: list[PrimitiveRouteEntry] = []
  for record in _facts_tensor_infos(model_facts):
    if (entry := _entry_from_record(record)) is not None: entries.append(entry)
  if not entries and meta is not None:
    for name, dims, typ, _off in meta.get("tensor_infos", ()):
      if not str(name).endswith(".weight"): continue
      shape = _shape_from_tensor_info(dims)
      if shape is None: continue
      if (entry := primitive_route_entry_for_tensor(str(name), int(typ), *shape)) is not None:
        entries.append(entry)
  return ModelRoutePlan(entries)


try:
  from tinygrad.llm.model_facts import ModelFacts as ModelFacts  # type: ignore
except Exception:
  ModelFacts = None  # type: ignore
