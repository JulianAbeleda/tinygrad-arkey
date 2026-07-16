from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable

@dataclass(frozen=True)
class PrimitiveRouteEntry:
  name:str; module_path:str; quant_label:str; rows:int; cols:int; role:str; parts:int; opts:tuple[str, ...]; family:str
  kernel_mode:str = "partial"

class ModelRoutePlan:
  def __init__(self, entries:Iterable[PrimitiveRouteEntry]=()): self._entries = {entry.name: entry for entry in entries}
  def primitive(self, name:str) -> PrimitiveRouteEntry|None: return self._entries.get(name)
  def __len__(self) -> int: return len(self._entries)
  def __iter__(self): return iter(self._entries.values())

def _module_path(name:str) -> str: return name[:-len(".weight")] if name.endswith(".weight") else name
def _role_family(role:str) -> str:
  if role in ("ffn_gate", "ffn_up", "ffn_gate_up"): return "ffn_gate_up"
  if role == "ffn_down": return role
  if role in ("attn_q", "attn_output", "attn_qo"): return "attn_qo"
  if role in ("attn_k", "attn_v", "attn_kv"): return "attn_kv"
  if role in ("output", "lm_head"): return "lm_head"
  return role

def _default(name:str, quant:str, role:str, rows:int, cols:int) -> tuple[int, tuple[str, ...]]|None:
  if rows <= 0 or cols <= 0 or cols % 256: return None
  family = _role_family(role)
  if quant == "Q4_K":
    if family in ("ffn_gate_up", "attn_qo", "attn_kv"): return 1, ("LOCAL:0:64",)
    if family == "ffn_down": return 4, ("LOCAL:0:32",)
  if quant == "Q6_K":
    if family == "ffn_down" or family == "lm_head" or name == "output.weight": return 1, ("LOCAL:0:64",)
    if family == "attn_kv" and _module_path(name).rsplit(".", 1)[-1] == "attn_v": return 4, ("LOCAL:0:32",)
  return None

def primitive_route_entry_for_tensor(name:str, typ:int, rows:int, cols:int, *, module_path:str|None=None,
                                     quant_label:str|None=None, role:str|None=None, **_unused) -> PrimitiveRouteEntry|None:
  module_path = module_path or _module_path(name); role = role or module_path.rsplit(".", 1)[-1]
  quant = quant_label or ("Q4_K" if typ == 12 else "Q6_K" if typ == 14 else "")
  if not quant or (policy := _default(name, quant, role, rows, cols)) is None: return None
  parts, opts = policy
  return PrimitiveRouteEntry(name, module_path, quant, rows, cols, role, parts, opts,
                             "q4_k_packed_u32" if typ == 12 else "q6_k_packed_u16")

def _record_entry(record:Any) -> PrimitiveRouteEntry|None:
  if isinstance(record, PrimitiveRouteEntry): return record
  get = record.get if isinstance(record, dict) else lambda key, default=None: getattr(record, key, default)
  name, typ = get("name", get("tensor")), get("ggml_type", get("typ"))
  rows, cols, dims = get("rows"), get("cols"), get("dims")
  if (rows is None or cols is None) and dims is not None and len(dims) == 2: rows, cols = reversed(dims)
  if name is None or typ is None or rows is None or cols is None: return None
  return primitive_route_entry_for_tensor(str(name), int(typ), int(rows), int(cols), module_path=get("module_path"),
                                          quant_label=get("quant_label"), role=get("role"))

def build_model_route_plan(meta:dict|None=None, model_facts:Any=None) -> ModelRoutePlan:
  records = getattr(model_facts, "tensors", ()) if model_facts is not None else ()
  entries = [entry for record in records if (entry := _record_entry(record)) is not None]
  if not entries and meta is not None:
    for name, dims, typ, _off in meta.get("tensor_infos", ()):
      if str(name).endswith(".weight") and len(dims) == 2 and (entry := primitive_route_entry_for_tensor(str(name), int(typ), int(dims[1]), int(dims[0]))) is not None:
        entries.append(entry)
  return ModelRoutePlan(entries)

try: from tinygrad.llm.model_facts import ModelFacts as ModelFacts
except Exception: ModelFacts = None
