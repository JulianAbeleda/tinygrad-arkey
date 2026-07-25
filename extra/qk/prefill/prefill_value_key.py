"""Canonical source-value identity for generated prefill LDS staging metadata."""
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any

@dataclass(frozen=True, slots=True)
class PrefillSourceValueKey:
  role:str; output_tile:tuple[Any, ...]; k_epoch:Any; k_phase:Any; vector_offset:Any; source_id:Any; buffer_id:Any
  def __post_init__(self) -> None:
    values = tuple(getattr(self, field.name) for field in fields(self) if field.name != "role")
    if self.role not in ("A", "B"): raise ValueError(f"source-value key requires role A or B, got {self.role!r}")
    if any(value is None for value in values): raise ValueError("source-value key fields must be complete")
    try: hash(self)
    except TypeError as exc: raise TypeError("source-value key fields must be hashable") from exc
  def tag_fields(self) -> tuple[tuple[str, Any], ...]: return tuple((field.name, getattr(self, field.name)) for field in fields(self))
  def to_json(self) -> dict[str, Any]:
    def plain(value:Any) -> Any: return [plain(x) for x in value] if isinstance(value, tuple) else value
    return {field.name: plain(getattr(self, field.name)) for field in fields(self)}

def stable_uop_token(uop:Any) -> tuple[str, str]:
  key = getattr(uop, "key", None)
  if not isinstance(key, bytes) or not key: raise ValueError("source-value key requires a stable UOp structural key")
  return ("uop_key", key.hex())

def single_buffer_stage_value_key(*, role:str, tile_idx:Any, tile_count:int, source:Any, lds_buffer_id:int) -> PrefillSourceValueKey:
  tile_token, source_token = stable_uop_token(tile_idx), stable_uop_token(source)
  return PrefillSourceValueKey(role, ("symbolic_output_tile", tile_token, tile_count), ("source_epoch", source_token),
    ("single_buffer", 0), ("operand_stage_wide", 0), source_token, ("lds", lds_buffer_id, 0))
