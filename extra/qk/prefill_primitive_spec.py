"""Small, JSON-safe contract shared by generated prefill primitives.

This module describes a candidate; lowering owns schedule selection.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json
from typing import Any

_ROLES = ("attn_qo", "attn_kv", "ffn_gate_up", "ffn_down", "lm_head", "test")

def target_capabilities(target: str) -> dict[str, int]:
  """Return the small hardware contract used by descriptor validation."""
  target = target.lower()
  if "gfx1100" in target: return {"wave_width": 32, "max_workgroup_size": 1024}
  return {"wave_width": 64, "max_workgroup_size": 1024}

@dataclass(frozen=True)
class PrimitiveABI:
  arguments: tuple[str, ...] = ("out", "weights", "activation")
  dtypes: tuple[str, ...] = ("float32", "uint8", "uint8")
  output_layout: str = "tokens_rows"

  def to_json(self) -> dict[str, Any]: return asdict(self) | {"arguments": list(self.arguments), "dtypes": list(self.dtypes)}

@dataclass(frozen=True)
class LaunchMetadata:
  workgroup_size: int
  waves: int
  grid: tuple[int, int, int] = (0, 0, 1)
  uniform_barriers: bool = True

  def to_json(self) -> dict[str, Any]: return asdict(self) | {"grid": list(self.grid)}

@dataclass(frozen=True)
class PrefillPrimitiveSpec:
  workload: str
  profile: str
  role: str
  quant_format: str
  activation_format: str
  weight_layout: str
  output_layout: str
  m: int
  n: int
  k: int
  parts: int = 1
  target: str = "amd_gfx1100"
  backend_strategy: str = "generated"
  schedule_options: tuple[tuple[str, Any], ...] = ()
  abi: PrimitiveABI = PrimitiveABI()
  launch: LaunchMetadata | None = None

  def validate(self) -> None:
    if not all(isinstance(x, str) and x for x in (self.workload, self.profile, self.role, self.quant_format, self.activation_format, self.weight_layout, self.output_layout, self.target, self.backend_strategy)):
      raise ValueError("primitive metadata strings must be non-empty")
    if self.role not in _ROLES: raise ValueError(f"unsupported role={self.role!r}")
    if min(self.m, self.n, self.k, self.parts) <= 0: raise ValueError("m/n/k/parts must be positive")
    if len(set(k for k, _ in self.schedule_options)) != len(self.schedule_options): raise ValueError("schedule option keys must be unique")
    if self.launch is not None:
      caps = target_capabilities(self.target)
      if self.launch.workgroup_size <= 0 or self.launch.waves <= 0: raise ValueError("invalid launch metadata")
      if self.launch.workgroup_size % caps["wave_width"] or self.launch.workgroup_size // caps["wave_width"] != self.launch.waves:
        raise ValueError(f"launch wave mapping is invalid for target {self.target}")
    if len(self.abi.arguments) != len(self.abi.dtypes) or len(set(self.abi.arguments)) != len(self.abi.arguments): raise ValueError("ABI arguments and dtypes must be parallel and unique")

  def to_json(self) -> dict[str, Any]:
    self.validate()
    d = asdict(self); d["schedule_options"] = {k: v for k, v in self.schedule_options}
    d["abi"] = self.abi.to_json(); d["launch"] = None if self.launch is None else self.launch.to_json()
    return d

  def canonical_payload(self) -> dict[str, Any]: return {"schema": "tinygrad.prefill_primitive.v1", **self.to_json()}
  def canonical_identity(self) -> str:
    raw = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

def canonical_payload(spec: PrefillPrimitiveSpec) -> dict[str, Any]: return spec.canonical_payload()
def canonical_identity(spec: PrefillPrimitiveSpec) -> str: return spec.canonical_identity()
