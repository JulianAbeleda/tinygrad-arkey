"""JSON-safe logical vocabulary for research MMQ candidates.

This module describes *what* an MMQ operation means.  It deliberately does
not describe a final lane/index schedule; a lowering is free to map these
logical axes to a backend later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

VOCABULARY_VERSION = "mmq-logical-vocabulary/1"


class _StrEnum(str, Enum):
  def __str__(self) -> str: return self.value


class DType(_StrEnum):
  Q4_PACKED = "q4_packed"; I8 = "i8"; F16 = "f16"; F32 = "f32"; I32 = "i32"


class Stage(_StrEnum):
  DIRECT = "direct"; REGISTERS = "registers"; LDS = "lds"


class DotOp(_StrEnum):
  DOT_I8_I8_I32 = "dot_i8_i8_i32"; WMMA_I8_I8_I32 = "wmma_i8_i8_i32"


class SyncScope(_StrEnum):
  NONE = "none"; WORKGROUP = "workgroup"


@dataclass(frozen=True)
class Axis:
  name: str
  extent: int | str
  tile: int | None = None
  unit: str = "elements"

  def __post_init__(self) -> None:
    if not self.name or self.name not in {"m", "n", "k", "group", "activation_block"}:
      raise ValueError("axis name must be one of m, n, k, group, activation_block")
    if isinstance(self.extent, int) and self.extent <= 0: raise ValueError("axis extent must be positive")
    if self.tile is not None and self.tile <= 0: raise ValueError("axis tile must be positive")


@dataclass(frozen=True)
class EdgePredicate:
  axis: str
  predicate: str = "index < extent"
  required: bool = True


@dataclass(frozen=True)
class Q4KDecode:
  block_elements: int = 256
  packed_words: int = 32
  scale_bits: int = 6
  minimum_bits: int = 6
  nibble_ownership: str = "two_values_per_byte_low_then_high"


@dataclass(frozen=True)
class Q8DS4Semantics:
  block_elements: int = 32
  values_dtype: DType = DType.I8
  scale_dtype: DType = DType.F32
  sum_policy: str = "derived"
  sum_operand: bool = False

  def __post_init__(self) -> None:
    if self.sum_policy not in {"derived", "supplied"}: raise ValueError("invalid Q8 sum policy")
    if self.sum_policy == "supplied" and not self.sum_operand: raise ValueError("supplied sums must be operands")


@dataclass(frozen=True)
class Operation:
  name: DotOp
  lhs_dtype: DType = DType.I8
  rhs_dtype: DType = DType.I8
  accumulator_dtype: DType = DType.I32
  legal: bool = True

  def __post_init__(self) -> None:
    if not self.legal or (self.lhs_dtype, self.rhs_dtype, self.accumulator_dtype) != (DType.I8, DType.I8, DType.I32):
      raise ValueError("MMQ operation must be a legal i8 x i8 -> i32 operation")


@dataclass(frozen=True)
class Staging:
  weights: Stage = Stage.DIRECT
  activations: Stage = Stage.DIRECT
  accumulator: Stage = Stage.REGISTERS
  lifetime: str = "k_tile"


@dataclass(frozen=True)
class Synchronization:
  scope: SyncScope = SyncScope.NONE
  points: tuple[str, ...] = ()
  uniform: bool = True

  def __post_init__(self) -> None:
    if not self.uniform: raise ValueError("MMQ synchronization must be uniform")


@dataclass(frozen=True)
class Ownership:
  accumulator_owner: str = "logical_mn_tile"
  writeback: str = "exactly_one_owner"

  def __post_init__(self) -> None:
    if self.writeback != "exactly_one_owner": raise ValueError("writeback must have one owner")


@dataclass(frozen=True)
class BackendCapability:
  backend: str
  device: str
  supported_ops: tuple[DotOp, ...] = (DotOp.DOT_I8_I8_I32,)
  wave_sizes: tuple[int, ...] = (32,)
  max_workgroup_size: int | None = None
  lds_bytes: int | None = None


@dataclass(frozen=True)
class PhysicalMapping:
  """Candidate choice, intentionally expressed without lane/index schedules."""
  wave_size: int
  workgroup_size: int
  tile_order: tuple[str, ...] = ("m", "n", "k")
  wmma_shape: tuple[int, int, int] = (16, 16, 16)
  lifecycle: str = "tiled"

  def __post_init__(self) -> None:
    if len(self.wmma_shape) != 3 or min(self.wmma_shape) <= 0:
      raise ValueError("WMMA shape must have three positive dimensions")
    if self.lifecycle not in {"tiled", "group", "scheduler", "packed_ds4"}:
      raise ValueError("unsupported MMQ lifecycle")


@dataclass(frozen=True)
class LogicalMMQDescriptor:
  axes: tuple[Axis, ...]
  q4k: Q4KDecode = field(default_factory=Q4KDecode)
  q8: Q8DS4Semantics = field(default_factory=Q8DS4Semantics)
  operation: Operation = field(default_factory=lambda: Operation(DotOp.WMMA_I8_I8_I32))
  staging: Staging = field(default_factory=Staging)
  synchronization: Synchronization = field(default_factory=Synchronization)
  ownership: Ownership = field(default_factory=Ownership)
  edge_predicates: tuple[EdgePredicate, ...] = ()
  abi: Mapping[str, Any] = field(default_factory=lambda: {"output_layout": "tokens_rows"})

  def __post_init__(self) -> None:
    names = [a.name for a in self.axes]
    if set(names) != {"m", "n", "k", "group", "activation_block"}: raise ValueError("all logical MMQ axes are required")
    if self.q4k.block_elements % self.q8.block_elements: raise ValueError("quant blocks must have compatible grouping")
    if not self.edge_predicates: raise ValueError("edge predicates must be explicit")


@dataclass(frozen=True)
class MMQCandidate:
  descriptor: LogicalMMQDescriptor
  mapping: PhysicalMapping
  capability: BackendCapability
  lowering_version: str = "logical-lowering/1"
  provenance: str = "research"
  rollback_identity: str = "direct-packed"

  def to_dict(self) -> dict[str, Any]:
    return _jsonable(asdict(self))

  def canonical_json(self) -> str:
    return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

  def identity(self) -> str:
    payload = {"vocabulary": VOCABULARY_VERSION, "candidate": self.to_dict()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "mmq-" + hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
  if isinstance(value, Enum): return value.value
  if isinstance(value, Mapping): return {str(k): _jsonable(v) for k, v in value.items()}
  if isinstance(value, (tuple, list)): return [_jsonable(v) for v in value]
  if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
  return value


__all__ = ["VOCABULARY_VERSION", "Axis", "BackendCapability", "DType", "DotOp", "EdgePredicate", "LogicalMMQDescriptor", "MMQCandidate", "Ownership", "PhysicalMapping", "Q4KDecode", "Q8DS4Semantics", "Stage", "Staging", "Synchronization", "SyncScope"]
