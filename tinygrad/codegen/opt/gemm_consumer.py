"""Consumer contracts for compiler-owned register-resident GEMM tiles.

The register lifecycle and allocator deliberately do not know which operation
consumes an A/B tile.  This module is the small consumer-owned seam: adapters
validate a logical tile and expose an instruction-family lowering hook.  The
adapters do not allocate physical registers, emit ISA, or claim GPU execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import Ops, UOp


@runtime_checkable
class LogicalTile(Protocol):
  """Duck-typed compiler-neutral tile descriptor.

  The concrete descriptor is owned by the register-storage layer.  Keeping
  this protocol structural avoids making that descriptor depend on WMMA,
  AMD, or any instruction-specific type.
  """
  role: str
  dtype: DType
  carrier_width: int
  tile_shape: tuple[int, ...]
  slot_count: int
  slot_addressing: str
  layout: str


def _tile_value(tile: object, name: str) -> Any:
  value = getattr(tile, name, None)
  if value is None:
    raise ValueError(f"logical tile is missing {name}")
  return value


def validate_logical_tile(tile: LogicalTile, *, identity: str, dtype: DType,
                          carrier_width: int, layout: str,
                          roles: tuple[str, ...] = ("A", "B")) -> None:
  """Validate fields that every consumer must agree on.

  This check is intentionally independent of storage and physical register
  assignment.  A consumer cannot opt into a tile solely because its dtype or
  shape happens to match; its layout and static-addressing contract must also
  be explicit.
  """
  if not isinstance(identity, str) or not identity:
    raise ValueError("consumer identity must be non-empty")
  actual_role = _tile_value(tile, "role")
  if actual_role not in roles:
    raise ValueError(f"{identity} does not accept tile role {actual_role!r}")
  if _tile_value(tile, "dtype") != dtype:
    raise ValueError(f"{identity} tile dtype does not match consumer ABI")
  width = _tile_value(tile, "carrier_width")
  if not isinstance(width, int) or isinstance(width, bool) or width != carrier_width:
    raise ValueError(f"{identity} tile carrier width does not match consumer ABI")
  shape = _tile_value(tile, "tile_shape")
  if not isinstance(shape, tuple) or not shape or any(not isinstance(x, int) or x <= 0 for x in shape):
    raise ValueError(f"{identity} tile shape is invalid")
  if _tile_value(tile, "layout") != layout:
    raise ValueError(f"{identity} tile layout is not proven")
  slots = _tile_value(tile, "slot_count")
  if slots not in (1, 2):
    raise ValueError(f"{identity} tile slot count must be one or two")
  addressing = _tile_value(tile, "slot_addressing")
  if addressing not in ("static", "sequential", "proven"):
    raise ValueError(f"{identity} tile slot addressing is not supported")
  # Dynamic modulo addressing cannot name a VGPR.  Consumers may only receive
  # a logical tile after storage has proven static/sequential mapping.
  if addressing == "static" and slots == 2:
    raise ValueError(f"{identity} double-buffer tile requires backend static-slot proof")


@runtime_checkable
class GemmConsumerAdapter(Protocol):
  """Minimal consumer adapter ABI shared by WMMA and non-WMMA consumers."""
  identity: str
  layout: str
  fragment_dtype: DType
  accumulator_dtype: DType
  carrier_width: int

  def validate_tile(self, tile: LogicalTile) -> None: ...

  def validate_fragment(self, fragment: UOp) -> None: ...


def validate_consumer_wait_coverage(adapter: GemmConsumerAdapter, policy, dependencies: tuple,
                                    required: tuple[tuple[str, int, int], ...] = ()):
  """Join a consumer identity to the existing typed wait proof.

  This does not emit a wait.  It prevents a consumer from being admitted to a
  register-resident route when its producer/load-group/consumer edges are
  absent or attributed to another policy.
  """
  from tinygrad.codegen.opt.compiler_policies import prove_wait_dependency_coverage
  consumer_adapter_identity(adapter)
  coverage = prove_wait_dependency_coverage(policy, dependencies, required)
  if not coverage.passed:
    raise ValueError(f"{adapter.identity} lacks complete wait coverage: {coverage.errors}")
  return coverage


@dataclass(frozen=True)
class WMMAConsumerAdapter:
  """RDNA3 WMMA adapter around the existing storage-independent validators."""
  identity: str = "amd.rdna3.wmma.fp16.v1"
  layout: str = "rdna3_wmma"
  fragment_dtype: DType = dtypes.half.vec(16)
  accumulator_dtype: DType = dtypes.float.vec(8)
  carrier_width: int = 16

  def validate_tile(self, tile: LogicalTile) -> None:
    validate_logical_tile(tile, identity=self.identity, dtype=dtypes.half,
                          carrier_width=self.carrier_width, layout=self.layout)

  def validate_descriptor(self, tc: object) -> None:
    from tinygrad.codegen.opt.kernel_lds import validate_rdna3_wmma_descriptor
    validate_rdna3_wmma_descriptor(tc)

  def validate_node(self, node: UOp) -> None:
    from tinygrad.codegen.opt.kernel_lds import validate_precontract_wmma_abi
    validate_precontract_wmma_abi(node, context=self.identity)

  def validate_fragment(self, fragment: UOp) -> None:
    if not isinstance(fragment, UOp) or fragment.dtype != self.fragment_dtype:
      raise ValueError(f"{self.identity} requires half.vec(16) fragments")


@dataclass(frozen=True)
class Dot2ConsumerAdapter:
  """Bounded fp16x2 dot2 consumer using the existing backend lowering hook."""
  identity: str = "amd.rdna3.dot2.fp16.v1"
  layout: str = "lane_pair"
  fragment_dtype: DType = dtypes.half.vec(2)
  accumulator_dtype: DType = dtypes.float
  carrier_width: int = 2

  def validate_tile(self, tile: LogicalTile) -> None:
    validate_logical_tile(tile, identity=self.identity, dtype=dtypes.half,
                          carrier_width=self.carrier_width, layout=self.layout)

  def lower(self, a: UOp, b: UOp, accumulator: UOp | None = None) -> UOp:
    """Build the canonical two-lane dot2 idiom and reuse `fdot2` lowering.

    The returned CUSTOMI is still a backend-owned lowering marker.  This
    helper only proves the adapter ABI; it intentionally performs no runtime
    compilation or execution.
    """
    if a.dtype != self.fragment_dtype or b.dtype != self.fragment_dtype:
      raise ValueError(f"{self.identity} requires half.vec(2) A/B carriers")
    if accumulator is not None and accumulator.dtype != self.accumulator_dtype:
      raise ValueError(f"{self.identity} requires scalar float accumulator")
    terms = []
    for lane in (0, 1):
      idx = UOp.const(dtypes.weakint, lane)
      av = a.index(idx, dtype=dtypes.half)
      bv = b.index(idx, dtype=dtypes.half)
      terms.append((av * bv).cast(dtypes.float))
    pair = UOp(Ops.ADD, dtypes.float, tuple(terms))
    if accumulator is not None:
      pair = UOp(Ops.ADD, dtypes.float, (accumulator, pair))
    from tinygrad.codegen.experimental import lower_fdot2_add
    lowered = lower_fdot2_add(pair)
    if lowered is None:
      raise ValueError(f"{self.identity} canonical pair was not accepted by fdot2 lowering")
    return lowered

  def validate_fragment(self, fragment: UOp) -> None:
    if not isinstance(fragment, UOp) or fragment.dtype != self.fragment_dtype:
      raise ValueError(f"{self.identity} requires half.vec(2) fragments")


WMMA_CONSUMER = WMMAConsumerAdapter()
DOT2_CONSUMER = Dot2ConsumerAdapter()


def consumer_adapter_identity(adapter: GemmConsumerAdapter) -> str:
  """Return a stable identity for evidence and machine-search joins."""
  if not isinstance(adapter, GemmConsumerAdapter):
    raise TypeError("expected typed GEMM consumer adapter")
  if not isinstance(adapter.identity, str) or not adapter.identity:
    raise ValueError("consumer adapter identity must be non-empty")
  return adapter.identity
