"""Profile-free bounded Q6_K prefill candidate contract.

The cooperative Q6_K x Q8_1/FP16 candidate described here is a capability
surface, not a production route.  In particular, describing a legal logical
grid does not prove that tinygrad lowers its tile lifecycle on gfx1100.  Until
that proof exists admission fails closed and names the existing direct-packed
FP16 emitter as the rollback implementation.

Packed Q6_K bytes are always the authoritative weights.  Every byte in the
candidate workspace is a function of tile geometry, never of the full N*K
matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import Any, Callable, Mapping, Sequence

from extra.qk.q6k_mmq_vocabulary import Q6K_BLOCK_BYTES, Q6K_BLOCK_ELEMENTS, q6k_weight


TARGET = "amd_gfx1100"
DIRECT_PACKED_FALLBACK = "q6k_direct_packed_fp16"
DIRECT_PACKED_Q8_FALLBACK = "q6k_direct_packed_q8_1"
BOUNDED_BLOCKER = (
  "generated packed Q6_K kernel lowers as one-thread direct owners, not the "
  "declared 64-thread cooperative matrix tile; no matching final gfx1100 proof"
)
COOPERATIVE_ROUTE = "q6k_bounded_packed_tiles"


class CapabilityStatus(str, Enum):
  SUPPORTED = "SUPPORTED"
  UNSUPPORTED = "UNSUPPORTED"
  REQUIRES_FALLBACK = "REQUIRES_FALLBACK"


@dataclass(frozen=True)
class Q6KPhysicalGrammar:
  """The physical Q6_K ABI; intentionally independent from Q4 grammars."""
  block_elements: int = 256
  block_bytes: int = 210
  ql_offset: int = 0
  ql_bytes: int = 128
  qh_offset: int = 128
  qh_bytes: int = 64
  scale_offset: int = 192
  scale_bytes: int = 16
  d_offset: int = 208
  d_bytes: int = 2
  zero_point: int = 32

  def validate(self) -> None:
    if tuple(vars(self).values()) != (256, 210, 0, 128, 128, 64, 192, 16, 208, 2, 32):
      raise ValueError("unsupported Q6_K physical decoder grammar")

  def to_json(self) -> dict[str, int]: return dict(vars(self))


@dataclass(frozen=True)
class Q6KBoundedResources:
  q6_packed_tile_bytes: int
  q6_decoded_tile_bytes: int
  activation_tile_bytes: int
  accumulator_tile_bytes: int
  lds_bytes: int
  global_workspace_bytes: int = 0
  scratch_bytes: int | None = None
  vgpr_count: int | None = None

  @property
  def bounded_live_bytes(self) -> int:
    return self.lds_bytes + self.accumulator_tile_bytes + self.global_workspace_bytes

  def to_json(self) -> dict[str, Any]:
    return {**vars(self), "bounded_live_bytes": self.bounded_live_bytes,
            "full_weight_decode_bytes": 0, "packed_weights_authoritative": True}


@dataclass(frozen=True)
class Q6KBoundedAdmission:
  status: CapabilityStatus
  route: str
  fallback_route: str
  reasons: tuple[str, ...]
  resources: Q6KBoundedResources
  production_coverage: bool = False

  @property
  def admitted(self) -> bool: return self.status is CapabilityStatus.SUPPORTED

  def to_json(self) -> dict[str, Any]:
    return {"status": self.status.value, "admitted": self.admitted, "route": self.route,
            "fallback_route": self.fallback_route, "reasons": list(self.reasons),
            "production_coverage": self.production_coverage, "resources": self.resources.to_json()}


@dataclass(frozen=True)
class Q6KGfx1100Proof:
  """Final-program facts required to dispatch the cooperative route.

  These are deliberately final code-object/ISA facts, not UOp estimates.  The
  identity fields prevent evidence collected for another kernel being reused.
  """
  candidate_id: str
  kernel_name: str
  target: str
  wavefront_size: int
  workgroup_threads: int
  vgpr: int
  lds_bytes: int
  scratch_bytes: int
  vgpr_spills: int
  sgpr_spills: int
  occupancy: float
  barrier_sites: int
  matrix_core_sites: int
  owner_map_sha256: str
  source: str = "final_code_object_and_isa"

  def validate(self, spec:"Q6KBoundedMMQSpec") -> None:
    if not self.candidate_id or not self.kernel_name: raise ValueError("resource proof identity is missing")
    if self.source != "final_code_object_and_isa": raise ValueError("resource proof is not final-program evidence")
    if self.target != TARGET or self.wavefront_size != spec.wave_size: raise ValueError("resource proof target/wave mismatch")
    if self.workgroup_threads != spec.workgroup_size: raise ValueError("resource proof workgroup mismatch")
    if min(self.vgpr, self.lds_bytes, self.scratch_bytes, self.vgpr_spills, self.sgpr_spills,
           self.barrier_sites, self.matrix_core_sites) < 0: raise ValueError("resource proof contains negative fields")
    if self.scratch_bytes or self.vgpr_spills or self.sgpr_spills: raise ValueError("resource proof contains scratch or spills")
    if self.lds_bytes > spec.resources().lds_bytes: raise ValueError("final LDS exceeds the bounded staging contract")
    if not 0 < self.occupancy <= 1: raise ValueError("resource proof occupancy is invalid")
    if spec.workgroup_size > spec.wave_size and self.barrier_sites == 0: raise ValueError("multi-wave proof has no barrier")
    if self.matrix_core_sites == 0: raise ValueError("proof has no matrix-core instruction")
    if self.owner_map_sha256 != owner_map_sha256(spec.tile_m, spec.tile_n, spec.workgroup_size):
      raise ValueError("owner/writeback proof does not match the exact tile mapping")


def owner_map(tile_m:int, tile_n:int, workgroup_size:int=64) -> tuple[tuple[int, int, int], ...]:
  """Canonical one-owner mapping: linear output index modulo workgroup size."""
  if min(tile_m, tile_n, workgroup_size) <= 0: raise ValueError("owner geometry must be positive")
  return tuple((i // tile_n, i % tile_n, i % workgroup_size) for i in range(tile_m * tile_n))


def owner_map_sha256(tile_m:int, tile_n:int, workgroup_size:int=64) -> str:
  import hashlib, json
  return hashlib.sha256(json.dumps(owner_map(tile_m, tile_n, workgroup_size), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Q6KRowRoute:
  row: int
  route: str
  reason: str | None = None


@dataclass(frozen=True)
class Q6KBoundedMMQSpec:
  """Logical X[M,K] @ W[N,K].T contract with explicit edge/grid facts."""
  m: int
  n: int
  k: int
  activation: str = "FP16"             # FP16 | Q8_1
  tile_m: int = 16
  tile_n: int = 16
  tile_k: int = 256
  target: str = TARGET
  backend: str = "AMD"
  wave_size: int = 32
  workgroup_size: int = 64
  accumulator: str = "FP32"
  output: str = "FP32"
  grammar: Q6KPhysicalGrammar = Q6KPhysicalGrammar()

  def validate(self) -> None:
    self.grammar.validate()
    if min(self.m, self.n, self.k, self.tile_m, self.tile_n, self.tile_k) <= 0:
      raise ValueError("logical dimensions and tile dimensions must be positive")
    if self.activation not in ("FP16", "Q8_1"): raise ValueError("activation must be FP16 or Q8_1")
    if self.accumulator != "FP32" or self.output != "FP32": raise ValueError("Q6 bounded MMQ requires FP32 accumulation/output")
    if self.tile_k % Q6K_BLOCK_ELEMENTS: raise ValueError("tile_k must contain complete Q6_K physical blocks")
    if self.wave_size <= 0 or self.workgroup_size <= 0 or self.workgroup_size % self.wave_size:
      raise ValueError("invalid wave/workgroup mapping")

  @property
  def k_blocks(self) -> int: return ceil(self.k / Q6K_BLOCK_ELEMENTS)
  @property
  def packed_row_stride_bytes(self) -> int: return self.k_blocks * Q6K_BLOCK_BYTES
  @property
  def packed_weight_bytes(self) -> int: return self.n * self.packed_row_stride_bytes
  @property
  def grid(self) -> tuple[int, int, int]: return (ceil(self.m/self.tile_m), ceil(self.n/self.tile_n), 1)
  @property
  def k_tiles(self) -> int: return ceil(self.k/self.tile_k)
  @property
  def tails(self) -> dict[str, int]: return {"M": self.m % self.tile_m, "N": self.n % self.tile_n, "K": self.k % self.tile_k}

  def resources(self) -> Q6KBoundedResources:
    # Both physical Q6 and decoded FP16 representations are listed exactly.
    # A lowering may choose one, but cannot silently allocate both as N*K.
    q6_packed = self.tile_n * (self.tile_k // Q6K_BLOCK_ELEMENTS) * Q6K_BLOCK_BYTES
    q6_decoded = self.tile_n * self.tile_k * 2
    if self.activation == "FP16": activation = self.tile_m * self.tile_k * 2
    else: activation = self.tile_m * (self.tile_k + (self.tile_k // 32) * 8)  # i8 values + f32 d/s
    accum = self.tile_m * self.tile_n * 4
    return Q6KBoundedResources(q6_packed, q6_decoded, activation, accum,
                               q6_packed + activation, scratch_bytes=None, vgpr_count=None)

  def admission(self, proof:Q6KGfx1100Proof|None=None) -> Q6KBoundedAdmission:
    self.validate()
    reasons: list[str] = []
    if (self.backend, self.target, self.wave_size) != ("AMD", TARGET, 32):
      reasons.append("target backend/architecture/wave is outside the gfx1100 capability")
    if proof is None: reasons.append(BOUNDED_BLOCKER)
    else:
      try: proof.validate(self)
      except ValueError as e: reasons.append(f"invalid final gfx1100 proof: {e}")
    admitted = not reasons
    return Q6KBoundedAdmission(CapabilityStatus.SUPPORTED if admitted else CapabilityStatus.REQUIRES_FALLBACK,
                               COOPERATIVE_ROUTE,
                               DIRECT_PACKED_FALLBACK if self.activation == "FP16" else DIRECT_PACKED_Q8_FALLBACK,
                               tuple(reasons), self.resources(), admitted)

  def to_json(self) -> dict[str, Any]:
    self.validate()
    return {"quant": "Q6_K", "activation": self.activation, "shape": {"M": self.m, "N": self.n, "K": self.k},
            "tile": {"M": self.tile_m, "N": self.tile_n, "K": self.tile_k}, "grid": list(self.grid),
            "k_tiles": self.k_tiles, "tails": self.tails, "target": self.target, "backend": self.backend,
            "wave_size": self.wave_size, "workgroup_size": self.workgroup_size, "accumulator": self.accumulator,
            "output": self.output, "packed_row_stride_bytes": self.packed_row_stride_bytes,
            "packed_weight_bytes": self.packed_weight_bytes, "grammar": self.grammar.to_json(),
            "resources": self.resources().to_json(), "admission": self.admission().to_json()}


def describe_q6k_bounded_mmq(m:int, n:int, k:int, **kwargs:Any) -> Q6KBoundedMMQSpec:
  spec = Q6KBoundedMMQSpec(m, n, k, **kwargs); spec.validate(); return spec


def emit_q6k_bounded_mmq_kernel(spec:Q6KBoundedMMQSpec, *, allow_direct_packed_fallback:bool = False):
  """Emit the production packed Q6_K custom-kernel callback.

  The physical K range is traversed in 256-element packed records.  Each
  logical (M,N) owner accumulates in FP32 and performs exactly one edge-safe
  write.  ``allow_direct_packed_fallback`` is retained for source compatibility
  but is no longer needed for FP16 or Q8_1.
  """
  spec.validate()
  from tinygrad import dtypes
  from tinygrad.uop.ops import AxisType, KernelInfo, UOp
  from extra.qk.layout import Q6K_HALFWORDS_PER_BLOCK
  from extra.qk.quant.q6_k_gemv_primitive import _f16_half, _i8, _q6k_byte
  blocks, physical_k, groups = spec.k_blocks, spec.k_blocks * Q6K_BLOCK_ELEMENTS, spec.k_blocks * 8

  def kernel(out:UOp, halfs:UOp, x:UOp, *q8_scales:UOp) -> UOp:
    if spec.activation == "Q8_1" and len(q8_scales) != 1:
      raise ValueError("Q8_1 custom kernel requires one FP32 scale operand")
    if spec.activation == "FP16" and q8_scales:
      raise ValueError("FP16 custom kernel does not accept Q8_1 scales")
    ni = UOp.range(spec.n, 0)
    mi = UOp.range(spec.m, 1)
    block = UOp.range(blocks, 2, axis_type=AxisType.REDUCE)
    lane2 = UOp.range(8, 3, axis_type=AxisType.REDUCE)
    base = (ni * blocks + block) * Q6K_HALFWORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    d = _f16_half(halfs[base + 104])
    for group in range(16):
      half, pgrp = group // 8, group % 8
      ql_word = halfs[base + half*32 + (pgrp%4)*8 + lane2]
      qh_word = halfs[base + 64 + half*16 + (pgrp%2)*8 + lane2]
      scale = _i8(_q6k_byte(halfs, base, 192 + group))
      for pair in range(2):
        pos = lane2 * 2 + pair
        ki = block * Q6K_BLOCK_ELEMENTS + group * 16 + pos
        valid = ki < spec.k
        xv = x[mi * physical_k + ki].cast(dtypes.float32)
        if spec.activation == "Q8_1": xv = xv * q8_scales[0][mi * groups + ki // 32]
        ql = ql_word.rshift(pair*8 + (4 if pgrp >= 4 else 0)).bitwise_and(0xf)
        qh = qh_word.rshift(pair*8 + (pgrp//2)*2).bitwise_and(0x3).lshift(4)
        weight = d * scale * (ql.bitwise_or(qh).cast(dtypes.float32) - 32.0)
        contrib = contrib + valid.where(weight * xv, 0.0)
    acc = out[mi, ni].set(0.0)
    acc = out[mi, ni].set(acc.after(block, lane2)[mi, ni] + contrib, end=lane2)
    name = f"q6k_bounded_{spec.activation.lower()}_{spec.m}_{spec.n}_{spec.k}"
    return acc.end(ni, mi, block).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def q6k_bounded_mmq(halfs, activations, spec:Q6KBoundedMMQSpec, *, q8_scales=None):
  """Launch the bounded emitter through ``Tensor.custom_kernel``."""
  from tinygrad import Tensor, dtypes
  spec.validate()
  expected_halfs = spec.n * spec.k_blocks * (Q6K_BLOCK_BYTES // 2)
  if tuple(halfs.shape) != (expected_halfs,) or halfs.dtype != dtypes.uint16:
    raise ValueError(f"packed Q6_K operand must be uint16[{expected_halfs}]")
  if tuple(activations.shape) != (spec.m, spec.k):
    raise ValueError(f"activation shape must be {(spec.m, spec.k)}")
  physical_k, pad_k = spec.k_blocks * Q6K_BLOCK_ELEMENTS, spec.k_blocks * Q6K_BLOCK_ELEMENTS - spec.k
  staged_activations = activations.pad(((0, 0), (0, pad_k))) if pad_k else activations
  operands = [halfs.contiguous(), staged_activations.reshape(-1).contiguous()]
  if spec.activation == "FP16":
    if activations.dtype != dtypes.float16 or q8_scales is not None:
      raise ValueError("FP16 route requires FP16 activations and no Q8_1 scales")
  else:
    logical_groups, physical_groups = ceil(spec.k / 32), physical_k // 32
    if activations.dtype != dtypes.int8 or q8_scales is None or tuple(q8_scales.shape) != (spec.m, logical_groups) or q8_scales.dtype != dtypes.float32:
      raise ValueError(f"Q8_1 route requires int8 activations and FP32 scales[{spec.m},{logical_groups}]")
    staged_scales = q8_scales.pad(((0, 0), (0, physical_groups-logical_groups))) if physical_groups != logical_groups else q8_scales
    operands.append(staged_scales.reshape(-1).contiguous())
  return Tensor.empty(spec.m, spec.n, dtype=dtypes.float32, device=halfs.device).custom_kernel(
    *operands, fxn=emit_q6k_bounded_mmq_kernel(spec))[0]


def q6k_row_routes(spec:Q6KBoundedMMQSpec, proof:Q6KGfx1100Proof|None=None) -> tuple[Q6KRowRoute, ...]:
  """Truthful deterministic coverage census; every logical weight row has one route."""
  gate = spec.admission(proof)
  reason = None if gate.admitted else "; ".join(gate.reasons)
  route = gate.route if gate.admitted else gate.fallback_route
  return tuple(Q6KRowRoute(row, route, reason) for row in range(spec.n))


def _stage_activation(row:Sequence[float], spec:Q6KBoundedMMQSpec, k0:int,
                      q8_scale_row:Sequence[float]|None) -> list[float]:
  k1 = min(k0 + spec.tile_k, spec.k)
  if spec.activation == "FP16": return [float(row[k]) for k in range(k0, k1)]
  assert q8_scale_row is not None
  return [float(row[k]) * float(q8_scale_row[k//32]) for k in range(k0, k1)]


def execute_q6k_bounded_mmq(packed:bytes|bytearray|memoryview, activations:Sequence[Sequence[float]],
                            spec:Q6KBoundedMMQSpec, *, q8_scales:Sequence[Sequence[float]]|None=None,
                            proof:Q6KGfx1100Proof|None=None,
                            cooperative_tile:Callable[[memoryview, Sequence[Sequence[float]], int, int, int, int],
                                                      Sequence[Sequence[float]]]|None=None
                            ) -> tuple[list[list[float]], tuple[Q6KRowRoute, ...]]:
  """Execute outer-M/N/K with tails, or an explicit direct-packed row fallback.

  A cooperative callback is usable only with a valid final proof. It receives
  one packed N panel, one dequantized activation panel, and (m0,n0,k0,k_len).
  Partial tiles are accumulated in FP32 and each output is written once.  With
  no proven callback, the canonical packed decoder executes every row directly.
  """
  spec.validate()
  # A proof cannot turn an absent executable into cooperative coverage.
  routes = q6k_row_routes(spec, proof if cooperative_tile is not None else None)
  if not spec.admission(proof).admitted or cooperative_tile is None:
    return q6k_bounded_reference(packed, activations, spec, q8_scales=q8_scales), routes
  if len(packed) != spec.packed_weight_bytes: raise ValueError(f"packed must contain exactly {spec.packed_weight_bytes} bytes")
  if len(activations) != spec.m or any(len(row) != spec.k for row in activations):
    raise ValueError(f"activations must have shape ({spec.m}, {spec.k})")
  groups = ceil(spec.k/32)
  if spec.activation == "Q8_1" and (q8_scales is None or len(q8_scales) != spec.m or
      any(len(row) != groups for row in q8_scales)):
    raise ValueError(f"q8_scales must have shape ({spec.m}, {groups})")
  import numpy as np
  out = np.zeros((spec.m, spec.n), dtype=np.float32); view = memoryview(packed)
  for m0 in range(0, spec.m, spec.tile_m):
    ml = min(spec.tile_m, spec.m-m0)
    for n0 in range(0, spec.n, spec.tile_n):
      nl = min(spec.tile_n, spec.n-n0); acc = np.zeros((ml, nl), dtype=np.float32)
      for k0 in range(0, spec.k, spec.tile_k):
        kl = min(spec.tile_k, spec.k-k0)
        panel = [_stage_activation(activations[m], spec, k0, None if q8_scales is None else q8_scales[m])
                 for m in range(m0, m0+ml)]
        # Physical rows remain packed. Include every physical block touched by this K tile.
        b0, b1 = k0//Q6K_BLOCK_ELEMENTS, ceil((k0+kl)/Q6K_BLOCK_ELEMENTS)
        packed_panel = bytearray()
        for n in range(n0, n0+nl):
          rb = n*spec.packed_row_stride_bytes
          packed_panel += view[rb+b0*Q6K_BLOCK_BYTES:rb+b1*Q6K_BLOCK_BYTES]
        part = np.asarray(cooperative_tile(memoryview(packed_panel), panel, m0, n0, k0, kl), dtype=np.float32)
        if part.shape != (ml, nl): raise ValueError(f"cooperative tile returned {part.shape}, expected {(ml, nl)}")
        acc = np.asarray(acc + part, dtype=np.float32)
      out[m0:m0+ml, n0:n0+nl] = acc
  return out.tolist(), routes


def q6k_bounded_reference(packed:bytes|bytearray|memoryview, activations:Sequence[Sequence[float]],
                          spec:Q6KBoundedMMQSpec, *, q8_scales:Sequence[Sequence[float]]|None = None) -> list[list[float]]:
  """Small/edge reference using packed weights and explicit FP32 accumulation.

  For Q8_1, ``activations`` contains signed integer values and ``q8_scales``
  contains one scale per 32 values.  Padded K values in the final physical Q6
  block are never read.
  """
  spec.validate()
  if len(packed) != spec.packed_weight_bytes: raise ValueError(f"packed must contain exactly {spec.packed_weight_bytes} bytes")
  if len(activations) != spec.m or any(len(row) != spec.k for row in activations):
    raise ValueError(f"activations must have shape ({spec.m}, {spec.k})")
  if spec.activation == "Q8_1":
    groups = ceil(spec.k/32)
    if q8_scales is None or len(q8_scales) != spec.m or any(len(row) != groups for row in q8_scales):
      raise ValueError(f"q8_scales must have shape ({spec.m}, {groups})")
  import numpy as np
  out = [[0.0] * spec.n for _ in range(spec.m)]
  view = memoryview(packed)
  for mi in range(spec.m):
    for ni in range(spec.n):
      acc = np.float32(0.0)
      row_base = ni * spec.packed_row_stride_bytes
      for ki in range(spec.k):
        block = ki // Q6K_BLOCK_ELEMENTS
        within = ki % Q6K_BLOCK_ELEMENTS
        bb = row_base + block * Q6K_BLOCK_BYTES
        weight = q6k_weight(view[bb:bb+Q6K_BLOCK_BYTES], within//16, within%16)
        act = activations[mi][ki]
        if spec.activation == "Q8_1": act = float(act) * float(q8_scales[mi][ki//32])  # type: ignore[index]
        acc = np.float32(acc + np.float32(weight) * np.float32(act))
      out[mi][ni] = float(acc)
  return out


__all__ = ["CapabilityStatus", "Q6KPhysicalGrammar", "Q6KBoundedResources", "Q6KBoundedAdmission",
           "Q6KBoundedMMQSpec", "describe_q6k_bounded_mmq", "emit_q6k_bounded_mmq_kernel",
           "q6k_bounded_reference", "execute_q6k_bounded_mmq", "q6k_row_routes", "Q6KRowRoute",
           "Q6KGfx1100Proof", "owner_map", "owner_map_sha256", "q6k_bounded_mmq", "DIRECT_PACKED_FALLBACK", "DIRECT_PACKED_Q8_FALLBACK", "COOPERATIVE_ROUTE",
           "BOUNDED_BLOCKER"]
