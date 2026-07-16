"""Descriptor-driven Q4_K x Q8_1 MMQ graph lowering.

The descriptor owns candidate geometry; the generated WMMA lowering owns the
Tensor graph and backend instruction selection.  This module intentionally
does not contain a schedule, route, or ISA implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec
from extra.qk.prefill_int8_wmma_spec import (
  Q4KInt8WMMAPrefillSpec, emit_q4k_int8_wmma_prefill_tensor,
  Q4KInt8WMMATiledPrefillSpec, emit_q4k_int8_wmma_tiled_lifecycle_tensor,
  emit_q4k_int8_wmma_tiled_scheduler_tensor,
)
from extra.qk.mmq_logical_vocabulary import MMQCandidate

Q4K_COOPERATIVE_LOWERING_CONTRACT = "q4k-cooperative-mmq-lowering-v1"


@dataclass(frozen=True)
class Q4KMultiWaveOwnershipPlan:
  """Typed contract for the cooperative lowering this emitter cannot express yet.

  M/N tiles are owned by waves; K is never split between waves, so each wave
  accumulates all groups before its one store.  This is the smallest legal
  ownership plan once Tensor has workgroup IDs, LDS, and barriers.
  """
  logical_axes: tuple[str, ...] = ("m", "n", "k", "group", "activation_block")
  wave_coordinates: str = "wave_id = m_tile_id * n_tiles + n_tile_id"
  k_group_schedule: str = "for k_block, group in lexicographic_order; every wave owns all groups"
  quant_load_ownership: str = "lane l loads Q4_K words/scales at l + wave_size*i, bounds checked"
  lds_ownership: str = "unsupported: no typed workgroup LDS allocation/producer contract"
  synchronization: str = "unsupported: no typed workgroup barrier contract"
  activation_lifetime: str = "wave-local for one (m_tile,n_tile), reused across groups"
  store_ownership: str = "wave owning (m_tile,n_tile) stores each output element exactly once"

  def validate(self) -> None:
    if self.logical_axes != ("m", "n", "k", "group", "activation_block"):
      raise ValueError("Q4 multi-wave contract has incomplete logical axes")
    if not self.wave_coordinates or not self.k_group_schedule or not self.quant_load_ownership:
      raise ValueError("Q4 multi-wave contract must define wave and quant scheduling")
    if not self.store_ownership.endswith("exactly once"):
      raise ValueError("Q4 multi-wave contract must define unique stores")


Q4K_MULTI_WAVE_PLAN = Q4KMultiWaveOwnershipPlan()


def audit_q4k_cooperative_lowering(candidate: MMQCandidate) -> dict[str, object]:
  """Return the facts a real multi-wave Q4 lowering must provide.

  The current Tensor emitter has no representation for these facts.  Keep
  this check at the shared-candidate boundary so a descriptor cannot be
  mistaken for a lowered cooperative kernel.
  """
  mapping = candidate.mapping
  if mapping.wave_size <= 0 or mapping.workgroup_size <= 0:
    raise ValueError("Q4 multi-wave mapping requires positive wave/workgroup sizes")
  if mapping.workgroup_size % mapping.wave_size:
    raise ValueError("Q4 multi-wave mapping workgroup must contain whole waves")
  waves = mapping.workgroup_size // mapping.wave_size
  Q4K_MULTI_WAVE_PLAN.validate()
  missing = {"logical_axes": Q4K_MULTI_WAVE_PLAN.logical_axes,
    "tile_wave_owner": Q4K_MULTI_WAVE_PLAN.wave_coordinates,
    "k_block_schedule": Q4K_MULTI_WAVE_PLAN.k_group_schedule,
    "quant_loads": Q4K_MULTI_WAVE_PLAN.quant_load_ownership,
    "lds_staging": Q4K_MULTI_WAVE_PLAN.lds_ownership, "barriers": Q4K_MULTI_WAVE_PLAN.synchronization,
    "activation_reuse": Q4K_MULTI_WAVE_PLAN.activation_lifetime, "stores": Q4K_MULTI_WAVE_PLAN.store_ownership}
  return {"contract": Q4K_COOPERATIVE_LOWERING_CONTRACT, "wave_size": mapping.wave_size,
          "workgroup_size": mapping.workgroup_size, "waves": waves,
          "implemented": False, "missing": missing,
          "plan": Q4K_MULTI_WAVE_PLAN, "reason": "compiler Tensor lowering has no multi-wave Q4 ownership/LDS contract"}


@dataclass(frozen=True)
class MMQEmitterCandidate:
  """The logical-to-generated lowering choices owned by a candidate.

  This is deliberately a small boundary until the shared logical vocabulary
  lands.  Nothing below the boundary is allowed to infer geometry, layout,
  staging, or lifecycle from workload shape.
  """
  spec: Q4KQ8MMQPrefillSpec
  wmma_m: int
  wmma_n: int
  wmma_k: int
  lifecycle: Literal["tiled", "group", "scheduler", "packed_ds4"]
  output_layout: str
  activation_layout: str
  tile_x_layout: str
  tile_y_layout: str
  staging_strategy: str
  writeback_strategy: str

  def validate(self) -> None:
    self.spec.validate()
    if min(self.wmma_m, self.wmma_n, self.wmma_k) <= 0:
      raise ValueError("candidate WMMA dimensions must be positive")
    if self.lifecycle not in ("tiled", "group", "scheduler", "packed_ds4"):
      raise ValueError(f"unsupported MMQ lifecycle {self.lifecycle!r}")
    if self.output_layout != self.spec.output_layout:
      raise ValueError("candidate output layout does not match descriptor")
    if (self.activation_layout, self.tile_x_layout, self.tile_y_layout) != (self.spec.activation_layout, self.spec.tile_x_layout, self.spec.tile_y_layout):
      raise ValueError("candidate layouts do not match descriptor")
    if (self.staging_strategy, self.writeback_strategy) != (self.spec.staging_strategy, self.spec.writeback_strategy):
      raise ValueError("candidate lifecycle policy does not match descriptor")
    if self.lifecycle == "tiled" and (self.spec.m % self.wmma_m or self.spec.n % self.wmma_n):
      raise ValueError("candidate shape is not divisible by declared WMMA geometry")


def _from_logical(candidate: MMQCandidate) -> MMQEmitterCandidate:
  d, mapping = candidate.descriptor, candidate.mapping
  if candidate.capability.backend != "amd" or mapping.wave_size not in candidate.capability.wave_sizes:
    raise ValueError("shared MMQ candidate capability does not cover its mapping")
  if candidate.capability.max_workgroup_size is not None and mapping.workgroup_size > candidate.capability.max_workgroup_size:
    raise ValueError("shared MMQ candidate workgroup exceeds capability")
  if d.operation.name not in candidate.capability.supported_ops:
    raise ValueError("shared MMQ candidate operation is not supported by capability")
  audit = audit_q4k_cooperative_lowering(candidate)
  if audit["waves"] > 1:
    raise ValueError(f"{Q4K_COOPERATIVE_LOWERING_CONTRACT} blocked: "
                     "multi-wave lowering is unavailable; missing " + ", ".join(audit["missing"]))
  axes = {axis.name: axis for axis in d.axes}
  required_abi = ("role", "shape", "output_layout", "weight_layout", "activation_layout",
                  "tile_x_layout", "tile_y_layout", "staging_strategy", "writeback_strategy")
  if any(key not in d.abi for key in required_abi):
    raise ValueError("shared MMQ descriptor ABI is missing explicit lowering fields")
  role = str(d.abi["role"])
  shape = d.abi["shape"]
  if shape != {"M": axes["m"].extent, "N": axes["n"].extent, "K": axes["k"].extent}:
    raise ValueError("shared MMQ descriptor ABI shape disagrees with logical axes")
  spec = Q4KQ8MMQPrefillSpec("logical_mmq", "logical", role, "Q4_K", "Q8_1", str(d.abi["weight_layout"]),
    str(d.abi["output_layout"]), axes["m"].extent, axes["n"].extent, axes["k"].extent,
    tile_m=axes["m"].tile, tile_n=axes["n"].tile, tile_k=axes["k"].tile,
    wave_width=mapping.wave_size, workgroup_size=mapping.workgroup_size,
    activation_layout=str(d.abi["activation_layout"]),
    tile_x_layout=str(d.abi["tile_x_layout"]), tile_y_layout=str(d.abi["tile_y_layout"]),
    staging_strategy=str(d.abi["staging_strategy"]), writeback_strategy=str(d.abi["writeback_strategy"]), lds_bytes=0)
  wm, wn, wk = mapping.wmma_shape
  return MMQEmitterCandidate(spec, wm, wn, wk, mapping.lifecycle, spec.output_layout,
    spec.activation_layout, spec.tile_x_layout, spec.tile_y_layout,
    spec.staging_strategy, spec.writeback_strategy)


def emit_q4k_q8_mmq_prefill(words: Tensor, xq: Tensor, xscales: Tensor,
                            candidate: MMQCandidate | MMQEmitterCandidate) -> Tensor:
  """Emit the descriptor-shaped graph without compiling or dispatching it."""
  if isinstance(candidate, MMQCandidate):
    candidate = _from_logical(candidate)
  if not isinstance(candidate, MMQEmitterCandidate):
    raise TypeError("MMQ emitter requires an MMQEmitterCandidate")
  candidate.validate()
  if candidate.lifecycle == "packed_ds4":
    raise ValueError("packed_ds4 candidates require emit_q4k_q8_mmq_ds4")
  spec = candidate.spec
  if spec.output_layout != "tokens_rows":
    raise ValueError("MMQ lowering only emits the canonical tokens_rows ABI layout")
  if any(size % tile for size, tile in ((spec.m, spec.tile_m), (spec.n, spec.tile_n),
                                        (spec.k, spec.tile_k))):
    raise ValueError("descriptor shape must be divisible by tile geometry")
  expected_words = spec.n * (spec.k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK
  expected_scales = (spec.m, spec.k // Q8_1_BLOCK_ELEMS)
  if tuple(words.shape) != (expected_words,):
    raise ValueError(f"words shape must be {(expected_words,)}, got {tuple(words.shape)}")
  if tuple(xq.shape) != (spec.m, spec.k):
    raise ValueError(f"xq shape must be {(spec.m, spec.k)}, got {tuple(xq.shape)}")
  if tuple(xscales.shape) != expected_scales:
    raise ValueError(f"xscales shape must be {expected_scales}, got {tuple(xscales.shape)}")
  if words.dtype != dtypes.uint32 or xq.dtype != dtypes.int8 or xscales.dtype != dtypes.float32:
    raise ValueError("MMQ operands have unsupported dtypes")
  if not (words.device == xq.device == xscales.device):
    raise ValueError("MMQ operands must be on the same device")

  # Use the generated tiled lifecycle whenever the candidate describes WMMA
  # tiles.  All tile/group ownership remains data in the translated spec.
  if candidate.lifecycle == "tiled":
    tiled = Q4KInt8WMMATiledPrefillSpec(n=spec.n, k=spec.k, m=spec.m, role=spec.role,
      wmma_m=candidate.wmma_m, wmma_n=candidate.wmma_n, wmma_k=candidate.wmma_k,
      m_tile=spec.tile_m, n_tile=spec.tile_n, group_tile=spec.tile_k // Q8_1_BLOCK_ELEMS)
    return emit_q4k_int8_wmma_tiled_lifecycle_tensor(words.contiguous(), xq.contiguous(),
                                                       xscales.contiguous(), tiled)

  if candidate.lifecycle == "scheduler":
    tiled = Q4KInt8WMMATiledPrefillSpec(n=spec.n, k=spec.k, m=spec.m, role=spec.role,
      wmma_m=candidate.wmma_m, wmma_n=candidate.wmma_n, wmma_k=candidate.wmma_k,
      m_tile=spec.tile_m, n_tile=spec.tile_n, group_tile=spec.tile_k // Q8_1_BLOCK_ELEMS)
    return emit_q4k_int8_wmma_tiled_scheduler_tensor(words.contiguous(), xq.contiguous(),
                                                       xscales.contiguous(), tiled)

  # Small graph/oracle shapes still use the same generated primitive, without
  # inventing a vector pointer base or a backend schedule.
  generated = Q4KInt8WMMAPrefillSpec(n=spec.n, k=spec.k, m=spec.m, role=spec.role,
                                     wmma_m=candidate.wmma_m, wmma_n=candidate.wmma_n,
                                     wmma_k=candidate.wmma_k, n_tile=spec.tile_n)
  return emit_q4k_int8_wmma_prefill_tensor(words.contiguous(), xq.contiguous(),
                                           xscales.contiguous(), generated, vectorized=False)


__all__ = ["MMQEmitterCandidate", "Q4KQ8MMQPrefillSpec", "Q4KMultiWaveOwnershipPlan",
           "Q4K_MULTI_WAVE_PLAN", "Q4K_COOPERATIVE_LOWERING_CONTRACT", "audit_q4k_cooperative_lowering",
           "emit_q4k_q8_mmq_prefill"]
