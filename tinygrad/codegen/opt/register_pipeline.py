"""Host-owned register-resident WMMA pipeline structure.

This module is deliberately limited to structural UOps.  It has no renderer,
ISA payload, local allocation, or route selection side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tinygrad.codegen.opt.compiler_policies import (PipelinePolicy, StoragePolicy, WaitDependency,
  prove_wait_dependency_coverage, wait_count_for_dependency)
from tinygrad.codegen.opt.gemm_consumer import WMMA_CONSUMER
from tinygrad.codegen.opt.kernel_lds import (PrecontractContractSpec, PrecontractOperandTemplate,
  derive_precontract_shape_factors, validate_precontract_carriers, validate_precontract_contracts,
  validate_precontract_operand_templates, validate_precontract_wmma_abi)
from tinygrad.codegen.opt.kernel_pipeline import (KernelStage1FragmentStage, KernelStage1LifecycleEvent,
  KernelStage1LifecycleProof, KernelStage1ProducerStage, Stage1StorageAdapter, prove_stage1_lifecycle,
  stage1_lifecycle_events)
from tinygrad.codegen.opt.register_contracts import LogicalRegisterTile
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelLDSWindow, KernelTileGeometry, Ops, UOp
from tinygrad.renderer.isa.amd_register_allocator import AMDStageBufferSpec


@dataclass(frozen=True)
class RegisterLogicalStagePlan:
  """Logical two-stage mapping with zero physical LDS slots."""
  buffer_count: int = 2
  slot_bytes: int = 0
  stage_count: int = 1
  roles: tuple[str, ...] = ("A", "B")

  @classmethod
  def from_policy(cls, policy: PipelinePolicy) -> "RegisterLogicalStagePlan":
    if not isinstance(policy, PipelinePolicy) or policy.storage_kind != "global_register_resident":
      raise ValueError("register logical stages require global_register_resident policy")
    if policy.logical_stage_count != 2 or policy.resources.lds_bytes != 0:
      raise ValueError("register logical stages require exactly two stages and zero LDS")
    return cls()

  def __post_init__(self) -> None:
    if self.buffer_count not in (1, 2) or self.slot_bytes != 0 or self.stage_count != 1 or self.roles != ("A", "B"):
      raise ValueError("register pipe requires one or two logical stages, zero LDS, and A/B roles")

  @property
  def active_lds_bytes(self) -> int: return 0

  def slot_for_epoch(self, epoch: int) -> int:
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0: raise ValueError("epoch must be non-negative")
    return epoch % self.buffer_count

  def slot_window(self, slot: int) -> tuple[int, int]:
    if not isinstance(slot, int) or not 0 <= slot < self.buffer_count: raise ValueError("invalid register stage")
    return (0, 0)


@dataclass(frozen=True)
class RegisterPipeTemplate:
  """Validated attn_qo register producer/fragment template."""
  tc: object
  geometry: KernelTileGeometry
  operands: tuple[PrecontractOperandTemplate, PrecontractOperandTemplate]
  contracts: tuple[PrecontractContractSpec, PrecontractContractSpec]
  shape: tuple[int, int, int] = (512, 4096, 4096)
  k_step: int = 16
  stages: int = 2
  pipe_tm: int = 2
  pipe_tn: int = 2
  schedule: Literal["double_buffer", "sequential"] = "double_buffer"

  def __post_init__(self) -> None:
    if self.shape != (512, 4096, 4096): raise ValueError("register vertical slice is attn_qo 512x4096x4096")
    if self.k_step != 16 or self.stages != 2 or (self.pipe_tm, self.pipe_tn) != (2, 2):
      raise ValueError("register template requires K step 16 and a two-stage 2x2 pipe")
    if self.schedule not in ("double_buffer", "sequential"):
      raise ValueError("register template schedule must be double_buffer or sequential")
    # Keep instruction-family validation behind the consumer seam.  The
    # adapter delegates to the existing storage-independent RDNA3 checks, so
    # graph output and failure behavior remain unchanged.
    self.consumer_adapter.validate_descriptor(self.tc)
    factors = derive_precontract_shape_factors(self.geometry, self.tc)
    if self.geometry.tile != (128, 128, 32) or (factors.subtiles_m, factors.subtiles_n) != (2, 4):
      raise ValueError("attn_qo register template requires 128x128x32 RDNA3 tile")
    validate_precontract_operand_templates(self.operands, context="register pipe")
    validate_precontract_contracts(self.tc, self.contracts, context="register pipe")
    validate_precontract_carriers(dtypes.half.vec(16), dtypes.float.vec(8), context="register pipe")
    for tile in self.logical_tiles:
      self.consumer_adapter.validate_tile(tile)

  @property
  def consumer_adapter(self):
    """The WMMA consumer contract for this template (no physical ISA data)."""
    return WMMA_CONSUMER

  @property
  def policy(self) -> PipelinePolicy:
    return PipelinePolicy.register_resident(stages=2)

  @property
  def loads_per_stage(self) -> int:
    # Each half.vec(16) carrier is two architectural b128 loads.
    return self.pipe_tm * 2 + self.pipe_tn * 2

  @property
  def body_readiness(self) -> str: return "sequential" if self.schedule == "sequential" else "matching"

  @property
  def logical_buffer_count(self) -> int:
    return 1 if self.schedule == "sequential" else 2

  @property
  def stage_width(self) -> int: return (self.pipe_tm + self.pipe_tn) * 16

  @property
  def logical_tiles(self) -> tuple[LogicalRegisterTile, LogicalRegisterTile]:
    """Compiler-neutral A/B tile contracts; backend packing is derived below."""
    return tuple(LogicalRegisterTile(
      role, dtypes.half, (fragments, 16), fragments, 16, 16,
      self.logical_buffer_count, "static" if self.schedule == "sequential" else "proven",
      layout=self.consumer_adapter.layout, alignment_bytes=32,
      ownership=("producer", "consumer", "slot"),
      lifetime=("produce", "consume", "release", "overwrite"))
      for role, fragments in (("A", self.pipe_tm), ("B", self.pipe_tn)))

  @property
  def stage_buffer_specs(self) -> tuple[AMDStageBufferSpec, AMDStageBufferSpec]:
    """Return the independent A/B logical register-buffer contracts.

    A and B are separate resources.  Their widths must not be derived from
    the combined pipe width: doing so doubles each role's allocation and
    hides the actual VGPR pressure from the backend resource gate.
    """
    return tuple(AMDStageBufferSpec(tile.role, tile.slot_count, tile.fragments, tile.carrier_width)
                 for tile in self.logical_tiles)

  @property
  def wait_dependencies(self) -> tuple[WaitDependency, WaitDependency]:
    """Typed A/B load edges consumed by the register-stage consumer.

    The storage template owns only provenance.  Counter selection still goes
    through :func:`wait_count_for_dependency`, and the AMD renderer remains
    the sole owner of the intrinsic lowering.
    """
    policy = self.policy.wait
    return tuple(WaitDependency(policy, f"global_load_{role}", "gemm_consumer", role,
                                producer_stage=0, consumer_stage=1, scope="per_stage")
                 for role in ("A", "B"))

  @property
  def wait_coverage(self):
    required = (("A", 0, 1), ("B", 0, 1))
    return prove_wait_dependency_coverage(self.policy, self.wait_dependencies, required)

  def _typed_load_wait(self, epoch: UOp, slot: UOp, loads: tuple[UOp, ...]) -> UOp:
    coverage = self.wait_coverage
    if not coverage.passed:
      raise ValueError(f"register load wait coverage failed: {coverage.errors}")
    # Both role groups use the same staged counter.  The dependency metadata
    # keeps the A/B ownership explicit without manufacturing backend ISA.
    count = wait_count_for_dependency(self.wait_dependencies[0], vmcnt=self.loads_per_stage)
    provenance = tuple((d.producer, d.consumer, d.load_group, d.producer_stage, d.consumer_stage)
                       for d in self.wait_dependencies)
    raw = UOp(Ops.WAIT, dtypes.void, loads, count,
              tag=("register_pipe_wait", epoch, slot, coverage.covered, provenance))
    return raw

  def _buffers(self) -> tuple[UOp, UOp]:
    # Keep A and B independent. A 2x2 pipe is 2*16 half elements per role and
    # slot, not one combined A+B allocation per role.
    return tuple(UOp.placeholder((spec.half_elements,), dtypes.half, 9300 + i, addrspace=AddrSpace.REG)
                 .replace(tag=("register_pipe_stage_buffer", spec.role, spec.slots, spec.fragments, spec.lane_width))
                 for i, spec in enumerate(self.stage_buffer_specs))

  def producer(self, epoch: UOp, slot: UOp, reuse: UOp | None = None) -> KernelStage1ProducerStage:
    if self.schedule == "sequential" and slot.op is not Ops.CONST:
      raise ValueError("sequential register producer requires a compile-time slot")
    buffers = self._buffers()
    nodes: list[UOp] = []
    loaded: list[UOp] = []
    for operand, count, buffer, spec in zip(self.operands, (self.pipe_tm, self.pipe_tn), buffers, self.stage_buffer_specs):
      vectors = []
      for frag in range(count):
        row = operand.row_tile_base + frag * 16
        values = tuple(operand.source.substitute({operand.row_axis: row,
          operand.k_axis: epoch * self.k_step + elem}) for elem in range(16))
        value = UOp(Ops.STACK, dtypes.half.vec(16), values,
          tag=("register_pipe_load", operand.role, frag, epoch, slot))
        loaded.append(value)
        offset = (UOp.const(dtypes.weakint, slot.arg * spec.role_width + frag * spec.lane_width)
                  if slot.op is Ops.CONST else slot * spec.role_width + frag * spec.lane_width)
        # Sequential mode reuses one physical buffer. Make every overwrite
        # pointer-dependent on the caller's reuse token; a GROUP alone does
        # not impose ordering between its sources. Put AFTER on the pointer
        # before INDEX so the STORE retains a valid pointer dtype.
        base = buffer.after(reuse) if reuse is not None and self.schedule == "sequential" else buffer
        target = base.index(offset, dtype=dtypes.half.vec(spec.lane_width))
        vectors.append(target.store(value))
      nodes.append(UOp.group(*vectors).replace(tag=("register_pipe_producer", operand.role, epoch, slot)))
    wait = self._typed_load_wait(epoch, slot, tuple(loaded))
    # GROUP is intentionally limited to stores/groups by the core UOp spec;
    # put the typed wait on the existing barrier seam instead of widening that
    # generic verifier for this route.
    ready = UOp.barrier(UOp.group(*nodes), wait, *(tuple() if reuse is None else (reuse,))).replace(
      tag=("register_pipe_ready", epoch, slot))
    return KernelStage1ProducerStage(epoch, slot, (nodes[0], nodes[1]), ready)

  def fragments(self, epoch: UOp, slot: UOp, ready: UOp) -> KernelStage1FragmentStage:
    if self.schedule == "sequential" and slot.op is not Ops.CONST:
      raise ValueError("sequential register fragments require a compile-time slot")
    if ready.op not in (Ops.GROUP, Ops.END, Ops.BARRIER):
      raise ValueError("register fragment consumer has no typed producer readiness")
    buffers = self._buffers()
    if isinstance(ready.tag, tuple) and ready.tag[:1] == ("register_pipe_ready",):
      ready_epoch, ready_slot = ready.tag[1], ready.tag[2]
      same = ready_epoch.render() == epoch.render() and ready_slot.render() == slot.render()
      nxt = ready_epoch.render() == (epoch + 1).render() and ready_slot.render() == ((slot + 1) % 2).render()
      initial = self.schedule == "sequential" and ready_epoch.op is Ops.CONST and ready_epoch.arg == 0 and \
        ready_slot.op is Ops.CONST and ready_slot.arg == 0
      if not (same or nxt or initial): raise ValueError("register fragment consumer readiness epoch/slot mismatch")
    out: list[UOp] = []
    for operand, contract, count, buffer, spec in zip(self.operands, self.contracts, (self.pipe_tm, self.pipe_tn), buffers,
                                                       self.stage_buffer_specs):
      for idx in range(count):
        offset = (UOp.const(dtypes.weakint, slot.arg * spec.role_width + idx * spec.lane_width)
                  if slot.op is Ops.CONST else slot * spec.role_width + idx * spec.lane_width)
        load = buffer.after(ready).index(offset, dtype=dtypes.half.vec(spec.lane_width)).load()
        out.append(UOp(Ops.CONTRACT, dtypes.half.vec(16), (load,), contract.arg,
                       tag=("register_pipe_fragment", operand.role, epoch, slot, idx)))
    return KernelStage1FragmentStage(epoch, slot, ready, tuple(out))


@dataclass(frozen=True)
class RegisterStorageAdapter(Stage1StorageAdapter):
  """Shared Stage1 callback adapter for zero-LDS register storage."""
  def __post_init__(self) -> None:
    if self.policy.kind != "global_register_resident" or self.policy.slot_bytes != 0:
      raise ValueError("register adapter requires zero-LDS global_register_resident policy")

  @classmethod
  def from_template(cls, template: RegisterPipeTemplate) -> "RegisterStorageAdapter":
    adapter = cls(template, template.policy.storage)
    RegisterLogicalStagePlan.from_policy(template.policy)
    return adapter

  @property
  def pipeline_policy(self) -> PipelinePolicy:
    """Expose the complete typed register policy without changing Stage1 ABI."""
    return PipelinePolicy.register_resident()

  @property
  def logical_plan(self) -> RegisterLogicalStagePlan:
    """Map the common policy to logical alternating stages, never LDS slots."""
    return RegisterLogicalStagePlan(buffer_count=getattr(self.callbacks, "logical_buffer_count", 2))


def register_geometry() -> KernelTileGeometry:
  """Canonical geometry metadata; windows are inert and never allocated."""
  return KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 10240, 80), KernelLDSWindow("B", 10240, 20480, 80)))


def prove_register_graph_no_lds(root: UOp) -> tuple[str, ...]:
  errors = []
  for u in root.toposort():
    if u.op is Ops.DEFINE_LOCAL: errors.append("register graph contains DEFINE_LOCAL")
    if u.op is Ops.INS: errors.append("register graph contains raw Ops.INS")
    if u.op is Ops.CONTRACT and u.dtype == dtypes.half.vec(16):
      if (len(u.src) != 1 or u.src[0].op is not Ops.LOAD or not u.src[0].src or
          not u.src[0].src[0].src or u.src[0].src[0].src[0].op is not Ops.AFTER):
        errors.append("register fragment is not ordered after producer readiness")
    if u.op is Ops.WMMA:
      try:
        validate_precontract_wmma_abi(u, context="register pipe")
      except ValueError as exc:
        errors.append(str(exc))
  return tuple(errors)


def register_lifecycle_events(k_tiles: int, *, buffer_count: int = 2) -> tuple[KernelStage1LifecycleEvent, ...]:
  """Return the shared ownership lifecycle for one- or two-buffer registers."""
  return stage1_lifecycle_events(RegisterLogicalStagePlan(buffer_count=buffer_count), k_tiles)


def prove_register_lifecycle(k_tiles: int, *, buffer_count: int = 2) -> KernelStage1LifecycleProof:
  """Prove producer/consume/release ownership for a register K loop."""
  plan = RegisterLogicalStagePlan(buffer_count=buffer_count)
  return prove_stage1_lifecycle(plan, k_tiles, stage1_lifecycle_events(plan, k_tiles))
