"""Host-owned register-resident WMMA pipeline structure.

This module is deliberately limited to structural UOps.  It has no renderer,
ISA payload, local allocation, or route selection side effects.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad.codegen.opt.compiler_policies import PipelinePolicy, StoragePolicy
from tinygrad.codegen.opt.kernel_lds import (PrecontractContractSpec, PrecontractOperandTemplate,
  derive_precontract_shape_factors, validate_precontract_carriers, validate_precontract_contracts,
  validate_precontract_operand_templates, validate_rdna3_wmma_descriptor)
from tinygrad.codegen.opt.kernel_pipeline import (KernelStage1FragmentStage, KernelStage1LifecycleEvent,
  KernelStage1LifecycleProof, KernelStage1ProducerStage, Stage1StorageAdapter, prove_stage1_lifecycle,
  stage1_lifecycle_events)
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelLDSWindow, KernelTileGeometry, Ops, UOp


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
    if (self.buffer_count, self.slot_bytes, self.stage_count, self.roles) != (2, 0, 1, ("A", "B")):
      raise ValueError("register pipe requires two logical stages, zero LDS, and A/B roles")

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

  def __post_init__(self) -> None:
    if self.shape != (512, 4096, 4096): raise ValueError("register vertical slice is attn_qo 512x4096x4096")
    if self.k_step != 16 or self.stages != 2 or (self.pipe_tm, self.pipe_tn) != (2, 2):
      raise ValueError("register template requires K step 16 and a two-stage 2x2 pipe")
    validate_rdna3_wmma_descriptor(self.tc)
    factors = derive_precontract_shape_factors(self.geometry, self.tc)
    if self.geometry.tile != (128, 128, 32) or (factors.subtiles_m, factors.subtiles_n) != (2, 4):
      raise ValueError("attn_qo register template requires 128x128x32 RDNA3 tile")
    validate_precontract_operand_templates(self.operands, context="register pipe")
    validate_precontract_contracts(self.tc, self.contracts, context="register pipe")
    validate_precontract_carriers(dtypes.half.vec(16), dtypes.float.vec(8), context="register pipe")

  @property
  def policy(self) -> PipelinePolicy:
    return PipelinePolicy.register_resident(stages=2)

  @property
  def loads_per_stage(self) -> int:
    # Each half.vec(16) carrier is two architectural b128 loads.
    return self.pipe_tm * 2 + self.pipe_tn * 2

  @property
  def body_readiness(self) -> str: return "matching"

  @property
  def stage_width(self) -> int: return (self.pipe_tm + self.pipe_tn) * 16

  def _buffers(self) -> tuple[UOp, UOp]:
    return (UOp.placeholder((self.stage_width * 2,), dtypes.half, 9300, addrspace=AddrSpace.REG),
            UOp.placeholder((self.stage_width * 2,), dtypes.half, 9301, addrspace=AddrSpace.REG))

  def producer(self, epoch: UOp, slot: UOp, reuse: UOp | None = None) -> KernelStage1ProducerStage:
    buffers = self._buffers()
    nodes: list[UOp] = []
    for operand, count, buffer in zip(self.operands, (self.pipe_tm, self.pipe_tn), buffers):
      vectors = []
      for frag in range(count):
        row = operand.row_tile_base + frag * 16
        values = tuple(operand.source.substitute({operand.row_axis: row,
          operand.k_axis: epoch * self.k_step + elem}) for elem in range(16))
        value = UOp(Ops.STACK, dtypes.half.vec(16), values,
          tag=("register_pipe_load", operand.role, frag, epoch, slot))
        vectors.append(buffer.index(slot * self.stage_width + frag * 16, dtype=dtypes.half.vec(16)).store(value))
      nodes.append(UOp.group(*vectors).replace(tag=("register_pipe_producer", operand.role, epoch, slot)))
    ready = UOp.group(*nodes, reuse).replace(tag=("register_pipe_ready", epoch, slot))
    return KernelStage1ProducerStage(epoch, slot, (nodes[0], nodes[1]), ready)

  def fragments(self, epoch: UOp, slot: UOp, ready: UOp) -> KernelStage1FragmentStage:
    if ready.op not in (Ops.GROUP, Ops.END, Ops.BARRIER):
      raise ValueError("register fragment consumer has no typed producer readiness")
    buffers = self._buffers()
    if isinstance(ready.tag, tuple) and ready.tag[:1] == ("register_pipe_ready",):
      ready_epoch, ready_slot = ready.tag[1], ready.tag[2]
      same = ready_epoch.render() == epoch.render() and ready_slot.render() == slot.render()
      nxt = ready_epoch.render() == (epoch + 1).render() and ready_slot.render() == ((slot + 1) % 2).render()
      if not (same or nxt): raise ValueError("register fragment consumer readiness epoch/slot mismatch")
    out: list[UOp] = []
    for operand, contract, count, buffer in zip(self.operands, self.contracts, (self.pipe_tm, self.pipe_tn), buffers):
      for idx in range(count):
        load = buffer.after(ready).index(slot * self.stage_width + idx * 16, dtype=dtypes.half.vec(16)).load()
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
    return RegisterLogicalStagePlan.from_policy(self.pipeline_policy)


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
  return tuple(errors)


def register_lifecycle_events(k_tiles: int) -> tuple[KernelStage1LifecycleEvent, ...]:
  """Return the shared ownership lifecycle for alternating register stages."""
  return stage1_lifecycle_events(RegisterLogicalStagePlan(), k_tiles)


def prove_register_lifecycle(k_tiles: int) -> KernelStage1LifecycleProof:
  """Prove producer/consume/release ownership for a register K loop."""
  plan = RegisterLogicalStagePlan()
  return prove_stage1_lifecycle(plan, k_tiles, stage1_lifecycle_events(plan, k_tiles))
