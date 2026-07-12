"""Core-neutral immutable compiler policy contracts."""
from dataclasses import dataclass
from typing import Literal

StorageKind = Literal["lds", "global_register_resident"]
WaitKind = Literal["full_barrier", "targeted_vmcnt"]
ResourceStage = Literal["host_estimate", "final_program"]

@dataclass(frozen=True)
class StoragePolicy:
  kind: StorageKind; buffer_count: int = 1; slot_bytes: int = 0; roles: tuple[str,...] = ("A", "B")
  def __post_init__(self):
    if self.kind not in ("lds", "global_register_resident"): raise ValueError("unsupported storage kind")
    if self.buffer_count not in (1,2): raise ValueError("buffer_count must be 1 or 2")
    if not isinstance(self.slot_bytes,int) or self.slot_bytes < 0: raise ValueError("slot_bytes must be non-negative")
    if self.kind == "lds" and self.slot_bytes <= 0: raise ValueError("LDS storage requires positive slot_bytes")
    if self.kind == "global_register_resident" and self.slot_bytes != 0: raise ValueError("register-resident storage cannot declare LDS slots")
    if self.roles != ("A","B"): raise ValueError("storage roles must be exactly ('A', 'B')")

@dataclass(frozen=True)
class WaitPolicy:
  kind: WaitKind; scope: str = "workgroup"
  def __post_init__(self):
    if self.kind not in ("full_barrier","targeted_vmcnt"): raise ValueError("unsupported wait kind")
    if self.kind == "full_barrier" and self.scope != "workgroup": raise ValueError("full barrier scope must be workgroup")
    if self.kind == "targeted_vmcnt" and self.scope != "per_stage": raise ValueError("targeted vmcnt scope must be per_stage")

@dataclass(frozen=True)
class WaitDependency:
  policy: WaitPolicy
  producer: str
  consumer: str
  load_group: str
  def __post_init__(self):
    if not isinstance(self.policy, WaitPolicy): raise ValueError("wait dependency requires WaitPolicy")
    if not all(isinstance(x, str) and x for x in (self.producer, self.consumer, self.load_group)): raise ValueError("wait dependency labels must be non-empty")

def amdllvm_wait_dependency(dep: WaitDependency) -> WaitDependency:
  """Fail closed: AMDLLVM can lower only full workgroup barriers today."""
  if not isinstance(dep, WaitDependency): raise TypeError("expected WaitDependency")
  if dep.policy.kind != "full_barrier" or dep.policy.scope != "workgroup":
    raise ValueError("targeted wait dependencies are unsupported by pure AMDLLVM")
  return dep

@dataclass(frozen=True)
class ResourcePlan:
  stage: ResourceStage; lds_bytes: int = 0; scratch_bytes: int = 0; vgpr: int|None = None; sgpr: int|None = None
  def __post_init__(self):
    if self.stage not in ("host_estimate","final_program"): raise ValueError("unsupported resource stage")
    if any(not isinstance(x,int) or x < 0 for x in (self.lds_bytes,self.scratch_bytes)): raise ValueError("resource bytes must be non-negative ints")
    if self.stage == "host_estimate" and (self.vgpr is not None or self.sgpr is not None): raise ValueError("host estimate cannot claim final register counts")
    if self.stage == "final_program" and (self.vgpr is None or self.sgpr is None): raise ValueError("final program requires VGPR and SGPR counts")

@dataclass(frozen=True)
class PipelinePolicy:
  """Complete, interchangeable policy for one compiler-owned pipeline.

  ``StoragePolicy.buffer_count`` describes physical local-memory slots.  The
  logical register-pipe stage count is kept separately because register
  stages have no LDS window and must not be misreported as local buffers.
  """
  storage: StoragePolicy
  wait: WaitPolicy
  resources: ResourcePlan
  stages: int = 1

  def __post_init__(self) -> None:
    if not isinstance(self.storage, StoragePolicy) or not isinstance(self.wait, WaitPolicy) or not isinstance(self.resources, ResourcePlan):
      raise ValueError("pipeline policy requires typed storage, wait, and resource contracts")
    if not isinstance(self.stages, int) or isinstance(self.stages, bool) or self.stages <= 0:
      raise ValueError("pipeline stages must be a positive int")
    if self.storage.kind == "global_register_resident":
      if self.storage.slot_bytes != 0 or self.resources.lds_bytes != 0:
        raise ValueError("register-resident policy cannot claim LDS storage")
    elif self.resources.lds_bytes < self.storage.buffer_count * self.storage.slot_bytes:
      raise ValueError("LDS resource plan does not cover physical storage slots")

  @property
  def storage_kind(self) -> StorageKind: return self.storage.kind

  @property
  def logical_stage_count(self) -> int: return self.stages

  @classmethod
  def lds(cls, *, buffer_count: int, slot_bytes: int, wait: WaitPolicy | None = None,
          resources: ResourcePlan | None = None, stages: int = 1) -> "PipelinePolicy":
    storage = StoragePolicy("lds", buffer_count=buffer_count, slot_bytes=slot_bytes)
    return cls(storage, wait or WaitPolicy("full_barrier"),
               resources or ResourcePlan("host_estimate", lds_bytes=storage.buffer_count * storage.slot_bytes), stages)

  @classmethod
  def register_resident(cls, *, stages: int = 2, wait: WaitPolicy | None = None,
                        resources: ResourcePlan | None = None) -> "PipelinePolicy":
    storage = StoragePolicy("global_register_resident")
    return cls(storage, wait or WaitPolicy("targeted_vmcnt", scope="per_stage"),
               resources or ResourcePlan("host_estimate"), stages)


def pipeline_policy_for_route(route_family: str, *, buffer_count: int = 1, slot_bytes: int = 0,
                              stages: int | None = None) -> PipelinePolicy:
  """Resolve route names once, at the compiler policy boundary.

  Callers may still carry a legacy ``route_family`` string, but all lowering
  code receives the same typed composition after this point.
  """
  if route_family == "lds":
    if slot_bytes <= 0: raise ValueError("LDS route requires positive slot_bytes")
    return PipelinePolicy.lds(buffer_count=buffer_count, slot_bytes=slot_bytes, stages=1 if stages is None else stages)
  if route_family == "pipe": return PipelinePolicy.register_resident(stages=2 if stages is None else stages)
  raise ValueError(f"unsupported pipeline route family {route_family!r}")

@dataclass(frozen=True)
class RegisterPipePlan:
  stages: int = 2; global_load_bytes: int = 16; storage: StoragePolicy = StoragePolicy("global_register_resident"); wait: WaitPolicy = WaitPolicy("targeted_vmcnt", "per_stage"); resources: ResourcePlan = ResourcePlan("host_estimate")
  def __post_init__(self):
    if self.stages != 2: raise ValueError("register pipe requires exactly two stages")
    if self.global_load_bytes != 16: raise ValueError("register pipe requires global b128 loads")
    if self.storage.kind != "global_register_resident" or self.storage.buffer_count != 1 or self.storage.slot_bytes != 0: raise ValueError("register pipe storage must be zero-LDS global-register-resident")
    if self.wait.kind != "targeted_vmcnt" or self.wait.scope != "per_stage": raise ValueError("register pipe requires per-stage targeted wait dependency")
    if self.resources.stage != "host_estimate" or self.resources.vgpr is not None or self.resources.sgpr is not None: raise ValueError("register pipe final resources are unproven")

  @property
  def policy(self) -> PipelinePolicy:
    """Expose the register plan through the common policy composition."""
    return PipelinePolicy(self.storage, self.wait, self.resources, stages=self.stages)
