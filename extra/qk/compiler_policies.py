"""Immutable compiler policy contracts; intentionally not wired into routing yet."""
from dataclasses import dataclass
from typing import Literal

StorageKind = Literal["lds", "global_register_resident"]
WaitKind = Literal["full_barrier", "targeted_vmcnt"]
ResourceStage = Literal["host_estimate", "final_program"]


@dataclass(frozen=True)
class StoragePolicy:
  kind: StorageKind
  buffer_count: int = 1
  slot_bytes: int = 0
  roles: tuple[str, ...] = ("A", "B")

  def __post_init__(self):
    if self.kind not in ("lds", "global_register_resident"): raise ValueError("unsupported storage kind")
    if self.buffer_count not in (1, 2): raise ValueError("buffer_count must be 1 or 2")
    if not isinstance(self.slot_bytes, int) or self.slot_bytes < 0: raise ValueError("slot_bytes must be non-negative")
    if self.kind == "lds" and self.slot_bytes <= 0: raise ValueError("LDS storage requires positive slot_bytes")
    if self.kind == "global_register_resident" and self.slot_bytes != 0: raise ValueError("register-resident storage cannot declare LDS slots")
    if self.roles != ("A", "B"): raise ValueError("storage roles must be exactly ('A', 'B')")


@dataclass(frozen=True)
class WaitPolicy:
  kind: WaitKind
  scope: str = "workgroup"

  def __post_init__(self):
    if self.kind not in ("full_barrier", "targeted_vmcnt"): raise ValueError("unsupported wait kind")
    if self.kind == "full_barrier" and self.scope != "workgroup": raise ValueError("full barrier scope must be workgroup")
    if self.kind == "targeted_vmcnt" and self.scope != "per_stage": raise ValueError("targeted vmcnt scope must be per_stage")


@dataclass(frozen=True)
class ResourcePlan:
  stage: ResourceStage
  lds_bytes: int = 0
  scratch_bytes: int = 0
  vgpr: int | None = None
  sgpr: int | None = None

  def __post_init__(self):
    if self.stage not in ("host_estimate", "final_program"): raise ValueError("unsupported resource stage")
    if any(not isinstance(x, int) or x < 0 for x in (self.lds_bytes, self.scratch_bytes)): raise ValueError("resource bytes must be non-negative ints")
    if self.stage == "host_estimate" and (self.vgpr is not None or self.sgpr is not None):
      raise ValueError("host estimate cannot claim final register counts")
    if self.stage == "final_program" and (self.vgpr is None or self.sgpr is None):
      raise ValueError("final program requires VGPR and SGPR counts")
    if any(x is not None and (not isinstance(x, int) or x < 0) for x in (self.vgpr, self.sgpr)):
      raise ValueError("register counts must be non-negative ints")


@dataclass(frozen=True)
class RegisterPipePlan:
  """Contract-only plan for a future two-stage register-resident pipe."""
  stages: int = 2
  global_load_bytes: int = 16
  storage: StoragePolicy = StoragePolicy("global_register_resident")
  wait: WaitPolicy = WaitPolicy("targeted_vmcnt", scope="per_stage")
  resources: ResourcePlan = ResourcePlan("host_estimate")

  def __post_init__(self):
    if self.stages != 2: raise ValueError("register pipe requires exactly two stages")
    if self.global_load_bytes != 16: raise ValueError("register pipe requires global b128 loads")
    if self.storage.kind != "global_register_resident" or self.storage.buffer_count != 1 or self.storage.slot_bytes != 0:
      raise ValueError("register pipe storage must be zero-LDS global-register-resident")
    if self.wait.kind != "targeted_vmcnt" or self.wait.scope != "per_stage":
      raise ValueError("register pipe requires per-stage targeted wait dependency")
    if self.resources.stage != "host_estimate" or self.resources.vgpr is not None or self.resources.sgpr is not None:
      raise ValueError("register pipe final resources are unproven")
