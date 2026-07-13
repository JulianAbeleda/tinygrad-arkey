"""Core-neutral immutable compiler policy contracts."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

StorageKind = Literal["lds", "global_register_resident"]
WaitKind = Literal["full_barrier", "targeted_vmcnt"]
ResourceStage = Literal["host_estimate", "final_program"]
OperandOwnership = Literal["workgroup_shared", "wave_private"]

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
  producer_stage: int | None = None
  consumer_stage: int | None = None
  scope: str | None = None
  def __post_init__(self):
    if not isinstance(self.policy, WaitPolicy): raise ValueError("wait dependency requires WaitPolicy")
    if not all(isinstance(x, str) and x for x in (self.producer, self.consumer, self.load_group)): raise ValueError("wait dependency labels must be non-empty")
    for name, value in (("producer_stage", self.producer_stage), ("consumer_stage", self.consumer_stage)):
      if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be a non-negative int when provided")
    if self.scope is not None and self.scope not in ("workgroup", "per_stage"):
      raise ValueError("wait dependency scope must be workgroup or per_stage")

@dataclass(frozen=True)
class WaitDependencyCoverage:
  """Lifecycle proof result for typed producer/consumer wait provenance."""
  passed: bool
  errors: tuple[str, ...]
  covered: tuple[tuple[str, int, int], ...]

  def to_json(self) -> dict[str, object]:
    """Serialize the proof without losing its typed stage-edge identity.

    Evidence consumers must receive the actual covered edges; a bare
    ``passed`` flag is insufficient to prove that the waits belong to this
    pipeline.  Keep this representation backend-neutral so AMD and other
    consumers can share the same artifact join.
    """
    return {"passed": self.passed, "errors": list(self.errors),
            "covered": [list(edge) for edge in self.covered]}

  @classmethod
  def from_json(cls, row: object) -> "WaitDependencyCoverage":
    """Parse serialized coverage fail-closed for artifact readers."""
    if not isinstance(row, dict):
      raise TypeError("wait dependency coverage must be an object")
    passed = row.get("passed")
    errors = row.get("errors", [])
    covered = row.get("covered", [])
    if not isinstance(passed, bool) or not isinstance(errors, list) or any(not isinstance(x, str) for x in errors):
      raise ValueError("malformed wait dependency coverage status")
    if not isinstance(covered, list):
      raise ValueError("malformed wait dependency coverage edges")
    edges: list[tuple[str, int, int]] = []
    for edge in covered:
      if not isinstance(edge, list) or len(edge) != 3 or not isinstance(edge[0], str) or not edge[0] or \
         any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in edge[1:]):
        raise ValueError("malformed wait dependency coverage edge")
      parsed = (edge[0], edge[1], edge[2])
      if parsed in edges:
        raise ValueError("duplicate wait dependency coverage edge")
      edges.append(parsed)
    return cls(passed, tuple(errors), tuple(edges))

def prove_wait_dependency_coverage(policy: PipelinePolicy, dependencies: tuple[WaitDependency, ...] | list[WaitDependency],
                                   required: tuple[tuple[str, int, int], ...] = ()) -> WaitDependencyCoverage:
  """Validate stage coverage without emitting or inferring a backend wait.

  ``required`` identifies the producer/load-group/consumer stage edges the
  lifecycle expects.  Targeted waits must name both stages and are rejected
  when duplicated or outside the policy's logical stage range.  The helper is
  deliberately backend-neutral; capability admission remains a separate
  fail-closed step.
  """
  errors: list[str] = []
  if not isinstance(policy, PipelinePolicy):
    raise TypeError("wait coverage requires PipelinePolicy")
  if not isinstance(dependencies, (tuple, list)):
    raise TypeError("wait dependencies must be a tuple or list")
  if any(not isinstance(x, WaitDependency) for x in dependencies):
    raise TypeError("wait coverage requires typed WaitDependency values")
  required_keys: set[tuple[str, int, int]] = set()
  for item in required:
    if not isinstance(item, tuple) or len(item) != 3 or not isinstance(item[0], str) or not item[0] or \
       any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in item[1:]):
      errors.append(f"invalid required wait edge {item!r}")
    elif item in required_keys:
      errors.append(f"duplicate required wait edge {item!r}")
    else:
      required_keys.add(item)

  covered: set[tuple[str, int, int]] = set()
  for index, dep in enumerate(dependencies):
    where = f"dependency {index}"
    if dep.policy != policy.wait:
      errors.append(f"{where}: policy does not match pipeline wait policy")
    expected_scope = "per_stage" if dep.policy.kind == "targeted_vmcnt" else "workgroup"
    if dep.scope is not None and dep.scope != expected_scope:
      errors.append(f"{where}: scope {dep.scope!r} does not match {expected_scope!r}")
    if dep.policy.kind == "targeted_vmcnt":
      if dep.producer_stage is None or dep.consumer_stage is None:
        errors.append(f"{where}: targeted wait requires producer and consumer stages")
        continue
      if dep.producer_stage >= policy.logical_stage_count or dep.consumer_stage >= policy.logical_stage_count:
        errors.append(f"{where}: stage is outside policy range 0..{policy.logical_stage_count - 1}")
      key = (dep.load_group, dep.producer_stage, dep.consumer_stage)
      if key in covered:
        errors.append(f"{where}: duplicate wait edge {key!r}")
      else:
        covered.add(key)
    elif dep.producer_stage is not None or dep.consumer_stage is not None:
      errors.append(f"{where}: full barrier wait must not claim stage-specific coverage")

  missing = sorted(required_keys - covered)
  errors.extend(f"missing wait coverage for {item!r}" for item in missing)
  return WaitDependencyCoverage(not errors, tuple(errors), tuple(sorted(covered)))

@dataclass(frozen=True)
class WaitCount:
  """Typed AMD wait-counter immediate.

  Fields use architectural counter values rather than an encoded instruction
  word, so any backend that owns wait lowering can reuse this contract.
  """
  vmcnt: int = 63
  lgkmcnt: int = 63
  expcnt: int = 7

  def __post_init__(self):
    if not isinstance(self.vmcnt, int) or isinstance(self.vmcnt, bool) or not 0 <= self.vmcnt <= 63:
      raise ValueError("vmcnt must be an integer in 0..63")
    if not isinstance(self.lgkmcnt, int) or isinstance(self.lgkmcnt, bool) or not 0 <= self.lgkmcnt <= 63:
      raise ValueError("lgkmcnt must be an integer in 0..63")
    if not isinstance(self.expcnt, int) or isinstance(self.expcnt, bool) or not 0 <= self.expcnt <= 7:
      raise ValueError("expcnt must be an integer in 0..7")

  @property
  def simm16(self) -> int:
    """Pack architectural fields into the AMD SOPP immediate."""
    return (self.vmcnt << 10) | (self.lgkmcnt << 4) | self.expcnt


def wait_count_for_dependency(dep: WaitDependency, *, younger_vmem_loads: int | None = None) -> WaitCount:
  """Create a sound AMD wait immediate from a typed staged dependency.

  AMD ``vmcnt(N)`` permits *N outstanding VMEM operations*; it does not mean
  "wait for N loads".  The sequential register pipeline therefore defaults to
  a full VMEM drain.  A non-zero value is accepted only when a caller supplies
  a backend-proven count of younger VMEM loads which are safe to leave in
  flight.  Physical instruction placement remains renderer-owned.
  """
  if not isinstance(dep, WaitDependency):
    raise TypeError("expected WaitDependency")
  if dep.policy.kind != "targeted_vmcnt" or dep.policy.scope != "per_stage":
    raise ValueError("targeted wait lowering requires a per-stage targeted dependency")
  if dep.producer_stage is None or dep.consumer_stage is None:
    raise ValueError("targeted wait lowering requires producer and consumer stages")
  if younger_vmem_loads is None: younger_vmem_loads = 0
  if not isinstance(younger_vmem_loads, int) or isinstance(younger_vmem_loads, bool) or younger_vmem_loads < 0:
    raise ValueError("younger_vmem_loads must be a non-negative backend-proven count")
  return WaitCount(vmcnt=younger_vmem_loads)

def amdllvm_wait_dependency(dep: WaitDependency) -> WaitDependency:
  """Validate the dependency contract still owned by the graph lifecycle.

  ``WaitCount`` has a backend intrinsic seam, but graph-level dependency
  scheduling has not yet been wired to emit it; keep this adapter fail-closed
  until that lifecycle lowering exists.
  """
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


@dataclass(frozen=True)
class GEMMWorkgroupPolicy:
  """Storage-aware work decomposition, independent of backend lowering.

  Registers are private to one wave, while LDS is shared by a workgroup.  Keep
  that ownership distinction in the schedule authority instead of asking a
  transport adapter to reinterpret one geometry after the fact.
  """
  tile: tuple[int, int, int]
  waves: tuple[int, int]
  threads: int
  wave_size: int
  ownership: OperandOwnership
  reuse: tuple[int, int]

  def __post_init__(self) -> None:
    if len(self.tile) != 3 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.tile):
      raise ValueError("GEMM tile must contain three positive ints")
    if len(self.waves) != 2 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.waves):
      raise ValueError("GEMM waves must contain two positive ints")
    if self.wave_size not in (32, 64) or self.threads != self.waves[0] * self.waves[1] * self.wave_size:
      raise ValueError("GEMM threads must equal waves_m*waves_n*wave_size")
    if self.ownership not in ("workgroup_shared", "wave_private"):
      raise ValueError("unsupported GEMM operand ownership")
    if len(self.reuse) != 2 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.reuse):
      raise ValueError("GEMM A/B reuse must contain two positive ints")
    if any(x % 16 for x in self.tile): raise ValueError("WMMA GEMM tile dimensions must be multiples of 16")
    if self.ownership == "wave_private" and (self.waves != (1, 1) or self.threads != self.wave_size):
      raise ValueError("wave-private operands require exactly one wave per workgroup")

  @classmethod
  def register_wave(cls, *, pipe_tm: int = 2, pipe_tn: int = 2, k_steps: int = 2,
                    wave_size: int = 32) -> "GEMMWorkgroupPolicy":
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in (pipe_tm, pipe_tn, k_steps)):
      raise ValueError("register wave factors must be positive ints")
    return cls((pipe_tm * 16, pipe_tn * 16, k_steps * 16), (1, 1), wave_size, wave_size,
               "wave_private", (pipe_tn, pipe_tm))

  @classmethod
  def cooperative_lds(cls, *, tile: tuple[int, int, int], waves: tuple[int, int],
                      wave_size: int = 32, reuse: tuple[int, int]) -> "GEMMWorkgroupPolicy":
    return cls(tile, waves, waves[0] * waves[1] * wave_size, wave_size, "workgroup_shared", reuse)


@dataclass(frozen=True)
class GEMMSchedulePolicy:
  """One public schedule composition shared by all storage transports."""
  workgroup: GEMMWorkgroupPolicy
  pipeline: PipelinePolicy

  def __post_init__(self) -> None:
    if not isinstance(self.workgroup, GEMMWorkgroupPolicy) or not isinstance(self.pipeline, PipelinePolicy):
      raise ValueError("GEMM schedule requires typed workgroup and pipeline policies")
    register = self.pipeline.storage_kind == "global_register_resident"
    if register != (self.workgroup.ownership == "wave_private"):
      raise ValueError("GEMM storage and operand ownership disagree")

  @classmethod
  def register_native(cls, *, pipe_tm: int = 2, pipe_tn: int = 2, k_steps: int = 2,
                      wave_size: int = 32) -> "GEMMSchedulePolicy":
    return cls(GEMMWorkgroupPolicy.register_wave(pipe_tm=pipe_tm, pipe_tn=pipe_tn,
      k_steps=k_steps, wave_size=wave_size), PipelinePolicy.register_resident(stages=2))

  @classmethod
  def lds_cooperative(cls, *, tile: tuple[int, int, int], waves: tuple[int, int], slot_bytes: int,
                      buffer_count: int = 2, wave_size: int = 32,
                      reuse: tuple[int, int]) -> "GEMMSchedulePolicy":
    return cls(GEMMWorkgroupPolicy.cooperative_lds(tile=tile, waves=waves, wave_size=wave_size, reuse=reuse),
               PipelinePolicy.lds(buffer_count=buffer_count, slot_bytes=slot_bytes, stages=1))


def pipeline_policy_for_route(route_family: str, *, buffer_count: int = 1, slot_bytes: int = 0,
                              stages: int | None = None) -> PipelinePolicy:
  """Resolve route names once, at the compiler policy boundary.

  Callers may still carry a legacy ``route_family`` string, but all lowering
  code receives the same typed composition after this point.
  """
  if route_family == "lds":
    if slot_bytes <= 0: raise ValueError("LDS route requires positive slot_bytes")
    return PipelinePolicy.lds(buffer_count=buffer_count, slot_bytes=slot_bytes, stages=1 if stages is None else stages)
  if route_family == "pipe":
    # The compiler-owned register pipe contract is specifically the proved
    # two-stage/b128 primitive. Do not silently construct a weaker variant.
    return RegisterPipePlan(stages=2 if stages is None else stages).policy
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

  @property
  def pipeline_policy(self) -> PipelinePolicy:
    """Candidate-pipeline protocol name consumed by postrange."""
    return self.policy

  @property
  def wait_coverage(self) -> WaitDependencyCoverage:
    deps = tuple(WaitDependency(self.wait, f"global_load_{role}", "gemm_consumer", role,
                                 producer_stage=0, consumer_stage=1, scope="per_stage")
                 for role in ("A", "B"))
    return prove_wait_dependency_coverage(self.policy, deps, (("A", 0, 1), ("B", 0, 1)))
