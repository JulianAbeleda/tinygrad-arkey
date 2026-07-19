"""Live allocation-census bridge for frozen staged C7 memory evidence.

The GPU harness owns allocation and dispatch.  This module supplies the small
set of hooks it needs to classify those allocations without teaching the
physical ledger about MMQ roles:

* one explicit semantic owner for every physical C7 allocation;
* exact route start/end markers;
* projection of physical lifetimes onto the measured route window; and
* content-addressed PM4/AQL observations for the existing C7 contract.

The module is CPU-only.  It never imports ``Device``, constructs a runtime, or
dispatches a program.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

from extra.qk.mmq_exact_role_spec import DEFAULT_INVENTORY, exact_role_spec
from extra.qk.mmq_frozen_staged_family import (
  QUEUE_MODES, FrozenStagedFamily, load_frozen_staged_family_manifest,
)
from extra.qk.mmq_staged_c7_c8_contract import (
  build_staged_c7_memory_ledger, physical_lifetime_rows,
  staged_c7_budget_identity, staged_c7_census_identity,
  staged_logical_memory_requirements, validate_staged_c7_memory_ledger,
)
from extra.qk.mmq_staged_c7_authority import load_staged_c7_authority_snapshot
from extra.qk.physical_memory_ledger import (
  AllocationOwner, PhysicalMemoryEvidence, PhysicalMemoryLedger,
  allocation_owner, bind_allocation_owner,
)


LOGICAL_CATEGORIES = (
  "full_q4_source", "full_q8_values_source", "full_q8_scales_source", "full_q8_sums_source",
  "compact_q4_stage", "compact_q8_values_stage", "compact_q8_scales_stage", "compact_q8_sums_stage",
  "output",
)
INFRASTRUCTURE_CATEGORIES = ("code_object", "runtime", "kernarg", "queue_state")
TEMPORARY_CATEGORIES = ("temporary_gather", "temporary_transfer")
REQUIRED_CATEGORIES = LOGICAL_CATEGORIES + INFRASTRUCTURE_CATEGORIES + TEMPORARY_CATEGORIES
FULL_ROUTE_CATEGORIES = LOGICAL_CATEGORIES + ("code_object", "runtime", "queue_state")
_ALLOWED_CATEGORIES = frozenset(REQUIRED_CATEGORIES + ("co_resident_model",))
_ZERO_ADMISSIBLE_CATEGORIES = frozenset(TEMPORARY_CATEGORIES)
C7_QUEUE_CAPTURE_SCHEMA = "tinygrad.mmq_q4k_q8_1.staged_c7_queue_capture_isolation.v1"


def _canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _nonempty(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value: raise ValueError(f"{label} must be a non-empty string")
  return value


def _positive(value: Any, label: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f"{label} must be a positive integer")
  return value


def staged_c7_memory_authority(*, device_identity: str, software_identity: str,
                               allocator_identity: str, allocation_granularity_bytes: int,
                               admitted_budget_bytes: int, budget_provenance: str) -> dict[str, Any]:
  """Build the one authority shared by PM4 and AQL C7 observations."""
  authority = {
    "device_identity": _nonempty(device_identity, "device_identity"),
    "software_identity": _nonempty(software_identity, "software_identity"),
    "allocator_identity": _nonempty(allocator_identity, "allocator_identity"),
    "allocation_granularity_bytes":
      _positive(allocation_granularity_bytes, "allocation_granularity_bytes"),
  }
  authority["budget_identity"] = staged_c7_budget_identity(
    **authority, admitted_budget_bytes=_positive(admitted_budget_bytes, "admitted_budget_bytes"),
    budget_provenance=_nonempty(budget_provenance, "budget_provenance"))
  return authority


@dataclass
class FrozenStagedC7QueueCapture:
  """One queue's explicit semantic ownership and physical route census.

  Allocations which exist before :meth:`begin_route` and are released after
  :meth:`end_route` are projected to the exact measured route window.  This is
  an intersection, not an inferred lifetime: the underlying ledger must prove
  that each physical allocation was live across that complete interval.
  """
  family: FrozenStagedFamily
  queue_mode: str
  ledger_device: str
  allocation_granularity_bytes: int
  ledger: PhysicalMemoryLedger = field(init=False)
  _categories: dict[str, str] = field(init=False, default_factory=dict)
  _owners: dict[tuple[str, str], AllocationOwner] = field(init=False, default_factory=dict)
  _external: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
  _route_start: int | None = field(init=False, default=None)
  _route_end: int | None = field(init=False, default=None)
  _capture_closed: bool = field(init=False, default=False)

  def __post_init__(self) -> None:
    staged_logical_memory_requirements(self.family)
    if self.queue_mode not in QUEUE_MODES:
      raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
    self.ledger_device = _nonempty(self.ledger_device, "ledger_device")
    self.allocation_granularity_bytes = _positive(
      self.allocation_granularity_bytes, "allocation_granularity_bytes")
    self.ledger = PhysicalMemoryLedger((self.ledger_device,))

  def owner(self, category: str, allocation_name: str) -> AllocationOwner:
    """Return the stable owner to bind around or onto one exact allocation."""
    category = _nonempty(category, "category")
    allocation_name = _nonempty(allocation_name, "allocation_name")
    if category == "dense_fp16_weight":
      raise ValueError("dense FP16 weight materialization is forbidden by frozen staged C7")
    if category not in _ALLOWED_CATEGORIES:
      raise ValueError(f"category {category!r} is not an admitted frozen staged C7 category")
    key = (category, allocation_name)
    if key not in self._owners:
      semantic = (
        f"frozen_staged_c7:{self.family.family_identity}:{self.queue_mode}:"
        f"{category}:{allocation_name}")
      if semantic in self._categories:
        raise ValueError("C7 semantic owner identity collision")
      lifetime = "route" if category in FULL_ROUTE_CATEGORIES else \
        "dispatch" if category == "kernarg" else "epoch"
      self._owners[key] = AllocationOwner(
        f"frozen_staged_{category}", lifetime,
        candidate_id=self.family.family_identity, semantic_owner_id=semantic)
      self._categories[semantic] = category
    return self._owners[key]

  @contextlib.contextmanager
  def allocation(self, category: str, allocation_name: str) -> Iterator[AllocationOwner]:
    """Ambient ownership hook for an allocation performed by the harness."""
    owner = self.owner(category, allocation_name)
    with allocation_owner(
        kind=owner.kind, lifetime=owner.lifetime, candidate_id=owner.candidate_id,
        semantic_owner_id=owner.semantic_owner_id):
      yield owner

  def bind_buffer(self, buffer: Any, category: str, allocation_name: str) -> AllocationOwner:
    """Persist ownership onto a lazy Buffer before its eventual allocation."""
    owner = self.owner(category, allocation_name)
    return bind_allocation_owner(buffer, owner)

  def record_external_full_route(self, category: str, allocation_name: str, allocation: Any, *,
                                 requested_bytes: int | None = None) -> None:
    """Record one real non-``Buffer`` HCQ allocation already live for the route.

    AMD program images, kernarg pools, signals, and queue rings are allocated
    below tinygrad's public ``Buffer`` layer.  They therefore cannot appear in
    :class:`PhysicalMemoryLedger`, but expose exact base addresses and sizes.
    This hook records those handles as runtime-allocation-census authority.
    """
    self.owner(category, allocation_name)
    if category not in INFRASTRUCTURE_CATEGORIES:
      raise ValueError("external C7 allocations are admitted only for runtime infrastructure")
    try: base = allocation.base
    except (AttributeError, TypeError): base = allocation
    physical = getattr(base, "size", getattr(base, "nbytes", None))
    address = getattr(base, "va_addr", getattr(base, "addr", None))
    physical = _positive(physical, f"{category} physical bytes")
    if physical % self.allocation_granularity_bytes:
      raise ValueError(f"{category} physical bytes differ from the scanned allocator granularity")
    if not isinstance(address, int) or isinstance(address, bool) or address <= 0:
      raise ValueError(f"{category} physical base address must be a positive integer")
    requested = physical if requested_bytes is None else _positive(
      requested_bytes, f"{category} requested bytes")
    if requested > physical:
      raise ValueError(f"{category} requested bytes exceed its physical allocation")
    physical_base = f"{self.ledger_device}:external_va:{address:x}"
    row = {
      "allocation_id": f"external:{self.queue_mode}:{category}:{allocation_name}",
      "physical_base_identity": physical_base, "category": category,
      "requested_bytes": requested, "physical_bytes": physical,
      "provenance": f"runtime_allocation_census:{category}:{allocation_name}",
    }
    if (existing := self._external.get(physical_base)) is not None and existing != row:
      raise ValueError("external C7 physical base has conflicting semantic ownership")
    self._external[physical_base] = row

  @contextlib.contextmanager
  def capture(self) -> Iterator["FrozenStagedC7QueueCapture"]:
    """Activate the physical ledger for preparation, route, and teardown."""
    if self._capture_closed or self._route_start is not None or self.ledger.events:
      raise RuntimeError("frozen staged C7 queue capture is single-use")
    with self.ledger.active(): yield self
    self._capture_closed = True
    if self._route_start is None or self._route_end is None:
      raise RuntimeError("frozen staged C7 route boundaries were not completed")

  def begin_route(self) -> int:
    """Mark the first logical event of the measured staged route."""
    if self._capture_closed or self._route_start is not None:
      raise RuntimeError("frozen staged C7 route start is invalid or duplicated")
    self._route_start = self.ledger.record_boundary(
      f"frozen_staged_c7:{self.queue_mode}:route_start")
    return self._route_start

  def end_route(self) -> int:
    """Mark the exclusive end of the measured staged route."""
    if self._capture_closed or self._route_start is None or self._route_end is not None:
      raise RuntimeError("frozen staged C7 route end is invalid or duplicated")
    self._route_end = self.ledger.record_boundary(
      f"frozen_staged_c7:{self.queue_mode}:route_end")
    return self._route_end

  @property
  def route_window(self) -> tuple[int, int]:
    if not self._capture_closed or self._route_start is None or self._route_end is None:
      raise RuntimeError("frozen staged C7 queue capture is not complete")
    return self._route_start, self._route_end

  def physical_evidence(self) -> PhysicalMemoryEvidence:
    """Export exact rounded physical sizes under the scanned granularity."""
    if not self._capture_closed: raise RuntimeError("frozen staged C7 queue capture is still active")
    return self.ledger.export_evidence(
      scanned_granularities={self.ledger_device: self.allocation_granularity_bytes})

  def observation(self, *, memory_authority: Mapping[str, Any]) -> dict[str, Any]:
    """Create one content-addressed queue observation for the C7 builder."""
    if memory_authority.get("allocation_granularity_bytes") != self.allocation_granularity_bytes:
      raise ValueError("queue capture granularity differs from the admitted memory authority")
    start, end = self.route_window
    evidence = self.physical_evidence()
    rows = physical_lifetime_rows(
      evidence, category_by_semantic_owner=self._categories)
    projected = []
    for row in rows:
      live_from, live_until = max(row["live_from"], start), min(row["live_until"], end)
      if live_from >= live_until:
        raise ValueError(
          f"physical allocation {row['allocation_id']} does not overlap the measured route")
      projected.append({**row, "live_from": live_from, "live_until": live_until})

    external_evidence_identity = _identity({
      "schema": "tinygrad.mmq_q4k_q8_1.staged_c7_external_runtime_census.v1",
      "family_identity": self.family.family_identity, "queue_mode": self.queue_mode,
      "allocations": sorted(self._external.values(), key=lambda row: row["allocation_id"]),
    })
    for row in self._external.values():
      projected.append({
        **row, "live_from": start, "live_until": end,
        "source": "runtime_allocation_census",
        "source_evidence_identity": external_evidence_identity,
      })
    source_identity = _identity(evidence.to_json())
    for category in TEMPORARY_CATEGORIES:
      if not any(row["category"] == category for row in projected):
        projected.append({
          "allocation_id": f"zero:{self.queue_mode}:{category}",
          "physical_base_identity": None, "category": category,
          "requested_bytes": 0, "physical_bytes": 0,
          "live_from": start, "live_until": end,
          "provenance": f"explicit_zero_measurement:{category}",
          "source": "explicit_zero_measurement",
          "source_evidence_identity": source_identity,
        })
    projected.sort(key=lambda row: (row["live_from"], row["allocation_id"]))
    authority = dict(memory_authority)
    return {
      "route_start": start, "route_end": end,
      "allocation_census_identity": staged_c7_census_identity(
        authority=authority, route_start=start, route_end=end, lifetimes=projected),
      "allocation_census_complete": True,
      # This is derived from a complete, explicit-owner ledger.  The owner API
      # rejects dense_fp16_weight and the physical ledger rejects unowned rows.
      "dense_fp16_weight_materialization": False,
      "authority": authority, "lifetimes": projected,
    }


@dataclass
class FrozenStagedC7HarnessAdapter:
  """Implementation of the staged harness's default-off lifecycle observer."""
  queue_capture: FrozenStagedC7QueueCapture
  _runtime_identity: int | None = field(init=False, default=None)
  _kernarg_base: tuple[int, int] | None = field(init=False, default=None)
  _launch_count: int = field(init=False, default=0)

  @contextlib.contextmanager
  def active(self) -> Iterator["FrozenStagedC7HarnessAdapter"]:
    with self.queue_capture.capture(): yield self

  def allocation(self, category: str, name: str) -> contextlib.AbstractContextManager[AllocationOwner]:
    return self.queue_capture.allocation(category, name)

  def bind_buffer(self, buffer: Any, category: str, name: str) -> AllocationOwner:
    """Persist the semantic owner when the harness's allocation is lazy."""
    return self.queue_capture.bind_buffer(buffer, category, name)

  def begin_route(self) -> int: return self.queue_capture.begin_route()
  def end_route(self) -> int: return self.queue_capture.end_route()

  def runtime(self, runtime: Any, device: Any) -> None:
    """Bind exact native program/device allocations which bypass ``Buffer``."""
    if self._runtime_identity is not None and self._runtime_identity != id(runtime):
      raise ValueError("frozen staged C7 observed more than one runtime object")
    self._runtime_identity = id(runtime)
    lib_gpu = getattr(runtime, "lib_gpu", None)
    kernargs = getattr(device, "kernargs_buf", None)
    timeline = getattr(getattr(device, "timeline_signal", None), "base_buf", None)
    if lib_gpu is None or kernargs is None or timeline is None:
      raise ValueError("frozen staged C7 runtime lacks code, kernarg, or timeline allocations")
    self.queue_capture.record_external_full_route(
      "code_object", "compact_program_image", lib_gpu,
      requested_bytes=len(getattr(runtime, "lib", b"")) or None)
    self.queue_capture.record_external_full_route("kernarg", "device_kernarg_pool", kernargs)
    self.queue_capture.record_external_full_route("runtime", "timeline_signal_pool", timeline)
    scratch = getattr(device, "scratch", None)
    if scratch is not None:
      self.queue_capture.record_external_full_route(
        "runtime", "device_scratch_backing", scratch)
    pm4_ibs = getattr(device, "pm4_ibs", None)
    if pm4_ibs is not None:
      self.queue_capture.record_external_full_route(
        "queue_state", "aql_pm4_indirect_buffer_pool", pm4_ibs)
    kernarg_base = kernargs.base
    self._kernarg_base = (int(kernarg_base.va_addr), int(kernarg_base.size))

    queue_count = 0
    for kind, queues in (
        ("compute", getattr(device, "compute_queues", {})),
        ("sdma", getattr(device, "sdma_queues", {}))):
      if not isinstance(queues, Mapping): continue
      for ordinal, descriptor in sorted(queues.items(), key=lambda item: str(item[0])):
        ring = getattr(descriptor, "ring", None)
        if ring is None: continue
        self.queue_capture.record_external_full_route(
          "queue_state", f"{kind}_ring_{ordinal}", ring)
        queue_count += 1
    if queue_count == 0:
      raise ValueError("frozen staged C7 runtime lacks an exact queue ring census")

  def launch(self, runtime: Any, launch: Mapping[str, Any]) -> None:
    """Bind the captured launch's kernarg view to the measured kernarg pool."""
    if self._runtime_identity != id(runtime) or self._kernarg_base is None:
      raise ValueError("frozen staged C7 launch preceded runtime allocation census")
    kernarg = launch.get("kernarg") if isinstance(launch, Mapping) else None
    if not isinstance(kernarg, Mapping):
      raise ValueError("frozen staged C7 launch lacks a kernarg receipt")
    va, size = kernarg.get("va"), kernarg.get("size")
    base, capacity = self._kernarg_base
    if not isinstance(va, int) or isinstance(va, bool) or \
       not isinstance(size, int) or isinstance(size, bool) or size <= 0 or \
       va < base or va + size > base + capacity:
      raise ValueError("frozen staged C7 kernarg receipt is outside the measured pool")
    self._launch_count += 1

  def observation(self, *, memory_authority: Mapping[str, Any]) -> dict[str, Any]:
    if self._runtime_identity is None or self._launch_count <= 0:
      raise ValueError("frozen staged C7 requires one measured runtime and target launch")
    return self.queue_capture.observation(memory_authority=memory_authority)


def capture_frozen_staged_c7_queue_probe(*, family: FrozenStagedFamily, queue_mode: str,
                                         ledger_device: str,
                                         allocation_granularity_bytes: int,
                                         memory_authority: Mapping[str, Any],
                                         probe_kwargs: Mapping[str, Any],
                                         probe_runner: Any | None = None) -> dict[str, Any]:
  """Run the current staged harness under a real C7 observer.

  Production callers invoke this inside their already guarded queue-specific
  child.  ``probe_runner`` exists so CPU tests can exercise the live seam with
  a mock harness and no device initialization.
  """
  if probe_runner is None:
    from extra.qk.mmq_llama_five_buffer_gpu_harness import run_full_grid_target_role_probe
    probe_runner = run_full_grid_target_role_probe
  if not callable(probe_runner): raise TypeError("probe_runner must be callable")
  kwargs = dict(probe_kwargs)
  if "staged_lifecycle_observer" in kwargs:
    raise ValueError("probe_kwargs must not replace the exact C7 lifecycle observer")
  capture = FrozenStagedC7QueueCapture(
    family, queue_mode, ledger_device, allocation_granularity_bytes)
  observer = FrozenStagedC7HarnessAdapter(capture)
  with observer.active():
    result = probe_runner(**kwargs, staged_lifecycle_observer=observer)
  if not isinstance(result, Mapping) or result.get("status") != "PASS":
    raise ValueError("frozen staged C7 harness probe did not pass")
  try:
    observation = observer.observation(memory_authority=memory_authority)
  except BaseException as exc:
    # Preserve the exact physical reconciliation when live evidence fails.
    # Without this structured child receipt the outer isolation layer can say
    # only that a child failed, losing the allocation IDs needed to repair C7.
    physical = capture.physical_evidence().to_json()
    return {
      "schema": "tinygrad.mmq_q4k_q8_1.staged_c7_queue_probe.v1",
      "status": "BLOCKED", "exact_blocker":
        "frozen staged C7 physical allocation census did not close",
      "exception": type(exc).__name__, "error": str(exc),
      "queue_mode": queue_mode, "family_identity": family.family_identity,
      "probe": dict(result), "queue_observation": None,
      "physical_memory_evidence": physical,
      "target_dispatch_attempted": observer._launch_count > 0,
      "target_dispatch_count": observer._launch_count,
      "production_dispatch_changed": False,
    }
  return {
    "schema": "tinygrad.mmq_q4k_q8_1.staged_c7_queue_probe.v1",
    "status": "PASS", "queue_mode": queue_mode,
    "family_identity": family.family_identity,
    "probe": dict(result), "queue_observation": observation,
    "target_dispatch_attempted": True,
    "target_dispatch_count": observer._launch_count,
    "production_dispatch_changed": False,
  }


def _run_frozen_staged_c7_queue_worker(
    role_name: str, role_shape: tuple[int, int, int],
    frozen_bundle: str, staged_family_manifest: str, queue_mode: str,
    inventory: str | Path | Mapping[str, Any], ledger_device: str,
    allocation_granularity_bytes: int, memory_authority: Mapping[str, Any],
    ) -> dict[str, Any]:
  """Spawn-only worker: the parent never imports or constructs a live device."""
  os.environ.update({
    "AMD_AQL": "1" if queue_mode == "AQL" else "0",
    "DEV": ledger_device,
  })
  role = exact_role_spec(role_name, shape=role_shape, inventory=inventory)
  family = load_frozen_staged_family_manifest(
    staged_family_manifest, role_spec=role,
    frozen_bundle=frozen_bundle, inventory=inventory)
  return capture_frozen_staged_c7_queue_probe(
    family=family, queue_mode=queue_mode, ledger_device=ledger_device,
    allocation_granularity_bytes=allocation_granularity_bytes,
    memory_authority=memory_authority,
    probe_kwargs={
      "role_spec": role, "warmups": 0, "rounds": 1,
      "epoch_limit": role.epochs, "n_chunk_tiles": role.program.grid[0],
      "epoch_start": 0, "host_accumulate": False,
      "in_kernel_accumulate": True, "per_epoch_check": False,
      "persistent_buffers": True, "preloaded_epochs": True,
      "sync_each_epoch": True, "stable_metadata_staging": True,
      "stable_epoch_staging": True, "wait_each_dispatch": True,
      "frozen_bundle": frozen_bundle,
    })


def run_frozen_staged_c7_queue_capture_isolated(
    *, family: FrozenStagedFamily, queue_mode: str,
    frozen_bundle: str | Path, staged_family_manifest: str | Path,
    runtime_canary_isolation: Mapping[str, Any],
    memory_authority: Mapping[str, Any], ledger_device: str = "AMD",
    timeout_seconds: float = 900.0,
    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
    isolated_runner: Callable[..., Any] | None = None,
    health_probe: Callable[[Mapping[str, str]], bool] | None = None,
    fault_collector: Callable[[float], tuple[list[str], Mapping[str, Any]]] | None = None,
    canary_validator: Callable[..., Mapping[str, Any]] | None = None,
    probe_validator: Callable[..., Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
  """Capture one queue behind the existing spawn/health/fault containment.

  A persisted, passing isolated C4 is mandatory before the child can launch.
  There is no retry and no queue fallback.
  """
  try:
    if not isinstance(family, FrozenStagedFamily):
      raise TypeError("family must be a loader-validated FrozenStagedFamily")
    role = family.binding.role_spec
    if queue_mode not in QUEUE_MODES:
      raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
    expected_aql = "1" if queue_mode == "AQL" else "0"
    if os.environ.get("AMD_AQL") != expected_aql:
      raise ValueError("AMD_AQL environment does not match the requested C7 queue")
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
      raise ValueError("timeout_seconds must be positive")
    granularity = _positive(
      memory_authority.get("allocation_granularity_bytes"),
      "memory_authority.allocation_granularity_bytes")
    if canary_validator is None:
      from extra.qk.mmq_frozen_staged_family_execution import \
        validate_frozen_staged_runtime_canary_isolation
      canary_validator = validate_frozen_staged_runtime_canary_isolation
    validated_canary = dict(canary_validator(
      runtime_canary_isolation, family, queue_mode=queue_mode))
    if validated_canary.get("status") != "PASS" or \
       validated_canary.get("family_identity") != family.family_identity or \
       validated_canary.get("program_key") != family.binding.program_key or \
       validated_canary.get("queue_mode") != queue_mode:
      raise ValueError("persisted isolated C4 differs from the exact C7 family or queue")
  except BaseException as exc:
    return {
      "schema": C7_QUEUE_CAPTURE_SCHEMA, "status": "BLOCKED",
      "exact_blocker": "isolated C7 queue capture admission failed",
      "exception": type(exc).__name__, "error": str(exc),
      "queue_mode": queue_mode, "launched": False,
      "target_dispatch_attempted": False,
      "production_dispatch_changed": False,
    }

  if isolated_runner is None:
    from tinygrad.runtime.process_isolated import run_isolated
    isolated_runner = run_isolated
  if health_probe is None:
    from extra.qk.mmq_target_epoch_orchestrator import spawned_tiny_health_probe
    health_probe = spawned_tiny_health_probe
  if fault_collector is None:
    from extra.qk.mmq_target_epoch_orchestrator import collect_kernel_fault_evidence
    fault_collector = collect_kernel_fault_evidence
  # The shared tiny health probe admits only the queue selector. The selected
  # DEV is already fixed in the parent capture environment and inherited by
  # its spawned health child; the target worker receives ledger_device
  # explicitly below.
  env_overrides = {"AMD_AQL": expected_aql}
  started = time.time()
  try: health_before = bool(health_probe(env_overrides))
  except BaseException: health_before = False
  isolated, runner_error = None, None
  if health_before:
    try:
      isolated = isolated_runner(
        _run_frozen_staged_c7_queue_worker,
        args=(
          role.role, role.shape, str(Path(frozen_bundle).resolve()),
          str(Path(staged_family_manifest).resolve()), queue_mode, inventory,
          ledger_device, granularity, dict(memory_authority),
        ), timeout_seconds=float(timeout_seconds), start_method="spawn")
    except BaseException as exc:
      runner_error = f"{type(exc).__name__}: {exc}"
  try: health_after = bool(health_probe(env_overrides))
  except BaseException: health_after = False
  try:
    faults, fault_evidence = fault_collector(started)
    faults, fault_evidence = list(faults), dict(fault_evidence)
  except BaseException as exc:
    faults = [f"kernel fault collection failed: {type(exc).__name__}: {exc}"]
    fault_evidence = {}

  # A returned isolation result proves that the runner started and reaped one
  # child.  A runner exception does not prove either state, so preserve it as
  # unknown instead of deriving "launched" from an unrelated health probe.
  launched: bool | None = False if not health_before else \
    None if runner_error is not None else isolated is not None
  child_status = "not_launched" if launched is False else \
    "runner_error" if runner_error is not None else getattr(isolated, "status", None)
  timed_out = bool(getattr(isolated, "timed_out", False))
  child_error = runner_error or getattr(isolated, "error", None)
  child = getattr(isolated, "result", None)
  # Only the worker's passing queue-probe receipt proves target dispatch.  A
  # failed or timed-out child may have reached the target, but supplies no
  # exact receipt, so record unknown rather than an unsupported Boolean.
  target_dispatch_attempted: bool | None = \
    True if isinstance(child, Mapping) and \
      child.get("schema") == "tinygrad.mmq_q4k_q8_1.staged_c7_queue_probe.v1" and \
      child.get("target_dispatch_attempted") is True else \
    False if launched is False else None
  blocker = None
  if not health_before: blocker = "isolated C7 queue capture preflight health failed"
  elif timed_out: blocker = "isolated C7 queue capture child timed out"
  elif child_status != "passed" or not isinstance(child, Mapping):
    blocker = child_error or "isolated C7 queue capture child returned no structured result"
  elif faults: blocker = "kernel fault/reset marker observed during isolated C7 queue capture"
  elif not health_after: blocker = "isolated C7 queue capture postflight health failed"
  elif child.get("status") == "BLOCKED":
    blocker = child.get("exact_blocker") or "isolated C7 child blocked"

  observation, raw_probe, probe_validation = None, None, None
  if isinstance(child, Mapping):
    raw_probe = child.get("probe") if isinstance(child.get("probe"), Mapping) else None
    observation = child.get("queue_observation") \
      if isinstance(child.get("queue_observation"), Mapping) else None
  if blocker is None:
    try:
      if child.get("schema") != "tinygrad.mmq_q4k_q8_1.staged_c7_queue_probe.v1" or \
         child.get("status") != "PASS" or child.get("queue_mode") != queue_mode or \
         child.get("family_identity") != family.family_identity:
        raise ValueError("isolated C7 child identity or PASS state differs")
      if not isinstance(observation, Mapping) or not isinstance(raw_probe, Mapping):
        raise ValueError("isolated C7 child lacks queue observation or raw probe")
      if observation.get("authority") != dict(memory_authority) or \
         observation.get("allocation_census_complete") is not True or \
         observation.get("dense_fp16_weight_materialization") is not False:
        raise ValueError("isolated C7 queue observation authority or census state differs")
      expected_census = staged_c7_census_identity(
        authority=observation["authority"], route_start=observation["route_start"],
        route_end=observation["route_end"], lifetimes=observation["lifetimes"])
      if observation.get("allocation_census_identity") != expected_census:
        raise ValueError("isolated C7 allocation census identity differs")
      if probe_validator is None:
        from extra.qk.mmq_frozen_staged_family_execution import _validate_probe_result
        probe_validator = _validate_probe_result
      # This C7 route inserts the lifecycle observer inside its own isolated
      # worker, so the worker's direct probe intentionally has no nested
      # health wrapper.  Bind the outer parent's actual queue request,
      # pre/post health, and fault window onto that exact probe before applying
      # the existing guarded-probe validator.
      raw_probe = {
        **dict(raw_probe),
        "health_before": health_before, "health_after": health_after,
        "mode_health_before": health_before, "mode_health_after": health_after,
        "child_env_overrides": dict(env_overrides),
        "kernel_faults": list(faults),
      }
      probe_validation = dict(probe_validator(
        raw_probe, family, prefix_epochs=role.epochs,
        queue_mode=queue_mode, frozen_bundle=frozen_bundle))
    except BaseException as exc:
      blocker = "isolated C7 queue evidence failed closed"
      child_error = f"{type(exc).__name__}: {exc}"

  return {
    "schema": C7_QUEUE_CAPTURE_SCHEMA,
    "status": "PASS" if blocker is None else "BLOCKED",
    "exact_blocker": blocker, "role": role.role, "shape": list(role.shape),
    "queue_mode": queue_mode, "family_identity": family.family_identity,
    "program_key": family.binding.program_key,
    "binary_sha256": family.binding.binary_sha256,
    "containment_authority": "outer_parent_fresh_process_guards",
    "c4_runtime_canary_isolation": validated_canary,
    "launched": launched, "target_dispatch_attempted": target_dispatch_attempted,
    "target_dispatch_attempted_authority":
      "passing_child_queue_probe" if target_dispatch_attempted is True else
      "preflight_prevented_child_launch" if target_dispatch_attempted is False else
      "unknown_without_structured_child_queue_probe",
    "child_status": child_status, "timed_out": timed_out,
    "error": child_error, "timeout_seconds": timeout_seconds,
    "elapsed_seconds": getattr(isolated, "elapsed_seconds", None),
    "health_before": health_before, "health_after": health_after,
    "kernel_faults": faults, "kernel_fault_evidence": fault_evidence,
    "queue_observation": observation, "raw_probe": raw_probe,
    "child_probe": dict(child) if isinstance(child, Mapping) else None,
    "probe_validation": probe_validation,
    "compile_performed": False, "requires_recompile": False,
    "production_dispatch_changed": False, "no_fallback": True,
  }


def build_frozen_staged_c7_census(*, family: FrozenStagedFamily,
                                  captures: Mapping[str, FrozenStagedC7QueueCapture],
                                  admitted_budget_bytes: int, budget_provenance: str,
                                  device_identity: str, software_identity: str,
                                  allocator_identity: str,
                                  allocation_granularity_bytes: int) -> dict[str, Any]:
  """Join complete PM4/AQL queue captures into the existing C7 contract."""
  if not isinstance(captures, Mapping) or set(captures) != set(QUEUE_MODES):
    raise ValueError(f"captures must contain exactly {QUEUE_MODES!r}")
  authority = staged_c7_memory_authority(
    device_identity=device_identity, software_identity=software_identity,
    allocator_identity=allocator_identity,
    allocation_granularity_bytes=allocation_granularity_bytes,
    admitted_budget_bytes=admitted_budget_bytes, budget_provenance=budget_provenance)
  observations = {}
  for queue in QUEUE_MODES:
    capture = captures[queue]
    if not isinstance(capture, FrozenStagedC7QueueCapture) or capture.family is not family or \
       capture.queue_mode != queue:
      raise ValueError(f"{queue} capture differs from the exact staged family or queue")
    observations[queue] = capture.observation(memory_authority=authority)
  return build_staged_c7_memory_ledger(
    family=family, queue_observations=observations,
    admitted_budget_bytes=admitted_budget_bytes,
    budget_provenance=budget_provenance, memory_authority=authority)


def build_frozen_staged_c7_from_observations(*, family: FrozenStagedFamily,
                                             queue_observations: Mapping[str, Any],
                                             admitted_budget_bytes: int,
                                             budget_provenance: str,
                                             device_identity: str,
                                             software_identity: str,
                                             allocator_identity: str,
                                             allocation_granularity_bytes: int) -> dict[str, Any]:
  """CLI-facing join for observations emitted by isolated live queue runs."""
  authority = staged_c7_memory_authority(
    device_identity=device_identity, software_identity=software_identity,
    allocator_identity=allocator_identity,
    allocation_granularity_bytes=allocation_granularity_bytes,
    admitted_budget_bytes=admitted_budget_bytes, budget_provenance=budget_provenance)
  if not isinstance(queue_observations, Mapping) or set(queue_observations) != set(QUEUE_MODES):
    raise ValueError(f"queue_observations must contain exactly {QUEUE_MODES!r}")
  for queue, observation in queue_observations.items():
    if not isinstance(observation, Mapping) or observation.get("authority") != authority:
      raise ValueError(f"{queue} observation authority differs from the requested C7 authority")
  return build_staged_c7_memory_ledger(
    family=family, queue_observations=queue_observations,
    admitted_budget_bytes=admitted_budget_bytes,
    budget_provenance=budget_provenance, memory_authority=authority)


def _load_family(args: argparse.Namespace) -> FrozenStagedFamily:
  role = exact_role_spec(args.role, inventory=args.inventory)
  return load_frozen_staged_family_manifest(
    args.staged_family_manifest, role_spec=role,
    frozen_bundle=args.frozen_bundle, inventory=args.inventory)


def _write_json(path: Path | None, value: Mapping[str, Any]) -> None:
  encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
  if path is None: print(encoded, end="")
  else:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
      prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
      with os.fdopen(fd, "w") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
      os.replace(temporary, path)
    except BaseException:
      try: os.unlink(temporary)
      except FileNotFoundError: pass
      raise


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument("--staged-family-manifest", type=Path, required=True)
  common.add_argument("--frozen-bundle", type=Path, required=True)
  common.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
  common.add_argument("--role", required=True)
  sub = parser.add_subparsers(dest="command", required=True)
  requirements = sub.add_parser("requirements", parents=[common])
  requirements.add_argument("--output", type=Path)
  validate = sub.add_parser("validate", parents=[common])
  validate.add_argument("--c7-ledger", type=Path, required=True)
  capture = sub.add_parser("capture", parents=[common])
  capture.add_argument("--queue-mode", choices=QUEUE_MODES, required=True)
  capture.add_argument("--runtime-canary-isolation", type=Path, required=True)
  capture.add_argument("--authority-snapshot", type=Path, required=True)
  capture.add_argument("--timeout-seconds", type=float, default=900.0)
  capture.add_argument("--output", type=Path, required=True)
  build = sub.add_parser("build", parents=[common])
  build.add_argument("--pm4-observation", type=Path, required=True)
  build.add_argument("--aql-observation", type=Path, required=True)
  build.add_argument("--authority-snapshot", type=Path, required=True)
  build.add_argument("--output", type=Path)
  args = parser.parse_args(argv)
  family = _load_family(args)
  if args.command == "requirements":
    result = staged_logical_memory_requirements(family)
    _write_json(args.output, result)
    return 0
  if args.command == "validate":
    result = validate_staged_c7_memory_ledger(
      json.loads(args.c7_ledger.read_text()), family=family)
    _write_json(None, result)
    return 0 if result["status"] == "PASS" else 1
  authority_snapshot = load_staged_c7_authority_snapshot(args.authority_snapshot)
  authority = authority_snapshot["memory_authority"]
  budget = authority_snapshot["budget"]
  if args.command == "capture":
    result = run_frozen_staged_c7_queue_capture_isolated(
      family=family, queue_mode=args.queue_mode,
      frozen_bundle=args.frozen_bundle,
      staged_family_manifest=args.staged_family_manifest,
      runtime_canary_isolation=json.loads(args.runtime_canary_isolation.read_text()),
      memory_authority=authority,
      ledger_device=authority_snapshot["selected_device"],
      timeout_seconds=args.timeout_seconds, inventory=args.inventory)
    _write_json(args.output, result)
    return 0 if result["status"] == "PASS" else 1
  def load_observation(path: Path, queue: str) -> Any:
    value = json.loads(path.read_text())
    if isinstance(value, Mapping) and value.get("schema") == C7_QUEUE_CAPTURE_SCHEMA:
      if value.get("status") != "PASS" or value.get("queue_mode") != queue:
        raise ValueError(f"{queue} isolated capture did not pass")
      value = value.get("queue_observation")
    return value
  observations = {
    "PM4": load_observation(args.pm4_observation, "PM4"),
    "AQL": load_observation(args.aql_observation, "AQL"),
  }
  result = build_frozen_staged_c7_from_observations(
    family=family, queue_observations=observations,
    admitted_budget_bytes=budget["admitted_bytes"],
    budget_provenance=budget["provenance"],
    device_identity=authority["device_identity"],
    software_identity=authority["software_identity"],
    allocator_identity=authority["allocator_identity"],
    allocation_granularity_bytes=authority["allocation_granularity_bytes"])
  _write_json(args.output, result)
  return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())


__all__ = [
  "FULL_ROUTE_CATEGORIES", "INFRASTRUCTURE_CATEGORIES", "LOGICAL_CATEGORIES",
  "REQUIRED_CATEGORIES", "TEMPORARY_CATEGORIES", "FrozenStagedC7HarnessAdapter",
  "FrozenStagedC7QueueCapture", "capture_frozen_staged_c7_queue_probe",
  "build_frozen_staged_c7_census", "build_frozen_staged_c7_from_observations",
  "run_frozen_staged_c7_queue_capture_isolated",
  "staged_c7_memory_authority",
]
