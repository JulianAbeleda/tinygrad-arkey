from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

import pytest

from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context

from extra.qk.mmq_frozen_staged_c7_census import (
  FULL_ROUTE_CATEGORIES, FrozenStagedC7QueueCapture,
  _write_json,
  build_frozen_staged_c7_census, build_frozen_staged_c7_from_observations,
  capture_frozen_staged_c7_queue_probe,
  main as c7_main,
  run_frozen_staged_c7_queue_capture_isolated,
  staged_c7_memory_authority,
)
from extra.qk.mmq_frozen_staged_family import FrozenStagedFamily, load_frozen_staged_family_manifest
from extra.qk.mmq_staged_c7_c8_contract import (
  staged_logical_memory_requirements, validate_staged_c7_memory_ledger,
)
from test.unit.test_mmq_frozen_staged_family import _loader, _produce


@pytest.fixture
def family(tmp_path) -> FrozenStagedFamily:
  role_spec, binding, output, _ = _produce(tmp_path)
  return load_frozen_staged_family_manifest(
    output, role_spec=role_spec, frozen_bundle="/frozen/bundle",
    binding_loader=_loader(binding))


def _authority(*, budget: int = 1_000_000_000, granularity: int = 4096) -> dict:
  return staged_c7_memory_authority(
    device_identity="mock-gfx1100-device-0",
    software_identity="tinygrad-test-revision",
    allocator_identity="mock-amd-allocator-0",
    allocation_granularity_bytes=granularity,
    admitted_budget_bytes=budget,
    budget_provenance="mock live device admission scan")


def _captured_queue(family: FrozenStagedFamily, queue: str, *,
                    omit: str | None = None, transfer_bytes: int = 0,
                    add_unowned: bool = False, granularity: int = 4096,
                    bind_output: bool = False) -> FrozenStagedC7QueueCapture:
  capture = FrozenStagedC7QueueCapture(family, queue, "CPU", granularity)
  requirements = staged_logical_memory_requirements(family)["components"]
  allocations: list[Buffer] = []
  with Context(LRU=0), capture.capture():
    for category, nbytes in requirements.items():
      if category == omit: continue
      buffer = Buffer("CPU", nbytes, dtypes.uint8)
      if bind_output and category == "output":
        capture.bind_buffer(buffer, category, "exact")
        buffer.allocate()
      else:
        with capture.allocation(category, "exact"): buffer.allocate()
      allocations.append(buffer)
    for category, nbytes in (
        ("code_object", 64 * 1024),
        ("runtime", 16 * 1024),
        ("queue_state", 32 * 1024)):
      if category == omit: continue
      buffer = Buffer("CPU", nbytes, dtypes.uint8)
      with capture.allocation(category, "exact"): buffer.allocate()
      allocations.append(buffer)

    capture.begin_route()
    if omit != "kernarg":
      kernarg = Buffer("CPU", 40, dtypes.uint8)
      with capture.allocation("kernarg", "epoch-0"): kernarg.allocate()
      kernarg.deallocate()
    if transfer_bytes:
      transfer = Buffer("CPU", transfer_bytes, dtypes.uint8)
      with capture.allocation("temporary_transfer", "epoch-0"): transfer.allocate()
      transfer.deallocate()
    if add_unowned:
      unowned = Buffer("CPU", 17, dtypes.uint8).allocate()
      unowned.deallocate()
    capture.end_route()
    for buffer in reversed(allocations): buffer.deallocate()
  return capture


def _build(family: FrozenStagedFamily, pm4: FrozenStagedC7QueueCapture,
           aql: FrozenStagedC7QueueCapture, *, budget: int = 1_000_000_000,
           granularity: int = 4096) -> dict:
  return build_frozen_staged_c7_census(
    family=family, captures={"PM4": pm4, "AQL": aql},
    admitted_budget_bytes=budget,
    budget_provenance="mock live device admission scan",
    device_identity="mock-gfx1100-device-0",
    software_identity="tinygrad-test-revision",
    allocator_identity="mock-amd-allocator-0",
    allocation_granularity_bytes=granularity)


def test_live_c7_capture_owns_all_categories_and_projects_exact_route_lifetimes(family):
  capture = _captured_queue(family, "PM4", transfer_bytes=8192, bind_output=True)
  observation = capture.observation(memory_authority=_authority())
  rows = observation["lifetimes"]
  assert observation["allocation_census_complete"] is True
  assert observation["dense_fp16_weight_materialization"] is False
  assert {row["category"] for row in rows} == {
    *staged_logical_memory_requirements(family)["components"],
    "code_object", "runtime", "kernarg", "queue_state",
    "temporary_gather", "temporary_transfer",
  }
  start, end = observation["route_start"], observation["route_end"]
  for category in FULL_ROUTE_CATEGORIES:
    row = next(row for row in rows if row["category"] == category)
    assert (row["live_from"], row["live_until"]) == (start, end)
    assert row["source"] == "physical_memory_ledger"
  gather = next(row for row in rows if row["category"] == "temporary_gather")
  transfer = next(row for row in rows if row["category"] == "temporary_transfer")
  assert gather["physical_bytes"] == 0 and gather["source"] == "explicit_zero_measurement"
  assert transfer["physical_bytes"] == 8192 and transfer["source"] == "physical_memory_ledger"
  assert all(row["physical_bytes"] % 4096 == 0 for row in rows)


def test_live_c7_pm4_aql_join_binds_budget_authority_and_deep_validates(family):
  pm4, aql = _captured_queue(family, "PM4"), _captured_queue(family, "AQL")
  report = _build(family, pm4, aql)
  assert report["status"] == "PASS"
  assert report["dense_fp16_weight_materialization"] is False
  assert report["production_dispatch_changed"] is False
  assert set(report["queues"]) == {"PM4", "AQL"}
  assert all(row["status"] == "PASS" and row["admitted"] for row in report["queues"].values())
  assert report["queues"]["PM4"]["authority"] == report["queues"]["AQL"]["authority"]
  assert validate_staged_c7_memory_ledger(report, family=family) == report

  observations = {
    queue: capture.observation(memory_authority=_authority())
    for queue, capture in (("PM4", pm4), ("AQL", aql))
  }
  rebuilt = build_frozen_staged_c7_from_observations(
    family=family, queue_observations=observations,
    admitted_budget_bytes=1_000_000_000,
    budget_provenance="mock live device admission scan",
    device_identity="mock-gfx1100-device-0",
    software_identity="tinygrad-test-revision",
    allocator_identity="mock-amd-allocator-0",
    allocation_granularity_bytes=4096)
  assert rebuilt == report


def test_live_c7_is_fail_closed_for_missing_unowned_dense_and_wrong_queue(family):
  missing = _captured_queue(family, "PM4", omit="compact_q4_stage")
  complete = _captured_queue(family, "AQL")
  report = _build(family, missing, complete)
  assert report["status"] == "BLOCKED"
  assert "missing exact C7 categories: compact_q4_stage" in report["queues"]["PM4"]["blockers"]

  unowned = _captured_queue(family, "PM4", add_unowned=True)
  with pytest.raises(ValueError, match="complete and blocker-free"):
    unowned.observation(memory_authority=_authority())

  capture = FrozenStagedC7QueueCapture(family, "PM4", "CPU", 4096)
  with pytest.raises(ValueError, match="dense FP16"):
    capture.owner("dense_fp16_weight", "forbidden")
  with pytest.raises(ValueError, match="captures must contain exactly"):
    build_frozen_staged_c7_census(
      family=family, captures={"PM4": missing},
      admitted_budget_bytes=1_000_000_000,
      budget_provenance="mock live device admission scan",
      device_identity="mock-gfx1100-device-0",
      software_identity="tinygrad-test-revision",
      allocator_identity="mock-amd-allocator-0",
      allocation_granularity_bytes=4096)


def test_live_c7_rejects_authority_drift_and_reports_budget_failure(family):
  pm4, aql = _captured_queue(family, "PM4"), _captured_queue(family, "AQL")
  with pytest.raises(ValueError, match="granularity differs"):
    pm4.observation(memory_authority=_authority(granularity=65536))

  passing = _build(family, pm4, aql)
  peak = max(row["peak_physical_bytes"] for row in passing["queues"].values())
  failed = _build(family, pm4, aql, budget=peak - 1)
  assert failed["status"] == "FAIL"
  assert all(row["status"] == "FAIL" and not row["admitted"] for row in failed["queues"].values())
  assert all("exceed admitted budget" in row["failures"][0] for row in failed["queues"].values())

  observations = {
    queue: capture.observation(memory_authority=_authority())
    for queue, capture in (("PM4", pm4), ("AQL", aql))
  }
  drifted = copy.deepcopy(observations)
  drifted["AQL"]["authority"]["allocator_identity"] = "wrong"
  with pytest.raises(ValueError, match="authority differs"):
    build_frozen_staged_c7_from_observations(
      family=family, queue_observations=drifted,
      admitted_budget_bytes=1_000_000_000,
      budget_provenance="mock live device admission scan",
      device_identity="mock-gfx1100-device-0",
      software_identity="tinygrad-test-revision",
      allocator_identity="mock-amd-allocator-0",
      allocation_granularity_bytes=4096)


def test_current_staged_harness_observer_seam_binds_real_logical_and_external_allocations(family):
  class Handle:
    def __init__(self, va: int, size: int):
      self.va_addr, self.size = va, size
      self.base = self

  runtime = SimpleNamespace(
    lib=b"frozen-elf", lib_gpu=Handle(0x100000, 4096))
  device = SimpleNamespace(
    kernargs_buf=Handle(0x200000, 4096),
    timeline_signal=SimpleNamespace(base_buf=Handle(0x300000, 4096)),
    scratch=Handle(0x350000, 8192),
    pm4_ibs=Handle(0x380000, 16384),
    compute_queues={0: SimpleNamespace(ring=SimpleNamespace(addr=0x400000, nbytes=4096))},
    sdma_queues={0: SimpleNamespace(ring=SimpleNamespace(addr=0x500000, nbytes=4096))},
  )

  def mock_current_harness(*, staged_lifecycle_observer):
    allocations = []
    requirements = staged_logical_memory_requirements(family)["components"]
    for category, nbytes in requirements.items():
      buffer = Buffer("CPU", nbytes, dtypes.uint8)
      with staged_lifecycle_observer.allocation(category, f"persistent_{category}"):
        buffer.allocate()
      allocations.append(buffer)
    staged_lifecycle_observer.begin_route()
    staged_lifecycle_observer.runtime(runtime, device)
    staged_lifecycle_observer.launch(
      runtime, {"kernarg": {"va": device.kernargs_buf.va_addr + 64, "size": 40}})
    staged_lifecycle_observer.end_route()
    for buffer in reversed(allocations): buffer.deallocate()
    return {"status": "PASS", "production_dispatch_changed": False}

  authority = _authority()
  with Context(LRU=0):
    result = capture_frozen_staged_c7_queue_probe(
      family=family, queue_mode="PM4", ledger_device="CPU",
      allocation_granularity_bytes=4096, memory_authority=authority,
      probe_kwargs={}, probe_runner=mock_current_harness)
  observation = result["queue_observation"]
  assert result["status"] == "PASS"
  assert observation["dense_fp16_weight_materialization"] is False
  assert {row["category"] for row in observation["lifetimes"]} == {
    *staged_logical_memory_requirements(family)["components"],
    "code_object", "runtime", "kernarg", "queue_state",
    "temporary_gather", "temporary_transfer",
  }
  assert sum(row["physical_bytes"] for row in observation["lifetimes"]
             if row["category"] == "queue_state") == 24576
  assert sum(row["physical_bytes"] for row in observation["lifetimes"]
             if row["category"] == "runtime") == 12288
  assert all(row["source"] == "runtime_allocation_census"
             for row in observation["lifetimes"]
             if row["category"] in ("code_object", "runtime", "kernarg", "queue_state"))


def test_external_runtime_census_rejects_physical_size_outside_scanned_granularity(family):
  capture = FrozenStagedC7QueueCapture(family, "PM4", "CPU", 4096)
  handle = SimpleNamespace(va_addr=0x100000, size=4097)
  handle.base = handle
  with pytest.raises(ValueError, match="scanned allocator granularity"):
    capture.record_external_full_route("code_object", "unaligned", handle)


def test_isolated_capture_requires_exact_c4_and_parent_health_fault_envelope(family, monkeypatch):
  monkeypatch.setenv("AMD_AQL", "0")
  authority = _authority()
  observation = _captured_queue(family, "PM4").observation(memory_authority=authority)
  child = {
    "schema": "tinygrad.mmq_q4k_q8_1.staged_c7_queue_probe.v1",
    "status": "PASS", "queue_mode": "PM4",
    "family_identity": family.family_identity,
    "probe": {"status": "PASS"}, "queue_observation": observation,
  }
  isolated = SimpleNamespace(
    status="passed", timed_out=False, error=None, result=child,
    elapsed_seconds=0.25)
  calls = {"runner": 0, "health": 0}

  def runner(*_args, **kwargs):
    calls["runner"] += 1
    assert kwargs["start_method"] == "spawn" and kwargs["timeout_seconds"] == 12.0
    return isolated

  def health(env):
    calls["health"] += 1
    assert env == {"AMD_AQL": "0"}
    return True

  canary = {
    "status": "PASS", "family_identity": family.family_identity,
    "program_key": family.binding.program_key, "queue_mode": "PM4",
  }
  result = run_frozen_staged_c7_queue_capture_isolated(
    family=family, queue_mode="PM4", frozen_bundle="/frozen/bundle",
    staged_family_manifest="/frozen/family.json",
    runtime_canary_isolation=canary, memory_authority=authority,
    ledger_device="CPU", timeout_seconds=12.0,
    isolated_runner=runner, health_probe=health,
    fault_collector=lambda _started: ([], {"window": "clean"}),
    canary_validator=lambda value, _family, *, queue_mode: value,
    probe_validator=lambda raw, _family, **_kwargs: {"status": raw["status"], "validated": True})
  assert result["status"] == "PASS"
  assert result["launched"] is True and result["target_dispatch_attempted"] is True
  assert result["target_dispatch_attempted_authority"] == "passing_child_queue_probe"
  assert result["containment_authority"] == "outer_parent_fresh_process_guards"
  assert result["health_before"] is result["health_after"] is True
  assert result["kernel_faults"] == []
  assert result["queue_observation"] == observation
  assert result["probe_validation"]["validated"] is True
  assert result["c4_runtime_canary_isolation"] == canary
  assert calls == {"runner": 1, "health": 2}

  blocked = run_frozen_staged_c7_queue_capture_isolated(
    family=family, queue_mode="PM4", frozen_bundle="/frozen/bundle",
    staged_family_manifest="/frozen/family.json",
    runtime_canary_isolation={**canary, "program_key": "wrong"},
    memory_authority=authority, ledger_device="CPU",
    isolated_runner=lambda *_args, **_kwargs: pytest.fail("must not launch"),
    health_probe=lambda _env: pytest.fail("must not probe health"),
    canary_validator=lambda value, _family, *, queue_mode: value)
  assert blocked["status"] == "BLOCKED"
  assert blocked["launched"] is False and blocked["target_dispatch_attempted"] is False

  runner_error = run_frozen_staged_c7_queue_capture_isolated(
    family=family, queue_mode="PM4", frozen_bundle="/frozen/bundle",
    staged_family_manifest="/frozen/family.json",
    runtime_canary_isolation=canary, memory_authority=authority,
    ledger_device="CPU",
    isolated_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("runner failed")),
    health_probe=lambda _env: True,
    fault_collector=lambda _started: ([], {}),
    canary_validator=lambda value, _family, *, queue_mode: value)
  assert runner_error["status"] == "BLOCKED"
  assert runner_error["launched"] is None
  assert runner_error["target_dispatch_attempted"] is None
  assert runner_error["target_dispatch_attempted_authority"] == \
    "unknown_without_structured_child_queue_probe"


def test_c7_json_evidence_is_published_atomically(tmp_path, monkeypatch):
  output = tmp_path / "nested" / "c7.json"
  original_replace = os.replace
  calls = []

  def replace(source, destination):
    calls.append((source, destination))
    return original_replace(source, destination)

  monkeypatch.setattr("extra.qk.mmq_frozen_staged_c7_census.os.replace", replace)
  _write_json(output, {"status": "PASS"})
  assert output.read_text() == '{\n  "status": "PASS"\n}\n'
  assert len(calls) == 1 and calls[0][1] == output
  assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_c7_capture_cli_consumes_only_validated_authority_snapshot(
    family, tmp_path, monkeypatch):
  authority = _authority()
  snapshot = {
    "selected_device": "AMD",
    "memory_authority": authority,
    "budget": {
      "admitted_bytes": 1_000_000_000,
      "provenance": "mock live device admission scan",
    },
  }
  canary = tmp_path / "c4.json"
  canary.write_text('{"status":"PASS"}')
  output = tmp_path / "c7-pm4.json"
  seen = {}

  monkeypatch.setattr(
    "extra.qk.mmq_frozen_staged_c7_census._load_family",
    lambda _args: family)
  monkeypatch.setattr(
    "extra.qk.mmq_frozen_staged_c7_census.load_staged_c7_authority_snapshot",
    lambda path: seen.setdefault("authority_path", path) and snapshot)

  def capture(**kwargs):
    seen.update(kwargs)
    return {"status": "PASS", "queue_mode": "PM4"}

  monkeypatch.setattr(
    "extra.qk.mmq_frozen_staged_c7_census."
    "run_frozen_staged_c7_queue_capture_isolated", capture)
  monkeypatch.setenv("AMD_AQL", "0")
  assert c7_main([
    "capture", "--staged-family-manifest", "/frozen/family.json",
    "--frozen-bundle", "/frozen/bundle", "--role", "attn_qo",
    "--queue-mode", "PM4", "--runtime-canary-isolation", str(canary),
    "--authority-snapshot", str(tmp_path / "authority.json"),
    "--output", str(output),
  ]) == 0
  assert seen["authority_path"] == tmp_path / "authority.json"
  assert seen["memory_authority"] == authority
  assert json.loads(output.read_text())["status"] == "PASS"
