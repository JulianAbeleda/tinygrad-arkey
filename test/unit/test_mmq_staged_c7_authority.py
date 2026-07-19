from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
from tinygrad.runtime.ops_amd import AMDAllocator, KFDIface

from extra.qk import mmq_staged_c7_authority as authority_module
from extra.qk.mmq_staged_c7_authority import (
  BUDGET_POLICY, BUDGET_PROVENANCE, SOFTWARE_SEMANTICS,
  build_staged_c7_authority_snapshot, collect_staged_c7_authority_snapshot,
  validate_staged_c7_authority_snapshot, write_json_atomic,
)


REPOSITORY = {
  "vcs": "git", "commit": "a" * 40, "tree": "b" * 40, "clean": True,
}


def _facts(queue: str, *, free: int, arch: str = "gfx1100",
           granularity: int = 4096) -> DeviceFacts:
  return DeviceFacts(
    "AMD", "AMD", arch, 100_000, free,
    DeviceCapabilities(
      wave_size=32, max_workgroup_threads=1024,
      max_workgroup_dimensions=(1024, 1024, 1024),
      lds_bytes=65536, lds_allocation_granularity=512,
      global_allocation_granularity=granularity),
    ProbeRecord("tinygrad-device + rocminfo", "2026-07-19T12:00:00+00:00"),
    ProbeRecord("rocm-smi --showmeminfo vram", "2026-07-19T12:00:01+00:00"),
    queue_mode=queue)


def _implementation() -> dict:
  return {
    "allocator": authority_module._class_source(AMDAllocator),
    "interface": authority_module._class_source(KFDIface),
  }


def _observations(*, pm4_free: int = 90_000, aql_free: int = 85_000,
                  aql_arch: str = "gfx1100", aql_granularity: int = 4096,
                  aql_implementation: dict | None = None) -> dict:
  implementation = _implementation()
  return {
    "PM4": {
      "queue_mode": "PM4", "facts": _facts("PM4", free=pm4_free).to_json(),
      "allocator_implementation": implementation,
    },
    "AQL": {
      "queue_mode": "AQL",
      "facts": _facts(
        "AQL", free=aql_free, arch=aql_arch,
        granularity=aql_granularity).to_json(),
      "allocator_implementation":
        implementation if aql_implementation is None else aql_implementation,
    },
  }


def _snapshot() -> dict:
  return build_staged_c7_authority_snapshot(
    _observations(), repository=REPOSITORY)


def test_dual_queue_authority_is_queue_neutral_content_addressed_and_conservative():
  snapshot = _snapshot()
  assert snapshot["status"] == "PASS"
  assert snapshot["software"]["semantics"] == SOFTWARE_SEMANTICS
  assert snapshot["budget"]["policy"] == BUDGET_POLICY
  assert snapshot["budget"]["provenance"] == BUDGET_PROVENANCE
  assert snapshot["queue_neutral_hardware"]["architecture"] == "gfx1100"
  assert "queue_mode" not in snapshot["queue_neutral_hardware"]
  assert snapshot["queue_observations"]["PM4"]["facts"]["queue_mode"] == "PM4"
  assert snapshot["queue_observations"]["AQL"]["facts"]["queue_mode"] == "AQL"
  # PM4: 90000 - align_up(10000,4096) = 77712.
  # AQL: 85000 - align_up(15000,4096) = 68616.
  assert snapshot["budget"]["queue_budgets"]["PM4"]["admitted_bytes"] == 77_712
  assert snapshot["budget"]["queue_budgets"]["AQL"]["admitted_bytes"] == 68_616
  assert snapshot["budget"]["admitted_bytes"] == 68_616
  assert snapshot["memory_authority"]["allocation_granularity_bytes"] == 4096
  assert snapshot["memory_authority"]["device_identity"] == snapshot["device_identity"]
  assert validate_staged_c7_authority_snapshot(
    snapshot, verify_current_software=False) == snapshot


@pytest.mark.parametrize(("observations", "message"), (
  (_observations(aql_arch="gfx1101"), "queue-neutral hardware differs"),
  (_observations(aql_granularity=8192), "global allocation granularity differs"),
  (_observations(aql_implementation={
    **_implementation(),
    "allocator": {
      **_implementation()["allocator"],
      "class": "tinygrad.runtime.ops_amd.DifferentAllocator",
    },
  }), "allocator implementation differs"),
))
def test_dual_queue_authority_rejects_hardware_granularity_or_allocator_drift(
    observations, message):
  with pytest.raises(ValueError, match=message):
    build_staged_c7_authority_snapshot(observations, repository=REPOSITORY)


def test_dual_queue_authority_requires_complete_positive_live_budgets():
  broken = _observations()
  broken["AQL"]["facts"]["free_vram_bytes"] = None
  with pytest.raises(ValueError, match="healthy effective AMD queue|positive integer"):
    build_staged_c7_authority_snapshot(broken, repository=REPOSITORY)

  zero = _observations(pm4_free=4096, aql_free=4096)
  with pytest.raises(ValueError, match="admitted budget must be a positive integer"):
    build_staged_c7_authority_snapshot(zero, repository=REPOSITORY)


def test_authority_validation_recomputes_raw_budgets_and_all_identities():
  snapshot = _snapshot()
  for mutate in (
      lambda row: row["budget"].__setitem__("admitted_bytes", 1),
      lambda row: row["queue_observations"]["PM4"]["facts"].__setitem__(
        "free_vram_bytes", 89_000),
      lambda row: row.__setitem__("device_identity", "sha256:" + "0" * 64),
      lambda row: row["allocator_implementation"]["allocator"].__setitem__(
        "source_sha256", "sha256:" + "0" * 64),
  ):
    changed = copy.deepcopy(snapshot)
    mutate(changed)
    with pytest.raises(ValueError):
      validate_staged_c7_authority_snapshot(
        changed, verify_current_software=False)


def test_authority_validation_binds_current_clean_repository_and_python():
  snapshot = _snapshot()
  assert validate_staged_c7_authority_snapshot(
    snapshot, repository_probe=lambda: REPOSITORY) == snapshot
  with pytest.raises(ValueError, match="current clean repository differs"):
    validate_staged_c7_authority_snapshot(
      snapshot, repository_probe=lambda: {
        **REPOSITORY, "commit": "c" * 40})
  with pytest.raises(ValueError, match="clean git revision"):
    validate_staged_c7_authority_snapshot(
      snapshot, repository_probe=lambda: {
        **REPOSITORY, "clean": False})


def test_collection_spawns_independent_pm4_aql_scans_and_never_opens_parent_device():
  observations, calls = _observations(), []

  def isolated_runner(callback, *, args, timeout_seconds, start_method):
    calls.append((callback, args, timeout_seconds, start_method))
    return SimpleNamespace(status="passed", result=observations[args[1]], error=None)

  snapshot = collect_staged_c7_authority_snapshot(
    selected_device="AMD", timeout_seconds=17,
    isolated_runner=isolated_runner, repository_probe=lambda: REPOSITORY)
  assert snapshot["status"] == "PASS"
  assert [(row[1][1], row[2], row[3]) for row in calls] == [
    ("PM4", 17.0, "spawn"), ("AQL", 17.0, "spawn")]
  assert all(row[0] is authority_module._scan_worker for row in calls)


def test_collection_stops_at_first_failed_queue_scan():
  calls = []

  def isolated_runner(_callback, *, args, **_kwargs):
    calls.append(args[1])
    return SimpleNamespace(status="failed", result=None, error="probe failed")

  with pytest.raises(ValueError, match="PM4 C7 authority scan failed"):
    collect_staged_c7_authority_snapshot(
      isolated_runner=isolated_runner, repository_probe=lambda: REPOSITORY)
  assert calls == ["PM4"]


def test_collection_rejects_repository_change_across_the_two_scans():
  observations = _observations()
  repositories = iter((REPOSITORY, {**REPOSITORY, "commit": "c" * 40}))

  def isolated_runner(_callback, *, args, **_kwargs):
    return SimpleNamespace(
      status="passed", result=observations[args[1]], error=None)

  with pytest.raises(ValueError, match="changed during dual-queue"):
    collect_staged_c7_authority_snapshot(
      isolated_runner=isolated_runner,
      repository_probe=lambda: next(repositories))


def test_authority_json_is_published_atomically(tmp_path, monkeypatch):
  output = tmp_path / "nested" / "authority.json"
  original_replace, calls = os.replace, []

  def replace(source, destination):
    calls.append((source, destination))
    return original_replace(source, destination)

  monkeypatch.setattr(
    "extra.qk.mmq_staged_c7_authority.os.replace", replace)
  write_json_atomic(output, _snapshot())
  assert json.loads(output.read_text())["status"] == "PASS"
  assert len(calls) == 1 and calls[0][1] == output
  assert not list(output.parent.glob(f".{output.name}.*.tmp"))
