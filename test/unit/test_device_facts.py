from datetime import datetime, timezone
import json

from tinygrad.llm.device_facts import _parse_rocminfo_gpu_capabilities, DeviceFacts, scan_device_facts
from extra.qk.memory_adaptive_device_facts import MemoryReservePolicy, calculate_admissible_budget


NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


def target(_device):
  return {"backend": "AMD", "architecture": "gfx1100", "wave_size": 32, "max_workgroup_threads": 1024,
          "max_workgroup_dimensions": (1024, 1024, 1024), "lds_bytes": 65536, "lds_allocation_granularity": 512,
          "provenance": "injected-target"}


def test_successful_scan_is_serializable_and_budget_policy_is_separate():
  facts = scan_device_facts("AMD:0", target_probe=target,
    memory_probe=lambda _: {"total_vram_bytes": 30_000, "free_vram_bytes": 20_000, "provenance": "injected-memory"}, clock=lambda: NOW)
  assert facts.state == "ok" and facts.architecture == "gfx1100" and facts.capabilities.wave_size == 32
  assert facts.target_probe.source == "injected-target" and facts.memory_probe.observed_at == NOW.isoformat()
  assert DeviceFacts.from_json(json.loads(json.dumps(facts.to_json()))) == facts
  budget = calculate_admissible_budget(facts, MemoryReservePolicy(fixed_bytes=1000, fraction_of_total=.1))
  assert (budget.reserve_bytes, budget.admissible_bytes, budget.state) == (4000, 16000, "ok")


def test_missing_rocm_smi_and_device_data_are_explicit_unknown_or_error():
  def missing(_): raise FileNotFoundError("rocm-smi")
  facts = scan_device_facts("AMD:0", target_probe=lambda _: {}, memory_probe=missing, clock=lambda: NOW)
  assert facts.state == "error" and facts.backend is None and facts.free_vram_bytes is None
  assert facts.target_probe.state == "unknown" and facts.memory_probe.state == "error"
  assert calculate_admissible_budget(facts, MemoryReservePolicy()).state == "error"


def test_inconsistent_free_above_total_is_an_error_and_not_admitted():
  facts = scan_device_facts("AMD:0", target_probe=target,
    memory_probe=lambda _: {"total_vram_bytes": 10, "free_vram_bytes": 11}, clock=lambda: NOW)
  assert facts.state == "error" and "exceeds total" in facts.errors[0]
  assert calculate_admissible_budget(facts, MemoryReservePolicy()).admissible_bytes is None


def test_changing_free_vram_changes_snapshot_but_not_canonical_hardware_identity():
  free = iter((9000, 7000))
  memory = lambda _: {"total_vram_bytes": 10000, "free_vram_bytes": next(free)}
  first = scan_device_facts("AMD:0", target_probe=target, memory_probe=memory, clock=lambda: NOW)
  second = scan_device_facts("AMD:0", target_probe=target, memory_probe=memory, clock=lambda: NOW)
  assert first.canonical_hardware_identity == second.canonical_hardware_identity
  assert first.planning_snapshot()["free_vram_bytes"] == 9000
  assert second.planning_snapshot()["free_vram_bytes"] == 7000
  assert "free_vram_bytes" not in first.canonical_hardware()


def test_planning_snapshot_is_stable_across_observation_time_only():
  later = datetime(2026, 7, 15, 12, 1, tzinfo=timezone.utc)
  memory = lambda _: {"total_vram_bytes": 10000, "free_vram_bytes": 9000, "provenance": "same-memory"}
  first = scan_device_facts("AMD:0", target_probe=target, memory_probe=memory, clock=lambda: NOW)
  second = scan_device_facts("AMD:0", target_probe=target, memory_probe=memory, clock=lambda: later)
  assert first.to_json() != second.to_json()
  assert first.planning_snapshot() == second.planning_snapshot()


def test_rocminfo_capabilities_are_selected_by_gpu_ordinal_not_architecture_table():
  output = """*******
Agent 1
*******
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1100
  Device Type:             GPU
  Wavefront Size:          32(0x20)
  Workgroup Max Size:      1024(0x400)
  Workgroup Max Size per Dimension:
    x                        1024(0x400)
    y                        512(0x200)
    z                        64(0x40)
  Pool Info:
      Segment:                 GLOBAL
      Alloc Granule:           4KB
      Segment:                 GROUP
      Size:                    64(0x40) KB
*******
Agent 3
*******
  Name:                    gfx-next
  Device Type:             GPU
  Wavefront Size:          64(0x40)
  Workgroup Max Size:      512(0x200)
  Workgroup Max Size per Dimension:
    x                        512(0x200)
    y                        256(0x100)
    z                        32(0x20)
  Pool Info:
      Segment:                 GLOBAL
      Alloc Granule:           8KB
      Segment:                 GROUP
      Size:                    32(0x20) KB
"""
  first = _parse_rocminfo_gpu_capabilities(output, 0)
  second = _parse_rocminfo_gpu_capabilities(output, 1)
  assert first == {"wave_size": 32, "max_workgroup_threads": 1024,
                   "max_workgroup_dimensions": (1024, 512, 64), "lds_bytes": 65536,
                   "lds_allocation_granularity": None, "global_allocation_granularity": 4096}
  assert second["wave_size"] == 64 and second["max_workgroup_threads"] == 512 and second["lds_bytes"] == 32768
  assert second["global_allocation_granularity"] == 8192
