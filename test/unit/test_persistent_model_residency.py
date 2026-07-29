import copy, importlib.util, json, pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("persistent_model_residency", ROOT / "extra/usbgpu/tests/persistent_model_residency.py")
residency = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(residency)
STATUS = json.loads((ROOT / "extra/usbgpu/protocol/fixtures/keepalive-status-v1.json").read_text())["valid"]
POWER = json.loads((ROOT / "extra/usbgpu/protocol/fixtures/power-residency-status-v4.json").read_text())["valid"]


def loaded(ticks=0, **patch):
  return copy.deepcopy(STATUS) | {"attempts":3600+ticks, "successes":3600+ticks,
                                  "last_attempt_monotonic_ns":3600000000000+ticks*1000000000,
                                  "last_success_monotonic_ns":3600000000000+ticks*1000000000,
                                  "active_workload_leases":1, "active_bar_mappings":6, "active_dma_allocations":42} | patch


def power(ticks=0, **patch):
  return copy.deepcopy(POWER) | {"last_canary_success_monotonic_ns":3600000000000+ticks*1000000000} | patch


def test_loaded_sample_requires_resident_resources_and_continuity():
  first, first_power = loaded(), power()
  assert residency.validate_loaded_sample(first, first_power) == 1
  assert residency.validate_loaded_sample(loaded(30), power(30), generation=1,
                                           previous_status=first, previous_power=first_power) == 1


@pytest.mark.parametrize("patch, message", [
  ({"active_workload_leases":0}, "one workload lease"),
  ({"active_bar_mappings":0}, "BAR mappings"),
  ({"active_dma_allocations":0}, "DMA/VRAM"),
  ({"provider_generation":2}, "generation changed"),
])
def test_loaded_sample_stops_on_lost_residency(patch, message):
  with pytest.raises(residency.QualificationError, match=message):
    residency.validate_loaded_sample(loaded(**patch), power(**({"provider_generation":2} if patch.get("provider_generation") == 2 else {})), generation=1)
