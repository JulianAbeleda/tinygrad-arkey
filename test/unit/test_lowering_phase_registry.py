import json

import pytest

from extra.qk import lowering_phase_registry as registry
from extra.qk import route_manifest, runtime_surface_registry


SEEDED = ()


def test_lowering_phase_registry_seed_rows_are_known_and_valid():
  by_id = {r["id"]: r for r in registry.rows()}
  assert registry.ids() == SEEDED
  assert set(by_id.keys()) == set(SEEDED)
  known = set(route_manifest.ROUTES) | set(runtime_surface_registry.surface_ids())
  assert set(by_id).issubset(known)
  for row in by_id.values():
    assert row["target_lowering_level"] in {"L3", "L4", "L5"}
    assert row["phase_name"]
    assert row["next_action"]
    assert "current_blocker_class" not in row


def test_lowering_phase_row_lookup_and_unknown_guard():
  with pytest.raises(KeyError):
    registry.row("not_a_known_lowering_id")


def test_lowering_phase_registry_build_is_json_serializable():
  report = registry.build()
  assert report["schema"] == "lowering-phase-registry.v1"
  assert report["total_rows"] == len(SEEDED)
  assert set(report["by_level"].keys()) <= {"L3", "L4", "L5"}
  assert report["by_phase"] == {}
  json.dumps(report)
