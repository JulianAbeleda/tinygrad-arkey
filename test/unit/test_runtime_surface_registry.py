import json

from extra.qk import runtime_surface_registry as registry


SEEDED = (
  "prefill_q6k_direct_packed_default_capable",
)


def test_runtime_surface_registry_contains_unmanifested_surfaces():
  assert registry.surface_ids() == SEEDED
  by_surface = {r["surface_id"]: r for r in registry.rows()}
  for surface_id in SEEDED:
    row = by_surface[surface_id]
    assert row["surface_class"] == "route_local_custom_kernel"
    assert row["writer_files"]
    assert row["reason"]
    assert row["replacement_scope"]


def test_runtime_surface_registry_build_is_json_serializable():
  report = registry.build()
  assert report["schema"] == "runtime-surface-registry.v1"
  assert report["total_surfaces"] == len(SEEDED)
  assert set(registry.surface_ids()) == set(SEEDED)
  json.dumps(report)
