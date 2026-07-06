import json

from extra.qk import generated_route_registry as registry
from extra.qk import route_manifest


SEEDED = ("decode_q4k_g3_generated", "decode_q6k_coop_generated", "prefill_q4k_direct_tile4x4_default",
          "prefill_q4k_reduce_out_research", "prefill_q6k_direct_generated")
REQUIRED_L3_FIELDS = {
  "route_id",
  "descriptor_artifact",
  "lowering_level",
  "owner",
  "writer_files",
  "emitter",
  "emitted_kernel_patterns",
  "authority_gate",
  "authority_artifacts",
  "selector_binding",
  "shape_role_policy",
  "manifest_profile_id",
  "manifest_status",
  "manifest_provenance",
  "manifest_purity_status",
  "route_attribution",
  "required_gates",
  "rollback_route",
}


def test_generated_route_registry_contains_positive_controls():
  by_route = {r["route_id"]: r for r in registry.rows()}
  assert by_route["decode_q4k_g3_generated"]["route_id"] == "decode_q4k_g3_generated"
  assert by_route["decode_q6k_coop_generated"]["route_id"] == "decode_q6k_coop_generated"
  assert by_route["prefill_q4k_direct_tile4x4_default"]["route_id"] == "prefill_q4k_direct_tile4x4_default"
  assert by_route["prefill_q4k_reduce_out_research"]["route_id"] == "prefill_q4k_reduce_out_research"
  assert by_route["prefill_q6k_direct_generated"]["route_id"] == "prefill_q6k_direct_generated"


def test_positive_controls_are_l3_descriptor_owned():
  by_route = {r["route_id"]: r for r in registry.rows()}

  for route_id in SEEDED:
    route_row = by_route[route_id]
    assert route_row["lowering_level"] == "L3"
    assert route_row["owner"] == "descriptor"


def test_seeded_rows_have_expanded_l3_scope_fields():
  for route_id in SEEDED:
    route_row = registry.row(route_id)
    manifest = route_manifest.ROUTES[route_id]
    assert REQUIRED_L3_FIELDS.issubset(route_row.keys())
    assert route_row["emitter"]
    assert route_row["emitted_kernel_patterns"] == list(manifest["expected_kernels"])
    assert route_row["authority_gate"] == manifest["authority_gate"]
    assert route_row["authority_artifacts"] == list(manifest["promotion_artifacts"])
    assert route_row["selector_binding"] == manifest["selector"]
    assert route_row["shape_role_policy"]["roles"] == list(manifest["roles"])
    assert route_row["shape_role_policy"]["shape_guards"] == list(manifest["shape_guards"])
    assert route_row["manifest_status"] == manifest["status"]
    assert route_row["manifest_provenance"] == route_manifest.route_provenance(route_id)
    assert route_row["manifest_purity_status"] == manifest["purity_status"]
    assert route_row["route_attribution"] == manifest["route_attribution"]


def test_rows_and_build_are_json_serializable():
  report = registry.build()
  assert report["schema"] == "generated-route-descriptor-registry.v2"
  assert registry.row("decode_q4k_g3_generated")["rollback_route"] is None
  assert set(registry.route_ids()) == set(SEEDED)
  json.dumps(report)
