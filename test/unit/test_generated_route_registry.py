import json
import pytest

from extra.qk.generated_candidates import builtin_registry
from extra.qk import generated_route_registry as registry
from extra.qk import route_manifest


SEEDED = ("decode_flash_block_tile_g5_konly", "decode_flash_live_split_g4_8b_kvboth",
          "decode_q4k_g3_generated", "decode_q6k_coop_generated", "prefill_q4k_direct_tile4x4_default",
          "prefill_q4k_reduce_out_research", "prefill_q6k_direct_generated",
          "prefill_14b_q4k_q8_1_hybrid_mmq_atom")
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
  "candidate_identity",
  "backend_strategy",
  "research_only",
}


def _manifest_required_gates(route_id):
  return [part.strip() for part in route_manifest.ROUTES[route_id]["authority_gate"].split(" + ") if part.strip()]


def test_generated_route_registry_contains_positive_controls():
  by_route = {r["route_id"]: r for r in registry.rows()}
  assert by_route["decode_flash_block_tile_g5_konly"]["route_id"] == "decode_flash_block_tile_g5_konly"
  assert by_route["decode_flash_live_split_g4_8b_kvboth"]["route_id"] == "decode_flash_live_split_g4_8b_kvboth"
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
    assert route_row["required_gates"] == _manifest_required_gates(route_id)


def test_generated_route_registry_route_ids_and_manifest_fields_are_consistent():
  assert set(registry.route_ids()) == {row["route_id"] for row in registry.rows()}
  for route_row in registry.rows():
    route_id = route_row["route_id"]
    assert route_id in route_manifest.ROUTES
    manifest = route_manifest.ROUTES[route_id]
    assert route_row["authority_gate"] == manifest["authority_gate"]
    assert route_row["authority_artifacts"] == list(manifest["promotion_artifacts"])
    assert route_row["selector_binding"] == manifest["selector"]
    assert route_row["shape_role_policy"]["roles"] == list(manifest["roles"])
    assert route_row["shape_role_policy"]["excluded_roles"] == list(manifest["excluded_roles"])
    assert route_row["shape_role_policy"]["quant"] == list(manifest["quant"])


def test_generated_candidates_and_route_registry_agree_on_shared_manifest_rows():
  candidates_by_route = {candidate.route_id: candidate for candidate in builtin_registry().all()}
  registry_by_route = {route_row["route_id"]: route_row for route_row in registry.rows()}
  shared = set(candidates_by_route) & set(registry_by_route)
  assert shared
  for route_id in shared:
    candidate = candidates_by_route[route_id]
    route_row = registry_by_route[route_id]
    assert candidate.route_id == route_row["route_id"]
    assert candidate.supported_quant_formats == tuple(route_row["shape_role_policy"]["quant"])
    assert candidate.provenance == route_row["manifest_provenance"]
    assert list(candidate.authority_gates) == route_row["required_gates"]


def test_rows_and_build_are_json_serializable():
  report = registry.build()
  assert report["schema"] == "generated-route-descriptor-registry.v2"
  assert registry.row("decode_q4k_g3_generated")["rollback_route"] is None
  assert set(registry.route_ids()) == set(SEEDED)
  json.dumps(report)


def test_mmq_candidate_is_research_only_and_rolls_back_to_direct_packed():
  row = registry.row("prefill_14b_q4k_q8_1_hybrid_mmq_atom")
  assert row["manifest_status"] == "research"
  assert row["manifest_purity_status"] == "research"
  assert row["research_only"] is True
  assert row["candidate_identity"] == "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
  assert row["backend_strategy"] == "q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0"
  assert row["rollback_route"] == "direct_packed"


def test_promoted_routes_cannot_bind_the_fixed_shape_mmq_atom():
  atom = "q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0"
  promoted = [route_manifest.ROUTES[route_id] for route_id in route_manifest.default_routes()]
  assert all(atom not in {str(pattern) for pattern in route.get("expected_kernels", ())} for route in promoted)
  assert route_manifest.ROUTES["prefill_14b_q4k_q8_1_hybrid_mmq_atom"]["research_only"] is True


def test_mmq_admission_boundary_keeps_direct_packed_as_default():
  """A research candidate must not silently become a runtime/default route."""
  mmq_id = "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
  mmq = route_manifest.ROUTES[mmq_id]

  assert mmq["status"] == "research"
  assert mmq["research_only"] is True
  assert mmq["selector"] == "research_descriptor_only"
  assert mmq_id not in route_manifest.default_routes()
  assert "direct_packed" in {
    mmq["rollback_route"], mmq["rollback"]["route"], mmq["baseline_route_id"]
  }
  direct_id = "prefill_q4k_direct_tile4x4_default"
  assert direct_id in route_manifest.default_routes()
  assert route_manifest.ROUTES[direct_id]["status"] == "promoted_default"
  assert route_manifest.ROUTES[direct_id]["baseline_route_id"] == "prefill_q4k_direct_packed_load_direct_out"


def test_existing_8b_routes_and_direct_packed_defaults_are_pinned():
  assert route_manifest.ROUTES["decode_flash_live_split_g4_8b_kvboth"]["status"] == "promoted_default"
  assert route_manifest.ROUTES["decode_flash_live_split_g4_8b_kvboth"]["env"] == {}
  assert route_manifest.ROUTES["prefill_q4k_direct_tile4x4_default"]["env"] == {}
  assert route_manifest.ROUTES["prefill_q4k_direct_tile4x4_default"]["rollback"] == {
    "PREFILL_Q4K_DIRECT_SCHEDULE": "legacy"}


def test_incomplete_manifest_evidence_cannot_create_registry_entry(monkeypatch):
  route_id = "prefill_q4k_direct_tile4x4_default"
  original = route_manifest.ROUTES[route_id]
  monkeypatch.setitem(route_manifest.ROUTES, route_id, {**original, "authority_gate": ""})
  with pytest.raises(ValueError, match="incomplete evidence"):
    registry.row(route_id)
