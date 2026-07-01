from extra.pure_machine_search_default_path_census import build_census
from extra.qk_route_manifest import default_purity_report, route_provenance, validate_manifest


def test_qk_route_manifest_purity_debt_is_explicit():
  assert validate_manifest() == []
  report = default_purity_report()
  assert report["verdict"] == "TINYGRAD_DEFAULT_PURITY_FAIL"
  assert route_provenance("decode_q4k_g3_generated") == "machine_authored_generated"
  assert set(report["transitional_default_routes"]) == {"decode_q6k_coop_shipped"}
  assert set(report["forbidden_default_routes"]) == {
    "decode_attention_owned_two_kernel",
    "prefill_pipe_role_selective_default",
  }


def test_default_path_census_uses_manifest_provenance():
  census = build_census()
  assert census["verdict"] == "PMS_R0_PASS_CENSUS_PINNED"
  assert census["strict_default_purity_verdict"] == "TINYGRAD_DEFAULT_PURITY_FAIL"
  by_route = {row["route_id"]: row for row in census["default_route_table"]}
  assert by_route["decode_q4k_g3_generated"]["final_default_allowed"] is True
  assert by_route["decode_q6k_coop_shipped"]["provenance"] == "hand_authored_uop_template"
  assert by_route["decode_attention_owned_two_kernel"]["provenance"] == "external_handwritten_kernel"
  assert by_route["prefill_pipe_role_selective_default"]["provenance"] == "external_handwritten_kernel"
