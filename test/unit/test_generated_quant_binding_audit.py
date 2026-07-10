from extra.qk.generated_candidates import builtin_registry
from extra.qk.runtime_specs import GENERATED_PROVENANCE
from extra.qk import route_manifest


def test_builtin_generated_candidates_point_at_known_generated_routes():
  rows = builtin_registry().all()
  assert len(rows) == 6
  by_id = {r.candidate_id: r for r in rows}

  assert by_id["quant_linear_prefill.prefill_v2_scheduler_matmul_default"].route_id == "prefill_v2_scheduler_matmul_default"
  assert by_id["quant_linear_prefill.q4k_int8_wmma_tensor_substrate"].route_id == "prefill_q4k_int8_wmma_generated_research"
  assert by_id["quant_linear_prefill.q4k_int8_wmma_tiled_substrate"].route_id == "prefill_q4k_int8_wmma_tiled_research"
  assert by_id["quant_linear_decode.q4k_g3_lanemap"].route_id == "decode_q4k_g3_generated"
  assert by_id["quant_linear_decode.q6k_generated_coop"].route_id == "decode_q6k_coop_generated"
  assert by_id["attention_decode.live_split_flash"].route_id == "decode_flash_live_split_g4_8b_kvboth"

  assert all(r.provenance in GENERATED_PROVENANCE for r in rows)
  assert all(r.route_id in route_manifest.ROUTES for r in rows)


def test_prefill_direct_packed_default_is_no_longer_transitional_debt():
  report = route_manifest.default_purity_report()
  rows = {r["route_id"]: r for r in report["rows"]}

  assert rows["prefill_q4k_direct_tile4x4_default"]["provenance"] == "machine_authored_generated"
  assert rows["prefill_q4k_direct_tile4x4_default"]["final_default_allowed"] is True
  assert rows["prefill_v2_scheduler_matmul_default"]["provenance"] == "tinygrad_scheduler_generated"
  assert rows["prefill_v2_scheduler_matmul_default"]["final_default_allowed"] is True
  assert report["verdict"] == "TINYGRAD_DEFAULT_PURITY_PASS"
