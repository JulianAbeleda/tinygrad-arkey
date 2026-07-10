from extra.qk.mmq_llama_store_probe import (
  LLAMA_MMQ_R4_STORE_ONLY_PROBE_ID, compare_llama_r4_store_probe_to_oracle, llama_r4_store_only_owner_map,
)
from extra.qk.mmq_q4k_q8_reference import Q8_1_MMQ_DS4_LAYOUT, describe_q4k_q8_1_mmq_tile


def _spec(m: int, n: int):
  return describe_q4k_q8_1_mmq_tile(
    role="ffn_gate_up", m=m, n=n, k=256, m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT,
  )


def test_llama_r4_store_only_probe_matches_oracle_for_full_8_wave_tile():
  report = compare_llama_r4_store_probe_to_oracle(_spec(128, 128))

  assert report["schema"] == "llama-mmq-r4-store-only-owner-map-probe.v1"
  assert report["probe_id"] == LLAMA_MMQ_R4_STORE_ONLY_PROBE_ID
  assert report["candidate_id"] == LLAMA_MMQ_R4_STORE_ONLY_PROBE_ID
  assert report["probe_kind"] == "store_only_owner_trace"
  assert report["status"] == "PASS"
  assert report["research_only"] is True
  assert report["production_dispatch_changed"] is False
  assert report["default_route"] == "direct_packed"
  assert report["store_only"] is True
  assert report["owner_fragment_count"] == 64
  assert report["store_count"] == 128 * 128
  assert report["covered_output_count"] == 128 * 128
  assert report["duplicate_store_count"] == 0
  assert report["missing_store_count"] == 0
  assert report["owner_mismatch_count"] == 0
  assert report["actual_owner_hash"] == report["expected_owner_hash"]
  assert len(report["source_hash"]) == 64
  assert report["matches_oracle"] is True


def test_llama_r4_store_only_probe_lane_schedule_is_bounded_to_wave_16x16_fragments():
  stores = llama_r4_store_only_owner_map(_spec(16, 16))

  assert len(stores) == 256
  assert {store.wave_id for store in stores} == {0}
  assert {store.lidx for store in stores} == set(range(32))
  assert {store.gidx for store in stores} == set(range(32))
  assert {store.store_iter for store in stores} == set(range(8))
  assert stores[0].to_json() == {
    "m": 0, "n": 0, "wave_id": 0, "lane_id": 0, "lidx": 0, "gidx": 0, "fragment_id": 0,
    "fragment_m_range": [0, 16], "fragment_n_range": [0, 16], "store_iter": 0,
    "asm_store": "global_store_b32 v_acc[0:0], dst[m=0,n=0]",
  }
  assert stores[-1].m == 15
  assert stores[-1].n == 15
  assert stores[-1].lane_id == 31
  assert stores[-1].store_iter == 7


def test_llama_r4_store_only_probe_matches_oracle_for_partial_tile():
  report = compare_llama_r4_store_probe_to_oracle(_spec(32, 48))

  assert report["store_count"] == 32 * 48
  assert report["covered_output_count"] == 32 * 48
  assert report["expected_output_count"] == 32 * 48
  assert report["matches_oracle"] is True
