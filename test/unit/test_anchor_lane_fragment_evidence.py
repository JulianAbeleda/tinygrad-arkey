from extra.qk.prefill.anchor_lane_fragment_evidence import (
  accumulator_c_rows, build_evidence, cooperative_stage_rows, summarize_evidence, wmma_fragment_rows)


def test_anchor_cooperative_stage_is_exact_and_in_bounds():
  rows = cooperative_stage_rows()
  assert len(rows) == 1024
  assert len([row for row in rows if row["operand"] == "A"]) == 512
  assert len([row for row in rows if row["operand"] == "B_transposed"]) == 512
  assert {(row["global_element"][0], row["global_element"][1]) for row in rows if row["operand"] == "A"} == {
    (m, k) for m in range(128) for k in range(0, 32, 8)}
  assert min(row["lds_byte"] for row in rows) == 0
  assert max(row["lds_byte"] + 16 for row in rows) == 20464


def test_anchor_wmma_fragment_map_matches_lane_contract():
  rows = wmma_fragment_rows()
  assert len(rows) == 256 * 2 * (2 + 4)
  a0 = next(row for row in rows if row["role"] == "A" and row["tid"] == 0 and row["k_substep"] == 0 and row["tile_index"] == 0)
  a16 = next(row for row in rows if row["role"] == "A" and row["tid"] == 16 and row["k_substep"] == 0 and row["tile_index"] == 0)
  b31 = next(row for row in rows if row["role"] == "B" and row["tid"] == 31 and row["k_substep"] == 1 and row["tile_index"] == 3)
  assert a0["fragment_elements"] == a16["fragment_elements"]
  assert a0["lane_replication"] is False and a16["lane_replication"] is True
  assert b31["fragment_elements"][0] == [63, 16]


def test_anchor_accumulator_to_c_is_full_bijection():
  rows = accumulator_c_rows()
  points = [tuple(row["c_tile_element"]) for row in rows]
  assert len(rows) == 128 * 128
  assert len(set(points)) == len(points)
  assert set(points) == {(m, n) for m in range(128) for n in range(128)}
  lane0 = [row for row in rows if row["tid"] == 0 and row["mi"] == 0 and row["ni"] == 0]
  lane16 = [row for row in rows if row["tid"] == 16 and row["mi"] == 0 and row["ni"] == 0]
  assert [row["c_tile_element"] for row in lane0] == [[m, 0] for m in range(0, 16, 2)]
  assert [row["c_tile_element"] for row in lane16] == [[m, 0] for m in range(1, 16, 2)]


def test_anchor_evidence_separates_calculation_from_missing_bank_measurement():
  evidence = build_evidence()
  assert evidence["schema"] == "prefill-anchor-lane-fragment-evidence.v1"
  assert evidence["proof_class"] == "calculated_from_existing_lds2_emitter_equations"
  assert evidence["invariants"]["accumulator_to_c_is_bijective"] is True
  assert evidence["invariants"]["lds_slot_identity"]["ok"] is True
  assert evidence["bank_evidence"]["status"] == "missing_measured_evidence"
  assert evidence["bank_evidence"]["proven"] is False


def test_compact_summary_binds_all_exhaustive_mapping_rows():
  evidence = summarize_evidence(build_evidence())
  hashes = evidence["mapping"]["sha256"]
  assert set(hashes) == {"cooperative_stage", "wmma_fragments", "accumulator_to_c"}
  assert all(len(value) == 64 for value in hashes.values())
