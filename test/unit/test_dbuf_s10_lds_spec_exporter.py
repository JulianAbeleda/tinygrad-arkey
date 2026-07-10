import json

from extra.qk.prefill.dbuf_epoch_lifecycle_checker import DBUFEvent, check_events
from extra.qk.prefill.dbuf_s10_lds_spec_exporter import (
  checker_compatible_events, export_s10_lds_spec, main, s10_lds_spec_dbuf_events)
from extra.qk.prefill_schedule_spec import PrefillGEMMScheduleSpec
from extra.qk.wmma_lds_spec import extract_wmma_lds_spec


def _prefill_spec() -> PrefillGEMMScheduleSpec:
  return PrefillGEMMScheduleSpec(
    m=512, n=12288, k=4096, route_family="lds", tile_m=128, tile_n=128, tile_k=32,
    waves_m=4, waves_n=2, wm=2, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=2, threads=256,
    dbuf=1, plra=0, plrab=1, pad=16, leanaddr=0, role="ffn_gate_up")


def _lds_spec():
  spec = extract_wmma_lds_spec(_prefill_spec())
  assert spec is not None
  return spec


def test_s10_lds_spec_exporter_emits_p2_byte_window_events():
  events = s10_lds_spec_dbuf_events(_lds_spec(), k_tiles=2)

  assert [event["op"] for event in events] == [
    "produce", "produce", "barrier", "consume", "consume", "produce", "produce", "barrier", "consume", "consume",
  ]
  assert events[0]["role"] == "A"
  assert events[0]["slot"] == 0
  assert events[0]["window"] == "A:slot0:0-10240"
  assert events[0]["byte_window"] == {
    "role": "A", "slot": 0, "window": "A:slot0:0-10240", "base": 0, "end": 10240,
    "bytes": 10240, "rows": 128, "row_stride_bytes": 80, "vector_bytes": 16, "total_vectors": 640,
  }
  assert events[0]["layout_key"]["role"] == "A"
  assert events[0]["layout_key"]["operand"] == "src0"
  assert events[0]["layout_key"]["lds_layout"] == "global_row_major_fp16_to_lds"
  assert events[6]["role"] == "B"
  assert events[6]["epoch"] == 1
  assert events[6]["slot"] == 1
  assert events[6]["window"] == "B:slot1:30720-40960"
  assert events[6]["layout_key"]["operand"] == "src1"
  assert events[6]["layout_key"]["lds_layout"] == "global_row_major_bt_fp16_to_lds"


def test_s10_lds_spec_exporter_events_are_checker_compatible():
  events = s10_lds_spec_dbuf_events(_lds_spec(), k_tiles=3)
  checker_rows = checker_compatible_events(events)
  report = check_events([DBUFEvent.from_json(row) for row in checker_rows])

  assert report["ok"] is True
  assert report["producer_count"] == 6
  assert report["consumer_count"] == 6
  assert checker_rows[0] == {
    "op": "produce", "step": 0, "role": "A", "epoch": 0, "slot": 0, "window": "A:slot0:0-10240",
    "lds_window": {"base": 0, "bytes": 10240, "stride": 80},
    "layout_key": {
      "role": "A", "operand": "src0", "lds_layout": "global_row_major_fp16_to_lds",
      "wmma_contract": "rdna3_wmma_f32_16x16x16_f16", "fragment_shape": [16, 16],
      "lane_map_id": "rdna3_wmma_f32_16x16x16_f16_lds2_static", "lane_count": 32,
      "lane_replication": "A_lanes_16_31_replicate", "per_lane_elements": 16,
      "vector_bytes": 16, "lds_row_stride_bytes": 80,
    },
  }


def test_export_s10_lds_spec_report_does_not_overclaim_cadence_or_value_proof():
  report = export_s10_lds_spec(_lds_spec(), k_tiles=2)

  assert report["schema"] == "dbuf-s10-lds-spec-export.v1"
  assert report["ok"] is True
  assert report["proof_schema"] == "wmma-lds-slot-identity-proof.v1"
  assert report["proof_coverage"]["P2_byte_window"] == "done"
  assert report["proof_coverage"]["P3_value_key"] == "not_proven"
  assert report["proof_coverage"]["P4_layout"] == "done_for_s10_lds_spec_static"
  assert report["proof_coverage"]["P5_wait_sync"] == "not_proven"
  assert report["proof_coverage"]["dbuf_cadence"] == "not_proven"
  assert report["event_counts"] == {"produce": 4, "consume": 4, "barrier": 2}
  assert [(w["role"], w["slot"], w["base"], w["end"]) for w in report["windows"]] == [
    ("A", 0, 0, 10240),
    ("B", 0, 10240, 20480),
    ("A", 1, 20480, 30720),
    ("B", 1, 30720, 40960),
  ]


def test_export_s10_lds_spec_reports_failed_identity_without_events():
  spec = _lds_spec()
  report = export_s10_lds_spec(spec, active_buffers=3, k_tiles=2)

  assert report["ok"] is False
  assert report["events"] == []
  assert "active_buffers must be 1 or 2" in report["errors"][0]


def test_s10_lds_spec_exporter_cli_reads_spec_json(tmp_path):
  path = tmp_path / "lds_spec.json"
  path.write_text(json.dumps(_lds_spec().to_json()))

  report = main(["--spec", str(path), "--k-tiles", "2", "--json"])

  assert report["ok"] is True
  assert report["event_counts"] == {"produce": 4, "consume": 4, "barrier": 2}
