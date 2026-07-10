import json

from extra.qk.prefill.dbuf_epoch_lifecycle_checker import (
  DBUFEvent, canonical_dbuf_events, check_events, events_from_epoch_primitive, events_from_s10_role_trace, main,
  s10_readiness_roadmap)


def test_canonical_dbuf_lifecycle_passes():
  report = check_events(canonical_dbuf_events(k_tiles=4))

  assert report["ok"] is True
  assert report["producer_count"] == 8
  assert report["consumer_count"] == 8
  assert report["barrier_count"] == 4
  assert report["errors"] == []


def test_consume_without_matching_epoch_fails():
  events = [
    DBUFEvent("produce", role="B", epoch=0, slot=0, step=0),
    DBUFEvent("barrier", step=1),
    DBUFEvent("consume", role="B", epoch=1, slot=0, step=2),
  ]

  report = check_events(events)

  assert report["ok"] is False
  assert report["errors"][0]["error"] == "consumer has no prior matching producer"


def test_consume_without_barrier_fails():
  events = [
    DBUFEvent("produce", role="A", epoch=0, slot=0, step=0),
    DBUFEvent("consume", role="A", epoch=0, slot=0, step=1),
  ]

  report = check_events(events)

  assert report["ok"] is False
  assert report["errors"][0]["error"] == "no barrier separates producer and consumer"


def test_slot_overwrite_before_consume_fails():
  events = [
    DBUFEvent("produce", role="B", epoch=0, slot=0, step=0),
    DBUFEvent("barrier", step=1),
    DBUFEvent("produce", role="B", epoch=2, slot=0, step=2),
    DBUFEvent("barrier", step=3),
    DBUFEvent("consume", role="B", epoch=0, slot=0, step=4),
    DBUFEvent("consume", role="B", epoch=2, slot=0, step=5),
  ]

  report = check_events(events)

  assert report["ok"] is False
  assert any("slot overwrite before consume" in err["error"] for err in report["errors"])


def test_duplicate_consume_fails():
  events = [
    DBUFEvent("produce", role="A", epoch=0, slot=0, step=0),
    DBUFEvent("barrier", step=1),
    DBUFEvent("consume", role="A", epoch=0, slot=0, step=2),
    DBUFEvent("consume", role="A", epoch=0, slot=0, step=3),
  ]

  report = check_events(events)

  assert report["ok"] is False
  assert any(err["error"] == "same producer consumed more than once" for err in report["errors"])


def test_matching_lds_windows_pass():
  window = {"base": 10240, "bytes": 10240, "stride": 80}
  events = [
    DBUFEvent("produce", role="B", epoch=0, slot=0, lds_window=window, step=0),
    DBUFEvent("barrier", step=1),
    DBUFEvent("consume", role="B", epoch=0, slot=0, lds_window=window, step=2),
  ]

  report = check_events(events)

  assert report["ok"] is True
  assert report["errors"] == []


def test_mismatched_lds_windows_fail():
  events = [
    DBUFEvent("produce", role="B", epoch=0, slot=0, lds_window={"base": 10240, "bytes": 10240, "stride": 80}, step=0),
    DBUFEvent("barrier", step=1),
    DBUFEvent("consume", role="B", epoch=0, slot=0, lds_window={"base": 20480, "bytes": 10240, "stride": 80}, step=2),
  ]

  report = check_events(events)

  assert report["ok"] is False
  assert any("consumer LDS window does not match producer" in err["error"] for err in report["errors"])


def test_incomplete_lds_window_fails():
  events = [
    DBUFEvent("produce", role="A", epoch=0, slot=0, lds_window={"base": 0}, step=0),
    DBUFEvent("barrier", step=1),
    DBUFEvent("consume", role="A", epoch=0, slot=0, lds_window={"base": 0}, step=2),
  ]

  report = check_events(events)

  assert report["ok"] is False
  assert any(err["error"] == "lds_window requires base and bytes" for err in report["errors"])


def test_cli_loads_event_json(tmp_path):
  path = tmp_path / "events.json"
  path.write_text(json.dumps({"events": [event.to_json() for event in canonical_dbuf_events(k_tiles=2)]}))

  report = main(["--input", str(path), "--json"])

  assert report["ok"] is True
  assert report["producer_count"] == 4


def test_epoch_primitive_exporter_builds_checkable_events():
  primitive = {
    "name": "s9_dbuf_epoch_coordinator",
    "nbuf": 2,
    "slot_expr": "epoch % 2",
  }

  events = events_from_epoch_primitive(primitive, roles=("A", "B"), k_tiles=3)
  report = check_events(events)

  assert report["ok"] is True
  assert report["producer_count"] == 6
  assert events[0] == DBUFEvent("produce", role="A", epoch=0, slot=0, step=0)
  assert events[-1] == DBUFEvent("consume", role="B", epoch=2, slot=0, step=14)


def test_s10_role_trace_exporter_uses_ffn_gate_up_epoch_primitive():
  trace = {
    "rows": [
      {"role": "attn_qo"},
      {
        "role": "ffn_gate_up",
        "hand_coded_epoch_primitive": {
          "name": "s9_dbuf_epoch_coordinator",
          "nbuf": 2,
          "slot_expr": "epoch % 2",
        },
      },
    ]
  }

  events = events_from_s10_role_trace(trace, k_tiles=2)
  report = check_events(events)

  assert report["ok"] is True
  assert report["producer_count"] == 4
  assert report["consumer_count"] == 4


def test_cli_exports_s10_role_trace(tmp_path):
  trace_path = tmp_path / "trace.json"
  trace_path.write_text(json.dumps({
    "rows": [{
      "role": "ffn_gate_up",
      "hand_coded_epoch_primitive": {"nbuf": 2, "slot_expr": "epoch % 2"},
    }]
  }))

  report = main(["--s10-role-trace", str(trace_path), "--k-tiles", "2", "--json"])

  assert report["ok"] is True
  assert report["source"]["kind"] == "s10_role_trace"
  assert len(report["events"]) == 10


def test_s10_roadmap_does_not_overclaim_readiness():
  roadmap = s10_readiness_roadmap()

  assert roadmap["schema"] == "dbuf-epoch-lifecycle-s10-roadmap.v1"
  assert roadmap["complete_for_s10"] is False
  layers = {layer["id"]: layer for layer in roadmap["proof_layers"]}
  exporters = {exporter["id"]: exporter for exporter in roadmap["exporters"]}
  assert layers["P1"]["status"] == "done"
  assert layers["P2"]["status"] == "done_for_s10_lds_spec"
  assert layers["P7"]["status"] == "pending"
  assert exporters["E1"]["status"] == "done"
  assert exporters["E5"]["status"] == "pending"


def test_cli_prints_s10_roadmap():
  report = main(["--roadmap", "--json"])

  assert report["complete_for_s10"] is False
  assert "optional LDS byte-window equality" in report["current_proof_coverage"]
