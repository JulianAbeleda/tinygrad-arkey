import json

from extra.qk.prefill.dbuf_epoch_lifecycle_checker import DBUFEvent, canonical_dbuf_events, check_events, main


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


def test_cli_loads_event_json(tmp_path):
  path = tmp_path / "events.json"
  path.write_text(json.dumps({"events": [event.to_json() for event in canonical_dbuf_events(k_tiles=2)]}))

  report = main(["--input", str(path), "--json"])

  assert report["ok"] is True
  assert report["producer_count"] == 4
