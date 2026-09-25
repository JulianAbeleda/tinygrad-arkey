import json

from extra.llm.bench.gameterm_self_training_eval import completed_text


def test_completed_text_reads_normalized_harness_event():
  events = "\n".join([
    json.dumps({"event":"ready"}),
    json.dumps({"event":"text_delta", "text":"RE"}),
    json.dumps({"event":"completed", "text":"READ"}),
  ])
  assert completed_text(events) == ("READ", None)


def test_completed_text_preserves_failure():
  assert completed_text(json.dumps({"event":"failed", "code":"Timeout"})) == (None, "Timeout")
