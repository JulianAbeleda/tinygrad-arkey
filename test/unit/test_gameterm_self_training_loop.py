import pathlib

import pytest

from extra.llm.bench.gameterm_self_training_loop import load_ledger, write_json_atomic


def test_controller_ledger_resumes_completed_rounds(tmp_path:pathlib.Path):
  path, config = tmp_path / "ledger.json", {"model":"qwen"}
  ledger = load_ledger(path, config)
  ledger["rounds"].append({"round":0, "adapter":"round-000/adapter"})
  write_json_atomic(path, ledger)
  resumed = load_ledger(path, config)
  assert len(resumed["rounds"]) == 1
  assert resumed["status"] == "running"


def test_controller_rejects_changed_resume_config(tmp_path:pathlib.Path):
  path = tmp_path / "ledger.json"
  write_json_atomic(path, {"config":{"model":"a"}, "rounds":[]})
  with pytest.raises(ValueError, match="config differs"):
    load_ledger(path, {"model":"b"})
