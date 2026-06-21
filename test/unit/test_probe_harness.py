#!/usr/bin/env python3
"""qk_probe_harness golden + no-new-clone guard (audit S1).

Pins that probe_io is byte-identical to the historical clone writer, and that no extra/*.py re-clones its own
write_json (the de-clone must stay done). No GPU / no tinygrad import.

Run: PYTHONPATH=. python -m unittest test.unit.test_probe_harness -v
"""
from __future__ import annotations
import json
import pathlib
import tempfile
import unittest

from extra.qk_probe_harness import probe_io, emit_verdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
# the helper itself + the generic JSONL/quality lib are the only allowed definers of a `write_json`.
ALLOWED_WRITE_JSON = {"extra/qk_probe_harness.py", "extra/llm_eval_common.py"}


class TestProbeHarness(unittest.TestCase):
  def test_write_json_byte_identical_to_clone_format(self):
    data = {"b": 2, "a": [1, 2], "verdict": "X"}
    with tempfile.TemporaryDirectory() as d:
      _, write_json = probe_io(d)
      write_json("r.json", data)
      got = (pathlib.Path(d) / "r.json").read_text()
    # the exact historical clone format: json.dumps(indent=2, sort_keys=True) + "\n"
    self.assertEqual(got, json.dumps(data, indent=2, sort_keys=True) + "\n")

  def test_read_json_default_and_roundtrip(self):
    with tempfile.TemporaryDirectory() as d:
      read_json, write_json = probe_io(d)
      self.assertEqual(read_json("bench/does/not/exist.json", default={"x": 1}), {"x": 1})
      # round-trip a repo-relative read of a known file
      self.assertIsInstance(read_json("bench/qk-decode-eval/candidates.json"), dict)

  def test_emit_verdict_shape(self):
    v = emit_verdict("p1", True, "proceed", extra_key=7)
    self.assertEqual(v, {"phase": "p1", "gate_pass": True, "next_action": "proceed", "extra_key": 7})

  def test_no_new_local_write_json_clone(self):
    """Guard: no extra/*.py re-clones write_json -- new probes must import qk_probe_harness.probe_io."""
    offenders = []
    for p in sorted(ROOT.glob("extra/*.py")):
      rel = p.relative_to(ROOT).as_posix()
      if rel in ALLOWED_WRITE_JSON: continue
      if any(line.startswith("def write_json") for line in p.read_text(errors="ignore").splitlines()):
        offenders.append(rel)
    self.assertEqual(offenders, [], f"these re-clone write_json instead of importing probe_io: {offenders}")


if __name__ == "__main__":
  unittest.main()
