"""Synthetic tests for the B3 demotion-search orchestrator (no GPU, no model).

`measure` (the only GPU-touching call) is patched with controlled tok/s + NLL so the candidate
enumeration, the quality-gated scorer, the AcceptedPolicy emission, and the frontier render are
all exercised deterministically.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from extra import qk_demote_search as ds


# baseline slow + good NLL; ffn_down/attn_v faster + tiny dNLL (free); output faster but BIG dNLL (sensitive).
FAKE = {
  "": {"tok_s": 54.5, "nll": 2.7795},
  "ffn_down": {"tok_s": 62.6, "nll": 2.7799},                 # dNLL +0.0004 -> within
  "ffn_down,attn_v": {"tok_s": 63.4, "nll": 2.7801},          # dNLL +0.0006 -> within
  "ffn_down,attn_v,output": {"tok_s": 66.0, "nll": 2.8300},   # dNLL +0.0505 -> FAILS quality
}


def _fake_measure(model, targets, *, bench, tokens, timeout):
  return dict(FAKE[targets])


class TestDemoteSearch(unittest.TestCase):
  @contextlib.contextmanager
  def _run(self, eps=0.01):
    with tempfile.TemporaryDirectory() as d, mock.patch.object(ds, "measure", _fake_measure):
      out = pathlib.Path(d)
      summary = ds.run_search("fake.gguf", epsilon=eps, bench=4, tokens=8, timeout=1, out_dir=out)
      yield summary, out

  def test_scoring_and_gate(self):
    with self._run(eps=0.01) as (summary, out):
      by = {r["label"]: r for r in summary["results"]}
      self.assertFalse(by["baseline"]["accepted"])                 # baseline is the reference, never accepted
      self.assertTrue(by["ffn_down"]["accepted"])                  # faster + within budget
      self.assertTrue(by["ffn_down+attn_v"]["accepted"])
      self.assertFalse(by["ffn_down+attn_v+output"]["within_quality"])  # dNLL 0.05 > 0.01
      self.assertFalse(by["ffn_down+attn_v+output"]["accepted"])   # rejected on quality despite being faster
      self.assertTrue(by["ffn_down+attn_v+output"]["faster"])
      self.assertAlmostEqual(by["ffn_down"]["dnll"], 0.0004, places=4)

  def test_accepted_policy_artifacts_written(self):
    with self._run(eps=0.01) as (_summary, out):
      self.assertTrue((out / "accepted-ffn_down.json").exists())
      self.assertTrue((out / "accepted-ffn_down+attn_v.json").exists())
      self.assertFalse((out / "accepted-ffn_down+attn_v+output.json").exists())  # rejected -> no artifact
      rec = json.loads((out / "accepted-ffn_down.json").read_text())
      self.assertEqual(rec["model"], "qwen3_8b")
      self.assertEqual(rec["targets"], "ffn_down")
      self.assertTrue(rec["exactness"].startswith("lossy"))

  def test_epsilon_zero_rejects_any_regression(self):
    # with epsilon 0, even the tiny +0.0004 dNLL fails -> nothing accepted (all demotions slightly raise NLL here)
    with self._run(eps=0.0) as (summary, _out):
      self.assertFalse(any(r["accepted"] for r in summary["results"]))

  def test_frontier_md_renders_rows(self):
    with self._run() as (summary, _out):
      md = ds.frontier_md(summary)
      for label in ("baseline", "ffn_down", "ffn_down+attn_v+output"):
        self.assertIn(label, md)
      self.assertIn("ACCEPT", md)


if __name__ == "__main__":
  unittest.main()
