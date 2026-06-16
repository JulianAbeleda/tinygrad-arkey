"""Reproduce-from-artifact golden lock for the flywheel row builders.

These tests regenerate the committed *portable* dataset artifacts through the
row-builder pipeline and assert the output is byte-identical, line for line.
They are the behavior proof for the row-builder consolidation: any change that
alters a single emitted byte fails here.

Only artifacts that use portable (repo-relative) source paths are locked. The
``kernel-triage-v0`` artifact embeds the benchmark machine's absolute path in
its ``accepted_runtime`` row ids (a pre-existing portability defect, see
coding-principles "Keep Artifacts And Fallbacks Portable"), so it is *not*
byte-reproducible from a different checkout and is deliberately excluded here.
"""
from __future__ import annotations

import hashlib, json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qk_flywheel_cost_model import run_cost_model
from extra.qk_flywheel_targeted_outcomes import build_targeted_rows, write_phase3f

REPO = pathlib.Path(__file__).resolve().parents[2]
PROOF = REPO / "bench/amd-decode-flywheel-proof-20260614"


def _lines(path: pathlib.Path) -> list[str]:
  return path.read_text().splitlines()


class TestFlywheelDatasetGolden(unittest.TestCase):
  def test_targeted_rows_byte_identical_to_committed(self):
    rows, _excluded = build_targeted_rows(REPO)
    regen = [json.dumps(row, sort_keys=True) for row in rows]
    committed = _lines(PROOF / "targeted-outcomes-v1/examples.jsonl")
    self.assertEqual(regen, committed, "build_targeted_rows output drifted from committed artifact")

  def test_phase3f_plus_dataset_byte_identical_to_committed(self):
    base = PROOF / "kernel-triage-v1-featured/examples.jsonl"
    with TemporaryDirectory() as raw_td:
      out = pathlib.Path(raw_td)
      write_phase3f(REPO, base, out / "targeted", out / "plus", out / "audit", out / "coverage")
      # Targeted batch (the row-builder output) and the assembled plus dataset must
      # both match committed byte for byte.
      self.assertEqual(
        _lines(out / "targeted/examples.jsonl"),
        _lines(PROOF / "targeted-outcomes-v1/examples.jsonl"),
        "targeted examples.jsonl drifted",
      )
      self.assertEqual(
        _lines(out / "plus/examples.jsonl"),
        _lines(PROOF / "kernel-triage-v1-featured-plus/examples.jsonl"),
        "plus examples.jsonl drifted",
      )
      self.assertEqual(
        _lines(out / "plus/prompts.jsonl"),
        _lines(PROOF / "kernel-triage-v1-featured-plus/prompts.jsonl"),
        "plus prompts.jsonl drifted",
      )

  def test_cost_model_centroid_output_is_pinned(self):
    """Lock the centroid-backend cost-model output (features + predictions +
    summary) from the committed portable examples. xgboost is not required on
    this checkout, so only the deterministic centroid backend is pinned; the
    feature extractor is the load-bearing piece either way.
    """
    examples = PROOF / "kernel-triage-v1-featured-plus/examples.jsonl"
    pinned = {
      "predictions.jsonl": "bf10c793a7e6c1b4cdd8672a2fec36341aa217a93d4341d6f639322e56673068",
      "features.jsonl": "c29f5cbd44b155ea22ff925767770d538f0eb6928ce688eb8116e72e552a4171",
      "feature-vocab.json": "be73e4f708847723250418c9b5cbade8091e2a1d54fc1e3bfe539509cf9d3f31",
    }
    with TemporaryDirectory() as raw_td:
      out = pathlib.Path(raw_td) / "cost-model"
      summary = run_cost_model(examples, out, backend="centroid", seed=20260614)
      self.assertEqual(summary["features"]["feature_count"], 232)
      self.assertEqual(summary["conclusion"], "no_signal")
      self.assertEqual(summary["backends"]["ran"][0]["backend"], "centroid")
      for name, want in pinned.items():
        got = hashlib.sha256((out / name).read_bytes()).hexdigest()
        self.assertEqual(got, want, f"cost-model {name} drifted")

  def test_no_committed_locked_artifact_uses_absolute_paths(self):
    """Guard the portability invariant for every artifact we lock against."""
    for rel in (
      "targeted-outcomes-v1/examples.jsonl",
      "kernel-triage-v1-featured-plus/examples.jsonl",
    ):
      for row in (json.loads(line) for line in _lines(PROOF / rel)):
        for src in row.get("source_files", []):
          self.assertFalse(
            str(src).startswith("/") or "home-ubuntu" in str(src) or "users-julianabeleda" in str(src),
            f"{rel}: non-portable source path {src!r}",
          )


if __name__ == "__main__":
  unittest.main()
