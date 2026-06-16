"""Golden tests locking the deterministic scorer to committed bench artifacts.

These are the safety anchor for the flywheel judging-tooling consolidation: they
assert byte-identical scorer output against scored artifacts already committed
under ``bench/``. Any change to the scorer that alters a single committed score
fails here, which is what proves later refactors are behavior-preserving.

Artifacts are discovered relative to the repo root so the test passes from any
checkout location (see coding-principles "Keep Artifacts And Fallbacks
Portable").
"""
from __future__ import annotations

import json, pathlib, unittest

from extra.llm_eval_common import quality_summary, score_prompt
from extra.llm_json_scorer import score_expected_json

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "bench"


def _read_jsonl(path: pathlib.Path) -> list[dict]:
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rollout_artifacts() -> list[pathlib.Path]:
  """Artifact dirs that carry both per-row scores and a quality summary."""
  out: list[pathlib.Path] = []
  for rollouts in sorted(BENCH.rglob("rollouts.jsonl")):
    summary = rollouts.parent / "summary.json"
    if not summary.exists():
      continue
    data = json.loads(summary.read_text())
    if "quality" in data:
      out.append(rollouts.parent)
  return out


class TestFlywheelScorerGolden(unittest.TestCase):
  def test_bench_artifacts_present(self):
    self.assertTrue(BENCH.is_dir(), f"missing bench dir at {BENCH}")
    artifacts = _rollout_artifacts()
    self.assertGreaterEqual(len(artifacts), 10, "expected committed rollout artifacts to lock against")

  def test_quality_summary_byte_identical(self):
    artifacts = _rollout_artifacts()
    self.assertTrue(artifacts)
    for art in artifacts:
      rows = _read_jsonl(art / "rollouts.jsonl")
      committed = json.loads((art / "summary.json").read_text())["quality"]
      recomputed = quality_summary(rows)
      self.assertEqual(
        json.dumps(recomputed, sort_keys=True),
        json.dumps(committed, sort_keys=True),
        f"quality_summary drifted for {art.relative_to(REPO_ROOT)}",
      )

  def test_score_expected_json_reproduces_committed_axes(self):
    checked = 0
    for art in _rollout_artifacts():
      for row in _read_jsonl(art / "rollouts.jsonl"):
        axes = row.get("score", {}).get("json_axes")
        if not isinstance(axes, dict):
          continue
        expected = axes.get("expected")
        text = row.get("text")
        if not isinstance(expected, dict) or text is None:
          continue
        case_insensitive = bool(axes.get("details", {}).get("case_insensitive", False))
        recomputed = score_expected_json(text, expected, case_insensitive=case_insensitive)
        self.assertEqual(
          json.dumps(recomputed, sort_keys=True),
          json.dumps(axes, sort_keys=True),
          f"score_expected_json drifted for {art.name}/{row.get('id')}",
        )
        checked += 1
    self.assertGreater(checked, 100, "expected many committed json_axes rows to lock against")

  def test_score_prompt_check_pass_flags_match_committed(self):
    """Re-derive each check's pass/fail from text and lock it against committed.

    Covers all four scorer axes (contains/regex/exact/json) present in the
    committed artifacts, without needing the original prompt spec: the committed
    check carries kind+value, which is enough to reconstruct the prompt spec for
    that single axis and re-score it.
    """
    kinds_seen: set[str] = set()
    checked = 0
    key = {"contains": "expected_contains", "regex": "expected_regex", "exact": "expected_exact"}
    artifacts = _rollout_artifacts()
    for art in artifacts:
      for row in _read_jsonl(art / "rollouts.jsonl"):
        text = row.get("text")
        if text is None:
          continue
        for check in row.get("score", {}).get("checks", []):
          kind, value, committed_pass = check.get("kind"), check.get("value"), check.get("passed")
          if kind == "json":
            spec = {"expected_json": value}
          elif kind in key:
            spec = {key[kind]: value}
          else:
            continue
          recomputed = score_prompt(spec, text)
          self.assertEqual(
            recomputed["checks"][0]["passed"],
            committed_pass,
            f"{kind} check drifted for {art.name}/{row.get('id')}",
          )
          kinds_seen.add(kind)
          checked += 1
    self.assertEqual(kinds_seen, {"contains", "regex", "exact", "json"}, "expected all check kinds covered")
    self.assertGreater(checked, 100)


if __name__ == "__main__":
  unittest.main()
