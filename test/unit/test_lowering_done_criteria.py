import json

import pytest

from extra.qk import lowering_done_criteria as criteria


def test_lowering_done_criteria_rows_are_valid_and_json_serializable():
  rows = criteria.rows()
  assert len(rows) == len(criteria.VALID_LOWERING_LEVELS)
  assert {r["target_lowering_level"] for r in rows} == set(criteria.VALID_LOWERING_LEVELS)

  for row in rows:
    assert set(row.keys()) == {"target_lowering_level", "required_criteria"}
    assert row["target_lowering_level"] in criteria.VALID_LOWERING_LEVELS
    assert row["required_criteria"]
    assert set(row["required_criteria"]) <= set(criteria.GENERIC_COMPLETION_CRITERIA)
    assert row == criteria.criteria_for_level(row["target_lowering_level"])
    json.dumps(row)


def test_criteria_for_level_validates_lowering_level():
  with pytest.raises(ValueError):
    criteria.criteria_for_level("L6")


def test_criteria_for_level_returns_json_serializable_payload():
  for level in criteria.VALID_LOWERING_LEVELS:
    payload = criteria.criteria_for_level(level)
    assert payload["target_lowering_level"] == level
    assert isinstance(payload["required_criteria"], list)
    json.dumps(payload)


def test_build_is_json_serializable_and_covers_required_criteria():
  report = criteria.build()
  assert report["schema"] == "lowering-done-criteria.v1"
  assert set(report["by_level"].keys()) == set(criteria.VALID_LOWERING_LEVELS)

  covered = set()
  for row in report["rows"]:
    covered.update(row["required_criteria"])

  assert covered == set(criteria.GENERIC_COMPLETION_CRITERIA)
  assert set(report["by_criterion"].keys()) == set(criteria.GENERIC_COMPLETION_CRITERIA)
  assert all(v >= 1 for v in report["by_criterion"].values())

  json.dumps(report)
