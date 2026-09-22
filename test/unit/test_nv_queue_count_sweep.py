"""Hermetic pin for the queue-count sweep that closes mechanism A.

The decisive numbers come from the recorded 596-node decode DAG. A third
compute queue can only capture the last 26.4 us of schedule slack, and any
larger queue count is exactly at the dependency critical path.
"""
from pathlib import Path

import pytest

from extra.llm_research.decode.nv_queue_count_sweep import load_dag, sweep


def test_recorded_dag_queue_count_ceiling():
  dag = Path("docs/task_workflow/evidence/nv-dag-duration-head-20260812.json")
  nodes, edges = load_dag(str(dag))
  result = sweep(nodes, edges, max_queues=8)

  by_queues = {row["queues"]: row for row in result["schedule"]}
  assert result["node_count"] == 596
  assert result["serialized_us"] == 5493.27
  assert result["critical_path_us"] == 4842.38
  assert by_queues[2]["span_us"] == 4868.78
  assert by_queues[3]["span_us"] == 4842.38
  assert by_queues[8]["span_us"] == 4842.38
  assert by_queues[3]["saving_vs_serial_us"] - by_queues[2]["saving_vs_serial_us"] == pytest.approx(26.4)


def test_synthetic_cycle_is_rejected():
  nodes = [
    {"id": 0, "duration_us": 1.0},
    {"id": 1, "duration_us": 1.0},
  ]
  edges = [{"from": 0, "to": 1}, {"from": 1, "to": 0}]
  try:
    sweep(nodes, edges, max_queues=2)
  except ValueError as exc:
    assert "cycle" in str(exc)
  else:
    raise AssertionError("expected a cycle error")
