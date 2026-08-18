"""S1 regression pin: the native NV decode DAG has runtime width at HEAD.

The S1 substrate row (DAG width) was measured present at HEAD
(`docs/task_workflow/evidence/nv-substrate-s1-runtime-width-head-20260817.json`):
the memory planner no longer aliases the q/k/v fan-out into one arena slot, so
the runtime dependency capture shows q/k/v as true siblings with a max-ready
width of 10. That is a regression guard: if a future planner change re-collapses
the decode DAG to width 1 (a pure chain), every overlap construction above it
(S2 multi-GPFIFO placement, S3 anchor+shadow, S4 PDL) is dead on arrival.

This is a hermetic pin over the committed evidence, not a GPU re-capture: the
full gate is the runtime dependency capture described in the evidence schema.
The pin asserts the committed record still proves the S1 gate (width >= 2 with
q/k/v siblings present), so a regenerated width-1 evidence file fails here.
"""
import json
import pathlib


EVIDENCE = pathlib.Path(__file__).parents[2] / "docs" / "task_workflow" / "evidence" / "nv-substrate-s1-runtime-width-head-20260817.json"


def _load_evidence():
  assert EVIDENCE.exists(), f"S1 runtime-width evidence missing: {EVIDENCE}"
  return json.loads(EVIDENCE.read_text())


def test_evidence_schema_marks_s1_present():
  d = _load_evidence()
  assert d["schema"] == "tinygrad.nv_substrate_s1_runtime_width.v1"
  assert d["per_token"]["replays_per_token"] >= 1


def test_runtime_width_gate_width_at_least_two():
  d = _load_evidence()
  assert d["per_token"]["runtime_deps_max_ready_width"] >= 2, \
    "decode DAG collapsed to a chain: S1 width regression"
  assert d["per_token"]["runtime_deps_critical_path_levels"] >= 2


def test_qkv_are_runtime_siblings():
  d = _load_evidence()
  sib = d["per_token"]["qkv_siblings"]
  assert "share the same" in sib and "neither is a predecessor" in sib, \
    f"q/k/v no longer runtime siblings: {sib}"
  assert "q4k_g3_lanemap_gemv_" in sib


def test_ready_set_histogram_shows_width_breadth():
  d = _load_evidence()
  hist = d["per_token"]["ready_set_histogram"]
  assert max(int(k) for k in hist) >= 2
  assert int(hist.get("1", 0)) < int(sum(hist.values())) * 0.8, \
    "majority of ready sets are width-1: DAG is chain-dominated"
