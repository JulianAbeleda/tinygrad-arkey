"""CPU-only regression tests for the NV captured-DAG support allowlist."""
import importlib.util, pathlib
from types import SimpleNamespace

from tinygrad.runtime.graph import hcq

PATH = pathlib.Path(__file__).resolve().parents[2] / "extra" / "llm_research" / "decode" / "nv_support_overlap_allowlist.py"
SPEC = importlib.util.spec_from_file_location("nv_support_overlap_allowlist", PATH)
MOD = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(MOD)


def test_derivation_excludes_semantic_and_long_nodes_and_uses_real_edges():
  # 0 and 1 are independent small support nodes.  2 joins them; semantic 3
  # is deliberately ineligible even though it is short.
  dag = {"nodes":[
    {"id":0, "name":"E_a", "duration_us":4.0, "metadata":None},
    {"id":1, "name":"r_b", "duration_us":4.0, "metadata":None},
    {"id":2, "name":"E_join", "duration_us":2.0, "metadata":None},
    {"id":3, "name":"E_semantic", "duration_us":1.0, "metadata":{"semantic":[{"role":"attn_kv"}]}},
    {"id":4, "name":"E_long", "duration_us":9.0, "metadata":None},
  ], "edges":[{"from":0,"to":2},{"from":1,"to":2},{"from":2,"to":3},{"from":3,"to":4}]}
  got = MOD.derive(dag)
  assert "E_semantic" not in got["selected"] and "E_long" not in got["selected"]
  assert set(got["selected"]) <= {"E_a", "r_b", "E_join"}
  assert got["predicted_saving_us"] > 0
  assert got["candidate"]["cross_queue_edges"] > 0


def test_hcq_picker_requires_explicit_name_or_prefix(monkeypatch):
  class Dev: device = "NV"
  dev = Dev()
  primary, secondary = object(), object()
  runner = SimpleNamespace(compute_queues={dev: [primary, secondary]}, compute_queue_load={primary:3, secondary:1})
  monkeypatch.setattr(hcq, "NV_MULTI_QUEUE_PROGRAMS", frozenset(("exact", "prefix:E_")))
  assert hcq.HCQGraph._pick_compute_queue(runner, dev, SimpleNamespace(name="exact")) is secondary
  assert hcq.HCQGraph._pick_compute_queue(runner, dev, SimpleNamespace(name="E_support")) is secondary
  assert hcq.HCQGraph._pick_compute_queue(runner, dev, SimpleNamespace(name="q4k_gemv")) is primary
