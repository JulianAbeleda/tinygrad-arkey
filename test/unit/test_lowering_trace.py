"""LR-010 acceptance: the lowering trace contract.

The trace exists because pass order is declared nowhere -- it is the statement order of three functions. These tests
pin the four things the scope asks for: a complete ordered trace on a real graph, identification of the pass that
materializes a producer, a trace that survives being taken in a subprocess, and *no* effect when disabled.
"""
import json, os, subprocess, sys, textwrap
import pytest

from tinygrad.uop import trace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _child(body: str, env_extra: dict) -> str:
  env = {**os.environ, "PYTHONPATH": ROOT, **env_extra}
  r = subprocess.run([sys.executable, "-c", textwrap.dedent(body)], cwd=ROOT, env=env, capture_output=True, text=True)
  assert r.returncode == 0, f"child failed: {r.stderr[-2000:]}"
  return r.stdout

# ----------------------------------------------------------------------------------- disabled by default ----
def test_disabled_by_default_and_inert():
  """Acceptance: existing output and cache behaviour are unchanged when tracing is off."""
  out = _child("""
    from tinygrad import Tensor
    from tinygrad.uop import trace
    t = (Tensor.rand(16,16)+1).sum(); t.schedule_linear()
    print("ENABLED", trace.ENABLED)
    print("ACTIVE", trace.active())
  """, {})
  assert "ENABLED False" in out
  assert "ACTIVE None" in out          # no collector is even constructed

def test_disabled_trace_prints_nothing():
  """It must never print on its own -- a silent default is part of the contract."""
  out = _child("""
    from tinygrad import Tensor
    (Tensor.rand(8,8)+1).sum().schedule_linear()
    print("SENTINEL")
  """, {})
  assert out.strip() == "SENTINEL"

# --------------------------------------------------------------------------------------- ordered trace ----
def test_fused_elementwise_reduce_graph_produces_an_ordered_trace():
  out = _child("""
    from tinygrad import Tensor
    from tinygrad.uop import trace
    ((Tensor.rand(64,64)+1.0)*2.0).sum().schedule_linear()
    tr = trace.active()
    print("EVENTS", len(tr.events))
    print("ORDERED", tr.order() == [e.pass_name for e in sorted(tr.events, key=lambda e: e.seq)])
    print("EFFECTIVE", len(tr.effective()) > 0)
  """, {"LOWER_TRACE": "1"})
  assert "ORDERED True" in out and "EFFECTIVE True" in out
  assert int(out.split("EVENTS")[1].split()[0]) > 20

def test_trace_identifies_the_materializing_pass():
  """Acceptance: the trace identifies the pass that materializes or preserves a producer."""
  out = _child("""
    from tinygrad import Tensor
    from tinygrad.uop import trace
    ((Tensor.rand(64,64)+1.0)*2.0).sum().schedule_linear()
    print("MATERIALIZERS", [e.pass_name for e in trace.active().materializers()])
  """, {"LOWER_TRACE": "1"})
  names = out.split("MATERIALIZERS")[1].strip()
  assert names not in ("[]", ""), "no pass was credited with materialization"

# ------------------------------------------------------------------------------------------ subprocess ----
def test_trace_is_complete_in_an_isolated_subprocess(tmp_path):
  """Acceptance: works in an isolated subprocess, not only in a parent that does not own execution.

  This repo measures everything in child processes, so an in-memory-only trace would be useless.
  """
  path = tmp_path / "trace.jsonl"
  _child("""
    from tinygrad import Tensor
    ((Tensor.rand(32,32)+1.0).sum()).schedule_linear()
  """, {"LOWER_TRACE": "1", "LOWER_TRACE_PATH": str(path)})
  lines = [json.loads(l) for l in path.read_text().splitlines()]
  header = [l for l in lines if l["record"] == "header"]
  events = [l for l in lines if l["record"] == "event"]
  assert len(header) == 1 and len(events) > 20
  assert header[0]["pid"] == events[0]["pid"]           # self-contained: the child owns the whole trace
  assert events == sorted(events, key=lambda e: e["seq"])

def test_header_records_effective_gate_values(tmp_path):
  """36 of 93 passes are env-gated; a trace without the effective values cannot explain another machine's result."""
  path = tmp_path / "trace.jsonl"
  _child("""
    from tinygrad import Tensor
    (Tensor.rand(8,8)+1).sum().schedule_linear()
  """, {"LOWER_TRACE": "1", "LOWER_TRACE_PATH": str(path), "WARP_REDUCE_LOWERING": "1"})
  header = json.loads(path.read_text().splitlines()[0])
  assert header["gates"]["WARP_REDUCE_LOWERING"] == "1"        # the override is captured
  assert header["gates"]["PREFILL_SOFTMAX_REDUCE_FUSE"] == "1" # ...and so is an unset gate's real default
  assert "PREFILL_SOFTMAX_REDUCE_FUSE" in header["gates_not_in_cache_key"]

# ------------------------------------------------------------------------------------------- unit-level ----
def test_summarize_counts_and_fingerprints():
  from tinygrad import Tensor
  a = Tensor.rand(8, 8).sum()
  s = trace.summarize(a.uop)
  assert s.nodes > 0 and s.fingerprint != "unavailable"
  assert sum(s.op_counts.values()) == s.nodes

def test_materializing_is_an_increase_not_a_level():
  """A pass that merely carries materialization forward must not be credited with creating it."""
  mk = lambda buffers: trace.GraphSummary(fingerprint="f", nodes=1, op_counts={}, ranges=0, reduces=0,
                                          buffers=buffers, defines=0, stores=0, contiguous=0)
  ev = lambda b, a: trace.LoweringTraceEvent(seq=0, pass_name="p", trace_version="v", pid=1, bottom_up=False,
                                             walk=False, changed=True, before=mk(b), after=mk(a), duration_us=0)
  assert ev(0, 1).materializing()
  assert not ev(1, 1).materializing()
  assert not ev(2, 1).materializing()

def test_reset_reresolves_the_enabled_flag(monkeypatch):
  monkeypatch.setenv("LOWER_TRACE", "1")
  trace.reset()
  assert trace.ENABLED and trace.active() is not None
  monkeypatch.setenv("LOWER_TRACE", "0")
  trace.reset()
  assert not trace.ENABLED and trace.active() is None
