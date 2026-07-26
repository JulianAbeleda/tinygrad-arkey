"""LR-042 acceptance: the buffer/storage decision (bufferize_to_store), recorded as a result rather than lost in
control flow. Mirrors test/unit/test_realization_plan.py's shape for LR-030."""
import os, subprocess, sys, textwrap

from tinygrad.schedule import buffer_plan
from tinygrad.schedule.buffer_plan import (BufferizeOpts, StorageDecision, BufferPlan, NEW_GLOBAL, NEW_LOCAL,
                                           REUSED_AFTER)
from tinygrad.schedule.indexing import BufferizeOpts as IndexingBufferizeOpts
from tinygrad.schedule.rangeify import BufferizeOpts as RangeifyBufferizeOpts
from tinygrad.dtype import AddrSpace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _child(body: str, env_extra: dict):
  env = {**os.environ, "PYTHONPATH": ROOT, **env_extra}
  return subprocess.run([sys.executable, "-c", textwrap.dedent(body)], cwd=ROOT, env=env, capture_output=True, text=True)

# ------------------------------------------------------------------------- the pure move is really a re-export ----
def test_bufferize_opts_is_the_same_class_everywhere():
  """LR-042: BufferizeOpts moved to buffer_plan.py; indexing.py and rangeify.py must still hand out the identical
  class object, not a copy -- this is what makes it a pure move rather than a fork."""
  assert IndexingBufferizeOpts is BufferizeOpts
  assert RangeifyBufferizeOpts is BufferizeOpts

def test_bufferize_opts_fields_unchanged():
  o = BufferizeOpts(device="AMD", addrspace=AddrSpace.LOCAL, removable=False, composite_consumer=True)
  assert (o.device, o.addrspace, o.removable, o.composite_consumer) == ("AMD", AddrSpace.LOCAL, False, True)

# ------------------------------------------------------------------------------------ inert by default ----
def test_recording_is_off_by_default():
  r = _child("""
    from tinygrad import Tensor
    from tinygrad.schedule import buffer_plan
    (Tensor.rand(16,16)+1).sum().schedule_linear()
    print("ENABLED", buffer_plan.ENABLED)
    print("ACTIVE", buffer_plan.active())
  """, {})
  assert "ENABLED False" in r.stdout and "ACTIVE None" in r.stdout

def test_recording_does_not_change_the_schedule():
  """Acceptance: the old and planned paths produce identical schedules -- BUFFER_PLAN only observes."""
  prog = """
    import hashlib
    from tinygrad import Tensor
    Tensor.manual_seed(1337)
    lin = ((Tensor.rand(64,64)+1.0)*2.0).sum().schedule_linear()
    print(hashlib.sha256(lin.key).hexdigest())
  """
  off, on = _child(prog, {}), _child(prog, {"BUFFER_PLAN": "1"})
  assert off.returncode == on.returncode == 0, (off.stderr[-800:], on.stderr[-800:])
  assert off.stdout.strip() == on.stdout.strip()

def test_plan_records_a_new_global_buffer():
  r = _child("""
    from tinygrad import Tensor
    from tinygrad.schedule import buffer_plan
    a, b = Tensor.rand(64,64), Tensor.rand(64,64)
    c = (a@b).relu()
    (c.sum()+c.max()).schedule_linear()
    p = buffer_plan.active()
    print("DECISIONS", len(p.decisions))
    print("GLOBAL", len(p.global_producers()))
    print("ADDRSPACE", p.by_addrspace())
  """, {"BUFFER_PLAN": "1"})
  assert r.returncode == 0, r.stderr[-1500:]
  assert int(r.stdout.split("DECISIONS")[1].split()[0]) > 0
  assert int(r.stdout.split("GLOBAL")[1].split()[0]) > 0

# ------------------------------------------------------------------------------------------- units ----
def test_explanations_name_the_actual_cause():
  assert "GLOBAL buffer" in StorageDecision("x", "GLOBAL", NEW_GLOBAL, ranges_closed=3, size=16).explain()
  assert "LOCAL buffer" in StorageDecision("x", "LOCAL", NEW_LOCAL, ranges_closed=2, size=8).explain()
  assert "reused the existing AFTER" in StorageDecision("x", "GLOBAL", REUSED_AFTER, ranges_closed=1).explain()

def test_by_addrspace_groups_producers():
  p = BufferPlan()
  p.record(StorageDecision("a", "GLOBAL", NEW_GLOBAL)); p.record(StorageDecision("b", "LOCAL", NEW_LOCAL))
  p.record(StorageDecision("c", "GLOBAL", REUSED_AFTER))
  assert p.by_addrspace() == {"GLOBAL": ["a", "c"], "LOCAL": ["b"]}

def test_plan_serializes():
  p = BufferPlan()
  p.record(StorageDecision("a", "LOCAL", NEW_LOCAL, ranges_closed=2, size=8))
  j = p.to_json()
  assert j["schema"] == "tinygrad.buffer_plan.v1"
  assert j["decisions"][0]["reason"] == NEW_LOCAL
