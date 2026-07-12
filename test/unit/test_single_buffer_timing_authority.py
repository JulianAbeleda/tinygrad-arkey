import pytest

from extra.qk.runtime_specs import ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH, GFX1100_SINGLE_BUFFER_CAPABILITY
from extra.qk.prefill.single_buffer_timing_authority import run_kernel_timing

def _execution(): return {"passed":True,"canonical_identity":ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH,
  "capability_id":GFX1100_SINGLE_BUFFER_CAPABILITY.capability_id,
  "structural_binding":{"pre_gpu_eligible":True},"program":{"binary_sha256":"a"*64},
  "runtime":{"executed_binary_sha256":"a"*64},"environment":{"git":{"revision":"deadbeef","dirty":False}}}

def test_kernel_only_timing_protocol_and_statistics():
  values=iter([.004,.003,.002,.001,.002]); waits=[]
  def kernel(*,wait): waits.append(wait); return next(values)
  report=run_kernel_timing(_execution(),kernel,warmups=2,rounds=3,
    telemetry=lambda window,**kw:{"window":window,"sensors":{"core_clock_hz":1,"power_uw":2}},
    clock_context=lambda: __import__("contextlib").nullcontext({"ok":True}))
  assert waits == [True]*5 and report["samples_ms"] == [2.0,1.0,2.0]
  assert report["median_ms"] == 2.0 and report["min_ms"] == 1.0
  assert report["protocol"]["compile_excluded"] and report["joins"] == {"candidate":True,"binary":True,"commit":True}

def test_timing_accepts_dynamic_canonical_identity_with_capability_join():
  execution=_execution();execution["canonical_identity"]="b"*64
  report=run_kernel_timing(execution,lambda **_:0.001,warmups=1,rounds=3,
    telemetry=lambda *a,**k:{},clock_context=lambda:__import__("contextlib").nullcontext(None))
  assert report["canonical_identity"]=="b"*64

@pytest.mark.parametrize("mutation,error",(
  (lambda x:x.update(passed=False),"passing"),(lambda x:x["structural_binding"].update(pre_gpu_eligible=False),"passing"),
  (lambda x:x["runtime"].update(executed_binary_sha256="b"*64),"binary"),
  (lambda x:x["environment"]["git"].update(dirty=True),"clean")))
def test_timing_fails_closed_on_broken_joins(mutation,error):
  execution=_execution();mutation(execution)
  with pytest.raises(ValueError,match=error): run_kernel_timing(execution,lambda **_:0.001,warmups=1,rounds=3)

def test_timing_rejects_invalid_samples():
  with pytest.raises(RuntimeError,match="positive finite"): run_kernel_timing(_execution(),lambda **_:0.0,warmups=1,rounds=3,
    telemetry=lambda *a,**k:{},clock_context=lambda:__import__("contextlib").nullcontext(None))
