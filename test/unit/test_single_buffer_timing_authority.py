import math, pytest

from extra.qk.runtime_specs import (ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH, GFX1100_SINGLE_BUFFER_CAPABILITY,
                                    GFX1100_TWO_BUFFER_STAGE1_CAPABILITY)
from extra.qk.prefill.single_buffer_timing_authority import run_kernel_timing

def _execution(): return {"passed":True,"canonical_identity":ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH,
  "capability_id":GFX1100_SINGLE_BUFFER_CAPABILITY.capability_id,
  "workload":{"role":"ffn_gate_up","shape":{"m":512,"n":12288,"k":4096}},
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

@pytest.mark.parametrize("role,shape",(
  ("ffn_gate_up",(512,12288,4096)),("ffn_down",(512,4096,12288)),
  ("attn_qo",(512,4096,4096)),("attn_kv",(512,1024,4096))))
def test_timing_derives_flops_from_exact_execution_workload(role,shape):
  execution=_execution();execution["workload"]={"role":role,"shape":dict(zip(("m","n","k"),shape))}
  report=run_kernel_timing(execution,lambda **_:0.002,warmups=1,rounds=3,
    telemetry=lambda *a,**k:{},clock_context=lambda:__import__("contextlib").nullcontext(None))
  assert report["workload"]==execution["workload"]
  assert report["median_tflops"]==2*math.prod(shape)/0.002/1e12

def test_timing_accepts_actual_two_buffer_capability():
  execution=_execution();execution["capability_id"]=GFX1100_TWO_BUFFER_STAGE1_CAPABILITY.capability_id
  report=run_kernel_timing(execution,lambda **_:0.001,warmups=1,rounds=3,
    telemetry=lambda *a,**k:{},clock_context=lambda:__import__("contextlib").nullcontext(None))
  assert report["capability_id"]==GFX1100_TWO_BUFFER_STAGE1_CAPABILITY.capability_id

@pytest.mark.parametrize("mutation,error",(
  (lambda x:x.update(passed=False),"passing"),(lambda x:x["structural_binding"].update(pre_gpu_eligible=False),"passing"),
  (lambda x:x["runtime"].update(executed_binary_sha256="b"*64),"binary"),
  (lambda x:x.update(workload={"role":"attention","shape":{"m":512,"n":4096,"k":4096}}),"role"),
  (lambda x:x.update(workload={"role":"attn_qo","shape":{"m":0,"n":4096,"k":4096}}),"shape"),
  (lambda x:x["environment"]["git"].update(dirty=True),"clean")))
def test_timing_fails_closed_on_broken_joins(mutation,error):
  execution=_execution();mutation(execution)
  with pytest.raises(ValueError,match=error): run_kernel_timing(execution,lambda **_:0.001,warmups=1,rounds=3)

def test_timing_rejects_invalid_samples():
  with pytest.raises(RuntimeError,match="positive finite"): run_kernel_timing(_execution(),lambda **_:0.0,warmups=1,rounds=3,
    telemetry=lambda *a,**k:{},clock_context=lambda:__import__("contextlib").nullcontext(None))
