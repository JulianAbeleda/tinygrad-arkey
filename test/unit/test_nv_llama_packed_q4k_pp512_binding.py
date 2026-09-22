from types import SimpleNamespace

import pytest
from tinygrad import Tensor,dtypes
from tinygrad.llm.model import _nv_llama_packed_q4k_capture
from extra.llm_research.prefill.nv_llama_packed_q4k_pp512_binding import (
  K,M,N,PAIRS_PER_MODEL,LlamaPackedQ4KPP512Capture,supports,
)


def test_exact_admission_is_fail_closed():
  base=dict(model_family="qwen3_8b",weight_type="Q4_K",m=M,n=N,k=K,device="NV")
  assert supports(**base)
  for mutation in ({"model_family":"qwen3_14b"},{"weight_type":"Q6_K"},{"m":256},{"n":4096},{"k":5120},{"device":"AMD"}):
    assert not supports(**(base|mutation))


def test_capture_epochs_are_model_and_jit_local():
  class Asset:
    def new_capture(self): return LlamaPackedQ4KPP512Capture(self)
  asset,a,b,j0,j1=Asset(),SimpleNamespace(),SimpleNamespace(),object(),object()
  a0=_nv_llama_packed_q4k_capture(a,j0,asset);a1=_nv_llama_packed_q4k_capture(a,j1,asset);b0=_nv_llama_packed_q4k_capture(b,j0,asset)
  assert a0 is _nv_llama_packed_q4k_capture(a,j0,asset)
  assert len({id(a0),id(a1),id(b0)}) == 3


def test_pair_buffers_are_lazy_graph_owned_and_census_bounded():
  class Program: pass
  capture=LlamaPackedQ4KPP512Capture(SimpleNamespace(producer=Program(),main=Program(),fixup=Program()))
  with pytest.raises(RuntimeError,match="begin_trace"): capture.project_pair(None,None,None,model_family="qwen3_8b")
  capture.begin_trace();capture.cursor=PAIRS_PER_MODEL
  with pytest.raises(RuntimeError,match="36-pair"): capture.project_pair(None,None,None,model_family="qwen3_8b")
  for t in (Tensor.empty(M*K//4,dtype=dtypes.uint32),Tensor.empty(M*N,dtype=dtypes.float32)):
    assert not t.uop.buf_uop.buffer.is_allocated()
