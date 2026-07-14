import os
from types import SimpleNamespace

import pytest

from extra.qk import prefill_graph_gemm_route as route


def test_prefill_v2_covered_linears_are_role_tagged():
  from tinygrad.llm.model import Transformer
  class Weight:
    def __init__(self, shape): self.shape = shape
  tr = object.__new__(Transformer)
  tr.blk = [SimpleNamespace(
    attn_q=SimpleNamespace(weight=Weight((4096,4096))), attn_k=SimpleNamespace(weight=Weight((1024,4096))),
    attn_v=SimpleNamespace(weight=Weight((1024,4096))), attn_output=SimpleNamespace(weight=Weight((4096,4096))),
    ffn_gate=SimpleNamespace(weight=Weight((12288,4096))), ffn_up=SimpleNamespace(weight=Weight((12288,4096))),
    ffn_down=SimpleNamespace(weight=Weight((4096,12288))))]
  covered = {name:getattr(tr.blk[0],name) for name in tr._PREFILL_V2_LINEARS if hasattr(tr.blk[0],name)}
  list(tr._prefill_v2_covered())
  assert covered["attn_q"]._prefill_graph_role == covered["attn_output"]._prefill_graph_role == "attn_qo"
  assert covered["attn_k"]._prefill_graph_role == covered["attn_v"]._prefill_graph_role == "attn_kv"
  assert covered["ffn_gate"]._prefill_graph_role == covered["ffn_up"]._prefill_graph_role == "ffn_gate_up"
  assert covered["ffn_down"]._prefill_graph_role == "ffn_down"


def test_prefill_lm_head_route_controls_resident_realize():
  from tinygrad.helpers import getenv
  from tinygrad.llm.model import Transformer
  class Weight:
    def __init__(self, shape): self.shape = shape
  def build():
    tr = object.__new__(Transformer); tr.blk = [SimpleNamespace()]
    tr.output = SimpleNamespace(weight=Weight((151936,4096)), q6k_storage=object(), prefill_packed_weight=lambda:None,
                                name="output.weight")
    return tr
  previous = os.environ.get("PREFILL_LM_HEAD_ROUTE")
  try:
    for policy, expected in ((None,False),("resident_fp16",True),("direct_packed",False)):
      if policy is None: os.environ.pop("PREFILL_LM_HEAD_ROUTE",None)
      else: os.environ["PREFILL_LM_HEAD_ROUTE"] = policy
      getenv.cache_clear(); tr=build()
      assert any(lin is tr.output for lin,_,_ in tr._prefill_v2_covered()) is expected
  finally:
    if previous is None: os.environ.pop("PREFILL_LM_HEAD_ROUTE",None)
    else: os.environ["PREFILL_LM_HEAD_ROUTE"] = previous
    getenv.cache_clear()


class CandidateTensor:
  def __init__(self,shape): self.shape=shape; self.device="CPU"
  @property
  def ndim(self): return len(self.shape)


@pytest.mark.parametrize(("role","shape"),(
  ("ffn_gate_up",(512,12288,4096)),("ffn_down",(512,4096,12288)),
  ("attn_qo",(512,4096,4096)),("attn_kv",(512,1024,4096))))
def test_candidate_set_exact_entries_bind(monkeypatch,role,shape):
  identity=(role.encode().hex()+"0"*64)[:64]; admission=SimpleNamespace(canonical_identity=identity)
  class Registry:
    def get(self,profile,selected_role,selected_shape,target):
      assert profile == "qwen3_8b_q4k_m_gfx1100" and target["arch"] == "gfx1100"
      return admission if (selected_role,selected_shape) == (role,shape) else None
  monkeypatch.setattr(route,"_candidate_registry_from_env",lambda:Registry())
  monkeypatch.setattr(route,"_install_candidate_matmul",lambda x,w,n,k,selected:selected.canonical_identity)
  lin=SimpleNamespace(_prefill_graph_role=role,_prefill_model_profile="qwen3_8b_q4k_m_gfx1100",bias=None)
  out=route.route_pf16_graph_gemm(lin,CandidateTensor((1,shape[0],shape[2])),w=CandidateTensor((shape[1],shape[2])))
  assert out == identity and lin._prefill_full_kernel_candidate_identity == identity


def test_missing_exact_candidate_falls_back(monkeypatch):
  monkeypatch.setattr(route,"_candidate_registry_from_env",lambda:SimpleNamespace(get=lambda *_:None))
  lin=SimpleNamespace(_prefill_graph_role="attn_qo",_prefill_model_profile="qwen3_8b_q4k_m_gfx1100",bias=None)
  assert route.route_pf16_graph_gemm(lin,CandidateTensor((1,512,4096)),w=CandidateTensor((2048,4096))) is None
  assert not hasattr(lin,"_prefill_full_kernel_candidate_identity")


def test_promoted_policy_applies_for_absent_or_explicit_on_and_zero_rolls_back():
  absent=route._candidate_policy_env({}); explicit=route._candidate_policy_env({"PREFILL_GRAPH_GEMM":"1"})
  assert absent["BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH"] == explicit["BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH"]
  assert route._candidate_policy_env({"PREFILL_GRAPH_GEMM":"0"}) == {"PREFILL_GRAPH_GEMM":"0"}


def _admission(role,n,k,identity):
  return SimpleNamespace(canonical_identity=identity,normalized_payload={"workload":{
    "profile":"qwen3_8b_q4k_m_gfx1100","role":role,"shape":{"m":512,"n":n,"k":k},
    "target":{"backend":"AMD","arch":"gfx1100","wave_size":32}}})


def test_candidate_route_census_requires_exact_bindings_and_counts_reuse():
  admissions=tuple(_admission(role,n,k,str(i)*64) for i,(role,n,k) in enumerate((
    ("ffn_gate_up",12288,4096),("ffn_down",4096,12288),("attn_qo",4096,4096),("attn_kv",1024,4096)),1))
  entries=tuple(SimpleNamespace(exact_key=("qwen3_8b_q4k_m_gfx1100",a.normalized_payload["workload"]["role"],512,
    a.normalized_payload["workload"]["shape"]["n"],a.normalized_payload["workload"]["shape"]["k"],"AMD","gfx1100",32)) for a in admissions)
  registry=SimpleNamespace(candidate_set=SimpleNamespace(entries=entries),admissions=admissions)
  with route.candidate_route_census() as collector:
    for admission in admissions: route._record_candidate_route(admission)
    route._record_candidate_route(admissions[2])
  report=route.finalize_candidate_route_census(collector,registry)
  assert report["passed"] and report["selected_entry_count"] == report["expected_entry_count"] == 4
  assert next(x for x in report["selected"] if x["role"] == "attn_qo")["bindings"] == 2
  with route.candidate_route_census() as incomplete: route._record_candidate_route(admissions[0])
  assert len(route.finalize_candidate_route_census(incomplete,registry)["missing"]) == 3
