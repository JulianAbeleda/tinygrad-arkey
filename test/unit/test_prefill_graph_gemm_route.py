from types import SimpleNamespace

import pytest

from extra.qk import prefill_graph_gemm_route as route
from extra.qk.route_manifest import canonical_candidate_set_identity
from tinygrad import Tensor, dtypes
from tinygrad.llm.memory_semantics import model_parameter


def test_prefill_v2_covered_linears_are_role_tagged():
  from tinygrad.llm.model import Transformer
  class Weight:
    def __init__(self, shape): self.shape = shape
  tr = object.__new__(Transformer)
  tr.config = SimpleNamespace(prefill_policy=None, prefill_ubatch=512,
                              prefill_device_facts=None, lm_head_route="lazy")
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
  def build(policy):
    tr = object.__new__(Transformer); tr.blk = [SimpleNamespace()]
    tr.config = SimpleNamespace(prefill_policy=None, prefill_ubatch=512,
                                prefill_device_facts=None, lm_head_route=policy)
    tr.output = SimpleNamespace(weight=Weight((151936,4096)), q6k_storage=object(), prefill_packed_weight=lambda:None,
                                name="output.weight")
    return tr
  for policy, expected in (("lazy",False),("resident_fp16",True),("direct_packed",False)):
    tr=build(policy)
    assert any(lin is tr.output for lin,_,_ in tr._prefill_v2_covered()) is expected


class CandidateTensor:
  def __init__(self,shape): self.shape=shape; self.device="CPU"
  @property
  def ndim(self): return len(self.shape)


def test_candidate_operand_reuses_semantic_resident_buffer():
  concrete = model_parameter(Tensor.empty(16, dtype=dtypes.float16, device="CPU").realize())
  assert concrete.uop.op.name == "MEMORY_SEMANTIC"
  assert route._contiguous_candidate_operand(concrete).uop is concrete.uop
  lazy = model_parameter((Tensor.empty(16, dtype=dtypes.float16, device="CPU") + 1))
  assert route._contiguous_candidate_operand(lazy).uop.op.name == "CONTIGUOUS"


@pytest.mark.parametrize(("role","shape"),(
  ("ffn_gate_up",(512,12288,4096)),("ffn_down",(512,4096,12288)),
  ("attn_qo",(512,4096,4096)),("attn_kv",(512,1024,4096))))
def test_candidate_set_exact_entries_bind(monkeypatch,role,shape):
  identity=(role.encode().hex()+"0"*64)[:64]; admission=_admission(role,shape[1],shape[2],identity)
  registry=_registry((admission,))
  monkeypatch.setattr(route,"_install_candidate_matmul",lambda x,w,n,k,selected,artifact:selected.canonical_identity)
  lin=SimpleNamespace(_prefill_graph_role=role,bias=None,
    _prefill_graph_gemm_binding=_binding(registry,admission))
  out=route.route_pf16_graph_gemm(lin,CandidateTensor((1,shape[0],shape[2])),w=CandidateTensor((shape[1],shape[2])))
  assert out == identity and lin._prefill_full_kernel_candidate_identity == identity


def test_missing_exact_candidate_falls_back(monkeypatch):
  admission=_admission("attn_qo",4096,4096,"1"*64); registry=_registry((admission,))
  lin=SimpleNamespace(_prefill_graph_role="attn_qo",bias=None,_prefill_graph_gemm_binding=_binding(registry,admission))
  assert route.route_pf16_graph_gemm(lin,CandidateTensor((1,512,4096)),w=CandidateTensor((2048,4096))) is None
  assert not hasattr(lin,"_prefill_full_kernel_candidate_identity")


def test_candidate_selector_requires_attached_scanned_policy(monkeypatch):
  monkeypatch.setenv("BOLTBEAM_MODEL_PROFILE", "must-not-bind")
  lin=SimpleNamespace(_prefill_graph_role="attn_qo",bias=None)
  assert route.route_pf16_graph_gemm(lin,CandidateTensor((1,512,4096)),w=CandidateTensor((4096,4096))) is None


@pytest.mark.parametrize("corrupt", ("facts", "inventory", "candidate_set", "candidate"))
def test_candidate_selector_fails_closed_on_nonexact_attachment(monkeypatch,corrupt):
  admission=_admission("attn_qo",4096,4096,"1"*64); registry=_registry((admission,)); binding=_binding(registry,admission)
  if corrupt == "facts": binding["scanned_target_facts"]["target"]["arch"] = "gfx1200"
  elif corrupt == "inventory": binding["selected_policy"]["inventory_identity"] = "inventory:sha256:"+"b"*64
  elif corrupt == "candidate_set": binding["candidate_set_identity"] = "candidate_set:sha256:"+"b"*64
  else: binding["selected_policy"]["candidate_identity"] = "2"*64
  lin=SimpleNamespace(_prefill_graph_role="attn_qo",bias=None,_prefill_graph_gemm_binding=binding)
  monkeypatch.setattr(route,"_install_candidate_matmul",lambda *_:pytest.fail("nonexact attachment installed"))
  assert route.route_pf16_graph_gemm(lin,CandidateTensor((1,512,4096)),w=CandidateTensor((4096,4096))) is None


def test_candidate_selector_passes_runtime_m_to_exact_admission(monkeypatch):
  admission=_admission("attn_qo",4096,4096,"1"*64); registry=_registry((admission,))
  lin=SimpleNamespace(_prefill_graph_role="attn_qo",bias=None,_prefill_graph_gemm_binding=_binding(registry,admission))
  assert route.route_pf16_graph_gemm(lin,CandidateTensor((1,256,4096)),w=CandidateTensor((4096,4096))) is None


def _admission(role,n,k,identity):
  return SimpleNamespace(canonical_identity=identity,normalized_payload={"workload":{
    "profile":"qwen3_8b_q4k_m_gfx1100","role":role,"shape":{"m":512,"n":n,"k":k},
    "target":{"backend":"AMD","arch":"gfx1100","wave_size":32}}})

def _registry(admissions):
  entries=tuple(SimpleNamespace(to_json=lambda a=a:{"canonical_identity":a.canonical_identity,"payload":a.normalized_payload}) for a in admissions)
  candidate_set=SimpleNamespace(entries=entries,to_json=lambda:{"schema":"boltbeam.full_kernel_candidate_set.v1",
    "entries":[x.to_json() for x in entries]})
  return SimpleNamespace(candidate_set=candidate_set,admissions=tuple(admissions))

def _binding(registry,admission):
  row=route._candidate_route_row(admission); set_identity=canonical_candidate_set_identity(registry.candidate_set.to_json())
  return {"candidate_registry":registry,"inventory_identity":"inventory:sha256:"+"a"*64,
    "candidate_set_identity":set_identity,"scanned_target_facts":{"target":row["target"]},
    "selected_policy":{"role":row["role"],"shape":row["shape"],"target":row["target"],
      "inventory_identity":"inventory:sha256:"+"a"*64,"candidate_set_identity":set_identity,
      "candidate_identity":admission.canonical_identity,"profile":"provenance-only"}}


def test_candidate_route_census_requires_exact_bindings_and_counts_reuse():
  admissions=tuple(_admission(role,n,k,str(i)*64) for i,(role,n,k) in enumerate((
    ("ffn_gate_up",12288,4096),("ffn_down",4096,12288),("attn_qo",4096,4096),("attn_kv",1024,4096)),1))
  entries=tuple(SimpleNamespace() for a in admissions)
  registry=SimpleNamespace(candidate_set=SimpleNamespace(entries=entries),admissions=admissions)
  with route.candidate_route_census() as collector:
    for admission in admissions: route._record_candidate_route(admission)
    route._record_candidate_route(admissions[2])
  report=route.finalize_candidate_route_census(collector,registry)
  assert report["passed"] and report["selected_entry_count"] == report["expected_entry_count"] == 4
  assert next(x for x in report["selected"] if x["role"] == "attn_qo")["bindings"] == 2
  with route.candidate_route_census() as incomplete: route._record_candidate_route(admissions[0])
  assert len(route.finalize_candidate_route_census(incomplete,registry)["missing"]) == 3
