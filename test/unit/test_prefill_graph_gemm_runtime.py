from pathlib import Path
from types import SimpleNamespace

import pytest

from tinygrad.llm import prefill_graph_gemm as route


class CandidateTensor:
  def __init__(self, shape): self.shape = shape
  @property
  def ndim(self): return len(self.shape)


def _admission(role, n, k, identity, *, register=False):
  schedule = {"residency":{"resident":["accumulator", "stage_ab_register"]}} if register else {}
  return SimpleNamespace(canonical_identity=identity, context=SimpleNamespace(packed_weight=None),
    normalized_payload={"workload":{"profile":"provenance-only", "role":role,
      "shape":{"m":512, "n":n, "k":k}, "target":{"backend":"AMD", "arch":"gfx1100", "wave_size":32}},
      "schedule":schedule})


def _registry(admissions):
  entries = tuple(SimpleNamespace(to_json=lambda admission=admission:
    {"canonical_identity":admission.canonical_identity, "payload":admission.normalized_payload}) for admission in admissions)
  candidate_set = SimpleNamespace(entries=entries, to_json=lambda:{"schema":"boltbeam.full_kernel_candidate_set.v1",
    "entries":[entry.to_json() for entry in entries]})
  return SimpleNamespace(candidate_set=candidate_set, admissions=tuple(admissions))


def _binding(registry, admission):
  row = route._candidate_route_row(admission)
  set_identity = route._canonical_candidate_set_identity(registry.candidate_set.to_json())
  inventory_identity = "inventory:sha256:" + "a" * 64
  return {"candidate_registry":registry, "inventory_identity":inventory_identity,
    "candidate_set_identity":set_identity, "scanned_target_facts":{"target":row["target"]},
    "selected_policy":{"role":row["role"], "shape":row["shape"], "target":row["target"],
      "inventory_identity":inventory_identity, "candidate_set_identity":set_identity,
      "candidate_identity":admission.canonical_identity, "profile":"provenance-only"}}


@pytest.mark.parametrize(("role", "shape"), (
  ("ffn_gate_up", (512, 12288, 4096)), ("ffn_down", (512, 4096, 12288)),
  ("attn_qo", (512, 4096, 4096)), ("attn_kv", (512, 1024, 4096))))
def test_exact_selected_shape_binds(monkeypatch, role, shape):
  admission = _admission(role, shape[1], shape[2], (role.encode().hex() + "0" * 64)[:64])
  registry = _registry((admission,))
  monkeypatch.setattr(route, "_install_candidate_matmul", lambda x,w,n,k,selected,artifact:selected.canonical_identity)
  lin = SimpleNamespace(_prefill_graph_role=role, bias=None, _prefill_graph_gemm_binding=_binding(registry, admission))
  assert route.route_pf16_graph_gemm(lin, CandidateTensor((1, shape[0], shape[2])),
                                     CandidateTensor((shape[1], shape[2]))) == admission.canonical_identity
  assert lin._prefill_full_kernel_candidate_identity == admission.canonical_identity


def test_unsupported_shape_and_nonexact_policy_decline_to_generic(monkeypatch):
  admission = _admission("attn_qo", 4096, 4096, "1" * 64)
  registry = _registry((admission,))
  binding = _binding(registry, admission)
  lin = SimpleNamespace(_prefill_graph_role="attn_qo", bias=None, _prefill_graph_gemm_binding=binding)
  monkeypatch.setattr(route, "_install_candidate_matmul", lambda *_:pytest.fail("declined route installed"))
  assert route.route_pf16_graph_gemm(lin, CandidateTensor((1, 256, 4096)), CandidateTensor((4096, 4096))) is None
  binding["candidate_set_identity"] = "candidate_set:sha256:" + "b" * 64
  assert route.route_pf16_graph_gemm(lin, CandidateTensor((1, 512, 4096)), CandidateTensor((4096, 4096))) is None


def test_register_resident_candidate_stays_default_closed():
  admission = _admission("attn_qo", 4096, 4096, "1" * 64, register=True)
  assert route._install_candidate_matmul(None, None, 4096, 4096, admission) is None


def test_route_census_preserves_selected_identity_and_reuse_count():
  admissions = tuple(_admission(role, n, k, str(index) * 64) for index,(role,n,k) in enumerate((
    ("ffn_gate_up", 12288, 4096), ("ffn_down", 4096, 12288),
    ("attn_qo", 4096, 4096), ("attn_kv", 1024, 4096)), 1))
  registry = _registry(admissions)
  with route.candidate_route_census() as collector:
    for admission in admissions: route._record_candidate_route(admission)
    route._record_candidate_route(admissions[2])
  report = route.finalize_candidate_route_census(collector, registry)
  assert report["passed"] and report["selected_entry_count"] == report["expected_entry_count"] == 4
  assert next(row for row in report["selected"] if row["role"] == "attn_qo")["bindings"] == 2


def test_production_module_has_no_research_import():
  source = Path(route.__file__).read_text()
  assert "extra.llm_research" not in source
