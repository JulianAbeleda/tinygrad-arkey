import ast
from pathlib import Path

from tinygrad import Tensor
from tinygrad.llm.memory_semantics import (MemorySemanticClass, PREFILL_ACTIVATION, memory_semantic_owner)
from tinygrad.llm.prefill_route_observer import PrefillDirectPackedBinding, PrefillRouteAttachment, prefill_route_scope
from tinygrad.llm import prefill_routes


class _Linear: pass


def _attachment(candidate_id="selected-structural-candidate"):
  return PrefillRouteAttachment("blk.0.attn_q", "route.structural", "blk.0.attn_q.weight",
                                {"candidate_id": candidate_id, "routes": {"blk.0.attn_q": "route.structural"}},
                                {"backend": "CPU"})


def test_candidate_workspace_identity_comes_only_from_selected_attachment():
  lin = _Linear()
  lin.name, lin.model_name, lin.shape_tier, lin.gpu, lin.vram = "candidate-looking-name", "model", "large", "GPU", "24GB"
  value = Tensor.empty(4, device="CPU")
  assert prefill_routes._attached_candidate_id(lin) is None
  assert prefill_routes._candidate_workspace_if_attached(value, lin) is value
  assert memory_semantic_owner(value) is None

  lin._prefill_route_attachment = _attachment()
  marked = prefill_routes._candidate_workspace_if_attached(Tensor.empty(4, device="CPU"), lin)
  owner = memory_semantic_owner(marked)
  assert owner is not None and owner.semantic_class is MemorySemanticClass.CANDIDATE_WORKSPACE
  assert owner.candidate_id == "selected-structural-candidate"


def test_malformed_or_empty_attachment_identity_does_not_create_workspace():
  for policy in ({"candidate_id": ""}, {"candidate_id": 7}, None):
    lin = _Linear()
    lin._prefill_route_attachment = PrefillRouteAttachment("i", "r", "t.weight", policy, {})
    value = prefill_routes._candidate_workspace_if_attached(Tensor.empty(1, device="CPU"), lin)
    assert memory_semantic_owner(value) is None


def test_direct_route_materialized_input_is_prefill_activation(monkeypatch):
  class Lin:
    bias, out_features, in_features, name = None, 3, 4, "attn_q"
    q4k_storage = object()
    def prefill_packed_weight(self): return Tensor.empty(1, device="CPU")

  observed = {}
  class Candidate:
    def matches(self, lin, spec): return True
    def run(self, lin, x, x_batch, spec):
      observed["owner"] = memory_semantic_owner(x_batch)
      return Tensor.zeros(1, spec.m, spec.n, device="CPU")

  monkeypatch.setattr(prefill_routes, "DIRECT_PACKED_PREFILL_CANDIDATES", (Candidate(),))
  lin = Lin()
  lin._prefill_route_attachment = PrefillRouteAttachment(
    "invocation", "direct-packed-baseline", "attn_q.weight",
    {"candidate_id": "direct-packed-baseline", "strategy": "DIRECT_PACKED_FALLBACK"}, {"backend": "CPU"})
  lin._prefill_graph_role = "attn_qo"
  lin._prefill_direct_packed_binding = PrefillDirectPackedBinding("invocation", "prefill", "attn_qo", (2, 3, 4))
  with prefill_route_scope(): out = prefill_routes.route_direct_packed_prefill(lin, Tensor.zeros(1, 2, 4, device="CPU"))
  assert out is not None and observed["owner"] == PREFILL_ACTIVATION


def test_every_local_tensor_empty_is_explicitly_marked_and_fixed_partials_are_scratch():
  path = Path(prefill_routes.__file__)
  source = path.read_text()
  tree = ast.parse(source)
  empty_lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and
                 isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
                 node.func.value.id == "Tensor" and node.func.attr == "empty"]
  assert len(empty_lines) == 4
  lines = source.splitlines()
  assert all("prefill_scratch(Tensor.empty" in lines[line - 1] or "prefill_output(Tensor.empty" in lines[line - 1]
             for line in empty_lines)
  assert source.count("partials = prefill_scratch(Tensor.empty") == 2


def test_only_unmarked_local_contiguous_is_cached_fused_weight_storage():
  source = Path(prefill_routes.__file__).read_text()
  tree = ast.parse(source)
  lines = source.splitlines()
  contiguous_lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and
                      isinstance(node.func, ast.Attribute) and node.func.attr == "contiguous"]
  unmarked = [line for line in contiguous_lines if "prefill_activation(" not in lines[line - 1]]
  assert len(unmarked) == 1
  assert "fused_words =" in lines[unmarked[0] - 1] and "_prefill_fused_gate_up_words" in source
