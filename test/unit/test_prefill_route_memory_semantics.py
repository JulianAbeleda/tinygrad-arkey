import ast
from pathlib import Path

from tinygrad import Tensor
from tinygrad.llm.memory_semantics import PREFILL_ACTIVATION, memory_semantic_owner
from tinygrad.llm import prefill_routes


def test_packed_wmma_materialized_input_is_prefill_activation(monkeypatch):
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

  monkeypatch.setattr(prefill_routes, "packed_wmma_prefill_enabled", lambda: True)
  monkeypatch.setattr(prefill_routes, "_attached_packed_wmma_spec",
                      lambda *_: prefill_routes.PrefillLinearRouteSpec("packed_wmma", "q4k", "attn_qo", 2, 3, 4))
  monkeypatch.setattr(prefill_routes, "select_packed_wmma_prefill_candidate", lambda *_: Candidate())
  out = prefill_routes.route_packed_wmma_prefill(Lin(), Tensor.zeros(1, 2, 4, device="CPU"))
  assert out is not None and observed["owner"] == PREFILL_ACTIVATION


def test_prefill_routes_owns_no_custom_kernel_output_or_partials_allocations():
  path = Path(prefill_routes.__file__)
  source = path.read_text()
  tree = ast.parse(source)
  empty_lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and
                 isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
                 node.func.value.id == "Tensor" and node.func.attr == "empty"]
  assert empty_lines == []
  assert "custom_kernel" not in source


def test_every_local_contiguous_materialization_is_a_prefill_activation():
  source = Path(prefill_routes.__file__).read_text()
  tree = ast.parse(source)
  lines = source.splitlines()
  contiguous_lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and
                      isinstance(node.func, ast.Attribute) and node.func.attr == "contiguous"]
  unmarked = [line for line in contiguous_lines if "prefill_activation(" not in lines[line - 1]]
  assert contiguous_lines and unmarked == []
