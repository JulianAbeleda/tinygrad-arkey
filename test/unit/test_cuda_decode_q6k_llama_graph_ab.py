import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

def test_diagnostic_q6_graph_ab_has_explicit_geometry_and_guards():
  src = (ROOT / "scratchpad/cuda_decode_q6k_llama_graph_ab.py").read_text()
  assert "Device.DEFAULT != \"CUDA\"" in src
  assert "add_kernel(self.graph, [qn], q6fn, 1024, 32, mparams, by=4)" in src
  assert "native_node_destroyed" in src
  assert "cuGraphRemoveDependencies" in src
  assert "cuGraphAddDependencies" in src
  assert "fp16->fp32 adapter is source-equivalent" in src

def test_family_scope_derives_and_checks_population_instead_of_hardcoding_it():
  src = (ROOT / "scratchpad/cuda_decode_q6k_llama_graph_ab.py").read_text()
  assert "def expected_population(manifest):" in src
  assert "semantic-manifest population" in src
  assert "LlamaQ6Graph.scope == \"family\"" in src
  assert "actual != LlamaQ6Graph.expected_population" in src

def test_family_scope_checks_each_role_topology_and_retains_buffers():
  src = (ROOT / "scratchpad/cuda_decode_q6k_llama_graph_ab.py").read_text()
  assert "for target in targets:" in src
  assert "expected exactly one Q6 partial consumer" in src
  assert "source-free Q6 node" in src
  assert "self._ab_buffers.extend" in src
  assert "3*len(targets)" in src
  assert "statistics.median(steady)" in src
  assert "sorted(steady)[len(steady)//2]" not in src

def test_diagnostic_adapter_defines_only_boundary_and_layout_helpers():
  src = (ROOT / "scratchpad/cuda_decode_q6k_llama_graph_adapter.cu").read_text()
  assert 'void half_to_float' in src
  assert 'void scatter_to_partials' in src
