import ast, pathlib

P=pathlib.Path(__file__).parents[2]/"scratchpad/cuda_decode_q4k_gate_up_llama_graph_ab.py"

def test_gate_up_ab_script_is_diagnostic_and_full_family():
  src=P.read_text(); tree=ast.parse(src)
  assert 'expected_population=36' in src
  assert 'names[i:i+4]==[GATE,GATE,SILU,MULCAST]' in src
  assert 'mapped!=LlamaGateUpGraph.expected_population' in src
  assert 'no production/default route change' in src
  assert any(isinstance(n,ast.ClassDef) and n.name=='LlamaGateUpGraph' for n in ast.walk(tree))

def test_gate_up_ab_preserves_exact_consumer_boundary():
  src=P.read_text()
  assert 'device_pointer(dst_h)' in src
  assert 'cuGraphDestroyNode' in src
  assert 'cuGraphAddDependencies' in src
  assert 'graph_nodes_before' in src and 'graph_nodes_after' in src
