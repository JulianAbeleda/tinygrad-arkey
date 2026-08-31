from pathlib import Path

def test_streamk_gate_uses_compiler_record_and_exact_local_geometry():
  source=Path("extra/llm_research/prefill/nv_compiler_q4k_streamk_gate.py").read_text()
  assert 'local_size=(32,2,4)' in source
  assert 'compiler.producer' in source and 'fixup.fixup_program' in source
  assert 'np.allclose(got,ref,rtol=2e-5,atol=2e-3)' in source
