from pathlib import Path

def test_streamk_gate_uses_compiler_record_and_exact_local_geometry():
  source=Path("extra/llm_research/prefill/nv_compiler_q4k_streamk_gate.py").read_text()
  assert 'local_size=(32,2,4)' in source
  assert 'compiler.producer' in source and 'active_fixup' in source
  assert 'np.allclose(got,ref,rtol=2e-5,atol=2e-3)' in source

def test_generated_plan_promotion_is_hash_bound_to_qualified_route():
  source=Path("extra/llm_research/generated_kernel_plan.py").read_text()
  assert 'transform_compiler_q4k_to_streamk(compiler_source,unroll=8)' in source
  assert 'bdf49e9bc6de56e843f39cc20f0ab087f4febef1b0d3d3c9992bfc01ab6fedbb' in source
  assert '9b3e400e942bf80f4f02184a6b4de0085ff0861ef5eb5bd4197e118a3ceb9acc' in source
  assert '"promotion_eligible":promotion_eligible' in source
