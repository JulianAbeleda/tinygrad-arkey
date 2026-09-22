from extra.llm_research.decode.q6k_nv_int_mma_substrate_gate import audit


def test_sm120_int_mma_qualified_lowering_is_ready_but_generic_admission_stays_closed():
  got=audit()
  assert got["schema"] == "tinygrad.q6k_nv_int_mma_substrate_gate.v2"
  assert got["hardware_compiler"]["pass"]
  assert got["hardware_compiler"]["fragment_registers"] == {"a_b32":4,"b_b32":2,"acc_s32":4}
  assert not got["tinygrad_descriptor"]["cuda_descriptor_present"]
  assert not got["tinygrad_descriptor"]["ptx_renderer_admits"]
  assert got["tinygrad_renderer"]["pass"]
  assert got["generic_descriptor_remains_fail_closed"]
  assert got["qualified_candidate_substrate_ready"]
  assert not got["blocker_localized_to_tinygrad_substrate"]
