from extra.llm_research.decode.q6k_nv_int_mma_substrate_gate import audit


def test_sm120_int_mma_exists_but_tinygrad_generic_substrate_is_missing():
  got=audit()
  assert got["hardware_compiler"]["pass"]
  assert got["hardware_compiler"]["fragment_registers"] == {"a_b32":4,"b_b32":2,"acc_s32":4}
  assert not got["tinygrad_descriptor"]["cuda_descriptor_present"]
  assert not got["tinygrad_descriptor"]["ptx_renderer_admits"]
  assert not got["tinygrad_renderer"]["pass"]
  assert got["blocker_localized_to_tinygrad_substrate"]
