from tinygrad.llm.pp512_qk_norm_rope import PP512QKNormRopeAdmission, validate_pp512_qk_shapes


def test_pp512_qk_contract_is_exact_and_sequence_tiled():
  admission = PP512QKNormRopeAdmission()
  validate_pp512_qk_shapes(admission, (512, 32, 128), (512, 8, 128))
  assert admission.tiles("q") == 512
  assert admission.tiles("k") == 512


def test_pp512_qk_contract_rejects_decode_shapes():
  admission = PP512QKNormRopeAdmission()
  try:
    validate_pp512_qk_shapes(admission, (1, 32, 128), (1, 8, 128))
  except ValueError:
    pass
  else:
    raise AssertionError("decode-shaped tensors must not enter pp512 admission")
