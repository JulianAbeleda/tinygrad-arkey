import pytest

from extra.qk.prefill.attn_qo_executable_preparation import compile_attn_qo_program


def test_exact_direct_attn_qo_compiles_to_real_program_and_passes_evidence():
  prepared = compile_attn_qo_program()
  assert prepared["transport"] == "direct_l2"
  assert prepared["compile_evidence"]["passed"] is True
  assert prepared["compile_evidence"]["capture"]["dispatch_permitted"] is False
  assert prepared["dispatch_performed"] is False


def test_exact_lds_preparation_does_not_fake_an_equivalent_program():
  with pytest.raises(NotImplementedError, match="LDS executable binding"):
    compile_attn_qo_program(transport="lds")
