import pytest

from extra.qk.mmq_mfma_lowering import adapt_mfma_evidence


def evidence():
  return {"target": {"backend": "AMD", "arch": "gfx942", "wave_size": 64},
          "lowering": {"op": "WMMA", "shape": [16, 16, 16], "input_dtype": "half", "accumulator_dtype": "float"},
          "operands": {"a_fragment": [4], "b_fragment": [4], "accumulator": [4]},
          "instruction_evidence": "call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16.f16(<4 x half>, <4 x half>, <4 x float>)"}


def test_cdna_contract_captures_exact_operand_abi_and_instruction():
  got = adapt_mfma_evidence(evidence())
  assert got.to_dict()["schema"] == "tinygrad.mmq.mfma_lowering.v1"
  assert got.wave_size == 64 and got.a_fragment == got.b_fragment == got.accumulator == (4,)


@pytest.mark.parametrize("change", [
  lambda e: e["lowering"].update(shape=[16, 16, 32]),
  lambda e: e["operands"].update(a_fragment=[16]),
  lambda e: e.update(instruction_evidence="v_add_f32"),
])
def test_missing_or_mismatched_compiler_capability_fails_closed(change):
  payload = evidence(); change(payload)
  with pytest.raises(ValueError): adapt_mfma_evidence(payload)
