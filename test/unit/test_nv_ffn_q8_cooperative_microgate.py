from extra.llm_research.decode.nv_ffn_q8_cooperative_microgate import NUMERICAL_CONTRACT, TOPOLOGY

def test_microgate_is_two_stage_candidate_and_declares_non_bitwise_numerics():
  assert TOPOLOGY["strictly_smaller"]
  assert TOPOLOGY["candidate_programs"] == 2 < TOPOLOGY["control_programs_min"]
  assert NUMERICAL_CONTRACT["kind"] == "shared_q8_semantic_not_bitwise"
