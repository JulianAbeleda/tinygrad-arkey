import json
from pathlib import Path

from tinygrad.llm.model_route_plan import decode_q4k_gate_up_four_warp_vector_promoted
from tinygrad.llm.q4k_gate_up_four_warp_mmvq import Q4KGateUpFourWarpAdmission


def test_gate_up_four_warp_vector_policy_is_nv_sm120_only():
  assert decode_q4k_gate_up_four_warp_vector_promoted(("NV", "sm_120"))
  assert not decode_q4k_gate_up_four_warp_vector_promoted(("NV", "sm_89"))
  assert not decode_q4k_gate_up_four_warp_vector_promoted(("AMD", "gfx1100"))
  record = Path(__file__).parents[2] / "tinygrad/llm/generated/decode-q4k-gate-up-four-warp-vector-route-policy.json"
  assert json.loads(record.read_text())["schema"] == "boltbeam.route_policy.v1"


def test_gate_up_four_warp_vector_admission_is_explicit():
  assert not Q4KGateUpFourWarpAdmission(3).vector_loads
  assert Q4KGateUpFourWarpAdmission(3, vector_loads=True).vector_loads
