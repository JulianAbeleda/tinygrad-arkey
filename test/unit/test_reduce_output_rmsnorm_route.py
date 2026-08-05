import json

from tinygrad.llm.model_route_plan import (decode_reduce_output_rmsnorm_promoted,
  load_decode_reduce_output_rmsnorm_promotion)

def test_shipped_reduce_output_policy_is_closed():
  assert not decode_reduce_output_rmsnorm_promoted(("NV", "sm_120"))
  assert not decode_reduce_output_rmsnorm_promoted(("AMD", "gfx1100"))

def test_policy_loader_is_target_exact(tmp_path):
  p=tmp_path/"policy.json"
  p.write_text(json.dumps({"schema":"boltbeam.route_policy.v1","route":"decode_reduce_output_rmsnorm",
                           "promoted_targets":[{"backend":"NV","architecture":"sm_120"}]}))
  got=load_decode_reduce_output_rmsnorm_promotion(str(p))
  assert got == frozenset({("NV","sm_120")})
