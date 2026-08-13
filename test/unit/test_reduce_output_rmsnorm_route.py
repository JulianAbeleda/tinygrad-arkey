import json

from tinygrad.llm.model_route_plan import (decode_reduce_output_rmsnorm_promoted,
  load_decode_reduce_output_rmsnorm_promotion)

def test_shipped_reduce_output_policy_pins_current_promotion():
  """The fp32 q/k route is promoted at HEAD for NV sm_120 only (a8b560457);
  the site-absorption P1 work must not change that promotion (no policy
  promotion until the GPU A/B clears the +50 us bar).  Pin the exact targets
  so a policy edit fails here before any CPU-side landing."""
  assert decode_reduce_output_rmsnorm_promoted(("NV", "sm_120"))
  assert not decode_reduce_output_rmsnorm_promoted(("AMD", "gfx1100"))
  assert not decode_reduce_output_rmsnorm_promoted(("CPU", ""))

def test_policy_loader_is_target_exact(tmp_path):
  p=tmp_path/"policy.json"
  p.write_text(json.dumps({"schema":"boltbeam.route_policy.v1","route":"decode_reduce_output_rmsnorm",
                           "promoted_targets":[{"backend":"NV","architecture":"sm_120"}]}))
  got=load_decode_reduce_output_rmsnorm_promotion(str(p))
  assert got == frozenset({("NV","sm_120")})
