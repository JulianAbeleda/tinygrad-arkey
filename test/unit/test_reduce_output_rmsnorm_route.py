import json

from tinygrad.llm.model_route_plan import (decode_reduce_output_rmsnorm_promoted,
  decode_qk_norm_rope_promoted, load_decode_qk_norm_rope_promotion,
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

def test_shipped_qk_norm_rope_policy_pins_accepted_promotion():
  assert decode_qk_norm_rope_promoted(("NV", "sm_120"))
  assert not decode_qk_norm_rope_promoted(("AMD", "gfx1100"))
  assert not decode_qk_norm_rope_promoted(("CUDA", "sm_120"))

def test_qk_norm_rope_policy_loader_is_closed_and_target_exact(tmp_path):
  p=tmp_path/"policy.json"
  p.write_text(json.dumps({"schema":"boltbeam.route_policy.v1","route":"decode_qk_norm_rope",
                           "promoted_targets":[{"backend":"NV","architecture":"sm_120"}]}))
  assert load_decode_qk_norm_rope_promotion(str(p)) == frozenset({("NV","sm_120")})
  p.write_text(json.dumps({"schema":"boltbeam.route_policy.v1","route":"decode_qk_norm_rope"}))
  assert load_decode_qk_norm_rope_promotion(str(p)) == frozenset()
