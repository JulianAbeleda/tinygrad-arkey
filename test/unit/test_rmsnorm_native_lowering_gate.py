"""Path 3 semantic RMSNorm native-lowering target and selective-site gate tests."""
import json, pathlib

from tinygrad.llm.model_route_plan import (decode_epilogue_fusion_promoted, decode_norm_fusion_promoted,
  decode_q4k_epilogue_fusion_promoted, decode_rmsnorm_native_lowering_promoted,
  decode_rmsnorm_native_lowering_site_promoted,
  load_decode_rmsnorm_native_lowering_promotion,
  _DECODE_RMSNORM_NATIVE_LOWERING_PROMOTED_TARGETS)


def _write_policy(path, *, targets="absent"):
  doc = {"schema": "boltbeam.route_policy.v1", "route": "decode_rmsnorm_native_lowering"}
  if targets != "absent": doc["promoted_targets"] = targets
  pathlib.Path(path).write_text(json.dumps(doc))
  return path


def test_closed_default_when_no_promoted_targets_key(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets="absent")
  assert load_decode_rmsnorm_native_lowering_promotion(p) == frozenset()


def test_closed_default_when_promoted_targets_empty(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[])
  assert load_decode_rmsnorm_native_lowering_promotion(p) == frozenset()


def test_loader_names_explicit_targets_only(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[{"backend": "NV", "architecture": "sm_120"}])
  assert load_decode_rmsnorm_native_lowering_promotion(p) == frozenset({("NV", "sm_120")})


def test_checked_in_rmsnorm_native_record_promotes_only_qualified_nv_sites_and_m2_record_stays():
  assert _DECODE_RMSNORM_NATIVE_LOWERING_PROMOTED_TARGETS == frozenset({("NV", "sm_120")})
  assert decode_rmsnorm_native_lowering_promoted(("NV", "sm_120"))
  assert not decode_rmsnorm_native_lowering_promoted(("AMD", "gfx1100"))
  assert not decode_rmsnorm_native_lowering_promoted((None, None))
  for site in ("attn_norm", "ffn_norm", "output_norm"):
    assert decode_rmsnorm_native_lowering_site_promoted(("NV", "sm_120"), site)
  for site in ("attn_q_norm", "attn_k_norm"):
    assert not decode_rmsnorm_native_lowering_site_promoted(("NV", "sm_120"), site)
  assert not decode_rmsnorm_native_lowering_site_promoted(("AMD", "gfx1100"), "attn_norm")
  assert decode_epilogue_fusion_promoted(("NV", "sm_120"))
  assert not decode_norm_fusion_promoted(("NV", "sm_120"))
  assert not decode_q4k_epilogue_fusion_promoted(("NV", "sm_120"))
