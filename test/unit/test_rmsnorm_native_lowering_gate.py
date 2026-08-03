"""Path 3 semantic RMSNorm native-lowering gate tests
(path3-semantic-rmsnorm-task-20260802.md): the record is CLOSED for every target, independently
of the M2 decode-epilogue record that stays promoted on NV sm_120, and independently of the
M3 opaque norm, M4 q4k, and M5 combine records."""
import json, pathlib

from tinygrad.llm.model_route_plan import (decode_epilogue_fusion_promoted, decode_norm_fusion_promoted,
  decode_q4k_epilogue_fusion_promoted, decode_rmsnorm_native_lowering_promoted,
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


def test_checked_in_rmsnorm_native_record_promotes_nothing_and_m2_record_stays():
  # Path 3 is closed-default (no measured win yet); M2's Q6K in-kernel merge keeps its NV
  # sm_120 opt-in; M3 opaque norm, M4 q4k, and M5 combine records stay closed.
  assert _DECODE_RMSNORM_NATIVE_LOWERING_PROMOTED_TARGETS == frozenset()
  assert not decode_rmsnorm_native_lowering_promoted(("NV", "sm_120"))
  assert not decode_rmsnorm_native_lowering_promoted(("AMD", "gfx1100"))
  assert not decode_rmsnorm_native_lowering_promoted((None, None))
  assert decode_epilogue_fusion_promoted(("NV", "sm_120"))
  assert not decode_norm_fusion_promoted(("NV", "sm_120"))
  assert not decode_q4k_epilogue_fusion_promoted(("NV", "sm_120"))
