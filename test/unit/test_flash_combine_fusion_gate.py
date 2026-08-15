"""L1 M5 flash-decode combine fp16 absorption gate tests
(m5-flash-combine-normalization-measurement-record-20260802.md): the combine fusion record is CLOSED
for every target, independently of the M2 decode-epilogue record that stays promoted on NV sm_120 for
the Q6K in-kernel merge, and independently of the M3 norm and M4 q4k records."""
import json, pathlib

from tinygrad.llm.model_route_plan import (decode_epilogue_fusion_promoted, decode_flash_combine_fusion_promoted,
  decode_norm_fusion_promoted, decode_q4k_epilogue_fusion_promoted, load_decode_flash_combine_fusion_promotion,
  _DECODE_FLASH_COMBINE_FUSION_PROMOTED_TARGETS)


def _write_policy(path, *, targets="absent"):
  doc = {"schema": "boltbeam.route_policy.v1", "route": "decode_flash_combine_fusion"}
  if targets != "absent": doc["promoted_targets"] = targets
  pathlib.Path(path).write_text(json.dumps(doc))
  return path


def test_closed_default_when_no_promoted_targets_key(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets="absent")
  assert load_decode_flash_combine_fusion_promotion(p) == frozenset()


def test_closed_default_when_promoted_targets_empty(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[])
  assert load_decode_flash_combine_fusion_promotion(p) == frozenset()


def test_loader_names_explicit_targets_only(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[{"backend": "NV", "architecture": "sm_120"}])
  assert load_decode_flash_combine_fusion_promotion(p) == frozenset({("NV", "sm_120")})


def test_checked_in_combine_record_promotes_nothing_and_other_records_stay():
  # M5's fp16 combine variant is promoted for NV sm_120 by the M2d wall bracket; M2's Q6K
  # in-kernel merge keeps its own NV sm_120 opt-in; M3 norm and M4 q4k records stay closed.
  assert _DECODE_FLASH_COMBINE_FUSION_PROMOTED_TARGETS == frozenset({("NV", "sm_120")})
  assert decode_flash_combine_fusion_promoted(("NV", "sm_120"))
  assert not decode_flash_combine_fusion_promoted(("AMD", "gfx1100"))
  assert not decode_flash_combine_fusion_promoted((None, None))
  assert decode_epilogue_fusion_promoted(("NV", "sm_120"))
  assert not decode_norm_fusion_promoted(("NV", "sm_120"))
  assert not decode_q4k_epilogue_fusion_promoted(("NV", "sm_120"))
