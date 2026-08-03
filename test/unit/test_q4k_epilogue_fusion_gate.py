"""L1 M4 q4k GEMV epilogue-fusion gate tests (m4-q4k-epilogue-measurement-record-20260802.md): the q4k
epilogue record is CLOSED for every target (measured non-landing), independently of the M2 decode-epilogue
record that stays promoted on NV sm_120 for the Q6K in-kernel merge, and independently of the M3 norm record."""
import json, pathlib

from tinygrad.llm.model_route_plan import (decode_epilogue_fusion_promoted, decode_norm_fusion_promoted,
  decode_q4k_epilogue_fusion_promoted, load_decode_q4k_epilogue_fusion_promotion,
  _DECODE_Q4K_EPILOGUE_FUSION_PROMOTED_TARGETS)


def _write_policy(path, *, targets="absent"):
  doc = {"schema": "boltbeam.route_policy.v1", "route": "decode_q4k_epilogue_fusion"}
  if targets != "absent": doc["promoted_targets"] = targets
  pathlib.Path(path).write_text(json.dumps(doc))
  return path


def test_closed_default_when_no_promoted_targets_key(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets="absent")
  assert load_decode_q4k_epilogue_fusion_promotion(p) == frozenset()


def test_closed_default_when_promoted_targets_empty(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[])
  assert load_decode_q4k_epilogue_fusion_promotion(p) == frozenset()


def test_loader_names_explicit_targets_only(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[{"backend": "NV", "architecture": "sm_120"}])
  assert load_decode_q4k_epilogue_fusion_promotion(p) == frozenset({("NV", "sm_120")})


def test_checked_in_q4k_record_promotes_nothing_and_m2_record_stays():
  # M4 measured non-landing (boundary copies regress the M2 baseline): the q4k epilogue record must
  # promote nothing, while the M2 decode-epilogue record keeps its measured NV sm_120 opt-in for the
  # Q6K in-kernel merge, and the M3 norm record stays closed.
  assert _DECODE_Q4K_EPILOGUE_FUSION_PROMOTED_TARGETS == frozenset()
  assert not decode_q4k_epilogue_fusion_promoted(("NV", "sm_120"))
  assert not decode_q4k_epilogue_fusion_promoted(("AMD", "gfx1100"))
  assert not decode_q4k_epilogue_fusion_promoted((None, None))
  assert decode_epilogue_fusion_promoted(("NV", "sm_120"))
  assert not decode_norm_fusion_promoted(("NV", "sm_120"))
