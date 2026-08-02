"""L1 P4 M1 gate tests (l1-decode-plumbing-fusion-design-20260802.md section 5):
the decode epilogue-fusion promotion record is CLOSED by default, the loader never
infers promotion from a target string, the checked-in record promotes nothing until
the first fused consumer lands, and the fused answers ride on the existing QK and
flash admissions without changing their legacy `admitted` routes."""
import json, pathlib

from tinygrad.llm.model_route_plan import (decode_epilogue_fusion_promoted, load_decode_epilogue_fusion_promotion,
  _DECODE_EPILOGUE_FUSION_PROMOTED_TARGETS)
from tinygrad.llm.flash_decode_attention import FlashDecodeAdmission, FlashDecodeCapability, FlashDecodeRouteConfig
from tinygrad.llm.qk_primitives import QKPrimitiveCapability, QKPrimitiveRouteAdmission


def _write_policy(path, *, targets="absent"):
  doc = {"schema": "boltbeam.route_policy.v1", "route": "decode_epilogue_fusion"}
  if targets != "absent": doc["promoted_targets"] = targets
  pathlib.Path(path).write_text(json.dumps(doc))
  return path


def test_closed_default_when_no_promoted_targets_key(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets="absent")
  assert load_decode_epilogue_fusion_promotion(p) == frozenset()


def test_closed_default_when_promoted_targets_empty(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[])
  assert load_decode_epilogue_fusion_promotion(p) == frozenset()


def test_loader_names_explicit_targets_only(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[{"backend": "NV", "architecture": "sm_120"}])
  promoted = load_decode_epilogue_fusion_promotion(p)
  assert ("NV", "sm_120") in promoted
  assert ("AMD", "gfx1100") not in promoted


def test_checked_in_record_is_closed_at_m1():
  # M1 promotes nothing: the NV entry lands in the same commit as the first fused consumer (M2).
  assert _DECODE_EPILOGUE_FUSION_PROMOTED_TARGETS == frozenset()
  assert not decode_epilogue_fusion_promoted(("NV", "sm_120"))
  assert not decode_epilogue_fusion_promoted(("AMD", "gfx1100"))
  assert not decode_epilogue_fusion_promoted((None, None))


def test_qk_admission_fused_answer_does_not_change_legacy_admitted():
  cap = QKPrimitiveCapability(backend="NV", architecture="sm_120", wave_size=32, supports_warp_shfl_xor=True)
  open_adm = QKPrimitiveRouteAdmission(cap, True, epilogue_fusion_promoted=True)
  assert open_adm.admitted and open_adm.fusion_admitted
  closed_adm = QKPrimitiveRouteAdmission(cap, True)  # default flag False
  assert closed_adm.admitted and not closed_adm.fusion_admitted
  not_promoted = QKPrimitiveRouteAdmission(cap, False, epilogue_fusion_promoted=True)
  assert not not_promoted.admitted and not not_promoted.fusion_admitted
  no_cap = QKPrimitiveRouteAdmission(QKPrimitiveCapability(), True, epilogue_fusion_promoted=True)
  assert not no_cap.admitted and not no_cap.fusion_admitted


def test_flash_admission_fused_answer_does_not_change_legacy_admitted():
  cap = FlashDecodeCapability(supports_warp_shfl_xor=True, supports_fdot2=True)
  open_adm = FlashDecodeAdmission(True, cap, True, epilogue_fusion_promoted=True)
  assert open_adm.admitted and open_adm.fusion_admitted
  closed_adm = FlashDecodeAdmission(True, cap, True)
  assert closed_adm.admitted and not closed_adm.fusion_admitted
  shape_bad = FlashDecodeAdmission(False, cap, True, epilogue_fusion_promoted=True)
  assert not shape_bad.admitted and not shape_bad.fusion_admitted


def test_flash_route_evaluate_defaults_to_closed_fusion(tmp_path):
  cfg = FlashDecodeRouteConfig("c", "r", 32, 48, None, 1)
  cap = FlashDecodeCapability(supports_warp_shfl_xor=True, supports_fdot2=True)
  adm = cfg.evaluate(1, 32, 8, 128, cap, True)
  assert adm.admitted and not adm.fusion_admitted
  adm_open = cfg.evaluate(1, 32, 8, 128, cap, True, epilogue_fusion_promoted=True)
  assert adm_open.admitted and adm_open.fusion_admitted
