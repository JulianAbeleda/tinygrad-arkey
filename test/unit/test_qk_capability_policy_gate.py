"""TG3: split the Q4_K/Q6_K quant gate into capability and policy (docs/task_workflow/input/
target-capability-policy-decoupling-scope-20260730.md). Pins that the two questions are independently
decidable with distinct, observable census reasons, that an unreported wave_size is never treated as 32, and
that AMD's admitted set is unchanged.

No AMD hardware is available on this machine (scope section 8): AMD is verified structurally, by feeding
faked DeviceFacts (real HIPRenderer facts values, not renderer instances) through the production install
path, the same pattern test_target_capability_facts.py and test_amd_allocator_granularity.py already use.
"""
import json

from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
from tinygrad.llm.model_route_plan import ModelRoutePlan, build_model_route_plan, load_qk_target_promotion
from tinygrad.llm.qk_primitives import (
  Q4KPrimitiveLinear, Q6KPrimitiveLinear, QKPrimitiveCapability, QKPrimitiveRouteAdmission,
  _install_q4k_primitives, _install_q6k_primitives, qk_primitive_capability_from_device_facts,
)


def _facts(*, backend, architecture, wave_size, supports_warp_shfl_xor):
  probe = ProbeRecord("test", "2026-07-15T00:00:00+00:00")
  return DeviceFacts(f"{backend}:0", backend, architecture, None, None,
                     DeviceCapabilities(wave_size=wave_size, supports_warp_shfl_xor=supports_warp_shfl_xor), probe, probe)


def _amd_facts(): return _facts(backend="AMD", architecture="gfx1100", wave_size=32, supports_warp_shfl_xor=True)
def _metal_facts(): return _facts(backend="METAL", architecture="Apple9", wave_size=None, supports_warp_shfl_xor=True)


def _q4_model_and_meta(tmp_path, name="blk.0.ffn_gate.weight"):
  import types
  from tinygrad import Tensor, dtypes
  gguf = tmp_path / "q4.bin"
  gguf.write_bytes(bytes((256 * 256) // 256 * 144))
  meta = {"data_start": 0, "tensor_infos": [(name, (256, 256), 12, 0)]}
  model = types.SimpleNamespace(blk=[types.SimpleNamespace(ffn_gate=types.SimpleNamespace(
    weight=Tensor.empty(256, 256, dtype=dtypes.float16), bias=None))])
  return model, gguf, meta


# ---- Fact: unreported wave_size is never treated as 32 (scope section 3.3) --------------------------------

def test_capability_requires_strict_wave_size_equality_not_truthiness():
  assert QKPrimitiveCapability("AMD", "gfx1100", 32, True).satisfied
  assert not QKPrimitiveCapability("METAL", "Apple9", None, True).satisfied   # unreported -- not treated as 32
  assert not QKPrimitiveCapability("AMD", "gfx942", 64, True).satisfied       # a real, different, reported width
  assert not QKPrimitiveCapability("AMD", "gfx1100", 32, False).satisfied     # wave32 alone is not enough
  assert not QKPrimitiveCapability("AMD", "gfx1100", 32, None).satisfied      # unreported shuffle support


def test_capability_from_device_facts_never_infers_from_backend_string():
  """The AMD-shaped backend/architecture strings alone must not satisfy capability -- only the actual
  wave_size/supports_warp_shfl_xor facts do (scope: capability must be read from TG2's facts, never inferred
  from a target string)."""
  facts = _facts(backend="AMD", architecture="gfx1100", wave_size=None, supports_warp_shfl_xor=True)
  assert not qk_primitive_capability_from_device_facts(facts).satisfied


# ---- Fact: capability and policy are independently decidable, with distinct census reasons -----------------

def test_capability_ok_but_not_promoted_yields_no_binding_with_its_own_reason(tmp_path, capsys):
  model, gguf, meta = _q4_model_and_meta(tmp_path)
  plan = build_model_route_plan(meta, promoted_targets=frozenset({("NV", "sm_80")}))  # explicit record, excludes AMD
  installed = _install_q4k_primitives(model, gguf, meta, route_plan=plan, device_facts=_amd_facts(), debug=True)
  assert installed == []
  out = capsys.readouterr().out
  assert "policy_target_not_promoted=1" in out
  assert "capability_missing" not in out


def test_promoted_but_capability_missing_yields_no_binding_with_a_different_reason(tmp_path, capsys):
  model, gguf, meta = _q4_model_and_meta(tmp_path)
  plan = build_model_route_plan(meta, promoted_targets=frozenset({("METAL", "Apple9")}))  # explicitly promotes Metal
  installed = _install_q4k_primitives(model, gguf, meta, route_plan=plan, device_facts=_metal_facts(), debug=True)
  assert installed == []
  out = capsys.readouterr().out
  assert "capability_missing=1" in out
  assert "policy_target_not_promoted" not in out


def test_admitted_when_both_capability_and_promotion_hold(tmp_path):
  """Proves the policy channel can affirmatively admit a non-AMD target once a promotion record names it --
  this is the exact hook TG4 (Metal promotion evidence) will populate; TG3 only wires the mechanism."""
  model, gguf, meta = _q4_model_and_meta(tmp_path)
  # capability satisfied by construction (a future target that DOES report wave_size==32 and shuffle support),
  # and explicitly promoted for that exact target.
  future_facts = _facts(backend="METAL", architecture="Apple9", wave_size=32, supports_warp_shfl_xor=True)
  plan = build_model_route_plan(meta, promoted_targets=frozenset({("METAL", "Apple9")}))
  installed = _install_q4k_primitives(model, gguf, meta, route_plan=plan, device_facts=future_facts)
  assert len(installed) == 1
  assert installed[0].route_admission.admitted


# ---- Today's real Metal facts: capability alone already fails closed, with a recorded reason ---------------

def test_real_metal_facts_today_fail_closed_on_capability_with_no_promotion_record_loaded(tmp_path, capsys):
  """No explicit promotion record is loaded in production yet (TG4 is a separate BoltBeam packet) --
  `promoted_targets` defaults to None (undecided-by-target, not denied). Today's real Metal facts
  (wave_size=None, unreported) mean capability alone already yields no binding, with an observable reason --
  never the pre-TG3 silence where _install_q4k_primitives was never even called."""
  model, gguf, meta = _q4_model_and_meta(tmp_path)
  plan = build_model_route_plan(meta)  # no promotion record loaded -> promoted_targets is None
  assert plan.promoted_targets is None
  installed = _install_q4k_primitives(model, gguf, meta, route_plan=plan, device_facts=_metal_facts(), debug=True)
  assert installed == []
  out = capsys.readouterr().out
  assert "capability_missing=1" in out
  assert "installed=0" in out


# ---- AMD structural non-regression: the admitted set is unchanged ------------------------------------------

def test_amd_admitted_set_is_unchanged_q4k_and_q6k(tmp_path):
  """Structural-only (scope section 8: no AMD hardware here). With real AMD gfx1100 wave32 facts and no
  promotion record loaded, every tensor the pre-TG3 gate would have admitted (i.e. every tensor the static
  route-plan default already selects by shape/role) is still admitted -- capability and policy both hold, so
  the split changes nothing about which tensors become primitives."""
  import types
  from tinygrad import Tensor, dtypes
  q4_gguf, q6_gguf = tmp_path / "q4.bin", tmp_path / "q6.bin"
  q4_gguf.write_bytes(bytes((256 * 256) // 256 * 144))
  q6_gguf.write_bytes(bytes((256 * 256) // 256 * 210))
  q4_meta = {"data_start": 0, "tensor_infos": [("blk.0.ffn_gate.weight", (256, 256), 12, 0)]}
  q6_meta = {"data_start": 0, "tensor_infos": [("blk.0.ffn_down.weight", (256, 256), 14, 0)]}
  model = types.SimpleNamespace(blk=[types.SimpleNamespace(
    ffn_gate=types.SimpleNamespace(weight=Tensor.empty(256, 256, dtype=dtypes.float16), bias=None),
    ffn_down=types.SimpleNamespace(weight=Tensor.empty(256, 256, dtype=dtypes.float16), bias=None))])

  amd = _amd_facts()
  q4_installed = _install_q4k_primitives(model, q4_gguf, q4_meta, route_plan=build_model_route_plan(q4_meta), device_facts=amd)
  q6_installed = _install_q6k_primitives(model, q6_gguf, q6_meta, route_plan=build_model_route_plan(q6_meta), device_facts=amd)

  assert [type(x) for x in q4_installed] == [Q4KPrimitiveLinear]
  assert [type(x) for x in q6_installed] == [Q6KPrimitiveLinear]
  assert q4_installed[0].route_admission.admitted and q6_installed[0].route_admission.admitted
  assert (q4_installed[0].name, q4_installed[0].parts) == ("blk.0.ffn_gate.weight", 1)
  assert (q6_installed[0].name, q6_installed[0].parts) == ("blk.0.ffn_down.weight", 1)


def test_model_route_plan_target_promoted_default_and_explicit_record():
  plan_no_record = ModelRoutePlan()
  assert plan_no_record.target_promoted(("METAL", "Apple9"))     # undecided-by-target -> admitted
  assert plan_no_record.target_promoted(("AMD", "gfx1100"))

  plan_with_record = ModelRoutePlan(promoted_targets=frozenset({("AMD", "gfx1100")}))
  assert plan_with_record.target_promoted(("AMD", "gfx1100"))
  assert not plan_with_record.target_promoted(("METAL", "Apple9"))

  plan_promotes_nothing = ModelRoutePlan(promoted_targets=frozenset())
  assert not plan_promotes_nothing.target_promoted(("AMD", "gfx1100"))


def test_load_qk_target_promotion_is_the_explicit_path_parser(tmp_path):
  """Target-promotion facts reach production through exactly one channel: an explicit JSON path on the same
  boltbeam.route_policy.v1 schema load_qk_route_policy already validates -- never a direct import of
  extra/llm_research/route_manifest.py (the Boundary Rule)."""
  path = tmp_path / "promotion.json"
  path.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "promoted_targets": [
    {"backend": "AMD", "architecture": "gfx1100"}, {"backend": "METAL", "architecture": "Apple9"},
  ]}))
  promoted = load_qk_target_promotion(str(path))
  assert promoted == frozenset({("AMD", "gfx1100"), ("METAL", "Apple9")})

  no_record_path = tmp_path / "no_record.json"
  no_record_path.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "routes": []}))
  assert load_qk_target_promotion(str(no_record_path)) is None


def test_route_admission_admitted_requires_both_capability_and_promotion():
  capability = QKPrimitiveCapability("AMD", "gfx1100", 32, True)
  assert QKPrimitiveRouteAdmission(capability, True).admitted
  assert not QKPrimitiveRouteAdmission(capability, False).admitted
  assert not QKPrimitiveRouteAdmission(QKPrimitiveCapability(), True).admitted
