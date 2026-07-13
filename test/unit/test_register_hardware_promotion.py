from extra.qk.prefill.register_hardware_promotion import (ENABLE_VALUE, EXACT_SHAPE, STAGES, TARGET,
  TOLERANCES, advance, prepare_authorization, promotion_plan)
from test.unit.test_pure_register_evaluation_gate import BINARY, IDENTITY, _compile, _runtime_binding


def _authorization(enable_value=ENABLE_VALUE):
  artifact = _compile(runtime_binding=_runtime_binding(),
    capture={"mode": "compile_only", "dispatch_permitted": False,
             "resource_authority": "final_code_object_descriptor", "allocator_authority": "final_regalloc"},
    target_evidence={"authority": "final_program", "target": "gfx1100", "abi": "amdgpu_kernel"},
    instruction_order_proof={"schema": "prefill-pure-register-instruction-order.v1",
                             "authority": "final_disassembly", "passed": True,
                             "disassembly_sha256": "d" * 64})
  return prepare_authorization({"canonical_identity": IDENTITY}, artifact,
                               profile=_runtime_binding()["profile"], enable_value=enable_value)


def _observation(index):
  stage = STAGES[index]
  return {"stage": stage["name"], "shape": list(stage["shape"]),
          "canonical_identity": IDENTITY, "binary_sha256": BINARY,
          "device_healthy_before": True, "device_healthy_after": True, "device_fault": False,
          "guards_intact": True, "inputs_unchanged": True, "numerics_passed": True,
          "full_output_compared": True, "nonconstant_inputs": True, "elapsed_seconds": 0.01,
          "rtol": TOLERANCES["rtol"], "atol": TOLERANCES["atol"]}


def test_plan_is_default_off_and_contains_no_dispatch_implementation():
  plan = promotion_plan()
  assert plan["enabled_by_default"] is False and plan["dispatch_implemented"] is False
  assert plan["stages"][-1]["shape"] == list(EXACT_SHAPE)
  assert plan["guards"]["prefix_bytes"] > 0 and plan["guards"]["suffix_bytes"] > 0
  assert plan["safety"]["revoke_on_timeout"] is True and plan["target"] == TARGET


def test_authorization_requires_explicit_opt_in_and_exact_compile_resources():
  assert _authorization(enable_value=None)["passed"] is False
  assert _authorization()["passed"] is True
  bad = _compile(runtime_binding={**_runtime_binding(), "shape": {"m": 1, "n": 128, "k": 128}})
  row = prepare_authorization({"canonical_identity": IDENTITY}, bad,
    profile=_runtime_binding()["profile"], enable_value=ENABLE_VALUE)
  assert row["passed"] is False and any("exact workload/target match" in error for error in row["errors"])


def test_progression_is_ordered_and_exact_shape_requires_every_canary():
  state = advance(_authorization(), [_observation(0), _observation(1), _observation(2)])
  assert state["passed"] is True and state["next_stage"] == "exact" and state["exact_shape_passed"] is False
  done = advance(_authorization(), [_observation(i) for i in range(len(STAGES))])
  assert done["passed"] is True and done["next_stage"] is None and done["exact_shape_passed"] is True


def test_fault_timeout_guard_or_numerical_failure_revokes():
  for mutation in ({"device_fault": True}, {"elapsed_seconds": 11}, {"guards_intact": False}, {"numerics_passed": False}):
    observation = _observation(0) | mutation
    state = advance(_authorization(), [observation])
    assert state["passed"] is False and state["revoked"] is True and state["next_stage"] is None


def test_identity_mismatch_and_skipped_canary_revoke_without_dispatch():
  mismatched = _observation(0) | {"binary_sha256": "c" * 64}
  assert advance(_authorization(), [mismatched])["revoked"] is True
  skipped = _observation(1)
  state = advance(_authorization(), [skipped])
  assert state["revoked"] is True and state["dispatch_performed"] is False


def test_advance_rejects_forged_or_non_dispatch_free_authorization():
  forged = _authorization() | {"compile_resource_evidence": {"passed": True}}
  state = advance(forged, [])
  assert state["revoked"] is True and state["next_stage"] is None
  dispatch_capable = _authorization() | {"dispatch_performed": True}
  assert advance(dispatch_capable, []) ["revoked"] is True


def test_nonfinite_timing_is_not_a_passing_observation():
  for elapsed in (float("nan"), float("inf"), float("-inf")):
    state = advance(_authorization(), [_observation(0) | {"elapsed_seconds": elapsed}])
    assert state["revoked"] is True


def test_invalid_hashes_cannot_authorize():
  artifact = _compile(canonical_identity="g" * 64)
  row = prepare_authorization({"canonical_identity": IDENTITY}, artifact,
    profile=_runtime_binding()["profile"], enable_value=ENABLE_VALUE)
  assert row["passed"] is False
